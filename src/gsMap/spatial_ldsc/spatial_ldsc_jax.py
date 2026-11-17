"""
JAX-optimized implementation of spatial LDSC.
"""

import logging
import time
from pathlib import Path

import anndata as ad
import jax

from gsMap.config import SpatialLDSCConfig
from gsMap.config.fingerprint import warn_if_outputs_differ
from gsMap.logging_config import logged_stage, stage_logging
from gsMap.spatial_ldsc.data.loading import (
    FeatherAnnData,
    generate_expected_output_filename,
    load_common_resources,
    load_common_resources_direct_ldscore,
    load_marker_scores_memmap_format,
    log_existing_result_statistics,
)
from gsMap.spatial_ldsc.execution.engine import SpatialLDSCEngine

logger = logging.getLogger("gsMap.spatial_ldsc_jax")

# ============================================================================
# Main entry point
# ============================================================================


@logged_stage("spatial_ldsc")
def run_spatial_ldsc_jax(config: SpatialLDSCConfig):
    """
    Run spatial LDSC for all traits in config.sumstats_config_dict.
    """
    from gsMap.utils.jax_setup import use_float32_jax

    use_float32_jax()
    traits_to_process = list(config.sumstats_config_dict.items())
    if not traits_to_process:
        raise ValueError("No traits to process. config.sumstats_config_dict is empty.")

    # Create output directory
    output_dir = config.ldsc_save_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    marker_score_adata = None
    ldscore_adata = None
    trait_failures: list[tuple[str, Exception]] = []
    completed_traits: set[str] = set()

    try:
        try:
            if config.ldscore_h5ad_path is not None:
                # ── Direct ldscore input mode ────────────────────────────────────
                # Bypasses LD score computation;
                # baseline annotations come from additional_baseline_h5ad_path_list.
                logger.info("Using direct ldscore input mode (pre-computed SNP × Cell h5ad).")
                baseline_ld, w_ld, ldscore_adata = load_common_resources_direct_ldscore(config)

                logger.debug("Initializing processor (direct ldscore mode)...")
                processor = SpatialLDSCEngine(
                    config=config,
                    output_dir=output_dir,
                    marker_score_adata=None,
                    snp_gene_weight_adata=None,
                    baseline_ld=baseline_ld,
                    w_ld=w_ld,
                    ldscore_adata=ldscore_adata,
                )
            else:
                # ── Weight-matrix mode ───────────────────────
                if config.marker_score_format not in ["memmap", "h5ad", "feather"]:
                    raise NotImplementedError(
                        f"Marker score format '{config.marker_score_format}' is not supported. "
                        "Only 'memmap', 'h5ad', and 'feather' are supported."
                    )

                # Load marker scores once (format-agnostic)
                logger.info(f"Loading marker scores (format: {config.marker_score_format})...")

                if config.marker_score_format == "memmap":
                    marker_score_adata = load_marker_scores_memmap_format(config)

                elif config.marker_score_format == "feather":
                    feather_path = Path(config.marker_score_feather_path)
                    logger.info(f"Loading marker scores from Feather: {feather_path}")
                    marker_score_adata = FeatherAnnData(
                        feather_path, index_col="HUMAN_GENE_SYM", transpose=True
                    )

                elif config.marker_score_format == "h5ad":
                    if not config.marker_score_h5ad_path:
                        raise ValueError(
                            "marker_score_h5ad_path must be provided when marker_score_format is 'h5ad'"
                        )

                    h5ad_path = Path(config.marker_score_h5ad_path)
                    if not h5ad_path.exists():
                        raise FileNotFoundError(f"Marker score H5AD file not found: {h5ad_path}")

                    logger.info(f"Loading marker scores from H5AD: {h5ad_path}")
                    marker_score_adata = ad.read_h5ad(h5ad_path, backed="r")

                # Load common resources once (baseline, weights, snp_gene_weights)
                baseline_ld, w_ld, snp_gene_weight_adata = load_common_resources(config)

                # Initialize processor with common resources
                logger.debug("Initializing processor (weight-matrix mode)...")
                processor = SpatialLDSCEngine(
                    config=config,
                    output_dir=output_dir,
                    marker_score_adata=marker_score_adata,
                    snp_gene_weight_adata=snp_gene_weight_adata,
                    baseline_ld=baseline_ld,
                    w_ld=w_ld,
                )

            num_gpus = getattr(config, "ldsc_num_gpus", 1)
            if num_gpus > 1 or (
                getattr(config, "ldsc_share_ld", False)
                and (
                    getattr(config, "ldsc_regression_backend", "auto") != "jax"
                    or (len(traits_to_process) > 1 and jax.default_backend() == "gpu")
                )
            ):
                from gsMap.spatial_ldsc.execution.shared import run_shared_engine

                run_shared_engine(processor, traits_to_process)
                _record_trait_fingerprints(config, output_dir, traits_to_process, set())
                return
            for idx, (trait_name, sumstats_file) in enumerate(traits_to_process):
                try:
                    with stage_logging(config, "spatial_ldsc", trait=trait_name):
                        logger.info("=" * 70)
                        logger.info("Running Spatial LDSC (JAX Implementation)")
                        logger.info(
                            f"Project: {config.project_name}, Trait: {trait_name} "
                            f"({idx + 1}/{len(traits_to_process)})"
                        )
                        if config.sample_filter:
                            logger.info(f"Sample filter: {config.sample_filter}")
                        if config.cell_indices_range:
                            logger.info(f"Cell indices range: {config.cell_indices_range}")
                        logger.info("=" * 70)

                        # Check if output already exists
                        expected_filename = generate_expected_output_filename(
                            config,
                            trait_name,
                            n_spots=_n_input_spots(config, marker_score_adata, ldscore_adata),
                        )
                        if expected_filename is not None:
                            expected_output_path = output_dir / expected_filename
                            if expected_output_path.exists():
                                logger.info(f"Output file already exists: {expected_output_path}")
                                logger.info(
                                    f"Skipping trait {trait_name} "
                                    f"({idx + 1}/{len(traits_to_process)})"
                                )

                                # Log statistics from existing result
                                log_existing_result_statistics(expected_output_path, trait_name)
                                warn_if_outputs_differ("spatial_ldsc", config, trait=trait_name)
                                continue

                        # Setup processor for current trait
                        processor.setup_trait(trait_name, sumstats_file)

                        # Process all chunks for current trait
                        start_time = time.time()
                        processor.process_all_chunks()
                        completed_traits.add(trait_name)

                        elapsed_time = time.time() - start_time
                        h, rem = divmod(elapsed_time, 3600)
                        m, s = divmod(rem, 60)
                        logger.info(
                            f"Trait {trait_name} completed in {int(h)}h {int(m)}m {s:.2f}s"
                        )
                except Exception as error:  # noqa: BLE001 isolate per-trait failures
                    logger.error(
                        f"Trait {trait_name} failed; continuing with the next trait: {error}"
                    )
                    trait_failures.append((trait_name, error))
                    continue  # Move on to the next trait even if this one fails
                finally:
                    # Setup/compute-pool initialization can fail before the
                    # engine's processing try/finally. Do not leave an output
                    # thread or partial stream behind when moving to a trait.
                    writer = getattr(processor, "result_writer", None)
                    if writer is not None and not writer.closed:
                        writer.abort()

            _record_trait_fingerprints(
                config,
                output_dir,
                [item for item in traits_to_process if item[0] in completed_traits],
                set(),
            )
            if trait_failures:
                failed_traits = ", ".join(name for name, _ in trait_failures)
                raise RuntimeError(
                    f"Spatial LDSC failed for {len(trait_failures)}/"
                    f"{len(traits_to_process)} traits: {failed_traits}"
                ) from trait_failures[0][1]

        finally:
            # Cleanup once: close memmap/adata if needed
            if marker_score_adata is not None:
                logger.info("Closing marker score resources...")
                # If it's our MemMap wrapper, close it explicitly
                if (
                    config.marker_score_format == "memmap"
                    and "memmap_manager" in marker_score_adata.uns
                ):
                    marker_score_adata.uns["memmap_manager"].close()
                # If it's backed AnnData, close the file
                if config.marker_score_format == "h5ad" and marker_score_adata.isbacked:
                    marker_score_adata.file.close()
            if config.ldscore_h5ad_path is not None and getattr(ldscore_adata, "isbacked", False):
                ldscore_adata.file.close()

    except Exception as error:
        logger.error(f"Spatial LDSC execution failed: {error}")
        raise


def _n_input_spots(config, marker_score_adata, ldscore_adata) -> int | None:
    """Input spot count, needed only to name a clamped ``cell_indices_range`` result."""
    if not config.cell_indices_range:
        return None
    if marker_score_adata is not None:
        return getattr(marker_score_adata, "n_obs", None)
    return getattr(ldscore_adata, "n_vars", None)


def _record_trait_fingerprints(config, output_dir, traits, failed: set[str]) -> None:
    """Fingerprint each trait whose complete result now exists (see config.fingerprint)."""
    from gsMap.config.fingerprint import record_fingerprint

    for trait_name, _ in traits:
        if trait_name in failed:
            continue
        name = generate_expected_output_filename(config, trait_name)
        if name is not None and (Path(output_dir) / name).is_file():
            record_fingerprint("spatial_ldsc", config, trait=trait_name)
