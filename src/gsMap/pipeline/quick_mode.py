import logging
import time
from pathlib import Path

from gsMap.cauchy_combination_test import run_Cauchy_combination
from gsMap.config import QuickModeConfig
from gsMap.config.cauchy_config import check_cauchy_done
from gsMap.config.find_latent_config import check_find_latent_done
from gsMap.config.fingerprint import warn_if_outputs_differ
from gsMap.config.latent2gene_config import check_latent2gene_done
from gsMap.config.quick_mode_config import check_report_done
from gsMap.config.spatial_ldsc_config import check_spatial_ldsc_done
from gsMap.find_latent import run_find_latent_representation
from gsMap.latent2gene import run_latent_to_gene
from gsMap.logging_config import logged_stage
from gsMap.report import run_report
from gsMap.spatial_ldsc.spatial_ldsc_jax import run_spatial_ldsc_jax

logger = logging.getLogger("gsMap.pipeline")


def format_duration(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m {int(seconds % 60)}s"


def _preflight(config, steps_to_run: list[str]) -> None:
    """Check settings of later steps before the long early steps start.

    Configuration errors that would otherwise surface only when spatial LDSC, Cauchy
    or the report starts (hours into a run) are raised here instead. Resources of a
    step whose outputs already exist are not checked, so reruns are unaffected.
    """
    traits = list(getattr(config, "sumstats_config_dict", None) or {})
    if "spatial_ldsc" in steps_to_run:
        from gsMap.config.spatial_ldsc_config import (
            resolve_w_ld_dir,
            validate_spatial_ldsc_settings,
        )

        validate_spatial_ldsc_settings(config)
        pending = [trait for trait in traits if not check_spatial_ldsc_done(config, trait)]
        if pending:
            w_ld_dir = resolve_w_ld_dir(config)
            if not w_ld_dir.is_dir():
                raise FileNotFoundError(f"Weights directory (--w-ld-dir) not found: {w_ld_dir}")
            for name in ("snp_gene_weight_adata_path", "ldscore_h5ad_path"):
                path = getattr(config, name, None)
                if path is not None and not Path(path).is_file():
                    raise FileNotFoundError(f"{name} not found: {path}")

    if "cauchy" in steps_to_run:
        annotations = list(getattr(config, "annotation_list", None) or [])
        reserved = {"spot", "sample_name", *annotations}
        clashes = sorted(reserved & set(traits))
        if clashes:
            raise ValueError(
                f"Trait names {clashes} are also spot metadata column names (spot, sample_name or "
                "an annotation); rename the traits."
            )
        cauchy_done = bool(traits) and all(check_cauchy_done(config, trait) for trait in traits)
        sources = {} if cauchy_done else _annotation_sources(config, steps_to_run)
        if annotations and sources:
            from gsMap.config.utils import validate_h5ad_structure

            validate_h5ad_structure(
                sources,
                {
                    f"annotation:{name}": ("obs", name, f"Cauchy annotation {name}")
                    for name in annotations
                },
            )

    if "report" in steps_to_run:
        from gsMap.report.report import _check_report_sample_file_names

        _check_report_sample_file_names(list(getattr(config, "sample_h5ad_dict", None) or {}))


def _annotation_sources(config, steps_to_run: list[str]) -> dict:
    """Files whose obs must hold the Cauchy annotations: inputs, or the latent AnnData."""
    if "find_latent" in steps_to_run or "latent2gene" in steps_to_run:
        return {
            name: Path(path)
            for name, path in (getattr(config, "sample_h5ad_dict", None) or {}).items()
        }
    latent = getattr(config, "concatenated_latent_adata_path", None)
    if latent is not None and Path(latent).is_file():
        return {"concatenated latent AnnData": Path(latent)}
    return {}


@logged_stage("quick_mode")
def run_quick_mode(config: QuickModeConfig):
    """
    Run the Quick Mode pipeline.
    """
    logger.info("Starting Quick Mode pipeline")
    pipeline_start_time = time.time()

    steps = ["find_latent", "latent2gene", "spatial_ldsc", "cauchy", "report"]
    try:
        start_idx = steps.index(config.start_step)
    except ValueError:
        raise ValueError(
            f"Invalid start_step: {config.start_step}. Must be one of {steps}"
        ) from None

    stop_idx = len(steps) - 1
    if config.stop_step:
        try:
            stop_idx = steps.index(config.stop_step)
        except ValueError:
            raise ValueError(
                f"Invalid stop_step: {config.stop_step}. Must be one of {steps}"
            ) from None

    if start_idx > stop_idx:
        raise ValueError(
            f"start_step ({config.start_step}) must be before or equal to stop_step ({config.stop_step})"
        )

    _preflight(config, steps[start_idx : stop_idx + 1])

    from gsMap.config.spatial import guard_spatial_contract

    if config.requires_spatial_inputs and config.start_step != "report":
        guard_spatial_contract(config, write=True)

    # Step 1: Find Latent Representations
    if start_idx <= 0 <= stop_idx:
        logger.info("=== Step 1: Find Latent Representations ===")
        start_time = time.time()

        if check_find_latent_done(config):
            logger.info(
                f"Find latent representations already done (verified via {config.find_latent_metadata_path}). Skipping..."
            )
        else:
            run_find_latent_representation(config.find_latent_config)

        logger.info(f"Step 1 completed in {format_duration(time.time() - start_time)}")

    # Step 2: Latent to Gene
    if start_idx <= 1 <= stop_idx:
        logger.info("=== Step 2: Latent to Gene Mapping ===")
        start_time = time.time()

        if check_latent2gene_done(config):
            logger.info("Latent to gene mapping already done. Skipping...")
            warn_if_outputs_differ("latent2gene", config.latent2gene_config)
        else:
            run_latent_to_gene(config.latent2gene_config)

        logger.info(f"Step 2 completed in {format_duration(time.time() - start_time)}")

    # Get lists of traits to process
    if not config.sumstats_config_dict:
        # Check if we should warn? Only if running step 3,4,5
        if start_idx <= 4 and stop_idx >= 2:
            logger.warning(
                "No summary statistics provided. Steps requiring GWAS data (Spatial LDSC, Cauchy, Report) may fail or do nothing if relying on them."
            )

    traits_to_process = config.sumstats_config_dict

    # Step 3: Spatial LDSC
    if start_idx <= 2 <= stop_idx:
        logger.info("=== Step 3: Spatial LDSC ===")
        start_time = time.time()

        traits_remaining = {}
        sldsc_config = config.spatial_ldsc_config
        for trait_name, sumstats_path in traits_to_process.items():
            if check_spatial_ldsc_done(config, trait_name):
                logger.info(f"Spatial LDSC result already exists for {trait_name}. Skipping...")
                warn_if_outputs_differ("spatial_ldsc", sldsc_config, trait=trait_name)
            else:
                traits_remaining[trait_name] = sumstats_path

        if not traits_remaining:
            logger.info("All traits have been processed for Spatial LDSC. Skipping step...")
        else:
            # Update config to run only remaining traits
            sldsc_config.sumstats_config_dict = traits_remaining
            run_spatial_ldsc_jax(sldsc_config)

        logger.info(f"Step 3 completed in {format_duration(time.time() - start_time)}")

    # Step 4: Cauchy Combination
    if start_idx <= 3 <= stop_idx:
        logger.info("=== Step 4: Cauchy Combination ===")
        start_time = time.time()

        cauchy_check_list = [
            check_cauchy_done(config, trait_name) for trait_name in traits_to_process
        ]
        if all(cauchy_check_list):
            logger.info("Cauchy combination already done for all traits. Skipping...")
            warn_if_outputs_differ("cauchy", config)
        else:
            cauchy_config = config.cauchy_config
            run_Cauchy_combination(cauchy_config)

        logger.info(f"Step 4 completed in {format_duration(time.time() - start_time)}")

    # Step 5: Report
    if start_idx <= 4 <= stop_idx:
        logger.info("=== Step 5: Generate Report ===")
        start_time = time.time()

        if getattr(config, "force_report_re_run", False):
            logger.info("force_report_re_run is set; regenerating the report.")
            run_report(config)
        elif check_report_done(config, verbose=True):
            logger.info("Report already exists. Skipping...")
            warn_if_outputs_differ("report", config)
        else:
            run_report(
                config,
            )

        logger.info(f"Step 5 completed in {format_duration(time.time() - start_time)}")

    logger.info(
        f"Pipeline completed successfully in {format_duration(time.time() - pipeline_start_time)}"
    )
