"""
Main entry point for the latent2gene subpackage
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.panel import Panel

from gsMap.config import LatentToGeneConfig
from gsMap.config.fingerprint import record_fingerprint, warn_if_outputs_differ
from gsMap.config.latent2gene_config import rank_checkpoint_present
from gsMap.logging_config import is_interactive_output, logged_stage

from .marker_scores import MarkerScoreCalculator
from .memmap_io import MemMapDense
from .rank_calculator import RankCalculator

logger = logging.getLogger(__name__)
console = Console()


def _latent2gene_metadata(
    config: LatentToGeneConfig,
    *,
    concatenated_latent_adata: str | Path,
    rank_memmap: str | Path,
    mean_frac: str | Path,
    marker_scores: str | Path,
    shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Build the final metadata object from completed output paths."""
    metadata: dict[str, Any] = {
        "config": config.to_dict_with_paths_as_strings(),
        "outputs": {
            "concatenated_latent_adata": str(concatenated_latent_adata),
            "rank_memmap": str(rank_memmap),
            "mean_frac": str(mean_frac),
            "marker_scores": str(marker_scores),
        },
        "n_sections": len(config.sample_h5ad_dict),
    }
    if shape is not None:
        metadata["n_cells"], metadata["n_genes"] = shape
    return metadata


def _write_metadata_atomic(path: Path, metadata: dict[str, Any]) -> None:
    """Publish YAML metadata atomically (and durably) so an interrupted write is recoverable."""
    from gsMap.utils.atomic_io import atomic_write_text

    atomic_write_text(path, yaml.safe_dump(metadata, default_flow_style=False, sort_keys=False))


@logged_stage("latent_to_gene")
def run_latent_to_gene(config: LatentToGeneConfig) -> dict[str, Any]:
    """
    Main entry point for latent to gene conversion

    This function orchestrates the complete pipeline:
    1. Calculate ranks and concatenate latent representations
    2. Calculate marker scores for each cell type

    Args:
        config: LatentToGeneConfig object with all necessary parameters

    Returns:
        Dictionary with paths to all outputs:
            - concatenated_latent_adata: Path to concatenated latent representations
            - rank_memmap: Path to rank memory map file
            - mean_frac: Path to mean expression fraction
            - marker_scores: Path to marker scores memory map
    """
    from gsMap.utils.jax_setup import use_float32_jax

    use_float32_jax()

    from gsMap.config.spatial import guard_spatial_contract

    guard_spatial_contract(config, write=True)

    # Setup output directory using config paths
    output_dir = Path(config.latent2gene_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if all outputs already exist using config paths
    expected_outputs = {
        "concatenated_latent_adata": Path(config.concatenated_latent_adata_path),
        "rank_memmap": Path(config.rank_memmap_path),
        "mean_frac": Path(config.mean_frac_path),
        "marker_scores": Path(config.marker_scores_memmap_path),
        "metadata": Path(config.latent2gene_metadata_path),
        "rank_meta": Path(config.rank_memmap_path).with_suffix(".meta.json"),
        "marker_scores_meta": Path(config.marker_scores_memmap_path).with_suffix(".meta.json"),
    }

    rank_memmap_complete, rank_meta = MemMapDense.check_complete(expected_outputs["rank_memmap"])
    marker_scores_complete, marker_meta = MemMapDense.check_complete(
        expected_outputs["marker_scores"]
    )
    sidecars_complete = (
        expected_outputs["concatenated_latent_adata"].is_file()
        and expected_outputs["mean_frac"].is_file()
    )
    # A removed rank checkpoint (intermediate, nothing downstream reads it) does not
    # invalidate completed marker scores; an incomplete one still blocks reuse below.
    rank_removed = not rank_checkpoint_present(expected_outputs["rank_memmap"])
    if marker_scores_complete and sidecars_complete and (rank_memmap_complete or rank_removed):
        if marker_meta is None or (rank_memmap_complete and rank_meta is None):
            raise RuntimeError("Completed latent-to-gene memmaps are missing metadata")
        marker_shape = tuple(marker_meta["shape"])
        if rank_removed:
            logger.info(
                "The rank checkpoint was removed; it is needed only to recompute marker "
                "scores, so the completed outputs are reused."
            )
            rank_shape = marker_shape
        else:
            rank_shape = tuple(rank_meta["shape"])
        if rank_shape != marker_shape:
            raise ValueError(
                f"Completed rank and marker-score shapes disagree: {rank_shape} != {marker_shape}"
            )

        metadata = _latent2gene_metadata(
            config,
            concatenated_latent_adata=expected_outputs["concatenated_latent_adata"],
            rank_memmap=expected_outputs["rank_memmap"],
            mean_frac=expected_outputs["mean_frac"],
            marker_scores=expected_outputs["marker_scores"],
            shape=rank_shape,
        )
        metadata_needs_repair = not expected_outputs["metadata"].is_file()
        if not metadata_needs_repair:
            try:
                with expected_outputs["metadata"].open(encoding="utf-8") as handle:
                    existing_metadata = yaml.safe_load(handle)
                metadata_needs_repair = not isinstance(existing_metadata, dict) or not isinstance(
                    existing_metadata.get("outputs"), dict
                )
            except (OSError, yaml.YAMLError):
                metadata_needs_repair = True
        if metadata_needs_repair:
            logger.warning(
                "Completed latent-to-gene arrays were found without valid final metadata; "
                "reconstructing metadata without recomputing arrays"
            )
            _write_metadata_atomic(expected_outputs["metadata"], metadata)

        logger.info(
            f"Found existing complete results for {rank_shape[0]} cells and {rank_shape[1]} genes"
        )
        warn_if_outputs_differ("latent2gene", config)
        return metadata["outputs"]

    if any(path.exists() for path in expected_outputs.values()):
        logger.warning(
            "Latent-to-gene outputs are partial or incomplete; completed checkpoints are "
            "reused and the remaining outputs are recomputed"
        )

    if marker_scores_complete:
        raise RuntimeError(
            "A completed marker-score matrix exists without a complete rank checkpoint and "
            "both rank sidecars. Refusing to combine outputs from inconsistent runs."
        )
    if rank_memmap_complete and not sidecars_complete:
        raise RuntimeError(
            "The rank checkpoint is complete but concatenated_latent_adata.h5ad or "
            "mean_frac.parquet is missing. Refusing to overwrite the completed checkpoint."
        )

    rank_outputs_ready = rank_memmap_complete and sidecars_complete
    if rank_outputs_ready:
        logger.info(
            "Reusing the completed rank checkpoint and sidecars; only marker scores will be "
            "recalculated"
        )
        rank_outputs = {
            "concatenated_latent_adata": str(expected_outputs["concatenated_latent_adata"]),
            "rank_memmap": str(expected_outputs["rank_memmap"]),
            "mean_frac": str(expected_outputs["mean_frac"]),
        }
    else:
        # Step 1: Calculate ranks and concatenate
        if is_interactive_output():
            console.print(
                Panel(
                    "[bold cyan]Step 1: Rank calculation and concatenation[/bold cyan]",
                    border_style="cyan",
                )
            )
        else:
            logger.info("Step 1/2: rank calculation and concatenation")

        rank_calculator = RankCalculator(config)

        # Use sample_h5ad_dict from config (already validated in config.__post_init__)
        logger.info(f"Found {len(config.sample_h5ad_dict)} samples to process")

        rank_outputs = rank_calculator.calculate_ranks_and_concatenate(
            sample_h5ad_dict=config.sample_h5ad_dict,
            annotation_key=config.annotation,
            data_layer=config.data_layer,
        )

    # Step 2: Calculate marker scores
    if is_interactive_output():
        console.print(
            Panel(
                "[bold cyan]Step 2: Marker score calculation[/bold cyan]",
                border_style="cyan",
            )
        )
    else:
        logger.info("Step 2/2: marker score calculation")

    marker_calculator = MarkerScoreCalculator(config)

    rank_memmap_handle = rank_outputs.get("rank_memmap_handle")
    rank_runtime_source = (
        rank_memmap_handle if rank_memmap_handle is not None else rank_outputs["rank_memmap"]
    )
    try:
        marker_scores_path = marker_calculator.calculate_marker_scores(
            adata_path=rank_outputs["concatenated_latent_adata"],
            rank_memmap_path=rank_runtime_source,
            mean_frac_path=rank_outputs["mean_frac"],
            output_path=expected_outputs["marker_scores"],
        )
    finally:
        if rank_memmap_handle is not None:
            rank_memmap_handle.close()

    final_rank_complete, final_rank_meta = MemMapDense.check_complete(
        expected_outputs["rank_memmap"]
    )
    if not final_rank_complete or final_rank_meta is None:
        raise RuntimeError("Rank output became incomplete before final metadata publication")
    final_marker_complete, final_marker_meta = MemMapDense.check_complete(marker_scores_path)
    if not final_marker_complete or final_marker_meta is None:
        raise RuntimeError(
            "Marker-score output is incomplete; final metadata will not be published"
        )
    final_rank_shape = tuple(final_rank_meta["shape"])
    final_marker_shape = tuple(final_marker_meta["shape"])
    if final_rank_shape != final_marker_shape:
        raise ValueError(
            "Rank and marker-score shapes disagree after marker calculation: "
            f"{final_rank_shape} != {final_marker_shape}"
        )
    metadata = _latent2gene_metadata(
        config,
        concatenated_latent_adata=rank_outputs["concatenated_latent_adata"],
        rank_memmap=rank_outputs["rank_memmap"],
        mean_frac=rank_outputs["mean_frac"],
        marker_scores=marker_scores_path,
        shape=final_rank_shape,
    )
    _write_metadata_atomic(expected_outputs["metadata"], metadata)
    record_fingerprint("latent2gene", config)

    logger.info(f"All outputs saved to: {output_dir}")

    return metadata["outputs"]
