"""
Main entry point for the latent2gene subpackage
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

from .rank_calculator import RankCalculator
from .marker_scores import MarkerScoreCalculator

logger = logging.getLogger(__name__)


def run_latent_to_gene(config) -> Dict[str, Any]:
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
            - rank_zarr: Path to rank zarr file
            - mean_frac: Path to mean expression fraction
            - marker_scores: Path to marker scores zarr
            - metadata: Path to metadata JSON
    """
    
    logger.info("=" * 60)
    logger.info("Starting latent to gene conversion pipeline")
    logger.info("=" * 60)
    
    # Setup output directory
    output_dir = Path(config.workdir) / "latent_to_gene"
    if config.project_name:
        output_dir = Path(config.workdir) / config.project_name / "latent_to_gene"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if all outputs already exist
    expected_outputs = {
        "concatenated_latent_adata": output_dir / "concatenated_latent_adata.h5ad",
        "rank_zarr": output_dir / "ranks.zarr",
        "mean_frac": output_dir / "mean_frac.parquet",
        "marker_scores": output_dir / "marker_scores.zarr",
        "metadata": output_dir / "metadata.json"
    }
    
    if all(Path(p).exists() for p in expected_outputs.values()):
        logger.info("All outputs already exist. Loading metadata...")
        with open(expected_outputs["metadata"], 'r') as f:
            existing_metadata = json.load(f)
        logger.info(f"Found existing results for {existing_metadata.get('n_cells', 'unknown')} cells "
                   f"and {existing_metadata.get('n_genes', 'unknown')} genes")
        return {k: str(v) for k, v in expected_outputs.items()}
    
    # Step 1: Calculate ranks and concatenate
    logger.info("\n" + "-" * 40)
    logger.info("Step 1: Rank calculation and concatenation")
    logger.info("-" * 40)
    
    rank_calculator = RankCalculator(config)
    
    # Get list of spatial files from latent directory
    latent_dir = Path(config.latent_dir)
    spe_files = []
    
    # Look for original spatial files based on latent file names
    for latent_file in latent_dir.glob("*_latent_adata.h5ad"):
        # Reconstruct original filename
        stem = latent_file.stem.replace("_latent_adata", "")
        # Try to find corresponding spatial file
        possible_paths = [
            latent_dir.parent / f"{stem}.h5ad",
            latent_dir.parent.parent / "spatial_data" / f"{stem}.h5ad",
            latent_dir.parent.parent / f"{stem}.h5ad"
        ]
        
        for path in possible_paths:
            if path.exists():
                spe_files.append(str(path))
                break
        else:
            logger.warning(f"Could not find spatial file for {latent_file}")
    
    if not spe_files:
        raise FileNotFoundError(f"No spatial files found corresponding to latent files in {latent_dir}")
    
    logger.info(f"Found {len(spe_files)} spatial files to process")
    
    rank_outputs = rank_calculator.calculate_ranks_and_concatenate(
        spe_file_list=spe_files,
        annotation_key=config.annotation,
        data_layer="counts"  # Or from config if available
    )
    
    # Step 2: Calculate marker scores
    logger.info("\n" + "-" * 40)
    logger.info("Step 2: Marker score calculation")
    logger.info("-" * 40)
    
    marker_calculator = MarkerScoreCalculator(config)
    
    marker_scores_path = marker_calculator.calculate_marker_scores(
        adata_path=rank_outputs["concatenated_latent_adata"],
        rank_zarr_path=rank_outputs["rank_zarr"],
        mean_frac_path=rank_outputs["mean_frac"],
        output_path=expected_outputs["marker_scores"]
    )
    
    # Create overall metadata
    metadata = {
        "config": {
            "workdir": str(config.workdir),
            "project_name": config.project_name,
            "sample_name": config.sample_name,
            "num_neighbour_spatial": config.num_neighbour_spatial,
            "num_anchor": config.num_anchor,
            "num_neighbour": config.num_neighbour,
            "batch_size": config.batch_size,
            "gpu_batch_size": config.gpu_batch_size,
            "num_read_workers": config.num_read_workers,
            "num_write_workers": config.num_write_workers
        },
        "outputs": {
            "concatenated_latent_adata": str(rank_outputs["concatenated_latent_adata"]),
            "rank_zarr": str(rank_outputs["rank_zarr"]),
            "mean_frac": str(rank_outputs["mean_frac"]),
            "marker_scores": str(marker_scores_path)
        },
        "n_sections": len(spe_files)
    }
    
    # Load marker scores metadata if it exists
    marker_metadata_path = Path(marker_scores_path).parent / f"{Path(marker_scores_path).stem}_metadata.json"
    if marker_metadata_path.exists():
        with open(marker_metadata_path, 'r') as f:
            marker_metadata = json.load(f)
            metadata.update(marker_metadata)
    
    # Save overall metadata
    with open(expected_outputs["metadata"], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("Latent to gene conversion complete!")
    logger.info(f"All outputs saved to: {output_dir}")
    logger.info("=" * 60)
    
    return metadata["outputs"]


def run_from_cli(args):
    """
    Entry point for CLI usage
    
    Args:
        args: Parsed command line arguments
    """
    from gsMap.config.dataclasses import LatentToGeneConfig
    
    # Create config from CLI args
    config = LatentToGeneConfig(**vars(args))
    
    # Run pipeline
    outputs = run_latent_to_gene(config)
    
    # Print results
    print("\nPipeline completed successfully!")
    print("Outputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")