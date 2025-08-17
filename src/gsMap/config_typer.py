"""
Refactored config.py using dataclass_typer decorator for more elegant CLI design.
This module provides a clean separation between configuration and CLI interface.
"""

import logging
import os
import sys
import time
from dataclasses import dataclass, fields
from functools import wraps
from pathlib import Path
from typing import Optional, Annotated, Literal
import inspect

import psutil
import pyfiglet
import typer
import yaml

from gsMap.__init__ import __version__
from gsMap.config import (
    get_gsMap_logger,
    track_resource_usage,
    ensure_path_exists,
    ConfigWithAutoPaths,
    verify_homolog_file_format,
)

logger = get_gsMap_logger("gsMap")

# Create the Typer app
app = typer.Typer(
    name="gsmap",
    help="gsMap: genetically informed spatial mapping of cells for complex traits",
    rich_markup_mode="rich",
    add_completion=False,
)


def dataclass_typer(func):
    """
    Decorator to convert a function that takes a dataclass config
    into a Typer command with individual CLI options.
    
    This decorator extracts the typer.Option annotations from the dataclass
    fields and creates a proper Typer command signature.
    """
    sig = inspect.signature(func)
    
    # Get the dataclass type from the function signature
    config_param = list(sig.parameters.values())[0]
    config_class = config_param.annotation
    
    @wraps(func)
    def wrapper(**kwargs):
        # Create the config instance
        config = config_class(**kwargs)
        return func(config)
    
    # Build new parameters from dataclass fields
    params = []
    for field in fields(config_class):
        # Check if field has a default value
        if field.default is not field.default_factory:
            # Field has a default value, use it as the parameter default
            param = inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=field.type,
                default=field.default
            )
        else:
            # No default, parameter is required
            param = inspect.Parameter(
                field.name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=field.type
            )
        params.append(param)
    
    # Update the wrapper's signature
    wrapper.__signature__ = inspect.Signature(params)
    
    # Preserve the original function's docstring
    wrapper.__doc__ = func.__doc__
    
    return wrapper


def show_banner_and_version(command_name: str):
    """Display gsMap banner and version information."""
    command_name = command_name.replace("_", " ")
    gsMap_main_logo = pyfiglet.figlet_format(
        "gsMap",
        font="doom",
        width=80,
        justify="center",
    ).rstrip()
    print(gsMap_main_logo, flush=True)
    version_number = "Version: " + __version__
    print(version_number.center(80), flush=True)
    print("=" * 80, flush=True)
    logger.info(f"Running {command_name}...")
    
    # Record start time for the log message
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Started at: {start_time}")


# ============================================================================
# Configuration dataclasses with Annotated typer.Option
# ============================================================================

@dataclass
class RunAllModeConfig(ConfigWithAutoPaths):
    """Configuration for running the complete gsMap pipeline."""
    
    # Required fields first (ConfigWithAutoPaths requires these)
    workdir: str
    project_name: str
    sample_name: str
    
    # Required paths and configurations
    gsMap_resource_dir: Annotated[Path, typer.Option(
        "--gsmap-resource-dir",
        help="Directory containing gsMap resources",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]
    
    hdf5_path: Annotated[Path, typer.Option(
        help="Path to the input spatial transcriptomics data (H5AD format)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )]
    
    annotation: Annotated[str, typer.Option(
        help="Name of the annotation in adata.obs to use"
    )]
    
    trait_name: Annotated[str, typer.Option(
        help="Name of the trait for GWAS analysis"
    )]
    
    sumstats_file: Annotated[Path, typer.Option(
        help="Path to GWAS summary statistics file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )]
    
    data_layer: Annotated[str, typer.Option(
        help="Data layer for gene expression"
    )] = "counts"
    
    n_comps: Annotated[int, typer.Option(
        help="Number of components",
        min=10,
        max=500
    )] = 300
    
    pearson_residuals: Annotated[bool, typer.Option(
        "--pearson-residuals",
        help="Use pearson residuals"
    )] = False
    
    gM_slices: Annotated[Optional[str], typer.Option(
        help="Path to the slice mean file"
    )] = None
    
    latent_representation: Annotated[Optional[str], typer.Option(
        help="Type of latent representation"
    )] = None
    
    num_neighbour: Annotated[int, typer.Option(
        help="Number of neighbors",
        min=1,
        max=100
    )] = 21
    
    num_neighbour_spatial: Annotated[int, typer.Option(
        help="Number of spatial neighbors",
        min=1,
        max=500
    )] = 101
    
    homolog_file: Annotated[Optional[Path], typer.Option(
        help="Path to homologous gene conversion file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    max_processes: Annotated[int, typer.Option(
        help="Maximum number of processes for parallel execution",
        min=1,
        max=50
    )] = 10
    
    use_jax: Annotated[bool, typer.Option(
        "--use-jax/--no-jax",
        help="Use JAX-accelerated spatial LDSC implementation"
    )] = True
    
    sumstats_config_file: Optional[str] = None  # Not exposed in CLI


@dataclass
class FindLatentRepresentationsConfig(ConfigWithAutoPaths):
    """Configuration for finding latent representations."""
    
    # Required parameters
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]
    
    sample_name: Annotated[str, typer.Option(
        help="Name of the sample"
    )]
    
    spe_file_list: Annotated[str, typer.Option(
        help="List of input ST (.h5ad) files"
    )]
    
    # Optional parameters with defaults
    project_name: Annotated[Optional[str], typer.Option(
        help="Project name"
    )] = None
    
    data_layer: Annotated[str, typer.Option(
        help="Gene expression data layer"
    )] = "count"
    
    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm storing spatial coordinates"
    )] = "spatial"
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation in adata.obs to use"
    )] = None
    
    # Feature extraction parameters (LGCN)
    n_neighbors: Annotated[int, typer.Option(
        help="Number of neighbors for LGCN",
        min=1,
        max=50
    )] = 10
    
    K: Annotated[int, typer.Option(
        help="Graph convolution depth for LGCN",
        min=1,
        max=10
    )] = 3
    
    feat_cell: Annotated[int, typer.Option(
        help="Number of top variable features to retain",
        min=100,
        max=10000
    )] = 2000
    
    pearson_residual: Annotated[bool, typer.Option(
        "--pearson-residual",
        help="Take the residuals of the input data"
    )] = False
    
    # Model dimension parameters
    hidden_size: Annotated[int, typer.Option(
        help="Units in the first hidden layer",
        min=32,
        max=512
    )] = 128
    
    embedding_size: Annotated[int, typer.Option(
        help="Size of the latent embedding layer",
        min=8,
        max=128
    )] = 32
    
    # Transformer module parameters
    use_tf: Annotated[bool, typer.Option(
        "--use-tf",
        help="Enable transformer module"
    )] = False
    
    module_dim: Annotated[int, typer.Option(
        help="Dimensionality of transformer modules",
        min=10,
        max=100
    )] = 30
    
    hidden_gmf: Annotated[int, typer.Option(
        help="Hidden units for global mean feature extractor",
        min=32,
        max=512
    )] = 128
    
    n_modules: Annotated[int, typer.Option(
        help="Number of transformer modules",
        min=4,
        max=64
    )] = 16
    
    nhead: Annotated[int, typer.Option(
        help="Number of attention heads in transformer",
        min=1,
        max=16
    )] = 4
    
    n_enc_layer: Annotated[int, typer.Option(
        help="Number of transformer encoder layers",
        min=1,
        max=8
    )] = 2
    
    # Training parameters
    distribution: Annotated[str, typer.Option(
        help="Distribution type for loss calculation",
        case_sensitive=False
    )] = "nb"
    
    n_cell_training: Annotated[int, typer.Option(
        help="Number of cells used for training",
        min=1000,
        max=1000000
    )] = 100000
    
    batch_size: Annotated[int, typer.Option(
        help="Batch size for training",
        min=32,
        max=4096
    )] = 1024
    
    itermax: Annotated[int, typer.Option(
        help="Maximum number of training iterations",
        min=10,
        max=1000
    )] = 100
    
    patience: Annotated[int, typer.Option(
        help="Early stopping patience",
        min=1,
        max=50
    )] = 10
    
    two_stage: Annotated[bool, typer.Option(
        "--two-stage/--single-stage",
        help="Tune the cell embeddings based on the provided annotation"
    )] = True
    
    do_sampling: Annotated[bool, typer.Option(
        "--do-sampling/--no-sampling",
        help="Down-sampling cells in training"
    )] = True
    
    homolog_file: Annotated[Optional[Path], typer.Option(
        help="Path to homologous gene conversion file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None


@dataclass
class SpatialLDSCConfig(ConfigWithAutoPaths):
    """Configuration for spatial LDSC analysis."""
    
    # Required parameters
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]
    
    sample_name: Annotated[str, typer.Option(
        help="Name of the sample"
    )]
    
    sumstats_file: Annotated[Path, typer.Option(
        help="Path to GWAS summary statistics file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )]
    
    trait_name: Annotated[str, typer.Option(
        help="Name of the trait being analyzed"
    )]
    
    # Optional parameters with defaults
    project_name: Annotated[Optional[str], typer.Option(
        help="Project name"
    )] = None
    
    w_file: Annotated[Optional[str], typer.Option(
        help="Path to regression weight file"
    )] = None
    
    n_blocks: Annotated[int, typer.Option(
        help="Number of blocks for jackknife resampling",
        min=10,
        max=500
    )] = 200
    
    chisq_max: Annotated[Optional[int], typer.Option(
        help="Maximum chi-square value for filtering SNPs"
    )] = None
    
    num_processes: Annotated[int, typer.Option(
        help="Number of processes for parallel computing",
        min=1,
        max=50
    )] = 4
    
    use_additional_baseline_annotation: Annotated[bool, typer.Option(
        "--use-baseline/--no-baseline",
        help="Use additional baseline annotations when provided"
    )] = True
    
    use_jax: Annotated[bool, typer.Option(
        "--use-jax/--no-jax",
        help="Use JAX-accelerated implementation"
    )] = True
    
    # Hidden parameters not exposed in CLI
    sumstats_config_file: Optional[str] = None
    not_M_5_50: bool = False
    all_chunk: Optional[int] = None
    chunk_range: Optional[tuple[int, int]] = None
    ldscore_save_format: Literal["feather", "quick_mode"] = "feather"
    spots_per_chunk_quick_mode: int = 1000
    snp_gene_weight_adata_path: Optional[str] = None


@dataclass
class ReportConfig(ConfigWithAutoPaths):
    """Configuration for generating reports."""
    
    # Required parameters
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]
    
    sample_name: Annotated[str, typer.Option(
        help="Name of the sample"
    )]
    
    trait_name: Annotated[str, typer.Option(
        help="Name of the trait to generate the report for"
    )]
    
    annotation: Annotated[str, typer.Option(
        help="Annotation layer name"
    )]
    
    sumstats_file: Annotated[Path, typer.Option(
        help="Path to GWAS summary statistics file",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True
    )]
    
    # Optional parameters with defaults
    project_name: Annotated[Optional[str], typer.Option(
        help="Project name"
    )] = None
    
    top_corr_genes: Annotated[int, typer.Option(
        help="Number of top correlated genes to display",
        min=1,
        max=500
    )] = 50
    
    selected_genes: Annotated[Optional[str], typer.Option(
        help="Comma-separated list of specific genes to include"
    )] = None
    
    fig_width: Annotated[Optional[int], typer.Option(
        help="Width of the generated figures in pixels"
    )] = None
    
    fig_height: Annotated[Optional[int], typer.Option(
        help="Height of the generated figures in pixels"
    )] = None
    
    point_size: Annotated[Optional[int], typer.Option(
        help="Point size for the figures"
    )] = None
    
    fig_style: Annotated[str, typer.Option(
        help="Style of the generated figures",
        case_sensitive=False
    )] = "light"
    
    # Hidden parameter
    plot_type: Literal["manhattan", "GSS", "gsMap", "all"] = "all"


# ============================================================================
# CLI Commands using dataclass_typer decorator
# ============================================================================

@app.command(name="quick-mode")
@dataclass_typer
@track_resource_usage
def quick_mode(config: RunAllModeConfig):
    """
    Run the complete gsMap pipeline with all steps.
    
    This command runs the full gsMap analysis pipeline including:
    - Data loading and preprocessing
    - Gene expression analysis
    - GWAS integration
    - Spatial mapping
    - Result generation
    """
    show_banner_and_version("quick mode")
    
    from gsMap.run_all_mode import run_pipeline
    
    logger.info(f"Running gsMap pipeline for sample: {config.sample_name}")
    logger.info(f"Trait: {config.trait_name}")
    logger.info(f"Working directory: {config.workdir}")
    
    try:
        run_pipeline(config)
        logger.info("✓ Pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


@app.command(name="find-latent")
@dataclass_typer
@track_resource_usage
def find_latent_representations(config: FindLatentRepresentationsConfig):
    """
    Find latent representations of each spot using Graph Neural Networks.
    
    This step:
    - Loads spatial transcriptomics data
    - Builds neighborhood graphs
    - Learns latent representations using GNN
    - Saves the model and embeddings
    """
    show_banner_and_version("find latent representations")
    
    from gsMap.find_latent_representation import run_find_latent_representation
    
    logger.info(f"Finding latent representations for sample: {config.sample_name}")
    
    if config.annotation and config.two_stage:
        logger.info(f"Using two-stage training with annotation: {config.annotation}")
    else:
        logger.info("Using single-stage training with reconstruction loss only")
    
    try:
        run_find_latent_representation(config)
        logger.info("✓ Latent representations computed successfully!")
    except Exception as e:
        logger.error(f"Failed to compute latent representations: {e}")
        raise


@app.command(name="spatial-ldsc")
@dataclass_typer
@track_resource_usage
def spatial_ldsc(config: SpatialLDSCConfig):
    """
    Run spatial LDSC analysis for genetic association.
    
    This step:
    - Loads LD scores and GWAS summary statistics
    - Performs spatial LDSC regression
    - Computes enrichment statistics
    - Saves results for downstream analysis
    """
    show_banner_and_version("spatial LDSC")
    
    if config.use_jax:
        from gsMap.spatial_ldsc_jax_final import run_spatial_ldsc_jax
        logger.info("Using JAX-accelerated spatial LDSC implementation")
        run_func = run_spatial_ldsc_jax
    else:
        from gsMap.spatial_ldsc_multiple_sumstats import run_spatial_ldsc
        logger.info("Using standard spatial LDSC implementation")
        run_func = run_spatial_ldsc
    
    logger.info(f"Running spatial LDSC for trait: {config.trait_name}")
    logger.info(f"Sample: {config.sample_name}")
    
    try:
        run_func(config)
        logger.info("✓ Spatial LDSC completed successfully!")
    except Exception as e:
        logger.error(f"Spatial LDSC failed: {e}")
        raise


@app.command(name="report")
@dataclass_typer
@track_resource_usage
def generate_report(config: ReportConfig):
    """
    Generate diagnostic plots and HTML report for gsMap results.
    
    Creates:
    - Manhattan plots
    - Gene set score (GSS) plots
    - Spatial enrichment visualizations
    - Interactive HTML report
    """
    show_banner_and_version("report generation")
    
    from gsMap.report import run_report
    
    logger.info(f"Generating report for trait: {config.trait_name}")
    logger.info(f"Sample: {config.sample_name}")
    logger.info(f"Annotation: {config.annotation}")
    
    # Parse selected genes if provided
    if config.selected_genes:
        config.selected_genes = [g.strip() for g in config.selected_genes.split(",")]
    
    try:
        run_report(config)
        logger.info("✓ Report generated successfully!")
        logger.info(f"Report saved to: {config.get_report_dir(config.trait_name)}")
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        typer.echo(f"gsMap version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[Optional[bool], typer.Option(
        "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit"
    )] = None,
):
    """
    gsMap: genetically informed spatial mapping of cells for complex traits.
    
    Use 'gsmap COMMAND --help' for more information on a specific command.
    """
    pass


if __name__ == "__main__":
    app()