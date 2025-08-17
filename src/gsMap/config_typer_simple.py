#!/usr/bin/env python
"""
Simplified version of config_typer.py demonstrating the dataclass_typer pattern.
This version doesn't depend on complex inheritance for testing purposes.
"""

import logging
import time
from dataclasses import dataclass, fields
from functools import wraps
from pathlib import Path
from typing import Optional, Annotated
import inspect

import typer

# Simple logger setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger("gsMap")

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


# ============================================================================
# Configuration dataclasses with Annotated typer.Option
# ============================================================================

@dataclass
class RunAllModeConfig:
    """Configuration for running the complete gsMap pipeline."""
    
    # Required paths and configurations
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
    
    gsmap_resource_dir: Annotated[Path, typer.Option(
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
    
    # Optional parameters with defaults
    project_name: Annotated[Optional[str], typer.Option(
        help="Project name"
    )] = None
    
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
    
    max_processes: Annotated[int, typer.Option(
        help="Maximum number of processes for parallel execution",
        min=1,
        max=50
    )] = 10
    
    use_jax: Annotated[bool, typer.Option(
        "--use-jax/--no-jax",
        help="Use JAX-accelerated spatial LDSC implementation"
    )] = True


@dataclass
class FindLatentRepresentationsConfig:
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
    
    feat_cell: Annotated[int, typer.Option(
        help="Number of top variable features to retain",
        min=100,
        max=10000
    )] = 2000
    
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
    
    # Training parameters
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


@dataclass
class SpatialLDSCConfig:
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
    
    num_processes: Annotated[int, typer.Option(
        help="Number of processes for parallel computing",
        min=1,
        max=50
    )] = 4
    
    use_jax: Annotated[bool, typer.Option(
        "--use-jax/--no-jax",
        help="Use JAX-accelerated implementation"
    )] = True


# ============================================================================
# CLI Commands using dataclass_typer decorator
# ============================================================================

@app.command(name="quick-mode")
@dataclass_typer
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
    logger.info(f"🚀 Running gsMap pipeline")
    logger.info(f"📊 Sample: {config.sample_name}")
    logger.info(f"🧬 Trait: {config.trait_name}")
    logger.info(f"📁 Working directory: {config.workdir}")
    
    if config.use_jax:
        logger.info("⚡ JAX acceleration enabled")
    
    try:
        # Simulate pipeline execution
        from gsMap.run_all_mode import run_pipeline
        run_pipeline(config)
        logger.info("✓ Pipeline completed successfully!")
    except (ImportError, AttributeError) as e:
        logger.info(f"Note: {e}")
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="find-latent")
@dataclass_typer
def find_latent_representations(config: FindLatentRepresentationsConfig):
    """
    Find latent representations of each spot using Graph Neural Networks.
    
    This step:
    - Loads spatial transcriptomics data
    - Builds neighborhood graphs
    - Learns latent representations using GNN
    - Saves the model and embeddings
    """
    logger.info(f"🔬 Finding latent representations")
    logger.info(f"📊 Sample: {config.sample_name}")
    logger.info(f"📁 Working directory: {config.workdir}")
    logger.info(f"📂 Input files: {config.spe_file_list}")
    
    if config.annotation:
        logger.info(f"🏷️ Using annotation: {config.annotation}")
    
    try:
        from gsMap.find_latent_representation import run_find_latent_representation
        run_find_latent_representation(config)
        logger.info("✓ Latent representations computed successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="spatial-ldsc")
@dataclass_typer
def spatial_ldsc(config: SpatialLDSCConfig):
    """
    Run spatial LDSC analysis for genetic association.
    
    This step:
    - Loads LD scores and GWAS summary statistics
    - Performs spatial LDSC regression
    - Computes enrichment statistics
    - Saves results for downstream analysis
    """
    logger.info(f"🧬 Running spatial LDSC")
    logger.info(f"📊 Trait: {config.trait_name}")
    logger.info(f"📊 Sample: {config.sample_name}")
    logger.info(f"📁 Working directory: {config.workdir}")
    
    if config.use_jax:
        logger.info("⚡ Using JAX-accelerated implementation")
    else:
        logger.info("🐢 Using standard implementation")
    
    try:
        if config.use_jax:
            from gsMap.spatial_ldsc_jax_final import run_spatial_ldsc_jax
            run_spatial_ldsc_jax(config)
        else:
            from gsMap.spatial_ldsc_multiple_sumstats import run_spatial_ldsc
            run_spatial_ldsc(config)
        logger.info("✓ Spatial LDSC completed successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        typer.echo("gsMap version 1.0.0")
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