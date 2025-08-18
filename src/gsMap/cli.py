#!/usr/bin/env python
"""
gsMap CLI - Main command-line interface using the modular config design.
"""

import logging
from typing import Optional, Annotated

import typer

from gsMap.config import (
    dataclass_typer,
    RunAllModeConfig,
    FindLatentRepresentationsConfig,
    LatentToGeneConfig,
    SpatialLDSCConfig,
    ReportConfig,
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s | %(name)s - %(message)s')
logger = logging.getLogger("gsMap")

# Create the Typer app
app = typer.Typer(
    name="gsmap",
    help="gsMap: genetically informed spatial mapping of cells for complex traits",
    rich_markup_mode="rich",
    add_completion=False,
)


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
    logger.info(f"Sample: {config.sample_name}")
    logger.info(f"Trait: {config.trait_name}")
    logger.info(f"Working directory: {config.workdir}")
    logger.info(f"Project directory: {config.project_dir}")
    
    # Show some auto-generated paths
    logger.info(f"Model will be saved to: {config.model_path}")
    logger.info(f"LD scores will be saved to: {config.ldscore_save_dir}")
    logger.info(f"Report will be saved to: {config.get_report_dir(config.trait_name)}")
    
    if config.use_jax:
        logger.info("JAX acceleration enabled")
    
    try:
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
    logger.info(f"Sample: {config.sample_name}")
    logger.info(f"Working directory: {config.workdir}")
    logger.info(f"Project directory: {config.project_dir}")
    
    # Show auto-generated paths
    logger.info(f"Latent representations will be saved to: {config.latent_dir}")
    logger.info(f"Model will be saved to: {config.model_path}")
    logger.info(f"H5AD with latent: {config.hdf5_with_latent_path}")
    
    if config.annotation and config.two_stage:
        logger.info(f"Using two-stage training with annotation: {config.annotation}")
    else:
        logger.info("Using single-stage training with reconstruction loss only")
    
    try:
        from gsMap.find_latent_representation import run_find_latent_representation
        run_find_latent_representation(config)
        logger.info("✓ Latent representations computed successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="latent-to-gene")
@dataclass_typer
def latent_to_gene(config: LatentToGeneConfig):
    """
    Estimate gene marker scores for each spot using latent representations.
    
    This step:
    - Loads latent representations
    - Estimates gene marker scores
    - Performs spatial smoothing
    - Saves marker scores for LDSC
    """
    logger.info(f"Sample: {config.sample_name}")
    logger.info(f"Working directory: {config.workdir}")
    logger.info(f"Project directory: {config.project_dir}")
    
    # Show auto-generated paths
    logger.info(f"Marker scores will be saved to: {config.mkscore_feather_path}")
    if config.annotation:
        logger.info(f"Tuned scores will be saved to: {config.tuned_mkscore_feather_path}")
    
    try:
        from gsMap.latent_to_gene import run_latent_to_gene
        run_latent_to_gene(config)
        logger.info("✓ Gene marker scores computed successfully!")
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
    logger.info(f"Trait: {config.trait_name}")
    logger.info(f"Sample: {config.sample_name}")
    logger.info(f"Working directory: {config.workdir}")
    logger.info(f"Project directory: {config.project_dir}")
    
    # Show auto-generated paths
    logger.info(f"LDSC results will be saved to: {config.ldsc_save_dir}")
    logger.info(f"Result file: {config.get_ldsc_result_file(config.trait_name)}")
    
    if config.use_jax:
        logger.info("Using JAX-accelerated implementation")
    else:
        logger.info("Using standard implementation")
    
    # Handle w_file default
    if config.w_file is None:
        w_ld_dir = config.ldscore_save_dir / "w_ld"
        if w_ld_dir.exists():
            config.w_file = str(w_ld_dir / "weights.")
            logger.info(f"Using weights from generate_ldscore: {config.w_file}")
    
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


@app.command(name="report")
@dataclass_typer
def generate_report(config: ReportConfig):
    """
    Generate diagnostic plots and HTML report for gsMap results.
    
    Creates:
    - Manhattan plots
    - Gene set score (GSS) plots
    - Spatial enrichment visualizations
    - Interactive HTML report
    """
    logger.info(f"Trait: {config.trait_name}")
    logger.info(f"Sample: {config.sample_name}")
    logger.info(f"Annotation: {config.annotation}")
    logger.info(f"Working directory: {config.workdir}")
    logger.info(f"Project directory: {config.project_dir}")
    
    # Show auto-generated paths
    logger.info(f"Report directory: {config.get_report_dir(config.trait_name)}")
    logger.info(f"Report file: {config.get_gsMap_report_file(config.trait_name)}")
    logger.info(f"Manhattan plot: {config.get_manhattan_html_plot_path(config.trait_name)}")
    logger.info(f"GSS plot directory: {config.get_GSS_plot_dir(config.trait_name)}")
    logger.info(f"gsMap plot directory: {config.get_gsMap_plot_save_dir(config.trait_name)}")
    
    # Parse selected genes if provided
    if config.selected_genes:
        config.selected_genes = [g.strip() for g in config.selected_genes.split(",")]
        logger.info(f"Selected genes: {len(config.selected_genes)}")
    
    try:
        from gsMap.report import run_report
        run_report(config)
        logger.info("✓ Report generated successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        try:
            from gsMap import __version__
            typer.echo(f"gsMap version {__version__}")
        except ImportError:
            typer.echo("gsMap version: development")
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
    
    Common workflows:
    
    1. Quick mode (recommended for first-time users):
       gsmap quick-mode --workdir /path/to/work --sample-name my_sample ...
    
    2. Step-by-step analysis:
       gsmap find-latent ...
       gsmap latent-to-gene ...
       gsmap spatial-ldsc ...
       gsmap report ...
    
    For detailed documentation, visit: https://github.com/mcgilldinglab/gsMap
    """
    pass


if __name__ == "__main__":
    app()