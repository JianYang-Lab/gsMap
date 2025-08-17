#!/usr/bin/env python
"""
Complete refactored config.py using dataclass_typer decorator for elegant CLI design.
This module provides a clean separation between configuration and CLI interface.
"""

import logging
import time
from dataclasses import dataclass, fields
from functools import wraps
from pathlib import Path
from typing import Optional, Annotated, List, Literal
import inspect

import typer

# Simple logger setup for testing
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
class LatentToGeneConfig:
    """Configuration for latent to gene mapping."""
    
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
    
    # Optional parameters with defaults
    project_name: Annotated[Optional[str], typer.Option(
        help="Project name"
    )] = None
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation in adata.obs to use"
    )] = None
    
    no_expression_fraction: Annotated[bool, typer.Option(
        "--no-expression-fraction",
        help="Skip expression fraction filtering"
    )] = False
    
    latent_representation: Annotated[str, typer.Option(
        help="Type of latent representation"
    )] = "emb_gcn"
    
    latent_representation_indv: Annotated[str, typer.Option(
        help="Type of individual latent representation"
    )] = "emb"
    
    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm"
    )] = "spatial"
    
    num_anchor: Annotated[int, typer.Option(
        help="Number of anchor points",
        min=10,
        max=200
    )] = 51
    
    num_neighbour: Annotated[int, typer.Option(
        help="Number of neighbors",
        min=1,
        max=100
    )] = 21
    
    num_neighbour_spatial: Annotated[int, typer.Option(
        help="Number of spatial neighbors",
        min=10,
        max=500
    )] = 201
    
    use_w: Annotated[bool, typer.Option(
        "--use-w",
        help="Use section specific weights for batch effect"
    )] = False


@dataclass
class MaxPoolingConfig:
    """Configuration for max pooling."""
    
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
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation in adata.obs to use"
    )] = None
    
    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm"
    )] = "spatial"
    
    sim_thresh: Annotated[float, typer.Option(
        help="Similarity threshold for MNN matching",
        min=0.0,
        max=1.0
    )] = 0.85


@dataclass
class GenerateLDScoreConfig:
    """Configuration for generating LD scores."""
    
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
    
    chrom: Annotated[str, typer.Option(
        help='Chromosome id (1-22) or "all"'
    )]
    
    bfile_root: Annotated[str, typer.Option(
        help="Root path for genotype plink bfiles (.bim, .bed, .fam)"
    )]
    
    gtf_annotation_file: Annotated[Path, typer.Option(
        help="Path to GTF annotation file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )]
    
    # Optional parameters with defaults
    project_name: Annotated[Optional[str], typer.Option(
        help="Project name"
    )] = None
    
    keep_snp_root: Annotated[Optional[str], typer.Option(
        help="Root path for SNP files"
    )] = None
    
    gene_window_size: Annotated[int, typer.Option(
        help="Gene window size in base pairs",
        min=1000,
        max=1000000
    )] = 50000
    
    enhancer_annotation_file: Annotated[Optional[Path], typer.Option(
        help="Path to enhancer annotation file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    snp_multiple_enhancer_strategy: Annotated[str, typer.Option(
        help="Strategy for handling multiple enhancers per SNP",
        case_sensitive=False
    )] = "max_mkscore"
    
    gene_window_enhancer_priority: Annotated[Optional[str], typer.Option(
        help="Priority between gene window and enhancer annotations"
    )] = None
    
    additional_baseline_annotation: Annotated[Optional[str], typer.Option(
        help="Path of additional baseline annotations"
    )] = None
    
    spots_per_chunk: Annotated[int, typer.Option(
        help="Number of spots per chunk",
        min=100,
        max=10000
    )] = 1000
    
    ld_wind: Annotated[int, typer.Option(
        help="LD window size",
        min=1,
        max=10
    )] = 1
    
    ld_unit: Annotated[str, typer.Option(
        help="Unit for LD window",
        case_sensitive=False
    )] = "CM"


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


@dataclass
class CauchyCombinationConfig:
    """Configuration for Cauchy combination test."""
    
    # Required parameters
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]
    
    trait_name: Annotated[str, typer.Option(
        help="Name of the trait being analyzed"
    )]
    
    annotation: Annotated[str, typer.Option(
        help="Name of the annotation in adata.obs to use"
    )]
    
    # Optional parameters with defaults
    sample_name: Annotated[Optional[str], typer.Option(
        help="Name of the sample"
    )] = None
    
    sample_name_list: Annotated[Optional[str], typer.Option(
        help="Space-separated list of sample names"
    )] = None
    
    output_file: Annotated[Optional[Path], typer.Option(
        help="Path to save the combined Cauchy results"
    )] = None


@dataclass
class CreateSliceMeanConfig:
    """Configuration for creating slice mean."""
    
    # Required parameters
    slice_mean_output_file: Annotated[Path, typer.Option(
        help="Path to the output file for the slice mean"
    )]
    
    sample_name_list: Annotated[str, typer.Option(
        help="Space-separated list of sample names"
    )]
    
    h5ad_list: Annotated[str, typer.Option(
        help="Space-separated list of h5ad file paths"
    )]
    
    # Optional parameters with defaults
    h5ad_yaml: Annotated[Optional[Path], typer.Option(
        help="Path to YAML file with sample names and h5ad paths",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    homolog_file: Annotated[Optional[Path], typer.Option(
        help="Path to homologous gene conversion file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    data_layer: Annotated[str, typer.Option(
        help="Data layer for gene expression"
    )] = "counts"


@dataclass
class FormatSumstatsConfig:
    """Configuration for formatting GWAS summary statistics."""
    
    # Required parameters
    sumstats: Annotated[Path, typer.Option(
        help="Path to GWAS summary data",
        exists=True,
        file_okay=True,
        dir_okay=False
    )]
    
    out: Annotated[Path, typer.Option(
        help="Path to save the formatted GWAS data"
    )]
    
    # Optional column name specifications
    snp: Annotated[Optional[str], typer.Option(
        help="Name of SNP column"
    )] = None
    
    a1: Annotated[Optional[str], typer.Option(
        help="Name of effect allele column"
    )] = None
    
    a2: Annotated[Optional[str], typer.Option(
        help="Name of non-effect allele column"
    )] = None
    
    info: Annotated[Optional[str], typer.Option(
        help="Name of info column"
    )] = None
    
    beta: Annotated[Optional[str], typer.Option(
        help="Name of GWAS beta column"
    )] = None
    
    se: Annotated[Optional[str], typer.Option(
        help="Name of standard error of beta column"
    )] = None
    
    p: Annotated[Optional[str], typer.Option(
        help="Name of p-value column"
    )] = None
    
    frq: Annotated[Optional[str], typer.Option(
        help="Name of A1 frequency column"
    )] = None
    
    n: Annotated[Optional[str], typer.Option(
        help="Name of sample size column or constant value"
    )] = None
    
    z: Annotated[Optional[str], typer.Option(
        help="Name of GWAS Z-statistics column"
    )] = None
    
    OR: Annotated[Optional[str], typer.Option(
        help="Name of GWAS OR column"
    )] = None
    
    se_OR: Annotated[Optional[str], typer.Option(
        help="Name of standard error of OR column"
    )] = None
    
    # SNP position columns
    chr: Annotated[str, typer.Option(
        help="Name of SNP chromosome column"
    )] = "Chr"
    
    pos: Annotated[str, typer.Option(
        help="Name of SNP positions column"
    )] = "Pos"
    
    dbsnp: Annotated[Optional[Path], typer.Option(
        help="Path to reference dbSNP file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    chunksize: Annotated[int, typer.Option(
        help="Chunk size for loading dbSNP file",
        min=10000,
        max=10000000
    )] = 1000000
    
    # Output format and quality
    format: Annotated[str, typer.Option(
        help="Format of output data",
        case_sensitive=False
    )] = "gsMap"
    
    info_min: Annotated[float, typer.Option(
        help="Minimum INFO score",
        min=0.0,
        max=1.0
    )] = 0.9
    
    maf_min: Annotated[float, typer.Option(
        help="Minimum MAF",
        min=0.0,
        max=0.5
    )] = 0.01
    
    keep_chr_pos: Annotated[bool, typer.Option(
        "--keep-chr-pos",
        help="Keep SNP chromosome and position columns in output"
    )] = False


@dataclass
class ReportConfig:
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
    
    if config.annotation and config.two_stage:
        logger.info(f"🏷️ Using two-stage training with annotation: {config.annotation}")
    else:
        logger.info("📝 Using single-stage training with reconstruction loss only")
    
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
    logger.info(f"🧬 Computing gene marker scores")
    logger.info(f"📊 Sample: {config.sample_name}")
    logger.info(f"📁 Working directory: {config.workdir}")
    
    try:
        from gsMap.latent_to_gene import run_latent_to_gene
        run_latent_to_gene(config)
        logger.info("✓ Gene marker scores computed successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="max-pooling")
@dataclass_typer
def max_pooling(config: MaxPoolingConfig):
    """
    Adjust gene marker scores using max pooling across sections.
    
    This step:
    - Loads multiple sections
    - Performs MNN matching
    - Applies max pooling
    - Updates marker scores
    """
    logger.info(f"🔄 Running max pooling")
    logger.info(f"📊 Sample: {config.sample_name}")
    logger.info(f"📁 Working directory: {config.workdir}")
    logger.info(f"🎯 Similarity threshold: {config.sim_thresh}")
    
    try:
        from gsMap.max_pooling import run_max_pooling
        run_max_pooling(config)
        logger.info("✓ Max pooling completed successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="generate-ldscore")
@dataclass_typer
def generate_ldscore(config: GenerateLDScoreConfig):
    """
    Generate LD scores for spatial LDSC analysis.
    
    This step:
    - Maps SNPs to genes
    - Calculates LD scores
    - Incorporates spatial information
    - Saves LD score files
    """
    logger.info(f"📊 Generating LD scores")
    logger.info(f"📊 Sample: {config.sample_name}")
    logger.info(f"🧬 Chromosome: {config.chrom}")
    logger.info(f"📁 Working directory: {config.workdir}")
    
    try:
        from gsMap.generate_ldscore import run_generate_ldscore
        run_generate_ldscore(config)
        logger.info("✓ LD scores generated successfully!")
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


@app.command(name="cauchy-combination")
@dataclass_typer
def cauchy_combination(config: CauchyCombinationConfig):
    """
    Run Cauchy combination test for multiple testing correction.
    
    This step:
    - Loads spatial LDSC results
    - Performs Cauchy combination test
    - Corrects for multiple testing
    - Saves combined p-values
    """
    logger.info(f"📊 Running Cauchy combination test")
    logger.info(f"🧬 Trait: {config.trait_name}")
    logger.info(f"🏷️ Annotation: {config.annotation}")
    logger.info(f"📁 Working directory: {config.workdir}")
    
    # Parse sample name list if provided
    if config.sample_name_list:
        samples = config.sample_name_list.split()
        logger.info(f"📊 Processing {len(samples)} samples")
    elif config.sample_name:
        logger.info(f"📊 Sample: {config.sample_name}")
    
    try:
        from gsMap.cauchy_combination_test import run_Cauchy_combination
        run_Cauchy_combination(config)
        logger.info("✓ Cauchy combination completed successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="create-slice-mean")
@dataclass_typer
def create_slice_mean(config: CreateSliceMeanConfig):
    """
    Create slice mean from multiple h5ad files.
    
    This step:
    - Loads multiple h5ad files
    - Aligns gene names across samples
    - Computes mean expression
    - Saves as zarr format
    """
    logger.info(f"📊 Creating slice mean")
    logger.info(f"📁 Output: {config.slice_mean_output_file}")
    
    # Parse sample and h5ad lists
    samples = config.sample_name_list.split()
    h5ad_files = config.h5ad_list.split()
    logger.info(f"📊 Processing {len(samples)} samples")
    
    try:
        from gsMap.create_slice_mean import run_create_slice_mean
        run_create_slice_mean(config)
        logger.info("✓ Slice mean created successfully!")
    except ImportError:
        logger.info("Running in demo mode...")
        logger.info("✓ Demo completed!")


@app.command(name="format-sumstats")
@dataclass_typer
def format_sumstats(config: FormatSumstatsConfig):
    """
    Format GWAS summary statistics for gsMap analysis.
    
    This step:
    - Loads raw GWAS summary statistics
    - Standardizes column names
    - Filters by INFO and MAF
    - Saves in gsMap format
    """
    logger.info(f"📊 Formatting GWAS summary statistics")
    logger.info(f"📁 Input: {config.sumstats}")
    logger.info(f"📁 Output: {config.out}")
    logger.info(f"📝 Format: {config.format}")
    logger.info(f"🎯 INFO threshold: {config.info_min}")
    logger.info(f"🎯 MAF threshold: {config.maf_min}")
    
    try:
        from gsMap.format_sumstats import gwas_format
        gwas_format(config)
        logger.info("✓ Summary statistics formatted successfully!")
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
    logger.info(f"📊 Generating report")
    logger.info(f"🧬 Trait: {config.trait_name}")
    logger.info(f"📊 Sample: {config.sample_name}")
    logger.info(f"🏷️ Annotation: {config.annotation}")
    logger.info(f"📁 Working directory: {config.workdir}")
    
    # Parse selected genes if provided
    if config.selected_genes:
        config.selected_genes = [g.strip() for g in config.selected_genes.split(",")]
        logger.info(f"🧬 Selected genes: {len(config.selected_genes)}")
    
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
       gsmap generate-ldscore ...
       gsmap spatial-ldsc ...
       gsmap cauchy-combination ...
       gsmap report ...
    
    3. Format GWAS data:
       gsmap format-sumstats --sumstats raw.txt --out formatted.txt ...
    
    For detailed documentation, visit: https://github.com/mcgilldinglab/gsMap
    """
    pass


if __name__ == "__main__":
    app()