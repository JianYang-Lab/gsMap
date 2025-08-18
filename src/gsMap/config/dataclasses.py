"""
Configuration dataclasses for gsMap commands.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated

import typer

from .base import ConfigWithAutoPaths


@dataclass
class RunAllModeConfig(ConfigWithAutoPaths):
    """Configuration for running the complete gsMap pipeline."""
    
    # Required from parent (no defaults)
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
    
    # Required paths and configurations
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
    
    # Optional parameter - must be after required fields
    project_name: str = None
    
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
    
    # Additional fields for compatibility
    latent_representation: Optional[str] = None
    gM_slices: Optional[str] = None
    sumstats_config_file: Optional[str] = None


@dataclass
class FindLatentRepresentationsConfig(ConfigWithAutoPaths):
    """Configuration for finding latent representations."""
    
    # Required from parent
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
    
    # Optional - must be after required fields
    project_name: str = None
    
    data_layer: Annotated[str, typer.Option(
        help="Gene expression data layer"
    )] = "count"
    
    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm storing spatial coordinates"
    )] = "spatial"
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation in adata.obs to use"
    )] = None
    
    # Feature extraction parameters
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
    
    # Model parameters
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
    
    # Transformer parameters
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
class LatentToGeneConfig(ConfigWithAutoPaths):
    """Configuration for latent to gene mapping."""
    
    # Required from parent
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
    
    # Optional - must be after required fields
    project_name: str = None
    
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
    
    # Additional fields
    gM_slices: Optional[str] = None
    homolog_file: Optional[Path] = None
    species: Optional[str] = None


@dataclass
class SpatialLDSCConfig(ConfigWithAutoPaths):
    """Configuration for spatial LDSC analysis."""
    
    # Required from parent
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
    
    # Optional - must be after required fields
    project_name: str = None
    
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
    
    # Hidden parameters
    sumstats_config_file: Optional[str] = None
    not_M_5_50: bool = False
    all_chunk: Optional[int] = None
    chunk_range: Optional[tuple[int, int]] = None
    ldscore_save_format: str = "feather"
    spots_per_chunk_quick_mode: int = 1000
    snp_gene_weight_adata_path: Optional[str] = None


@dataclass
class ReportConfig(ConfigWithAutoPaths):
    """Configuration for generating reports."""
    
    # Required from parent
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
    
    # Optional - must be after required fields
    project_name: str = None
    
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
    plot_type: str = "all"


# Add other config classes as needed...