"""
Refactored configuration system using Pydantic Settings with CLI support.
"""
import sys
from pathlib import Path
from typing import Literal, Optional, Union, Any
from dataclasses import field

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict, CliApp
import yaml

from gsMap import __version__


# ============================================================================
# Base Settings Classes
# ============================================================================

class GsMapBaseSettings(BaseSettings):
    """Base settings class with common configuration."""
    
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_prog_name="gsMap",
        cli_use_class_docs_for_groups=True,
        cli_hide_none_type=True,
        cli_avoid_json=True,
        cli_enforce_required=True,
        cli_implicit_flags=True,
        cli_kebab_case=True,
        env_prefix="GSMAP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        validate_default=True,
        extra="forbid"
    )


class WorkdirMixin(BaseModel):
    """Mixin for settings that require workdir and sample paths."""
    
    workdir: Path = Field(
        ...,
        description="Path to the working directory",
        validation_alias="workdir"
    )
    sample_name: str = Field(
        ...,
        description="Name of the sample",
        validation_alias="sample_name"
    )
    project_name: Optional[str] = Field(
        None,
        description="Project name (optional)",
        validation_alias="project_name"
    )
    
    # Computed paths
    project_dir: Optional[Path] = None
    
    @model_validator(mode='after')
    def setup_paths(self) -> 'WorkdirMixin':
        """Setup project directory based on workdir and project_name."""
        if self.project_name:
            self.project_dir = self.workdir / self.project_name
        else:
            self.project_dir = self.workdir
        
        # Create directories if they don't exist
        self.project_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        return self
    
    @property
    def latent_dir(self) -> Path:
        """Directory for latent representations."""
        path = self.project_dir / "find_latent_representations"
        path.mkdir(parents=True, exist_ok=True, mode=0o755)
        return path
    
    @property
    def model_path(self) -> Path:
        """Path to the LGCN model."""
        path = self.project_dir / "LGCN_model" / "gsMap_LGCN_.pt"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        return path
    
    @property
    def ldscore_save_dir(self) -> Path:
        """Directory for LD scores."""
        path = self.project_dir / "generate_ldscore" / self.sample_name
        path.mkdir(parents=True, exist_ok=True, mode=0o755)
        return path
    
    @property
    def ldsc_save_dir(self) -> Path:
        """Directory for LDSC results."""
        path = self.project_dir / "spatial_ldsc" / self.sample_name
        path.mkdir(parents=True, exist_ok=True, mode=0o755)
        return path


# ============================================================================
# Command-specific Settings
# ============================================================================

class FindLatentRepresentationsSettings(GsMapBaseSettings, WorkdirMixin):
    """Settings for finding latent representations using GNN."""
    
    # Input files
    spe_file_list: str = Field(
        ...,
        description="List of input ST (.h5ad) files",
        validation_alias="spe_file_list"
    )
    data_layer: str = Field(
        "count",
        description="Gene expression data layer",
        validation_alias="data_layer"
    )
    spatial_key: str = Field(
        "spatial",
        description="Spatial key in adata.obsm storing spatial coordinates",
        validation_alias="spatial_key"
    )
    annotation: Optional[str] = Field(
        None,
        description="Annotation in adata.obs to use",
        validation_alias="annotation"
    )
    
    # Feature extraction parameters (LGCN)
    n_neighbors: int = Field(
        10,
        description="Number of neighbors for LGCN",
        validation_alias="n_neighbors"
    )
    K: int = Field(
        3,
        description="Graph convolution depth for LGCN",
        validation_alias="K"
    )
    feat_cell: int = Field(
        2000,
        description="Number of top variable features to retain",
        validation_alias="feat_cell"
    )
    pearson_residual: bool = Field(
        False,
        description="Take the residuals of the input data",
        validation_alias="pearson_residual"
    )
    
    # Model dimension parameters
    hidden_size: int = Field(
        128,
        description="Units in the first hidden layer",
        validation_alias="hidden_size"
    )
    embedding_size: int = Field(
        32,
        description="Size of the latent embedding layer",
        validation_alias="embedding_size"
    )
    
    # Transformer module parameters
    use_tf: bool = Field(
        False,
        description="Enable transformer module",
        validation_alias="use_tf"
    )
    module_dim: int = Field(
        30,
        description="Dimensionality of transformer modules",
        validation_alias="module_dim"
    )
    hidden_gmf: int = Field(
        128,
        description="Hidden units for global mean feature extractor",
        validation_alias="hidden_gmf"
    )
    n_modules: int = Field(
        16,
        description="Number of transformer modules",
        validation_alias="n_modules"
    )
    nhead: int = Field(
        4,
        description="Number of attention heads in transformer",
        validation_alias="nhead"
    )
    n_enc_layer: int = Field(
        2,
        description="Number of transformer encoder layers",
        validation_alias="n_enc_layer"
    )
    
    # Training parameters
    distribution: Literal["nb", "zinb", "gaussian"] = Field(
        "nb",
        description="Distribution type for loss calculation",
        validation_alias="distribution"
    )
    n_cell_training: int = Field(
        100000,
        description="Number of cells used for training",
        validation_alias="n_cell_training"
    )
    batch_size: int = Field(
        1024,
        description="Batch size for training",
        validation_alias="batch_size"
    )
    itermax: int = Field(
        100,
        description="Maximum number of training iterations",
        validation_alias="itermax"
    )
    patience: int = Field(
        10,
        description="Early stopping patience",
        validation_alias="patience"
    )
    two_stage: bool = Field(
        True,
        description="Tune the cell embeddings based on the provided annotation",
        validation_alias="two_stage"
    )
    do_sampling: bool = Field(
        True,
        description="Down-sampling cells in training",
        validation_alias="do_sampling"
    )
    
    # Homolog transformation
    homolog_file: Optional[Path] = Field(
        None,
        description="Path to homologous gene conversion file (optional)",
        validation_alias="homolog_file"
    )
    
    def cli_cmd(self) -> None:
        """Execute the find latent representations command."""
        from gsMap.find_latent_representation import run_find_latent_representation
        run_find_latent_representation(self)


class LatentToGeneSettings(GsMapBaseSettings, WorkdirMixin):
    """Settings for converting latent representations to gene scores."""
    
    annotation: Optional[str] = Field(
        None,
        description="Annotation in adata.obs to use",
        validation_alias="annotation"
    )
    no_expression_fraction: bool = Field(
        False,
        description="Skip expression fraction filtering",
        validation_alias="no_expression_fraction"
    )
    latent_representation: str = Field(
        "emb_gcn",
        description="Type of latent representation",
        validation_alias="latent_representation"
    )
    latent_representation_indv: str = Field(
        "emb",
        description="Type of individual latent representation",
        validation_alias="latent_representation_indv"
    )
    spatial_key: str = Field(
        "spatial",
        description="Spatial key in adata.obsm storing spatial coordinates",
        validation_alias="spatial_key"
    )
    num_anchor: int = Field(
        51,
        description="Number of anchor points",
        validation_alias="num_anchor"
    )
    num_neighbour: int = Field(
        21,
        description="Number of neighbors",
        validation_alias="num_neighbour"
    )
    num_neighbour_spatial: int = Field(
        201,
        description="Number of spatial neighbors",
        validation_alias="num_neighbour_spatial"
    )
    use_w: bool = Field(
        False,
        description="Use section-specific weights to account for across-section batch effects",
        validation_alias="use_w"
    )
    
    def cli_cmd(self) -> None:
        """Execute the latent to gene command."""
        from gsMap.latent_to_gene import run_latent_to_gene
        run_latent_to_gene(self)


class GenerateLDScoreSettings(GsMapBaseSettings, WorkdirMixin):
    """Settings for generating LD scores."""
    
    chrom: Union[int, Literal["all"]] = Field(
        ...,
        description='Chromosome id (1-22) or "all"',
        validation_alias="chrom"
    )
    bfile_root: Path = Field(
        ...,
        description="Root path for genotype plink bfiles (.bim, .bed, .fam)",
        validation_alias="bfile_root"
    )
    keep_snp_root: Optional[Path] = Field(
        None,
        description="Root path for SNP files",
        validation_alias="keep_snp_root"
    )
    gtf_annotation_file: Path = Field(
        ...,
        description="Path to GTF annotation file",
        validation_alias="gtf_annotation_file"
    )
    gene_window_size: int = Field(
        50000,
        description="Gene window size in base pairs",
        validation_alias="gene_window_size"
    )
    enhancer_annotation_file: Optional[Path] = Field(
        None,
        description="Path to enhancer annotation file (optional)",
        validation_alias="enhancer_annotation_file"
    )
    snp_multiple_enhancer_strategy: Literal["max_mkscore", "nearest_TSS"] = Field(
        "max_mkscore",
        description="Strategy for handling multiple enhancers per SNP",
        validation_alias="snp_multiple_enhancer_strategy"
    )
    gene_window_enhancer_priority: Optional[Literal["gene_window_first", "enhancer_first", "enhancer_only"]] = Field(
        None,
        description="Priority between gene window and enhancer annotations",
        validation_alias="gene_window_enhancer_priority"
    )
    spots_per_chunk: int = Field(
        1000,
        description="Number of spots per chunk",
        validation_alias="spots_per_chunk"
    )
    ld_wind: int = Field(
        1,
        description="LD window size",
        validation_alias="ld_wind"
    )
    ld_unit: Literal["SNP", "KB", "CM"] = Field(
        "CM",
        description="Unit for LD window",
        validation_alias="ld_unit"
    )
    additional_baseline_annotation: Optional[Path] = Field(
        None,
        description="Path of additional baseline annotations",
        validation_alias="additional_baseline_annotation"
    )
    
    @field_validator('chrom')
    @classmethod
    def validate_chrom(cls, v):
        """Validate chromosome value."""
        if isinstance(v, str):
            if v.lower() == "all":
                return "all"
            elif v.isdigit() and 1 <= int(v) <= 22:
                return int(v)
        elif isinstance(v, int) and 1 <= v <= 22:
            return v
        raise ValueError(f"Invalid chromosome value: {v}. Must be 1-22 or 'all'")
    
    def cli_cmd(self) -> None:
        """Execute the generate LD score command."""
        from gsMap.generate_ldscore import run_generate_ldscore
        run_generate_ldscore(self)


class SpatialLDSCSettings(GsMapBaseSettings, WorkdirMixin):
    """Settings for spatial LDSC analysis."""
    
    sumstats_file: Path = Field(
        ...,
        description="Path to GWAS summary statistics file",
        validation_alias="sumstats_file"
    )
    w_file: Optional[Path] = Field(
        None,
        description="Path to regression weight file",
        validation_alias="w_file"
    )
    trait_name: str = Field(
        ...,
        description="Name of the trait being analyzed",
        validation_alias="trait_name"
    )
    n_blocks: int = Field(
        200,
        description="Number of blocks for jackknife resampling",
        validation_alias="n_blocks"
    )
    chisq_max: Optional[int] = Field(
        None,
        description="Maximum chi-square value for filtering SNPs",
        validation_alias="chisq_max"
    )
    num_processes: int = Field(
        4,
        description="Number of processes for parallel computing",
        validation_alias="num_processes"
    )
    use_additional_baseline_annotation: bool = Field(
        True,
        description="Use additional baseline annotations when provided",
        validation_alias="use_additional_baseline_annotation"
    )
    use_jax: bool = Field(
        True,
        description="Use JAX-accelerated implementation",
        validation_alias="use_jax"
    )
    
    @model_validator(mode='after')
    def setup_w_file(self) -> 'SpatialLDSCSettings':
        """Setup w_file if not provided."""
        if self.w_file is None:
            w_ld_dir = self.ldscore_save_dir / "w_ld"
            if w_ld_dir.exists():
                self.w_file = w_ld_dir / "weights."
            else:
                raise ValueError(
                    "No w_file provided and no weights found in generate_ldscore output. "
                    "Either provide --w-file or run generate_ldscore first."
                )
        return self
    
    def cli_cmd(self) -> None:
        """Execute the spatial LDSC command."""
        if self.use_jax:
            from gsMap.spatial_ldsc_jax_final import run_spatial_ldsc_jax
            run_spatial_ldsc_jax(self)
        else:
            from gsMap.spatial_ldsc_multiple_sumstats import run_spatial_ldsc
            run_spatial_ldsc(self)


class CauchyCombinationSettings(GsMapBaseSettings, WorkdirMixin):
    """Settings for Cauchy combination analysis."""
    
    trait_name: str = Field(
        ...,
        description="Name of the trait being analyzed",
        validation_alias="trait_name"
    )
    annotation: str = Field(
        ...,
        description="Name of the annotation in adata.obs to use",
        validation_alias="annotation"
    )
    sample_name_list: Optional[list[str]] = Field(
        None,
        description="List of sample names to process",
        validation_alias="sample_name_list"
    )
    output_file: Optional[Path] = Field(
        None,
        description="Path to save the combined Cauchy results",
        validation_alias="output_file"
    )
    
    def cli_cmd(self) -> None:
        """Execute the Cauchy combination command."""
        from gsMap.cauchy_combination_test import run_Cauchy_combination
        run_Cauchy_combination(self)


class ReportSettings(GsMapBaseSettings, WorkdirMixin):
    """Settings for generating reports."""
    
    trait_name: str = Field(
        ...,
        description="Name of the trait to generate the report for",
        validation_alias="trait_name"
    )
    annotation: str = Field(
        ...,
        description="Annotation layer name",
        validation_alias="annotation"
    )
    top_corr_genes: int = Field(
        50,
        description="Number of top correlated genes to display",
        validation_alias="top_corr_genes"
    )
    selected_genes: Optional[list[str]] = Field(
        None,
        description="List of specific genes to include in the report",
        validation_alias="selected_genes"
    )
    sumstats_file: Path = Field(
        ...,
        description="Path to GWAS summary statistics file",
        validation_alias="sumstats_file"
    )
    fig_width: Optional[int] = Field(
        None,
        description="Width of the generated figures in pixels",
        validation_alias="fig_width"
    )
    fig_height: Optional[int] = Field(
        None,
        description="Height of the generated figures in pixels",
        validation_alias="fig_height"
    )
    point_size: Optional[int] = Field(
        None,
        description="Point size for the figures",
        validation_alias="point_size"
    )
    fig_style: Literal["dark", "light"] = Field(
        "light",
        description="Style of the generated figures",
        validation_alias="fig_style"
    )
    
    def cli_cmd(self) -> None:
        """Execute the report generation command."""
        from gsMap.report import run_report
        run_report(self)


class FormatSumstatsSettings(GsMapBaseSettings):
    """Settings for formatting GWAS summary statistics."""
    
    sumstats: Path = Field(
        ...,
        description="Path to GWAS summary data",
        validation_alias="sumstats"
    )
    out: Path = Field(
        ...,
        description="Path to save the formatted GWAS data",
        validation_alias="out"
    )
    
    # Column name specifications
    snp: Optional[str] = Field(
        None,
        description="Name of SNP column",
        validation_alias="snp"
    )
    a1: Optional[str] = Field(
        None,
        description="Name of effect allele column",
        validation_alias="a1"
    )
    a2: Optional[str] = Field(
        None,
        description="Name of non-effect allele column",
        validation_alias="a2"
    )
    info: Optional[str] = Field(
        None,
        description="Name of info column",
        validation_alias="info"
    )
    beta: Optional[str] = Field(
        None,
        description="Name of GWAS beta column",
        validation_alias="beta"
    )
    se: Optional[str] = Field(
        None,
        description="Name of GWAS standard error of beta column",
        validation_alias="se"
    )
    p: Optional[str] = Field(
        None,
        description="Name of p-value column",
        validation_alias="p"
    )
    frq: Optional[str] = Field(
        None,
        description="Name of A1 frequency column",
        validation_alias="frq"
    )
    n: Optional[Union[str, int]] = Field(
        None,
        description="Name of sample size column or sample size value",
        validation_alias="n"
    )
    z: Optional[str] = Field(
        None,
        description="Name of GWAS Z-statistics column",
        validation_alias="z"
    )
    OR: Optional[str] = Field(
        None,
        description="Name of GWAS OR column",
        validation_alias="OR"
    )
    se_OR: Optional[str] = Field(
        None,
        description="Name of standard error of OR column",
        validation_alias="se_OR"
    )
    
    # SNP to rsid conversion
    chr: str = Field(
        "Chr",
        description="Name of SNP chromosome column",
        validation_alias="chr"
    )
    pos: str = Field(
        "Pos",
        description="Name of SNP positions column",
        validation_alias="pos"
    )
    dbsnp: Optional[Path] = Field(
        None,
        description="Path to reference dbSNP file",
        validation_alias="dbsnp"
    )
    chunksize: int = Field(
        1000000,
        description="Chunk size for loading dbSNP file",
        validation_alias="chunksize"
    )
    
    # Output format and quality
    format: Literal["gsMap", "COJO"] = Field(
        "gsMap",
        description="Format of output data",
        validation_alias="format"
    )
    info_min: float = Field(
        0.9,
        description="Minimum INFO score",
        validation_alias="info_min"
    )
    maf_min: float = Field(
        0.01,
        description="Minimum MAF",
        validation_alias="maf_min"
    )
    keep_chr_pos: bool = Field(
        False,
        description="Keep SNP chromosome and position columns in the output data",
        validation_alias="keep_chr_pos"
    )
    
    def cli_cmd(self) -> None:
        """Execute the format sumstats command."""
        from gsMap.format_sumstats import gwas_format
        gwas_format(self)


class QuickModeSettings(GsMapBaseSettings, WorkdirMixin):
    """Settings for running gsMap in quick mode."""
    
    gsMap_resource_dir: Path = Field(
        ...,
        description="Directory containing gsMap resources",
        validation_alias="gsMap_resource_dir"
    )
    hdf5_path: Path = Field(
        ...,
        description="Path to the input spatial transcriptomics data (H5AD format)",
        validation_alias="hdf5_path"
    )
    annotation: str = Field(
        ...,
        description="Name of the annotation in adata.obs to use",
        validation_alias="annotation"
    )
    data_layer: str = Field(
        "counts",
        description="Data layer for gene expression",
        validation_alias="data_layer"
    )
    trait_name: Optional[str] = Field(
        None,
        description="Name of the trait for GWAS analysis",
        validation_alias="trait_name"
    )
    sumstats_file: Optional[Path] = Field(
        None,
        description="Path to GWAS summary statistics file",
        validation_alias="sumstats_file"
    )
    sumstats_config_file: Optional[Path] = Field(
        None,
        description="Path to GWAS summary statistics config file",
        validation_alias="sumstats_config_file"
    )
    homolog_file: Optional[Path] = Field(
        None,
        description="Path to homologous gene conversion file",
        validation_alias="homolog_file"
    )
    max_processes: int = Field(
        10,
        description="Maximum number of processes for parallel execution",
        validation_alias="max_processes"
    )
    latent_representation: Optional[str] = Field(
        None,
        description="Type of latent representation",
        validation_alias="latent_representation"
    )
    num_neighbour: int = Field(
        21,
        description="Number of neighbors",
        validation_alias="num_neighbour"
    )
    num_neighbour_spatial: int = Field(
        101,
        description="Number of spatial neighbors",
        validation_alias="num_neighbour_spatial"
    )
    gM_slices: Optional[Path] = Field(
        None,
        description="Path to the slice mean file",
        validation_alias="gM_slices"
    )
    pearson_residuals: bool = Field(
        False,
        description="Use pearson residuals",
        validation_alias="pearson_residuals"
    )
    use_jax: bool = Field(
        True,
        description="Use JAX-accelerated spatial LDSC implementation",
        validation_alias="use_jax"
    )
    
    @model_validator(mode='after')
    def validate_sumstats(self) -> 'QuickModeSettings':
        """Validate summary statistics configuration."""
        if self.sumstats_file is None and self.sumstats_config_file is None:
            raise ValueError("Either sumstats_file or sumstats_config_file is required")
        if self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError("Only one of sumstats_file or sumstats_config_file should be provided")
        if self.sumstats_file is not None and self.trait_name is None:
            raise ValueError("trait_name is required when sumstats_file is provided")
        return self
    
    def cli_cmd(self) -> None:
        """Execute the quick mode pipeline."""
        from gsMap.run_all_mode import run_pipeline
        run_pipeline(self)


class CreateSliceMeanSettings(GsMapBaseSettings):
    """Settings for creating slice mean from multiple samples."""
    
    sample_name_list: list[str] = Field(
        ...,
        description="List of sample names to process",
        validation_alias="sample_name_list"
    )
    h5ad_list: Optional[list[Path]] = Field(
        None,
        description="List of h5ad file paths",
        validation_alias="h5ad_list"
    )
    h5ad_yaml: Optional[Path] = Field(
        None,
        description="Path to YAML file containing sample names and h5ad paths",
        validation_alias="h5ad_yaml"
    )
    slice_mean_output_file: Path = Field(
        ...,
        description="Path to the output file for the slice mean",
        validation_alias="slice_mean_output_file"
    )
    homolog_file: Optional[Path] = Field(
        None,
        description="Path to homologous gene conversion file",
        validation_alias="homolog_file"
    )
    data_layer: str = Field(
        "counts",
        description="Data layer for gene expression",
        validation_alias="data_layer"
    )
    
    @model_validator(mode='after')
    def validate_inputs(self) -> 'CreateSliceMeanSettings':
        """Validate input configuration."""
        if self.h5ad_list is None and self.h5ad_yaml is None:
            raise ValueError("Either h5ad_list or h5ad_yaml must be provided")
        if self.h5ad_yaml is not None:
            # Load and validate YAML file
            with open(self.h5ad_yaml) as f:
                h5ad_dict = yaml.safe_load(f)
            if len(h5ad_dict) != len(set(h5ad_dict)):
                raise ValueError("Sample names must be unique")
            if len(h5ad_dict) < 2:
                raise ValueError("At least two samples are required")
        return self
    
    def cli_cmd(self) -> None:
        """Execute the create slice mean command."""
        from gsMap.create_slice_mean import run_create_slice_mean
        run_create_slice_mean(self)


# ============================================================================
# CLI Application Entry Point
# ============================================================================

def main():
    """Main entry point for the gsMap CLI."""
    import pyfiglet
    
    # Display banner
    banner = pyfiglet.figlet_format("gsMap", font="doom", width=80, justify="center").rstrip()
    print(banner, flush=True)
    version_str = f"Version: {__version__}"
    print(version_str.center(80), flush=True)
    print("=" * 80, flush=True)
    
    # Parse command from sys.argv
    if len(sys.argv) < 2:
        print("Usage: gsMap <command> [options]")
        print("\nAvailable commands:")
        print("  quick_mode                    - Run the entire pipeline in quick mode")
        print("  find_latent_representations   - Find latent representations using GNN")
        print("  latent_to_gene               - Convert latent representations to gene scores")
        print("  generate_ldscore             - Generate LD scores")
        print("  spatial_ldsc                 - Run spatial LDSC analysis")
        print("  cauchy_combination           - Run Cauchy combination")
        print("  report                       - Generate diagnostic reports")
        print("  format_sumstats              - Format GWAS summary statistics")
        print("  create_slice_mean            - Create slice mean from multiple samples")
        print("\nUse 'gsMap <command> --help' for command-specific help")
        sys.exit(1)
    
    command = sys.argv[1]
    
    # Map commands to settings classes
    command_map = {
        "quick_mode": QuickModeSettings,
        "find_latent_representations": FindLatentRepresentationsSettings,
        "latent_to_gene": LatentToGeneSettings,
        "generate_ldscore": GenerateLDScoreSettings,
        "spatial_ldsc": SpatialLDSCSettings,
        "cauchy_combination": CauchyCombinationSettings,
        "report": ReportSettings,
        "format_sumstats": FormatSumstatsSettings,
        "create_slice_mean": CreateSliceMeanSettings,
    }
    
    if command not in command_map:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(command_map.keys())}")
        sys.exit(1)
    
    # Remove command from argv and run
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    settings_class = command_map[command]
    
    # Run the command using CliApp
    CliApp.run(settings_class)


if __name__ == "__main__":
    main()