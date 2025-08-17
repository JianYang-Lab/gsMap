"""
Refactored configuration system using Pydantic Settings with CLI support.
Includes subcommand support and YAML configuration management.
"""
import sys
from pathlib import Path
from typing import Literal, Optional, Union, Any, Dict
import json

from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from pydantic_settings import (
    BaseSettings, 
    SettingsConfigDict, 
    CliApp,
    CliSubCommand,
    CliPositionalArg,
    get_subcommand,
    SettingsError
)
import yaml

from gsMap import __version__


# ============================================================================
# Mixin Classes (without BaseModel inheritance to avoid MRO issues)
# ============================================================================

class ConfigFileMixin:
    """Mixin for YAML configuration management."""
    
    config_file: Optional[Path] = Field(
        default=None,
        description="Path to YAML configuration file to load settings from"
    )
    save_config: Optional[Path] = Field(
        default=None,
        description="Path to save current configuration to YAML file"
    )
    
    def save_to_yaml(self, path: Path) -> None:
        """Save current configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get model dump excluding None values and computed fields
        config_dict = self.model_dump(
            exclude_none=True,
            exclude={'config_file', 'save_config', 'project_dir'},
            mode='json'  # Ensures Path objects are converted to strings
        )
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Configuration saved to: {path}")
    
    @classmethod
    def load_from_yaml(cls, path: Path, **overrides):
        """Load configuration from YAML file with optional overrides."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Merge with overrides
        config_dict.update(overrides)
        
        return cls(**config_dict)
    
    @classmethod
    def create_template(cls, path: Path) -> None:
        """Create a template configuration file with example values."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get field info and create example dict
        example_dict = {}
        for field_name, field_info in cls.model_fields.items():
            # Skip config management fields
            if field_name in ['config_file', 'save_config', 'project_dir']:
                continue
                
            if field_info.is_required():
                # Add placeholder for required fields
                if 'Path' in str(field_info.annotation):
                    example_dict[field_name] = f"/path/to/{field_name}"
                elif 'str' in str(field_info.annotation):
                    example_dict[field_name] = f"example_{field_name}"
                elif 'int' in str(field_info.annotation):
                    example_dict[field_name] = 100
                elif 'float' in str(field_info.annotation):
                    example_dict[field_name] = 0.5
                elif 'bool' in str(field_info.annotation):
                    example_dict[field_name] = False
                elif 'list' in str(field_info.annotation):
                    example_dict[field_name] = ["example1", "example2"]
                else:
                    example_dict[field_name] = "REQUIRED"
            else:
                # Add default values for optional fields
                default = field_info.get_default()
                if default is not None and default != ...:
                    example_dict[field_name] = default
        
        with open(path, 'w') as f:
            yaml.dump(example_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Template configuration created: {path}")
        print(f"  Edit this file and use with --config-file {path}")


class WorkdirFieldsMixin:
    """Mixin for workdir-related fields and properties."""
    
    workdir: Path = Field(
        ...,
        description="Path to the working directory"
    )
    sample_name: str = Field(
        ...,
        description="Name of the sample"
    )
    project_name: Optional[str] = Field(
        default=None,
        description="Project name (optional)"
    )
    
    # Will be set in validator
    project_dir: Optional[Path] = None
    
    @model_validator(mode='after')
    def setup_paths(self):
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
# Base Command Class
# ============================================================================

class BaseCommand(BaseModel, ConfigFileMixin):
    """Base class for all commands with config file support."""
    
    def cli_cmd(self) -> None:
        """Default CLI command handler."""
        # If save_config is specified, just save template
        if self.save_config:
            self.save_to_yaml(self.save_config)
            return
        
        # Otherwise, execute the command
        self.execute()
    
    def execute(self):
        """Execute the actual command. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")


class BaseWorkdirCommand(BaseModel, ConfigFileMixin, WorkdirFieldsMixin):
    """Base class for commands that require workdir."""
    
    def cli_cmd(self) -> None:
        """Default CLI command handler."""
        # If save_config is specified, just save template
        if self.save_config:
            self.save_to_yaml(self.save_config)
            return
        
        # Otherwise, execute the command
        self.execute()
    
    def execute(self):
        """Execute the actual command. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement execute()")


# ============================================================================
# Subcommand Models
# ============================================================================

class FindLatentCommand(BaseWorkdirCommand):
    """Find latent representations of each spot using GNN."""
    
    # Input files
    spe_file_list: str = Field(
        ...,
        description="List of input ST (.h5ad) files"
    )
    data_layer: str = Field(
        default="count",
        description="Gene expression data layer"
    )
    spatial_key: str = Field(
        default="spatial",
        description="Spatial key in adata.obsm storing spatial coordinates"
    )
    annotation: Optional[str] = Field(
        default=None,
        description="Annotation in adata.obs to use"
    )
    
    # Feature extraction parameters (LGCN)
    n_neighbors: int = Field(
        default=10,
        description="Number of neighbors for LGCN"
    )
    K: int = Field(
        default=3,
        description="Graph convolution depth for LGCN"
    )
    feat_cell: int = Field(
        default=2000,
        description="Number of top variable features to retain"
    )
    pearson_residual: bool = Field(
        default=False,
        description="Take the residuals of the input data"
    )
    
    # Model dimension parameters
    hidden_size: int = Field(
        default=128,
        description="Units in the first hidden layer"
    )
    embedding_size: int = Field(
        default=32,
        description="Size of the latent embedding layer"
    )
    
    # Transformer module parameters
    use_tf: bool = Field(
        default=False,
        description="Enable transformer module"
    )
    module_dim: int = Field(
        default=30,
        description="Dimensionality of transformer modules"
    )
    hidden_gmf: int = Field(
        default=128,
        description="Hidden units for global mean feature extractor"
    )
    n_modules: int = Field(
        default=16,
        description="Number of transformer modules"
    )
    nhead: int = Field(
        default=4,
        description="Number of attention heads in transformer"
    )
    n_enc_layer: int = Field(
        default=2,
        description="Number of transformer encoder layers"
    )
    
    # Training parameters
    distribution: Literal["nb", "zinb", "gaussian"] = Field(
        default="nb",
        description="Distribution type for loss calculation"
    )
    n_cell_training: int = Field(
        default=100000,
        description="Number of cells used for training"
    )
    batch_size: int = Field(
        default=1024,
        description="Batch size for training"
    )
    itermax: int = Field(
        default=100,
        description="Maximum number of training iterations"
    )
    patience: int = Field(
        default=10,
        description="Early stopping patience"
    )
    two_stage: bool = Field(
        default=True,
        description="Tune the cell embeddings based on the provided annotation"
    )
    do_sampling: bool = Field(
        default=True,
        description="Down-sampling cells in training"
    )
    
    # Homolog transformation
    homolog_file: Optional[Path] = Field(
        default=None,
        description="Path to homologous gene conversion file (optional)"
    )
    
    def execute(self) -> None:
        """Execute the find latent representations command."""
        from gsMap.find_latent_representation import run_find_latent_representation
        run_find_latent_representation(self)


class LatentToGeneCommand(BaseWorkdirCommand):
    """Estimate gene marker scores for each spot using latent representations."""
    
    annotation: Optional[str] = Field(
        default=None,
        description="Annotation in adata.obs to use"
    )
    no_expression_fraction: bool = Field(
        default=False,
        description="Skip expression fraction filtering"
    )
    latent_representation: str = Field(
        default="emb_gcn",
        description="Type of latent representation"
    )
    latent_representation_indv: str = Field(
        default="emb",
        description="Type of individual latent representation"
    )
    spatial_key: str = Field(
        default="spatial",
        description="Spatial key in adata.obsm storing spatial coordinates"
    )
    num_anchor: int = Field(
        default=51,
        description="Number of anchor points"
    )
    num_neighbour: int = Field(
        default=21,
        description="Number of neighbors"
    )
    num_neighbour_spatial: int = Field(
        default=201,
        description="Number of spatial neighbors"
    )
    use_w: bool = Field(
        default=False,
        description="Use section-specific weights to account for across-section batch effects"
    )
    
    def execute(self) -> None:
        """Execute the latent to gene command."""
        from gsMap.latent_to_gene import run_latent_to_gene
        run_latent_to_gene(self)


class GenerateLDScoreCommand(BaseWorkdirCommand):
    """Generate LD scores for each spot."""
    
    chrom: Union[int, Literal["all"]] = Field(
        ...,
        description='Chromosome id (1-22) or "all"'
    )
    bfile_root: Path = Field(
        ...,
        description="Root path for genotype plink bfiles (.bim, .bed, .fam)"
    )
    keep_snp_root: Optional[Path] = Field(
        default=None,
        description="Root path for SNP files"
    )
    gtf_annotation_file: Path = Field(
        ...,
        description="Path to GTF annotation file"
    )
    gene_window_size: int = Field(
        default=50000,
        description="Gene window size in base pairs"
    )
    enhancer_annotation_file: Optional[Path] = Field(
        default=None,
        description="Path to enhancer annotation file (optional)"
    )
    snp_multiple_enhancer_strategy: Literal["max_mkscore", "nearest_TSS"] = Field(
        default="max_mkscore",
        description="Strategy for handling multiple enhancers per SNP"
    )
    gene_window_enhancer_priority: Optional[Literal["gene_window_first", "enhancer_first", "enhancer_only"]] = Field(
        default=None,
        description="Priority between gene window and enhancer annotations"
    )
    spots_per_chunk: int = Field(
        default=1000,
        description="Number of spots per chunk"
    )
    ld_wind: int = Field(
        default=1,
        description="LD window size"
    )
    ld_unit: Literal["SNP", "KB", "CM"] = Field(
        default="CM",
        description="Unit for LD window"
    )
    additional_baseline_annotation: Optional[Path] = Field(
        default=None,
        description="Path of additional baseline annotations"
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
    
    def execute(self) -> None:
        """Execute the generate LD score command."""
        from gsMap.generate_ldscore import run_generate_ldscore
        run_generate_ldscore(self)


class SpatialLDSCCommand(BaseWorkdirCommand):
    """Run spatial LDSC for each spot."""
    
    sumstats_file: Path = Field(
        ...,
        description="Path to GWAS summary statistics file"
    )
    w_file: Optional[Path] = Field(
        default=None,
        description="Path to regression weight file"
    )
    trait_name: str = Field(
        ...,
        description="Name of the trait being analyzed"
    )
    n_blocks: int = Field(
        default=200,
        description="Number of blocks for jackknife resampling"
    )
    chisq_max: Optional[int] = Field(
        default=None,
        description="Maximum chi-square value for filtering SNPs"
    )
    num_processes: int = Field(
        default=4,
        description="Number of processes for parallel computing"
    )
    use_additional_baseline_annotation: bool = Field(
        default=True,
        description="Use additional baseline annotations when provided"
    )
    use_jax: bool = Field(
        default=True,
        description="Use JAX-accelerated implementation"
    )
    
    @model_validator(mode='after')
    def setup_w_file(self):
        """Setup w_file if not provided."""
        if self.w_file is None:
            w_ld_dir = self.ldscore_save_dir / "w_ld"
            if w_ld_dir.exists():
                self.w_file = w_ld_dir / "weights."
        return self
    
    def execute(self) -> None:
        """Execute the spatial LDSC command."""
        if self.use_jax:
            from gsMap.spatial_ldsc_jax_final import run_spatial_ldsc_jax
            run_spatial_ldsc_jax(self)
        else:
            from gsMap.spatial_ldsc_multiple_sumstats import run_spatial_ldsc
            run_spatial_ldsc(self)


class CauchyCommand(BaseWorkdirCommand):
    """Run Cauchy combination for each annotation."""
    
    trait_name: str = Field(
        ...,
        description="Name of the trait being analyzed"
    )
    annotation: str = Field(
        ...,
        description="Name of the annotation in adata.obs to use"
    )
    sample_name_list: Optional[list[str]] = Field(
        default=None,
        description="List of sample names to process"
    )
    output_file: Optional[Path] = Field(
        default=None,
        description="Path to save the combined Cauchy results"
    )
    
    def execute(self) -> None:
        """Execute the Cauchy combination command."""
        from gsMap.cauchy_combination_test import run_Cauchy_combination
        run_Cauchy_combination(self)


class ReportCommand(BaseWorkdirCommand):
    """Generate diagnostic plots and tables."""
    
    trait_name: str = Field(
        ...,
        description="Name of the trait to generate the report for"
    )
    annotation: str = Field(
        ...,
        description="Annotation layer name"
    )
    top_corr_genes: int = Field(
        default=50,
        description="Number of top correlated genes to display"
    )
    selected_genes: Optional[list[str]] = Field(
        default=None,
        description="List of specific genes to include in the report"
    )
    sumstats_file: Path = Field(
        ...,
        description="Path to GWAS summary statistics file"
    )
    fig_width: Optional[int] = Field(
        default=None,
        description="Width of the generated figures in pixels"
    )
    fig_height: Optional[int] = Field(
        default=None,
        description="Height of the generated figures in pixels"
    )
    point_size: Optional[int] = Field(
        default=None,
        description="Point size for the figures"
    )
    fig_style: Literal["dark", "light"] = Field(
        default="light",
        description="Style of the generated figures"
    )
    
    def execute(self) -> None:
        """Execute the report generation command."""
        from gsMap.report import run_report
        run_report(self)


class FormatSumstatsCommand(BaseCommand):
    """Format GWAS summary statistics."""
    
    sumstats: Path = Field(
        ...,
        description="Path to GWAS summary data"
    )
    out: Path = Field(
        ...,
        description="Path to save the formatted GWAS data"
    )
    
    # Column name specifications
    snp: Optional[str] = Field(default=None, description="Name of SNP column")
    a1: Optional[str] = Field(default=None, description="Name of effect allele column")
    a2: Optional[str] = Field(default=None, description="Name of non-effect allele column")
    info: Optional[str] = Field(default=None, description="Name of info column")
    beta: Optional[str] = Field(default=None, description="Name of GWAS beta column")
    se: Optional[str] = Field(default=None, description="Name of GWAS standard error of beta column")
    p: Optional[str] = Field(default=None, description="Name of p-value column")
    frq: Optional[str] = Field(default=None, description="Name of A1 frequency column")
    n: Optional[Union[str, int]] = Field(default=None, description="Name of sample size column or sample size value")
    z: Optional[str] = Field(default=None, description="Name of GWAS Z-statistics column")
    OR: Optional[str] = Field(default=None, description="Name of GWAS OR column")
    se_OR: Optional[str] = Field(default=None, description="Name of standard error of OR column")
    
    # SNP to rsid conversion
    chr: str = Field(default="Chr", description="Name of SNP chromosome column")
    pos: str = Field(default="Pos", description="Name of SNP positions column")
    dbsnp: Optional[Path] = Field(default=None, description="Path to reference dbSNP file")
    chunksize: int = Field(default=1000000, description="Chunk size for loading dbSNP file")
    
    # Output format and quality
    format: Literal["gsMap", "COJO"] = Field(default="gsMap", description="Format of output data")
    info_min: float = Field(default=0.9, description="Minimum INFO score")
    maf_min: float = Field(default=0.01, description="Minimum MAF")
    keep_chr_pos: bool = Field(default=False, description="Keep SNP chromosome and position columns in the output data")
    
    def execute(self) -> None:
        """Execute the format sumstats command."""
        from gsMap.format_sumstats import gwas_format
        gwas_format(self)


class QuickModeCommand(BaseWorkdirCommand):
    """Run the entire gsMap pipeline in quick mode."""
    
    gsMap_resource_dir: Path = Field(
        ...,
        description="Directory containing gsMap resources"
    )
    hdf5_path: Path = Field(
        ...,
        description="Path to the input spatial transcriptomics data (H5AD format)"
    )
    annotation: str = Field(
        ...,
        description="Name of the annotation in adata.obs to use"
    )
    data_layer: str = Field(
        default="counts",
        description="Data layer for gene expression"
    )
    trait_name: Optional[str] = Field(
        default=None,
        description="Name of the trait for GWAS analysis"
    )
    sumstats_file: Optional[Path] = Field(
        default=None,
        description="Path to GWAS summary statistics file"
    )
    sumstats_config_file: Optional[Path] = Field(
        default=None,
        description="Path to GWAS summary statistics config file"
    )
    homolog_file: Optional[Path] = Field(
        default=None,
        description="Path to homologous gene conversion file"
    )
    max_processes: int = Field(
        default=10,
        description="Maximum number of processes for parallel execution"
    )
    latent_representation: Optional[str] = Field(
        default=None,
        description="Type of latent representation"
    )
    num_neighbour: int = Field(
        default=21,
        description="Number of neighbors"
    )
    num_neighbour_spatial: int = Field(
        default=101,
        description="Number of spatial neighbors"
    )
    gM_slices: Optional[Path] = Field(
        default=None,
        description="Path to the slice mean file"
    )
    pearson_residuals: bool = Field(
        default=False,
        description="Use pearson residuals"
    )
    use_jax: bool = Field(
        default=True,
        description="Use JAX-accelerated spatial LDSC implementation"
    )
    
    @model_validator(mode='after')
    def validate_sumstats(self):
        """Validate summary statistics configuration."""
        # Skip validation if we're just saving config
        if hasattr(self, 'save_config') and self.save_config:
            return self
            
        if self.sumstats_file is None and self.sumstats_config_file is None:
            raise ValueError("Either sumstats_file or sumstats_config_file is required")
        if self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError("Only one of sumstats_file or sumstats_config_file should be provided")
        if self.sumstats_file is not None and self.trait_name is None:
            raise ValueError("trait_name is required when sumstats_file is provided")
        return self
    
    def execute(self) -> None:
        """Execute the quick mode pipeline."""
        from gsMap.run_all_mode import run_pipeline
        run_pipeline(self)


class CreateSliceMeanCommand(BaseCommand):
    """Create slice mean from multiple h5ad files."""
    
    sample_name_list: list[str] = Field(
        ...,
        description="List of sample names to process"
    )
    h5ad_list: Optional[list[Path]] = Field(
        default=None,
        description="List of h5ad file paths"
    )
    h5ad_yaml: Optional[Path] = Field(
        default=None,
        description="Path to YAML file containing sample names and h5ad paths"
    )
    slice_mean_output_file: Path = Field(
        ...,
        description="Path to the output file for the slice mean"
    )
    homolog_file: Optional[Path] = Field(
        default=None,
        description="Path to homologous gene conversion file"
    )
    data_layer: str = Field(
        default="counts",
        description="Data layer for gene expression"
    )
    
    @model_validator(mode='after')
    def validate_inputs(self):
        """Validate input configuration."""
        # Skip validation if we're just saving config
        if hasattr(self, 'save_config') and self.save_config:
            return self
            
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
    
    def execute(self) -> None:
        """Execute the create slice mean command."""
        from gsMap.create_slice_mean import run_create_slice_mean
        run_create_slice_mean(self)


class ConfigManageCommand(BaseModel):
    """Manage gsMap configuration files."""
    
    action: Literal["validate", "convert", "merge", "template"] = Field(
        ...,
        description="Action to perform on configuration files"
    )
    input_file: Optional[Path] = Field(
        default=None,
        description="Input configuration file"
    )
    output_file: Optional[Path] = Field(
        default=None,
        description="Output configuration file"
    )
    merge_file: Optional[Path] = Field(
        default=None,
        description="Additional configuration file to merge"
    )
    format: Optional[Literal["yaml", "json"]] = Field(
        default="yaml",
        description="Output format for configuration files"
    )
    command: Optional[str] = Field(
        default=None,
        description="Command name for template generation"
    )
    
    def cli_cmd(self) -> None:
        """Execute configuration management commands."""
        if self.action == "template":
            if not self.command or not self.output_file:
                raise ValueError("Both command and output_file required for template action")
            create_template_config(self.command, self.output_file)
            return
        
        if not self.input_file:
            raise ValueError(f"Input file required for {self.action} action")
            
        # Load input configuration
        with open(self.input_file, 'r') as f:
            if self.input_file.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(f)
            elif self.input_file.suffix.lower() == '.json':
                config = json.load(f)
            else:
                # Try to detect format
                content = f.read()
                f.seek(0)
                try:
                    config = yaml.safe_load(f)
                except:
                    f.seek(0)
                    config = json.load(f)
        
        if self.action == "validate":
            print(f"✓ Configuration file is valid: {self.input_file}")
            print(f"  Contains {len(config)} top-level keys")
            for key in config:
                print(f"    - {key}")
        
        elif self.action == "convert":
            if not self.output_file:
                raise ValueError("Output file required for convert action")
            
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_file, 'w') as f:
                if self.format == "yaml":
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(config, f, indent=2)
            
            print(f"✓ Configuration converted to {self.format}: {self.output_file}")
        
        elif self.action == "merge":
            if not self.merge_file or not self.output_file:
                raise ValueError("Both merge_file and output_file required for merge action")
            
            # Load merge configuration
            with open(self.merge_file, 'r') as f:
                if self.merge_file.suffix.lower() in ['.yml', '.yaml']:
                    merge_config = yaml.safe_load(f)
                else:
                    merge_config = json.load(f)
            
            # Merge configurations (merge_config takes precedence)
            config.update(merge_config)
            
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.output_file, 'w') as f:
                if self.format == "yaml":
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                else:
                    json.dump(config, f, indent=2)
            
            print(f"✓ Configurations merged to {self.format}: {self.output_file}")


# ============================================================================
# Root CLI Command
# ============================================================================

class GsMapCLI(BaseSettings):
    """
    gsMap: Genetically informed spatial mapping of cells for complex traits
    
    A comprehensive toolkit for integrating spatial transcriptomics data with GWAS
    to map cells to human complex traits in a spatially resolved manner.
    """
    
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_prog_name="gsMap",
        cli_use_class_docs_for_groups=True,
        cli_exit_on_error=False,  # We'll handle errors ourselves
        cli_hide_none_type=True,
        cli_avoid_json=True,
        cli_implicit_flags=True,
        cli_kebab_case=False
    )
    
    # Subcommands (required fields - no default values)
    quick_mode: CliSubCommand[QuickModeCommand]
    find_latent: CliSubCommand[FindLatentCommand]
    latent_to_gene: CliSubCommand[LatentToGeneCommand]
    generate_ldscore: CliSubCommand[GenerateLDScoreCommand]
    spatial_ldsc: CliSubCommand[SpatialLDSCCommand]
    cauchy: CliSubCommand[CauchyCommand]
    report: CliSubCommand[ReportCommand]
    format_sumstats: CliSubCommand[FormatSumstatsCommand]
    create_slice_mean: CliSubCommand[CreateSliceMeanCommand]
    config: CliSubCommand[ConfigManageCommand]
    
    def cli_cmd(self) -> None:
        """Execute the selected subcommand."""
        import pyfiglet
        
        # Display banner
        banner = pyfiglet.figlet_format("gsMap", font="doom", width=80, justify="center").rstrip()
        print(banner, flush=True)
        version_str = f"Version: {__version__}"
        print(version_str.center(80), flush=True)
        print("=" * 80, flush=True)
        
        # Run the selected subcommand
        try:
            subcommand = get_subcommand(self)
            if subcommand:
                # Check if this is just a save-config operation
                if hasattr(subcommand, 'save_config') and subcommand.save_config:
                    # Create template instead of failing validation
                    subcommand.__class__.create_template(subcommand.save_config)
                else:
                    CliApp.run_subcommand(self)
        except SettingsError as e:
            # No subcommand selected
            print("\nNo subcommand selected. Use --help to see available commands.")
            print("\nAvailable commands:")
            print("  quick_mode         - Run the entire pipeline in quick mode")
            print("  find_latent        - Find latent representations using GNN")
            print("  latent_to_gene     - Convert latent representations to gene scores")
            print("  generate_ldscore   - Generate LD scores")
            print("  spatial_ldsc       - Run spatial LDSC analysis")
            print("  cauchy             - Run Cauchy combination")
            print("  report             - Generate diagnostic reports")
            print("  format_sumstats    - Format GWAS summary statistics")
            print("  create_slice_mean  - Create slice mean from multiple samples")
            print("  config             - Manage configuration files")
            sys.exit(1)


# ============================================================================
# Utility Functions
# ============================================================================

def create_template_config(command: str, output_path: Path) -> None:
    """
    Create a template configuration file for a specific command.
    
    Args:
        command: Name of the command to create template for
        output_path: Path to save the template configuration
    """
    templates = {
        'quick-mode': QuickModeCommand,
        'quick_mode': QuickModeCommand,
        'find-latent': FindLatentCommand,
        'find_latent': FindLatentCommand,
        'latent-to-gene': LatentToGeneCommand,
        'latent_to_gene': LatentToGeneCommand,
        'generate-ldscore': GenerateLDScoreCommand,
        'generate_ldscore': GenerateLDScoreCommand,
        'spatial-ldsc': SpatialLDSCCommand,
        'spatial_ldsc': SpatialLDSCCommand,
        'cauchy': CauchyCommand,
        'report': ReportCommand,
        'format-sumstats': FormatSumstatsCommand,
        'format_sumstats': FormatSumstatsCommand,
        'create-slice-mean': CreateSliceMeanCommand,
        'create_slice_mean': CreateSliceMeanCommand,
    }
    
    if command not in templates:
        raise ValueError(f"Unknown command: {command}. Available: {list(set(templates.keys()))}")
    
    # Use the class's create_template method
    model_class = templates[command]
    model_class.create_template(output_path)


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

# For backward compatibility with old names
FindLatentRepresentations = FindLatentCommand
LatentToGene = LatentToGeneCommand
GenerateLDScore = GenerateLDScoreCommand
SpatialLDSC = SpatialLDSCCommand
CauchyCombination = CauchyCommand
Report = ReportCommand
FormatSumstats = FormatSumstatsCommand
QuickMode = QuickModeCommand
CreateSliceMean = CreateSliceMeanCommand


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the gsMap CLI."""
    try:
        # Special handling for save-config
        if '--save_config' in sys.argv or '--save-config' in sys.argv:
            # Extract command and save_config path
            import argparse
            parser = argparse.ArgumentParser(add_help=False)
            parser.add_argument('command', nargs='?')
            parser.add_argument('--save_config', '--save-config', type=str)
            args, _ = parser.parse_known_args()
            
            if args.command and args.save_config:
                # Map command to class
                command_map = {
                    'quick_mode': QuickModeCommand,
                    'find_latent': FindLatentCommand,
                    'latent_to_gene': LatentToGeneCommand,
                    'generate_ldscore': GenerateLDScoreCommand,
                    'spatial_ldsc': SpatialLDSCCommand,
                    'cauchy': CauchyCommand,
                    'report': ReportCommand,
                    'format_sumstats': FormatSumstatsCommand,
                    'create_slice_mean': CreateSliceMeanCommand,
                }
                
                if args.command in command_map:
                    # Create template directly
                    command_class = command_map[args.command]
                    command_class.create_template(Path(args.save_config))
                    sys.exit(0)
        
        # Normal CLI execution
        CliApp.run(GsMapCLI)
        
    except ValidationError as e:
        # Handle validation errors more gracefully
        print("\n❌ Configuration Error:", file=sys.stderr)
        for error in e.errors():
            field_path = " → ".join(str(x) for x in error['loc'])
            print(f"  • {field_path}: {error['msg']}", file=sys.stderr)
        print("\n💡 Tip: Use --save_config <file.yaml> to create a configuration template", file=sys.stderr)
        print("  Example: gsMap format_sumstats --save_config config.yaml", file=sys.stderr)
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()