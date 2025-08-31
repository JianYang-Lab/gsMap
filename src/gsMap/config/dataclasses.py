"""
Configuration dataclasses for gsMap commands.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Annotated, List, Literal
import yaml
import logging
from pathlib import Path
import h5py


import typer

from .base import ConfigWithAutoPaths

def get_gsMap_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[{asctime}] {levelname:.5s} | {name} - {message}", style="{")
    )
    logger.addHandler(handler)
    return logger


logger = get_gsMap_logger("gsMap")

def get_anndata_shape(h5ad_path: str):
    """Get the shape (n_obs, n_vars) of an AnnData file without loading it."""
    with h5py.File(h5ad_path, 'r') as f:
        # 1. Verify it's a valid AnnData file by checking metadata
        if f.attrs.get('encoding-type') != 'anndata':
            logger.error(f"File '{h5ad_path}' does not appear to be a valid AnnData file.")
            return None

        # 2. Determine n_obs and n_vars from the primary metadata sources
        if 'obs' not in f or 'var' not in f:
            logger.error("AnnData file is missing 'obs' or 'var' group.")
            return None

        # Get the name of the index column from attributes
        obs_index_key = f['obs'].attrs.get('_index', None)
        var_index_key = f['var'].attrs.get('_index', None)

        if not obs_index_key or obs_index_key not in f['obs']:
            logger.error("Could not determine index for 'obs'.")
            return None
        if not var_index_key or var_index_key not in f['var']:
            logger.error("Could not determine index for 'var'.")
            return None

        # The shape is the length of these index arrays
        obs_obj = f['obs'][obs_index_key]
        if isinstance(obs_obj, h5py.Group):
            obs_obj = obs_obj['categories']
        n_obs = obs_obj.shape[0]

        var_obj  = f['var'][var_index_key]
        if isinstance(var_obj, h5py.Group):
            var_obj = var_obj['categories']
        n_vars = var_obj.shape[0]

        return n_obs, n_vars


def inspect_h5ad_structure(filename):
    """
    Inspect the structure of an h5ad file without loading data.
    
    Returns dict with keys present in each slot.
    """
    structure = {}
    
    with h5py.File(filename, 'r') as f:
        # Check main slots
        slots = ['obs', 'var', 'obsm', 'varm', 'obsp', 'varp', 'uns', 'layers', 'X', 'raw']
        
        for slot in slots:
            if slot in f:
                if slot in ['obsm', 'varm', 'obsp', 'varp', 'layers', 'uns']:
                    # These are groups containing multiple keys
                    structure[slot] = list(f[slot].keys())
                elif slot in ['obs', 'var']:
                    # These are dataframes - get column names
                    if 'column-order' in f[slot].attrs:
                        structure[slot] = list(f[slot].attrs['column-order'])
                    else:
                        structure[slot] = list(f[slot].keys())
                else:
                    # X, raw - just note they exist
                    structure[slot] = True
    
    return structure


def validate_h5ad_structure(sample_h5ad_dict, required_fields, optional_fields=None):
    """
    Validate h5ad files have required structure.
    
    Args:
        sample_h5ad_dict: OrderedDict of {sample_name: h5ad_path}
        required_fields: Dict of {field_name: (slot, field_key, error_msg_template)}
            e.g., {'spatial': ('obsm', 'spatial', 'Spatial key')}
        optional_fields: Dict of {field_name: (slot, field_key)} for fields to warn about
    
    Returns:
        None, raises ValueError if required fields are missing
    """
    for sample_name, h5ad_path in sample_h5ad_dict.items():
        if not h5ad_path.exists():
            raise FileNotFoundError(f"H5AD file not found for sample '{sample_name}': {h5ad_path}")
        
        # Inspect h5ad structure
        structure = inspect_h5ad_structure(h5ad_path)
        
        # Check required fields
        for field_name, (slot, field_key, error_msg) in required_fields.items():
            if field_key is None:  # Skip if field not specified
                continue
                
            # Special handling for data_layer
            if field_name == 'data_layer' and field_key != 'X':
                if 'layers' not in structure or field_key not in structure.get('layers', []):
                    logger.warning(
                        f"Data layer '{field_key}' not found in layers for sample '{sample_name}'. "
                        f"Available layers: {structure.get('layers', [])}"
                    )
            elif field_name == 'data_layer' and field_key == 'X':
                if 'X' not in structure:
                    raise ValueError(f"X matrix not found in h5ad file for sample '{sample_name}'")
            else:
                # Standard validation for obsm, obs, etc.
                if slot not in structure or field_key not in structure.get(slot, []):
                    available = structure.get(slot, [])
                    raise ValueError(
                        f"{error_msg} '{field_key}' not found in {slot} for sample '{sample_name}'. "
                        f"Available keys in {slot}: {available}"
                    )
        
        # Check optional fields (warn only)
        if optional_fields:
            for field_name, (slot, field_key) in optional_fields.items():
                if field_key is None:  # Skip if field not specified
                    continue
                    
                if slot not in structure or field_key not in structure.get(slot, []):
                    available = structure.get(slot, [])
                    logger.warning(
                        f"Optional field '{field_key}' not found in {slot} for sample '{sample_name}'. "
                        f"Available keys in {slot}: {available}"
                    )


def process_h5ad_inputs(config, input_options):
    """
    Process h5ad input options and create sample_h5ad_dict.
    
    Args:
        config: Configuration object with h5ad input fields
        input_options: Dict mapping option names to (field_name, processing_type)
            e.g., {'h5ad_yaml': ('h5ad_yaml', 'yaml'), 
                   'h5ad': ('h5ad', 'list'),
                   'h5ad_list_file': ('h5ad_list_file', 'file')}
    
    Returns:
        OrderedDict of {sample_name: h5ad_path}
    """
    from collections import OrderedDict
    
    sample_h5ad_dict = OrderedDict()
    
    # Check which options are provided
    options_provided = []
    for option_name, (field_name, _) in input_options.items():
        if hasattr(config, field_name) and getattr(config, field_name):
            options_provided.append(option_name)
    
    # Ensure at most one option is provided
    if len(options_provided) > 1:
        assert False, (
            f"At most one input option can be provided. Got {len(options_provided)}: {', '.join(options_provided)}. "
            f"Please provide only one of: {', '.join(input_options.keys())}"
        )
    
    # Process the provided input option
    for option_name, (field_name, processing_type) in input_options.items():
        field_value = getattr(config, field_name, None)
        if not field_value:
            continue
            
        if processing_type == 'yaml':
            logger.info(f"Using {option_name}: {field_value}")
            with open(field_value) as f:
                h5ad_data = yaml.safe_load(f)
                for sample_name, h5ad_path in h5ad_data.items():
                    sample_h5ad_dict[sample_name] = Path(h5ad_path)
                    
        elif processing_type == 'list':
            logger.info(f"Using {option_name} with {len(field_value)} files")
            for h5ad_path in field_value:
                h5ad_path = Path(h5ad_path)
                sample_name = h5ad_path.stem
                if sample_name in sample_h5ad_dict:
                    logger.warning(f"Duplicate sample name: {sample_name}, will be overwritten")
                sample_h5ad_dict[sample_name] = h5ad_path
                
        elif processing_type == 'file':
            logger.info(f"Using {option_name}: {field_value}")
            with open(field_value) as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        h5ad_path = Path(line)
                        sample_name = h5ad_path.stem
                        if sample_name in sample_h5ad_dict:
                            logger.warning(f"Duplicate sample name: {sample_name}, will be overwritten")
                        sample_h5ad_dict[sample_name] = h5ad_path
        break
    
    return sample_h5ad_dict

def verify_homolog_file_format(config):
    if config.homolog_file is not None:
        logger.info(
            f"User provided homolog file to map gene names to human: {config.homolog_file}"
        )
        # check the format of the homolog file
        with open(config.homolog_file) as f:
            first_line = f.readline().strip()
            _n_col = len(first_line.split())
            if _n_col != 2:
                raise ValueError(
                    f"Invalid homolog file format. Expected 2 columns, first column should be other species gene name, second column should be human gene name. "
                    f"Got {_n_col} columns in the first line."
                )
            else:
                first_col_name, second_col_name = first_line.split()
                config.species = first_col_name
                logger.info(
                    f"Homolog file provided and will map gene name from column1:{first_col_name} to column2:{second_col_name}"
                )
    else:
        logger.info("No homolog file provided. Run in human mode.")



@dataclass
class RunAllModeConfig:
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
    species: Optional[str] = None


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

    project_name: Annotated[str, typer.Option(
        help="Name of the project"
    )]

    sample_name: Annotated[str, typer.Option(
        help="Name of the sample"
    )]

    h5ad_path: Annotated[Optional[List[Path]], typer.Option(
        help="Space-separated list of h5ad file paths. Sample names are derived from file names without suffix.",
        exists=True,
        file_okay=True,
    )] = None

    h5ad_yaml: Annotated[Path, typer.Option(
        help="YAML file with sample names and h5ad paths",
        exists=True,
        file_okay=True,
        dir_okay=False,
    )] = None

    h5ad_list_file: Annotated[Optional[str], typer.Option(
        help="Each row is a h5ad file path, sample name is the file name without suffix",
        exists = True,
        file_okay = True,
        dir_okay = False,
    )] = None

    data_layer: Annotated[str, typer.Option(
        help="Gene expression raw counts data layer in h5ad layers, e.g., 'count', 'counts'. Other wise use 'X' for adata.X"
    )] = "X"
    
    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm storing spatial coordinates"
    )] = "spatial"
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation of cell type in adata.obs to use"
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

    species: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        
        # Define input options
        input_options = {
            'h5ad_yaml': ('h5ad_yaml', 'yaml'),
            'h5ad_path': ('h5ad_path', 'list'),
            'h5ad_list_file': ('h5ad_list_file', 'file'),
        }
        
        # Process h5ad inputs
        self.sample_h5ad_dict = process_h5ad_inputs(self, input_options)

        if not self.sample_h5ad_dict:
            raise ValueError(
                "At least one of h5ad_yaml, h5ad_path, h5ad_list_file, or spe_file_list must be provided"
            )
        
        # Define required and optional fields for validation
        required_fields = {
            'data_layer': ('layers', self.data_layer, 'Data layer'),
            'spatial_key': ('obsm', self.spatial_key, 'Spatial key'),
        }
        
        # Add annotation as required if provided
        if self.annotation:
            required_fields['annotation'] = ('obs', self.annotation, 'Annotation')
        
        # Validate h5ad structure
        validate_h5ad_structure(self.sample_h5ad_dict, required_fields)
        
        # Log final sample count
        logger.info(f"Loaded and validated {len(self.sample_h5ad_dict)} samples")
        
        # Check if at least one sample is provided
        if len(self.sample_h5ad_dict) == 0:
            raise ValueError("No valid samples found in the provided input")
        
        # Verify homolog file format if provided
        verify_homolog_file_format(self)


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
    
    project_name: Annotated[str, typer.Option(
        help="Name of the project"
    )]

    # --------input h5ad file paths which have the latent representations
    h5ad: Annotated[Optional[List[Path]], typer.Option(
        help="Space-separated list of h5ad file paths. Sample names are derived from file names without suffix.",
        exists=True,
        file_okay=True,
    )] = None

    h5ad_yaml: Annotated[Path, typer.Option(
        help="YAML file with sample names and h5ad paths",
        exists=True,
        file_okay=True,
        dir_okay=False,
    )] = None

    h5ad_list_file: Annotated[Optional[str], typer.Option(
        help="Each row is a h5ad file path, sample name is the file name without suffix",
        exists = True,
        file_okay = True,
        dir_okay = False,
    )] = None


    # --------input h5ad obs, obsm, layers keys

    annotation: Annotated[Optional[str], typer.Option(
        help="Cell type annotation in adata.obs to use. This would constrain finding homogeneous spots within each cell type"
    )] = None

    data_layer: Annotated[str, typer.Option(
        help="Gene expression raw counts data layer in h5ad layers, e.g., 'count', 'counts'. Other wise use 'X' for adata.X"
    )] = "X"

    
    latent_representation_niche: Annotated[str, typer.Option(
        help="Key for spatial niche embedding in obsm"
    )] = "emb_niche"

    latent_representation_cell: Annotated[str, typer.Option(
        help="Key for cell identity embedding in obsm"
    )] = "emb_cell"
    
    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm"
    )] = "spatial"

    # --------parameters for finding homogeneous spots
    num_neighbour_spatial: Annotated[int, typer.Option(
        help="k1: Number of spatial neighbors for initial graph",
        min=10,
        max=500
    )] = 201

    num_anchor: Annotated[int, typer.Option(
        help="k2: Number of spatial anchors per cell",
        min=10,
        max=200
    )] = 51
    
    num_neighbour: Annotated[int, typer.Option(
        help="k3: Number of homogeneous spots to find",
        min=1,
        max=100
    )] = 21
    
    no_expression_fraction: Annotated[bool, typer.Option(
        "--no-expression-fraction",
        help="Skip expression fraction filtering"
    )] = False


    # -------- IO parameters
    rank_batch_size:int = 1000
    rank_write_interval:int = 10

    rank_read_workers: Annotated[int, typer.Option(
        help="Number of parallel reader threads of rank zarr",
        min=1,
        max=16
    )] = 2
    
    mkscore_write_workers: Annotated[int, typer.Option(
        help="Number of parallel writer threads of marker scores",
        min=1,
        max=16
    )] = 2
    
    mkscore_batch_size: Annotated[int, typer.Option(
        help="Batch size for GPU to calculate the marker score to avoid CUDA OOM",
        min=100,
        max=5000
    )] = 500
    

    chunks_cells: Annotated[Optional[int], typer.Option(
        help="Chunk size for cells dimension (None for optimal)"
    )] = None
    
    chunks_genes: Annotated[Optional[int], typer.Option(
        help="Chunk size for genes dimension (None for optimal)"
    )] = None
    
    use_jax: Annotated[bool, typer.Option(
        "--use-jax/--no-jax",
        help="Use JAX acceleration for computations"
    )] = True
    
    cache_size_mb: Annotated[int, typer.Option(
        help="Cache size in MB for data reading",
        min=100,
        max=10000
    )] = 1000

    zarr_group_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize and validate configuration"""
        super().__post_init__()
        from collections import OrderedDict
        
        # Define input options
        input_options = {
            'h5ad_yaml': ('h5ad_yaml', 'yaml'),
            'h5ad': ('h5ad', 'list'),
            'h5ad_list_file': ('h5ad_list_file', 'file'),
        }
        
        # Process h5ad inputs
        self.sample_h5ad_dict = process_h5ad_inputs(self, input_options)
        
        # Auto-detect from latent directory if no inputs provided
        if not self.sample_h5ad_dict:
            self.sample_h5ad_dict = OrderedDict()
            latent_dir = self.latent_dir
            logger.info(f"No input options provided. Auto-detecting h5ad files from latent directory: {latent_dir}")
            
            # Look for latent files and derive sample names
            latent_files = list(latent_dir.glob("*_latent_adata.h5ad"))
            if not latent_files:
                # Try alternative pattern
                latent_files = list(latent_dir.glob("*_add_latent.h5ad"))
            
            if not latent_files:
                raise ValueError(
                    f"No h5ad files found in latent directory {latent_dir}. "
                    f"Please run the find latent representation first. "
                    f"Or provide one of: h5ad_yaml, h5ad, or h5ad_list_file, which points to h5ad files which contain the latent embedding."
                )
            
            for latent_file in latent_files:
                # Extract sample name by removing suffix patterns
                filename = latent_file.stem
                if filename.endswith("_latent_adata"):
                    sample_name = filename[:-len("_latent_adata")]
                elif filename.endswith("_add_latent"):
                    sample_name = filename[:-len("_add_latent")]
                else:
                    sample_name = filename
                
                # Use the latent file itself as the h5ad path
                self.sample_h5ad_dict[sample_name] = latent_file
                
            logger.info(f"Auto-detected {len(self.sample_h5ad_dict)} samples from latent directory")
        
        # Define required and optional fields for validation
        required_fields = {
            'latent_representation_cell': ('obsm', self.latent_representation_cell, 'Latent representation of cell identity'),
            'spatial_key': ('obsm', self.spatial_key, 'Spatial key'),
        }
        
        # Add annotation as required if provided
        if self.annotation:
            required_fields['annotation'] = ('obs', self.annotation, 'Annotation')
        
        # Optional fields
        if self.latent_representation_niche:
            required_fields['latent_representation_niche'] = ('obsm', self.latent_representation_niche, 'Latent representation of spatial niche')
        
        # Validate h5ad structure
        validate_h5ad_structure(self.sample_h5ad_dict, required_fields,)
        
        # Log final sample count
        logger.info(f"Loaded and validated {len(self.sample_h5ad_dict)} samples")
        
        # Check if at least one sample is provided
        if len(self.sample_h5ad_dict) == 0:
            raise ValueError("No valid samples found in the provided input")
        
        # Validate configuration constraints
        assert self.num_neighbour <= self.num_anchor, \
            f"num_neighbour ({self.num_neighbour}) must be <= num_anchor ({self.num_anchor})"
        assert self.num_anchor <= self.num_neighbour_spatial, \
            f"num_anchor ({self.num_anchor}) must be <= num_neighbour_spatial ({self.num_neighbour_spatial})"




@dataclass
class SpatialLDSCConfig(ConfigWithAutoPaths):
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]

    project_name: Annotated[str, typer.Option(
        help="Name of the project"
    )]


    w_file: str | None = None
    # ldscore_save_dir: str
    use_additional_baseline_annotation: bool = True
    trait_name: str | None = None
    sumstats_file: str | None = None
    sumstats_config_file: str | None = None
    num_processes: int = 4
    not_M_5_50: bool = False
    n_blocks: int = 200
    chisq_max: int | None = None
    cell_indices_range: tuple[int, int] | None = None  # 0-based range [start, end) of cell indices to process
    sample_name: str | None = None  # Field for filtering by sample name

    ldscore_save_format: Literal["feather", "quick_mode"] = "feather"

    spots_per_chunk_quick_mode: int = 1_000

    ldscore_save_dir: str  | Path | None = None
    quick_mode_resource_dir: str | Path | None = None
    use_jax: bool = True
    
    marker_score_format: Literal[ "memmap", "feather"] = "memmap"
    mkscore_feather_path: str | Path | None = None

    def __post_init__(self):
        super().__post_init__()
        

        # Validate cell_indices_range is 0-based
        if self.cell_indices_range is not None:
            # Validate exclusivity between sample_name and cell_indices_range

            if self.sample_name is not None:
                raise ValueError(
                    "Only one of sample_name or cell_indices_range can be provided, not both. "
                    "Use sample_name to filter by sample, or cell_indices_range to process specific cell indices."
                )

            start, end = self.cell_indices_range
            
            # Check that indices are 0-based
            if start < 0:
                raise ValueError(f"cell_indices_range start must be >= 0, got {start}")
            if start == 1:
                logger.warning(
                    "cell_indices_range appears to be 1-based (start=1). "
                    "Please ensure indices are 0-based. Adjusting start to 0."
                )
                start = 0

            # Check that start < end
            if start >= end:
                raise ValueError(f"cell_indices_range start ({start}) must be less than end ({end})")
            
            # Validate against actual data shape if in quick_mode
            if self.ldscore_save_format == "quick_mode" and self.quick_mode_resource_dir is not None:
                # Check if concatenated latent adata exists
                concat_adata_path = Path(self.workdir) / self.project_name / "latent2gene" / "concatenated_latent_adata.h5ad"
                assert concat_adata_path.exists(), f"Concatenated latent adata not found at {concat_adata_path}. The latent to gene step must be run first."
                shape = get_anndata_shape(str(concat_adata_path))
                if shape is not None:
                    n_obs, _ = shape
                    if end > n_obs:
                        logger.warning(
                            f"cell_indices_range end ({end}) exceeds number of observations ({n_obs}) "
                            f"Set end to {n_obs}."
                            )
                        end = n_obs
            self.cell_indices_range = (start, end)
            logger.info(f"Processing cell_indices_range: [{start}, {end})")

        
        if self.sumstats_file is None and self.sumstats_config_file is None:
            raise ValueError("One of sumstats_file and sumstats_config_file must be provided.")
        if self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError(
                "Only one of sumstats_file and sumstats_config_file must be provided."
            )
        if self.sumstats_file is not None and self.trait_name is None:
            raise ValueError("trait_name must be provided if sumstats_file is provided.")
        if self.sumstats_config_file is not None and self.trait_name is not None:
            raise ValueError(
                "trait_name must not be provided if sumstats_config_file is provided."
            )
        self.sumstats_config_dict = {}
        # load the sumstats config file
        if self.sumstats_config_file is not None:
            import yaml

            with open(self.sumstats_config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for _trait_name, sumstats_file in config.items():
                assert Path(sumstats_file).exists(), f"{sumstats_file} does not exist."
                self.sumstats_config_dict[_trait_name] = sumstats_file
        # load the sumstats file
        elif self.sumstats_file is not None:
            self.sumstats_config_dict[self.trait_name] = self.sumstats_file
        else:
            raise ValueError("One of sumstats_file and sumstats_config_file must be provided.")

        for sumstats_file in self.sumstats_config_dict.values():
            assert Path(sumstats_file).exists(), f"{sumstats_file} does not exist."

        if self.quick_mode_resource_dir is not None:
            logger.info(
                f"quick_mode_resource_dir is provided: {self.quick_mode_resource_dir}"
            )
            self.ldscore_save_dir = self.quick_mode_resource_dir
            # Fix the path - quick_mode_resource_dir already points to quick_mode directory
            self.snp_gene_weight_adata_path = Path(self.quick_mode_resource_dir) / "snp_gene_weight_matrix.h5ad"

        # Handle w_file
        if self.w_file is None:
            w_ld_dir = Path(self.ldscore_save_dir) / "w_ld"
            if w_ld_dir.exists():
                self.w_file = str(w_ld_dir / "weights.")
                logger.info(f"Using weights generated in the generate_ldscore step: {self.w_file}")
            else:
                raise ValueError(
                    "No w_file provided and no weights found in generate_ldscore output. "
                    "Either provide --w_file or run generate_ldscore first."
                )
        else:
            logger.info(f"Using provided weights file: {self.w_file}")

        if self.use_additional_baseline_annotation:
            self.process_additional_baseline_annotation()

    def process_additional_baseline_annotation(self):
        additional_baseline_annotation = Path(self.ldscore_save_dir) / "additional_baseline"
        dir_exists = additional_baseline_annotation.exists()

        if not dir_exists:
            self.use_additional_baseline_annotation = False
        else:
            logger.info(
                "------Additional baseline annotation is provided. It will be used with the default baseline annotation."
            )
            logger.info(
                f"------Additional baseline annotation directory: {additional_baseline_annotation}"
            )

            chrom_list = range(1, 23)
            for chrom in chrom_list:
                baseline_annotation_path = (
                    additional_baseline_annotation / f"baseline.{chrom}.l2.ldscore.feather"
                )
                if not baseline_annotation_path.exists():
                    raise FileNotFoundError(
                        f"baseline.{chrom}.annot.gz is not found in {additional_baseline_annotation}."
                    )
        return None


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


@dataclass
class MaxPoolingConfig(ConfigWithAutoPaths):
    """Configuration for max pooling across sections."""
    
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
class GenerateLDScoreConfig(ConfigWithAutoPaths):
    """Configuration for generating LD scores."""
    
    # Required from parent
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
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

    sample_name: Annotated[str, typer.Option(
        help="Name of the sample"
    )] = None

    # Optional - must be after required fields
    project_name: str = None
    
    keep_snp_root: Optional[str] = None  # Internal field
    
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
    
    # Additional fields
    ldscore_save_format: str = "feather"
    save_pre_calculate_snp_gene_weight_matrix: bool = False
    baseline_annotation_dir: Optional[str] = None
    SNP_gene_pair_dir: Optional[str] = None


@dataclass
class CauchyCombinationConfig(ConfigWithAutoPaths):
    """Configuration for Cauchy combination test."""
    
    # Required from parent
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
    
    # Optional - must be after required fields
    sample_name: str = None
    project_name: str = None
    
    sample_name_list: Annotated[Optional[str], typer.Option(
        help="Space-separated list of sample names"
    )] = None
    
    output_file: Annotated[Optional[Path], typer.Option(
        help="Path to save the combined Cauchy results"
    )] = None
    
    def __post_init__(self):
        super().__post_init__()
        # Handle sample_name_list
        if self.sample_name_list and isinstance(self.sample_name_list, str):
            self.sample_name_list = self.sample_name_list.split()
        
        if self.sample_name is not None:
            if self.sample_name_list and len(self.sample_name_list) > 0:
                raise ValueError("Only one of sample_name and sample_name_list must be provided.")
            else:
                # Single sample case
                self.sample_name_list = [self.sample_name]
                if self.output_file is None:
                    # Use single sample naming convention
                    self.output_file = Path(
                        f"{self.cauchy_save_dir}/{self.project_name}_single_sample_{self.sample_name}_{self.trait_name}.Cauchy.csv.gz"
                    )
        else:
            # Multiple samples or all samples case
            if not (self.sample_name_list and len(self.sample_name_list) > 0):
                raise ValueError("At least one sample name must be provided via sample_name or sample_name_list.")
            
            if self.output_file is None:
                # Use all samples naming convention when no specific sample_name provided
                self.output_file = Path(
                    f"{self.cauchy_save_dir}/{self.project_name}_all_samples_{self.trait_name}.Cauchy.csv.gz"
                )


@dataclass
class CreateSliceMeanConfig:
    """Configuration for creating slice mean from multiple h5ad files."""
    
    slice_mean_output_file: Annotated[Path, typer.Option(
        help="Path to the output file for the slice mean"
    )]
    
    sample_name_list: Annotated[str, typer.Option(
        help="Space-separated list of sample names"
    )]
    
    h5ad_list: Annotated[str, typer.Option(
        help="Space-separated list of h5ad file paths"
    )]
    
    # Optional parameters
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
    
    species: Optional[str] = None
    h5ad_dict: Optional[dict] = None
    
    def __post_init__(self):

        
        # Parse lists if provided as strings
        if isinstance(self.sample_name_list, str):
            self.sample_name_list = self.sample_name_list.split()
        if isinstance(self.h5ad_list, str):
            self.h5ad_list = self.h5ad_list.split()
        
        if self.h5ad_list is None and self.h5ad_yaml is None:
            raise ValueError("At least one of --h5ad_list or --h5ad_yaml must be provided.")
        
        if self.h5ad_yaml is not None:
            if isinstance(self.h5ad_yaml, (str, Path)):
                logger.info(f"Reading h5ad yaml file: {self.h5ad_yaml}")
                with open(self.h5ad_yaml) as f:
                    h5ad_dict = yaml.safe_load(f)
            else:
                h5ad_dict = self.h5ad_yaml
        elif self.sample_name_list and self.h5ad_list:
            logger.info("Reading sample name list and h5ad list")
            h5ad_dict = dict(zip(self.sample_name_list, self.h5ad_list, strict=False))
        else:
            raise ValueError(
                "Please provide either h5ad_yaml or both sample_name_list and h5ad_list."
            )
        
        # Check if sample names are unique
        assert len(h5ad_dict) == len(set(h5ad_dict)), "Sample names must be unique."
        assert len(h5ad_dict) > 1, "At least two samples are required."
        
        logger.info(f"Input h5ad files: {h5ad_dict}")
        
        # Check if all files exist
        self.h5ad_dict = {}
        for sample_name, h5ad_file in h5ad_dict.items():
            h5ad_file = Path(h5ad_file)
            if not h5ad_file.exists():
                raise FileNotFoundError(f"{h5ad_file} does not exist.")
            self.h5ad_dict[sample_name] = h5ad_file
        
        self.slice_mean_output_file = Path(self.slice_mean_output_file)
        self.slice_mean_output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Verify homolog file format if provided
        if self.homolog_file is not None:
            verify_homolog_file_format(self)


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
class DiagnosisConfig(ConfigWithAutoPaths):
    """Configuration for diagnostic plots and analysis."""
    
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
    
    annotation: Annotated[str, typer.Option(
        help="Annotation layer name"
    )]
    
    trait_name: Annotated[str, typer.Option(
        help="Name of the trait"
    )]
    
    sumstats_file: Annotated[Path, typer.Option(
        help="Path to GWAS summary statistics file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )]
    
    # Optional - must be after required fields
    project_name: str = None
    
    plot_type: Annotated[str, typer.Option(
        help="Type of diagnostic plot to generate",
        case_sensitive=False
    )] = "all"
    
    top_corr_genes: Annotated[int, typer.Option(
        help="Number of top correlated genes",
        min=1,
        max=500
    )] = 50
    
    selected_genes: Annotated[Optional[str], typer.Option(
        help="Comma-separated list of specific genes"
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
    
    customize_fig: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        import logging
        logger = logging.getLogger("gsMap")
        
        if any([self.fig_width, self.fig_height, self.point_size]):
            logger.info("Customizing the figure size and point size.")
            assert all([self.fig_width, self.fig_height, self.point_size]), (
                "All of fig_width, fig_height, and point_size must be provided."
            )
            self.customize_fig = True
        else:
            self.customize_fig = False


@dataclass
class VisualizeConfig(ConfigWithAutoPaths):
    """Configuration for visualization."""
    
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
        help="Name of the trait"
    )]
    
    # Optional - must be after required fields
    project_name: str = None
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation layer name"
    )] = None
    
    fig_title: Annotated[Optional[str], typer.Option(
        help="Figure title"
    )] = None
    
    fig_height: Annotated[int, typer.Option(
        help="Height of the figure",
        min=100,
        max=5000
    )] = 600
    
    fig_width: Annotated[int, typer.Option(
        help="Width of the figure",
        min=100,
        max=5000
    )] = 800
    
    point_size: Annotated[Optional[int], typer.Option(
        help="Point size for the figure"
    )] = None
    
    fig_style: Annotated[str, typer.Option(
        help="Style of the figure",
        case_sensitive=False
    )] = "light"


@dataclass
class RunLinkModeConfig(ConfigWithAutoPaths):
    """Configuration for running gsMap in link mode (pre-computed LD scores)."""
    
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
    
    gsmap_resource_dir: Annotated[Path, typer.Option(
        "--gsmap-resource-dir",
        help="Directory containing gsMap resources",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]
    
    # Optional - must be after required fields
    project_name: str = None
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation in adata.obs to use"
    )] = None
    
    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm"
    )] = "spatial"
    
    trait_name: Annotated[Optional[str], typer.Option(
        help="Name of the trait for GWAS analysis"
    )] = None
    
    sumstats_file: Annotated[Optional[Path], typer.Option(
        help="Path to GWAS summary statistics file",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    sumstats_config_file: Annotated[Optional[Path], typer.Option(
        help="Path to YAML file with trait names and sumstats paths",
        exists=True,
        file_okay=True,
        dir_okay=False
    )] = None
    
    max_processes: Annotated[int, typer.Option(
        help="Maximum number of processes for parallel execution",
        min=1,
        max=50
    )] = 10
    
    use_pooling: Annotated[bool, typer.Option(
        "--use-pooling",
        help="Use pooling across sections"
    )] = False
    
    # Hidden parameters (populated in __post_init__)
    gtffile: Optional[Path] = None
    bfile_root: Optional[str] = None
    keep_snp_root: Optional[str] = None
    w_file: Optional[str] = None
    snp_gene_weight_adata_path: Optional[str] = None
    baseline_annotation_dir: Optional[Path] = None
    SNP_gene_pair_dir: Optional[Path] = None
    sumstats_config_dict: Optional[dict] = None
    
    def __post_init__(self):
        super().__post_init__()
        from pathlib import Path
        import yaml
        
        # Set resource paths
        self.gtffile = Path(f"{self.gsmap_resource_dir}/genome_annotation/gtf/gencode.v46lift37.basic.annotation.gtf")
        self.bfile_root = f"{self.gsmap_resource_dir}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC"
        self.keep_snp_root = f"{self.gsmap_resource_dir}/LDSC_resource/hapmap3_snps/hm"
        self.w_file = f"{self.gsmap_resource_dir}/LDSC_resource/weights_hm3_no_hla/weights."
        self.snp_gene_weight_adata_path = f"{self.gsmap_resource_dir}/quick_mode/snp_gene_weight_matrix.h5ad"
        self.baseline_annotation_dir = Path(f"{self.gsmap_resource_dir}/quick_mode/baseline").resolve()
        self.SNP_gene_pair_dir = Path(f"{self.gsmap_resource_dir}/quick_mode/SNP_gene_pair").resolve()
        
        # Check resource files exist
        if not self.gtffile.exists():
            raise FileNotFoundError(f"GTF file {self.gtffile} does not exist.")
        
        # Validate sumstats inputs
        if self.sumstats_file is None and self.sumstats_config_file is None:
            raise ValueError("One of sumstats_file and sumstats_config_file must be provided.")
        if self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError("Only one of sumstats_file and sumstats_config_file must be provided.")
        if self.sumstats_file is not None and self.trait_name is None:
            raise ValueError("trait_name must be provided if sumstats_file is provided.")
        if self.sumstats_config_file is not None and self.trait_name is not None:
            raise ValueError("trait_name must not be provided if sumstats_config_file is provided.")
        
        self.sumstats_config_dict = {}
        
        # Load sumstats config
        if self.sumstats_config_file is not None:
            with open(self.sumstats_config_file) as f:
                config = yaml.safe_load(f)
            for trait_name, sumstats_file in config.items():
                sumstats_path = Path(sumstats_file)
                if not sumstats_path.exists():
                    raise FileNotFoundError(f"{sumstats_file} does not exist.")
                self.sumstats_config_dict[trait_name] = sumstats_file
        elif self.sumstats_file is not None and self.trait_name is not None:
            self.sumstats_config_dict[self.trait_name] = str(self.sumstats_file)
        
        # Verify all sumstats files exist
        for sumstats_file in self.sumstats_config_dict.values():
            if not Path(sumstats_file).exists():
                raise FileNotFoundError(f"{sumstats_file} does not exist.")


@dataclass
class ThreeDCombineConfig:
    """Configuration for 3D visualization and combination."""
    
    workdir: Annotated[Path, typer.Option(
        help="Path to the working directory",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True
    )]
    
    # Optional parameters
    trait_name: Annotated[Optional[str], typer.Option(
        help="Name of the trait"
    )] = None
    
    adata_3d: Annotated[Optional[str], typer.Option(
        help="Path to 3D anndata file"
    )] = None
    
    project_name: Annotated[Optional[str], typer.Option(
        help="Project name"
    )] = None
    
    st_id: Annotated[Optional[str], typer.Option(
        help="Spatial transcriptomics ID"
    )] = None
    
    annotation: Annotated[Optional[str], typer.Option(
        help="Annotation layer name"
    )] = None
    
    spatial_key: Annotated[str, typer.Option(
        help="Spatial key in adata.obsm"
    )] = "spatial"
    
    cmap: Annotated[Optional[str], typer.Option(
        help="Colormap for visualization"
    )] = None
    
    point_size: Annotated[float, typer.Option(
        help="Point size for 3D visualization",
        min=0.001,
        max=1.0
    )] = 0.01
    
    background_color: Annotated[str, typer.Option(
        help="Background color for visualization"
    )] = "white"
    
    n_snapshot: Annotated[int, typer.Option(
        help="Number of snapshots to generate",
        min=1,
        max=1000
    )] = 200
    
    show_outline: Annotated[bool, typer.Option(
        "--show-outline",
        help="Show outline in visualization"
    )] = False
    
    save_mp4: Annotated[bool, typer.Option(
        "--save-mp4",
        help="Save as MP4 video"
    )] = False
    
    save_gif: Annotated[bool, typer.Option(
        "--save-gif",
        help="Save as GIF animation"
    )] = False
    
    project_dir: Optional[Path] = None
    
    def __post_init__(self):
        from pathlib import Path
        
        if self.workdir is None:
            raise ValueError('workdir must be provided.')
        work_dir = Path(self.workdir)
        if self.project_name is not None:
            self.project_dir = work_dir / self.project_name
        else:
            self.project_dir = work_dir


# Helper function for homolog file verification
def verify_homolog_file_format(config):
    """Verify the format of homolog file."""
    import logging
    logger = logging.getLogger("gsMap")
    
    if config.homolog_file is not None:
        logger.info(
            f"User provided homolog file to map gene names to human: {config.homolog_file}"
        )
        # Check the format of the homolog file
        with open(config.homolog_file) as f:
            first_line = f.readline().strip()
            _n_col = len(first_line.split())
            if _n_col != 2:
                raise ValueError(
                    f"Invalid homolog file format. Expected 2 columns, first column should be other species gene name, "
                    f"second column should be human gene name. Got {_n_col} columns in the first line."
                )
            else:
                first_col_name, second_col_name = first_line.split()
                config.species = first_col_name
                logger.info(
                    f"Homolog file provided and will map gene name from column1:{first_col_name} to column2:{second_col_name}"
                )
    else:
        logger.info("No homolog file provided. Run in human mode.")