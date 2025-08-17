#!/usr/bin/env python
"""
Complete refactored config.py using dataclass_typer decorator with proper inheritance,
resource tracking, and banner display for elegant CLI design.
"""

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, fields
from functools import wraps
from pathlib import Path
from typing import Optional, Annotated, List, Literal
import inspect

import psutil
import pyfiglet
import typer

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s | %(name)s - %(message)s')
logger = logging.getLogger("gsMap")

# Version (mock for testing)
__version__ = "1.73.5"

# Create the Typer app
app = typer.Typer(
    name="gsmap",
    help="gsMap: genetically informed spatial mapping of cells for complex traits",
    rich_markup_mode="rich",
    add_completion=False,
)


def ensure_path_exists(func):
    """Decorator to ensure path exists when accessing properties."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, Path):
            if result.suffix:
                result.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            else:  # It's a directory path
                result.mkdir(parents=True, exist_ok=True, mode=0o755)
        return result
    return wrapper


def process_cpu_time(proc: psutil.Process):
    """Calculate total CPU time for a process."""
    cpu_times = proc.cpu_times()
    total = cpu_times.user + cpu_times.system
    return total


def track_resource_usage(func):
    """
    Decorator to track resource usage during function execution.
    Logs memory usage, CPU time, and wall clock time at the end of the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the current process
        process = psutil.Process(os.getpid())
        
        # Initialize tracking variables
        peak_memory = 0
        cpu_percent_samples = []
        stop_thread = False
        
        # Function to monitor resource usage
        def resource_monitor():
            nonlocal peak_memory, cpu_percent_samples
            while not stop_thread:
                try:
                    # Get current memory usage in MB
                    current_memory = process.memory_info().rss / (1024 * 1024)
                    peak_memory = max(peak_memory, current_memory)
                    
                    # Get CPU usage percentage
                    cpu_percent = process.cpu_percent(interval=None)
                    if cpu_percent > 0:  # Skip initial zero readings
                        cpu_percent_samples.append(cpu_percent)
                    
                    time.sleep(0.5)
                except Exception:
                    pass
        
        # Start resource monitoring in a separate thread
        monitor_thread = threading.Thread(target=resource_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Get start times
        start_wall_time = time.time()
        start_cpu_time = process_cpu_time(process)
        
        try:
            # Run the actual function
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop the monitoring thread
            stop_thread = True
            monitor_thread.join(timeout=1.0)
            
            # Calculate elapsed times
            end_wall_time = time.time()
            end_cpu_time = process_cpu_time(process)
            
            wall_time = end_wall_time - start_wall_time
            cpu_time = end_cpu_time - start_cpu_time
            
            # Calculate average CPU percentage
            avg_cpu_percent = (
                sum(cpu_percent_samples) / len(cpu_percent_samples) if cpu_percent_samples else 0
            )
            
            # Format memory for display
            if peak_memory < 1024:
                memory_str = f"{peak_memory:.2f} MB"
            else:
                memory_str = f"{peak_memory / 1024:.2f} GB"
            
            # Format times for display
            if wall_time < 60:
                wall_time_str = f"{wall_time:.2f} seconds"
            elif wall_time < 3600:
                wall_time_str = f"{wall_time / 60:.2f} minutes"
            else:
                wall_time_str = f"{wall_time / 3600:.2f} hours"
            
            if cpu_time < 60:
                cpu_time_str = f"{cpu_time:.2f} seconds"
            elif cpu_time < 3600:
                cpu_time_str = f"{cpu_time / 60:.2f} minutes"
            else:
                cpu_time_str = f"{cpu_time / 3600:.2f} hours"
            
            # Log the resource usage
            logger.info("Resource usage summary:")
            logger.info(f"  • Wall clock time: {wall_time_str}")
            logger.info(f"  • CPU time: {cpu_time_str}")
            logger.info(f"  • Average CPU utilization: {avg_cpu_percent:.1f}%")
            logger.info(f"  • Peak memory usage: {memory_str}")
    
    return wrapper


def show_banner(command_name: str):
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


def dataclass_typer(func):
    """
    Decorator to convert a function that takes a dataclass config
    into a Typer command with individual CLI options.
    """
    sig = inspect.signature(func)
    
    # Get the dataclass type from the function signature
    config_param = list(sig.parameters.values())[0]
    config_class = config_param.annotation
    
    @wraps(func)
    @track_resource_usage  # Add resource tracking
    def wrapper(**kwargs):
        # Show banner
        show_banner(func.__name__)
        
        # Create the config instance
        config = config_class(**kwargs)
        result = func(config)
        
        # Record end time
        end_time = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Finished at: {end_time}")
        
        return result
    
    # Build new parameters from dataclass fields
    params = []
    for field in fields(config_class):
        # Skip fields that are part of parent class and handled internally
        if field.name in ['project_dir']:
            continue
            
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
# Base Configuration Class with Auto Paths
# ============================================================================

class ConfigWithAutoPaths:
    """Base configuration class with automatic path generation."""
    # These will be provided by subclasses
    workdir: str
    project_name: str
    sample_name: str
    
    def __post_init__(self):
        if self.workdir is None:
            raise ValueError('workdir must be provided.')
        work_dir = Path(self.workdir)
        if self.project_name is not None:
            self.project_dir = work_dir / self.project_name
        else:
            self.project_dir = work_dir
    
    @property
    @ensure_path_exists
    def latent_dir(self) -> Path:
        return self.project_dir / "find_latent_representations"
    
    @property
    @ensure_path_exists
    def zarr_group_path(self) -> Path:
        return self.project_dir / "slice_mean.zarr"
    
    @property
    @ensure_path_exists
    def model_path(self) -> Path:
        return Path(f'{self.project_dir}/LGCN_model/gsMap_LGCN_.pt')
    
    @property
    @ensure_path_exists
    def hdf5_with_latent_path(self) -> Path:
        return Path(
            f"{self.project_dir}/find_latent_representations/{self.sample_name}_add_latent.h5ad"
        )
    
    @property
    @ensure_path_exists
    def mkscore_feather_path(self) -> Path:
        return Path(f'{self.project_dir}/latent_to_gene/mk_score/{self.sample_name}_gene_marker_score.feather')
    
    @property
    @ensure_path_exists
    def tuned_mkscore_feather_path(self) -> Path:
        return Path(f'{self.project_dir}/latent_to_gene/mk_score_pooling/{self.sample_name}_gene_marker_score.feather')
    
    @property
    @ensure_path_exists
    def ldscore_save_dir(self) -> Path:
        return Path(f'{self.project_dir}/generate_ldscore/{self.sample_name}')
    
    @property
    @ensure_path_exists
    def ldsc_save_dir(self) -> Path:
        return Path(f'{self.project_dir}/spatial_ldsc/{self.sample_name}')
    
    @property
    @ensure_path_exists
    def cauchy_save_dir(self) -> Path:
        return Path(f'{self.project_dir}/cauchy_combination/{self.sample_name}')
    
    @ensure_path_exists
    def get_report_dir(self, trait_name: str) -> Path:
        return Path(f'{self.project_dir}/report/{self.sample_name}/{trait_name}')
    
    def get_gsMap_report_file(self, trait_name: str) -> Path:
        return (
            self.get_report_dir(trait_name)
            / f"{self.sample_name}_{trait_name}_gsMap_Report.html"
        )
    
    @ensure_path_exists
    def get_manhattan_html_plot_path(self, trait_name: str) -> Path:
        return Path(
            f'{self.project_dir}/report/{self.sample_name}/{trait_name}/manhattan_plot/{self.sample_name}_{trait_name}_Diagnostic_Manhattan_Plot.html')
    
    @ensure_path_exists
    def get_GSS_plot_dir(self, trait_name: str) -> Path:
        return Path(f'{self.project_dir}/report/{self.sample_name}/{trait_name}/GSS_plot')
    
    def get_GSS_plot_select_gene_file(self, trait_name: str) -> Path:
        return self.get_GSS_plot_dir(trait_name) / "plot_genes.csv"
    
    @ensure_path_exists
    def get_ldsc_result_file(self, trait_name: str) -> Path:
        return Path(f"{self.ldsc_save_dir}/{self.sample_name}_{trait_name}.csv.gz")
    
    @ensure_path_exists
    def get_cauchy_result_file(self, trait_name: str) -> Path:
        return Path(
            f"{self.cauchy_save_dir}/{self.sample_name}_{trait_name}.Cauchy.csv.gz"
        )
    
    @ensure_path_exists
    def get_gene_diagnostic_info_save_path(self, trait_name: str) -> Path:
        return Path(
            f'{self.project_dir}/report/{self.sample_name}/{trait_name}/{self.sample_name}_{trait_name}_Gene_Diagnostic_Info.csv')
    
    @ensure_path_exists
    def get_gsMap_plot_save_dir(self, trait_name: str) -> Path:
        return Path(f'{self.project_dir}/report/{self.sample_name}/{trait_name}/gsMap_plot')
    
    def get_gsMap_html_plot_save_path(self, trait_name: str) -> Path:
        return (
            self.get_gsMap_plot_save_dir(trait_name)
            / f"{self.sample_name}_{trait_name}_gsMap_plot.html"
        )


# ============================================================================
# Configuration dataclasses inheriting from ConfigWithAutoPaths
# ============================================================================

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
    
    two_stage: Annotated[bool, typer.Option(
        "--two-stage/--single-stage",
        help="Tune the cell embeddings based on the provided annotation"
    )] = True


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
    
    num_processes: Annotated[int, typer.Option(
        help="Number of processes for parallel computing",
        min=1,
        max=50
    )] = 4
    
    use_jax: Annotated[bool, typer.Option(
        "--use-jax/--no-jax",
        help="Use JAX-accelerated implementation"
    )] = True


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
    
    Common workflows:
    
    1. Quick mode (recommended for first-time users):
       gsmap quick-mode --workdir /path/to/work --sample-name my_sample ...
    
    2. Step-by-step analysis:
       gsmap find-latent ...
       gsmap spatial-ldsc ...
       gsmap report ...
    
    For detailed documentation, visit: https://github.com/mcgilldinglab/gsMap
    """
    pass


if __name__ == "__main__":
    app()