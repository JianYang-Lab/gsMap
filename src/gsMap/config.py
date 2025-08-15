import argparse
import dataclasses
import functools
import logging
import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict, namedtuple
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from pprint import pprint
from typing import Literal, Optional

import psutil
import pyfiglet
import yaml

from gsMap.__init__ import __version__

# Global registry to hold functions
cli_function_registry = OrderedDict()
subcommand = namedtuple("subcommand", ["name", "func", "add_args_function", "description"])


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


@functools.cache
def macos_timebase_factor():
    """
    On MacOS, `psutil.Process.cpu_times()` is not accurate, check activity monitor instead.
    see: https://github.com/giampaolo/psutil/issues/2411#issuecomment-2274682289
    """
    default_factor = 1
    ioreg_output_lines = []

    try:
        result = subprocess.run(
            ["ioreg", "-p", "IODeviceTree", "-c", "IOPlatformDevice"],
            capture_output=True,
            text=True,
            check=True,
        )
        ioreg_output_lines = result.stdout.splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        return default_factor

    if not ioreg_output_lines:
        return default_factor

    for line in ioreg_output_lines:
        if "timebase-frequency" in line:
            match = re.search(r"<([0-9a-fA-F]+)>", line)
            if not match:
                return default_factor
            byte_data = bytes.fromhex(match.group(1))
            timebase_freq = int.from_bytes(byte_data, byteorder="little")
            # Typically, it should be 1000/24.
            return pow(10, 9) / timebase_freq
    return default_factor


def process_cpu_time(proc: psutil.Process):
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
                except Exception:  # Catching all exceptions here because... # noqa: BLE001
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

            if sys.platform == "darwin":
                cpu_time *= macos_timebase_factor()
                avg_cpu_percent *= macos_timebase_factor()

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
            import logging

            logger = logging.getLogger("gsMap")
            logger.info("Resource usage summary:")
            logger.info(f"  • Wall clock time: {wall_time_str}")
            logger.info(f"  • CPU time: {cpu_time_str}")
            logger.info(f"  • Average CPU utilization: {avg_cpu_percent:.1f}%")
            logger.info(f"  • Peak memory usage: {memory_str}")

    return wrapper


# Decorator to register functions for cli parsing
def register_cli(name: str, description: str, add_args_function: Callable) -> Callable:
    def decorator(func: Callable) -> Callable:
        @track_resource_usage  # Use enhanced resource tracking
        @wraps(func)
        def wrapper(*args, **kwargs):
            name.replace("_", " ")
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
            logger.info(f"Running {name}...")

            # Record start time for the log message
            start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Started at: {start_time}")

            func(*args, **kwargs)

            # Record end time for the log message
            end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Finished running {name} at: {end_time}.")

        cli_function_registry[name] = subcommand(
            name=name, func=wrapper, add_args_function=add_args_function, description=description
        )
        return wrapper

    return decorator


def str_or_float(value):
    try:
        return int(value)
    except ValueError:
        return value

def add_shared_args(parser):
    parser.add_argument('--workdir', type=str, required=True,
                        help='Path to the working directory.')
    parser.add_argument('--sample_name', type=str,
                        required=True, help='Name of the sample.')
    parser.add_argument('--project_name', type=str,
                        required=False, help='Project name')


def add_qc_stdata_args(parser):
    add_shared_args(parser)
    parser.add_argument('--spe_file_list', required=True,
                        type=str, help='List of input ST (.h5ad) files.')
    parser.add_argument('--data_layer', type=str,
                        default='count', help='Gene expression data layer.')

def add_find_latent_representations_args(parser):
    add_shared_args(parser)

    # File paths and general settings
    parser.add_argument(
        "--spe_file_list",
        required=True,
        type=str,
        help="List of input ST (.h5ad) files."
    )
    # parser.add_argument('--workdir', required=True, type=str, help='Working directory of gsMap3D.')
    parser.add_argument(
        "--data_layer", type=str, default="count", help="Gene expression data layer."
    )
    parser.add_argument(
        "--spatial_key",
        type=str,
        default="spatial",
        required=False,
        help="spatial key in adata.obsm storing spatial coordinats."
    )
    parser.add_argument(
        "--annotation",
        type=str,
        default=None,
        help="Annotation in adata.obs to use."
    )

    # Feature extraction parameters (LGCN)
    parser.add_argument(
        "--n_neighbors", type=int, default=10, help="Number of neighbors for LGCN."
    )
    parser.add_argument(
        "--K", type=int, default=3, help="Graph convolution depth for LGCN."
    )
    parser.add_argument(
        "--feat_cell",
        type=int,
        default=2000,
        help="Number of top variable features to retain.",
    )
    parser.add_argument(
        "--pearson_residual", action="store_true", help="Take the residuals of the input data."
    )

    # Model dimension parameters
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Units in the first hidden layer.",
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        default=32,
        help="Size of the latent embedding layer.",
    )

    # Transformer module parameters
    parser.add_argument(
        "--use_tf", action="store_true", help="Enable transformer module."
    )
    parser.add_argument(
        "--module_dim",
        type=int,
        default=30,
        help="Dimensionality of transformer modules.",
    )
    parser.add_argument(
        "--hidden_gmf",
        type=int,
        default=128,
        help="Hidden units for global mean feature extractor.",
    )
    parser.add_argument(
        "--n_modules", type=int, default=16, help="Number of transformer modules."
    )
    parser.add_argument(
        "--nhead", type=int, default=4, help="Number of attention heads in transformer."
    )
    parser.add_argument(
        "--n_enc_layer",
        type=int,
        default=2,
        help="Number of transformer encoder layers.",
    )

    # Training parameters
    parser.add_argument(
        "--distribution",
        type=str,
        choices=["nb", "zinb", "gaussian"],
        default="nb",
        help='Distribution type for loss calculation (e.g., "nb").',
    )
    parser.add_argument(
        "--n_cell_training",
        type=int,
        default=100000,
        help="Number of cells used for training.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Batch size for training."
    )
    parser.add_argument(
        "--itermax",
        type=int,
        default=100,
        help="Maximum number of training iterations.",
    )
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience."
    )
    parser.add_argument(
        "--two_stage",
        type=bool,
        default=True,
        help="Tune the cell embeddings based on the provided annotation",
    )
    parser.add_argument(
        "--do_sampling",
        type=bool,
        default=True,
        help="Donw-sampling cells in training.",
    )

    # Homologs transformation
    parser.add_argument(
        '--homolog_file', type=str,
        help='Path to homologous gene conversion file (optional).'
    )


def chrom_choice(value):
    if value.isdigit():
        ivalue = int(value)
        if 1 <= ivalue <= 22:
            return ivalue
    elif value.lower() == "all":
        return value
    else:
        raise argparse.ArgumentTypeError(
            f"'{value}' is an invalid chromosome choice. Choose from 1-22 or 'all'."
        )


def filter_args_for_dataclass(args_dict, data_class: dataclass):
    return {k: v for k, v in args_dict.items() if k in data_class.__dataclass_fields__}


def get_dataclass_from_parser(args: argparse.Namespace, data_class: dataclass):
    remain_kwargs = filter_args_for_dataclass(vars(args), data_class)
    print(f"Using the following arguments for {data_class.__name__}:", flush=True)
    pprint(remain_kwargs, indent=4)
    sys.stdout.flush()
    return data_class(**remain_kwargs)


def add_latent_to_gene_args(parser):
    add_shared_args(parser)
    parser.add_argument(
        "--annotation",
        type=str,
        default=None,
        help="Annotation in adata.obs to use."
    )
    parser.add_argument(
        "--no_expression_fraction",
        action="store_true",
        help="Skip expression fraction filtering.",
    )
    parser.add_argument(
        "--latent_representation",
        type=str,
        default="emb_gcn",
        required=False,
        help="Type of latent representation.",
    )
    parser.add_argument(
        "--latent_representation_indv",
        type=str,
        default="emb",
        required=False,
        help="Type of latent representation.",
    )
    parser.add_argument(
        "--spatial_key",
        type=str,
        default="spatial",
        required=False,
        help="spatial key in adata.obsm storing spatial coordinats.",
    )
    parser.add_argument(
        "--num_anchor", type=int, default=51, help="Number of neighbors."
    )
    parser.add_argument(
        "--num_neighbour", type=int, default=21, help="Number of neighbors."
    )
    parser.add_argument(
        "--num_neighbour_spatial",
        type=int,
        default=201,
        help="Number of spatial neighbors.",
    )
    parser.add_argument(
        "--use_w",
        action="store_true",
        default=False,
        help="Using section specific weights to acount for across section batch effect.",
    )

def add_max_pooling_args(parser):
    add_shared_args(parser)
    parser.add_argument(
        "--spe_file_list",required=True,type=str,help="List of input ST (.h5ad) files."
    )
    parser.add_argument(
        "--annotation", type=str, default=None,help="Annotation in adata.obs to use."
    )
    parser.add_argument(
        "--spatial_key",type=str, default="spatial", help="spatial key in adata.obsm storing spatial coordinats."
    )
    parser.add_argument(
        "--sim_thresh",type=float, default=0.85,help="Similarity threshold for MNN matching.",
    )


def add_generate_ldscore_args(parser):
    add_shared_args(parser)
    parser.add_argument("--chrom", type=str, required=True, help='Chromosome id (1-22) or "all".')
    parser.add_argument(
        "--bfile_root",
        type=str,
        required=True,
        help="Root path for genotype plink bfiles (.bim, .bed, .fam).",
    )
    parser.add_argument(
        "--keep_snp_root", type=str, required=False, help="Root path for SNP files"
    )
    parser.add_argument(
        "--gtf_annotation_file", type=str, required=True, help="Path to GTF annotation file."
    )
    parser.add_argument(
        "--gene_window_size", type=int, default=50000, help="Gene window size in base pairs."
    )
    parser.add_argument(
        "--enhancer_annotation_file", type=str, help="Path to enhancer annotation file (optional)."
    )
    parser.add_argument(
        "--snp_multiple_enhancer_strategy",
        type=str,
        choices=["max_mkscore", "nearest_TSS"],
        default="max_mkscore",
        help="Strategy for handling multiple enhancers per SNP.",
    )
    parser.add_argument(
        "--gene_window_enhancer_priority",
        type=str,
        choices=["gene_window_first", "enhancer_first", "enhancer_only"],
        help="Priority between gene window and enhancer annotations.",
    )
    parser.add_argument(
        "--spots_per_chunk", type=int, default=1000, help="Number of spots per chunk."
    )
    parser.add_argument("--ld_wind", type=int, default=1, help="LD window size.")
    parser.add_argument(
        "--ld_unit",
        type=str,
        choices=["SNP", "KB", "CM"],
        default="CM",
        help="Unit for LD window.",
    )
    parser.add_argument(
        "--additional_baseline_annotation",
        type=str,
        default=None,
        help="Path of additional baseline annotations",
    )


def add_spatial_ldsc_args(parser):
    add_shared_args(parser)
    parser.add_argument(
        "--sumstats_file", type=str, required=True, help="Path to GWAS summary statistics file."
    )
    parser.add_argument(
        "--w_file",
        type=str,
        required=False,
        default=None,
        help="Path to regression weight file. If not provided, will use weights generated in the generate_ldscore step.",
    )
    parser.add_argument(
        "--trait_name", type=str, required=True, help="Name of the trait being analyzed."
    )
    parser.add_argument(
        "--n_blocks", type=int, default=200, help="Number of blocks for jackknife resampling."
    )
    parser.add_argument(
        "--chisq_max", type=int, help="Maximum chi-square value for filtering SNPs."
    )
    parser.add_argument(
        "--num_processes", type=int, default=4, help="Number of processes for parallel computing."
    )
    parser.add_argument(
        "--use_additional_baseline_annotation",
        type=bool,
        nargs="?",
        const=True,
        default=True,
        help="Use additional baseline annotations when provided",
    )
    parser.add_argument(
        "--use_jax",
        type=bool,
        nargs="?",
        const=True,
        default=True,
        help="Use JAX-accelerated implementation (default: True). Set to False to use standard implementation.",
    )


def add_Cauchy_combination_args(parser):
    parser.add_argument(
        "--workdir", type=str, required=True, help="Path to the working directory."
    )
    parser.add_argument("--sample_name", type=str, required=False, help="Name of the sample.")

    parser.add_argument(
        "--trait_name", type=str, required=True, help="Name of the trait being analyzed."
    )
    parser.add_argument(
        "--annotation", type=str, required=True, help="Name of the annotation in adata.obs to use."
    )

    parser.add_argument(
        "--sample_name_list",
        type=str,
        nargs="+",
        required=False,
        help="List of sample names to process. Provide as a space-separated list.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        help="Path to save the combined Cauchy results. Required when using multiple samples.",
    )


def add_report_args(parser):
    add_shared_args(parser)
    parser.add_argument(
        "--trait_name",
        type=str,
        required=True,
        help="Name of the trait to generate the report for.",
    )
    parser.add_argument("--annotation", type=str, required=True, help="Annotation layer name.")
    # parser.add_argument('--plot_type', type=str, choices=['manhattan', 'GSS', 'gsMap', 'all'], default='all',
    #                     help="Type of diagnostic plot to generate. Choose from 'manhattan', 'GSS', 'gsMap', or 'all'.")
    parser.add_argument(
        "--top_corr_genes", type=int, default=50, help="Number of top correlated genes to display."
    )
    parser.add_argument(
        "--selected_genes",
        type=str,
        nargs="*",
        help="List of specific genes to include in the report (optional).",
    )
    parser.add_argument(
        "--sumstats_file", type=str, required=True, help="Path to GWAS summary statistics file."
    )

    # Optional arguments for customization
    parser.add_argument(
        "--fig_width", type=int, default=None, help="Width of the generated figures in pixels."
    )
    parser.add_argument(
        "--fig_height", type=int, default=None, help="Height of the generated figures in pixels."
    )
    parser.add_argument("--point_size", type=int, default=None, help="Point size for the figures.")
    parser.add_argument(
        "--fig_style",
        type=str,
        default="light",
        choices=["dark", "light"],
        help="Style of the generated figures.",
    )


def add_create_slice_mean_args(parser):
    parser.add_argument(
        "--sample_name_list",
        type=str,
        nargs="+",
        required=True,
        help="List of sample names to process. Provide as a space-separated list.",
    )

    parser.add_argument(
        "--h5ad_list",
        type=str,
        nargs="+",
        help="List of h5ad file paths corresponding to the sample names. Provide as a space-separated list.",
    )
    parser.add_argument(
        "--h5ad_yaml",
        type=str,
        default=None,
        help="Path to the YAML file containing sample names and associated h5ad file paths",
    )
    parser.add_argument(
        "--slice_mean_output_file",
        type=str,
        required=True,
        help="Path to the output file for the slice mean",
    )
    parser.add_argument(
        "--homolog_file", type=str, help="Path to homologous gene conversion file (optional)."
    )
    parser.add_argument(
        "--data_layer",
        type=str,
        default="counts",
        required=True,
        help='Data layer for gene expression (e.g., "count", "counts", "log1p").',
    )


def add_format_sumstats_args(parser):
    # Required arguments
    parser.add_argument("--sumstats", required=True, type=str, help="Path to gwas summary data")
    parser.add_argument(
        "--out", required=True, type=str, help="Path to save the formatted gwas data"
    )

    # Arguments for specify column name
    parser.add_argument(
        "--snp",
        default=None,
        type=str,
        help="Name of snp column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--a1",
        default=None,
        type=str,
        help="Name of effect allele column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--a2",
        default=None,
        type=str,
        help="Name of none-effect allele column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--info",
        default=None,
        type=str,
        help="Name of info column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--beta",
        default=None,
        type=str,
        help="Name of gwas beta column (if not a name that gsMap understands).",
    )
    parser.add_argument(
        "--se",
        default=None,
        type=str,
        help="Name of gwas standar error of beta column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--p",
        default=None,
        type=str,
        help="Name of p-value column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--frq",
        default=None,
        type=str,
        help="Name of A1 ferquency column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--n",
        default=None,
        type=str_or_float,
        help="Name of sample size column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--z",
        default=None,
        type=str,
        help="Name of gwas Z-statistics column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--OR",
        default=None,
        type=str,
        help="Name of gwas OR column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--se_OR",
        default=None,
        type=str,
        help="Name of standar error of OR column (if not a name that gsMap understands)",
    )

    # Arguments for convert SNP (chr, pos) to rsid
    parser.add_argument(
        "--chr",
        default="Chr",
        type=str,
        help="Name of SNP chromosome column (if not a name that gsMap understands)",
    )
    parser.add_argument(
        "--pos",
        default="Pos",
        type=str,
        help="Name of SNP positions column (if not a name that gsMap understands)",
    )
    parser.add_argument("--dbsnp", default=None, type=str, help="Path to reference dnsnp file")
    parser.add_argument(
        "--chunksize", default=1e6, type=int, help="Chunk size for loading dbsnp file"
    )

    # Arguments for output format and quality
    parser.add_argument(
        "--format",
        default="gsMap",
        type=str,
        help="Format of output data",
        choices=["gsMap", "COJO"],
    )
    parser.add_argument("--info_min", default=0.9, type=float, help="Minimum INFO score.")
    parser.add_argument("--maf_min", default=0.01, type=float, help="Minimum MAF.")
    parser.add_argument(
        "--keep_chr_pos",
        action="store_true",
        default=False,
        help="Keep SNP chromosome and position columns in the output data",
    )


def add_run_all_mode_args(parser):
    add_shared_args(parser)

    # Required paths and configurations
    parser.add_argument(
        "--gsMap_resource_dir",
        type=str,
        required=True,
        help="Directory containing gsMap resources (e.g., genome annotations, LD reference panel, etc.).",
    )
    parser.add_argument(
        "--hdf5_path",
        type=str,
        required=True,
        help="Path to the input spatial transcriptomics data (H5AD format).",
    )
    parser.add_argument(
        "--annotation", type=str, required=True, help="Name of the annotation in adata.obs to use."
    )
    parser.add_argument(
        "--data_layer",
        type=str,
        default="counts",
        required=True,
        help='Data layer for gene expression (e.g., "count", "counts", "log1p").',
    )

    # GWAS Data Parameters
    parser.add_argument(
        "--trait_name",
        type=str,
        help="Name of the trait for GWAS analysis (required if sumstats_file is provided).",
    )
    parser.add_argument(
        "--sumstats_file",
        type=str,
        help="Path to GWAS summary statistics file. Either sumstats_file or sumstats_config_file is required.",
    )
    parser.add_argument(
        "--sumstats_config_file",
        type=str,
        help="Path to GWAS summary statistics config file. Either sumstats_file or sumstats_config_file is required.",
    )

    # Homolog Data Parameters
    parser.add_argument(
        "--homolog_file",
        type=str,
        help="Path to homologous gene for converting gene names from different species to human (optional, used for cross-species analysis).",
    )

    # Maximum number of processes
    parser.add_argument(
        "--max_processes",
        type=int,
        default=10,
        help="Maximum number of processes for parallel execution.",
    )

    parser.add_argument(
        "--latent_representation",
        type=str,
        default=None,
        help="Type of latent representation. This should exist in the h5ad obsm.",
    )
    parser.add_argument("--num_neighbour", type=int, default=21, help="Number of neighbors.")
    parser.add_argument(
        "--num_neighbour_spatial", type=int, default=101, help="Number of spatial neighbors."
    )
    parser.add_argument(
        "--gM_slices", type=str, default=None, help="Path to the slice mean file (optional)."
    )
    parser.add_argument(
        "--pearson_residuals", action="store_true", help="Using the pearson residuals."
    )
    parser.add_argument(
        "--use_jax",
        type=bool,
        nargs="?",
        const=True,
        default=True,
        help="Use JAX-accelerated spatial LDSC implementation (default: True). Set to False to use standard implementation.",
    )


def ensure_path_exists(func):
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


class PostInitMeta(type):
    def __new__(cls, name, bases, namespace):
        original_post_init = namespace.get("__post_init__", None)

        def wrapped_post_init(self):
            # Execute base class's __post_init__ if it exists
            for base in bases:
                if hasattr(base, "__post_init__"):
                    base.__post_init__(self)

            # Execute core logic
            if self.workdir is None:
                raise ValueError("workdir must be provided.")

            work_dir = Path(self.workdir)
            if self.project_name is not None:
                self.project_dir = work_dir / self.project_name
            else:
                self.project_dir = work_dir

            # Call the class's original __post_init__ if it exists
            if original_post_init:
                original_post_init(self)

        namespace["__post_init__"] = wrapped_post_init
        return super().__new__(cls, name, bases, namespace)

@dataclass
class ConfigWithAutoPaths(metaclass=PostInitMeta):
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
        return Path(f'{self.project_dir}/LGCN_model/gsMap3D_LGCN_.pt')

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

    def get_gsMap3D_report_file(self, trait_name: str) -> Path:
        return (
                self.get_report_dir(trait_name)
                / f"{self.sample_name}_{trait_name}_gsMap3D_Report.html"
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
    def get_gsMap3D_plot_save_dir(self, trait_name: str) -> Path:
        return Path(f'{self.project_dir}/report/{self.sample_name}/{trait_name}/gsMap3D_plot')

    def get_gsMap3D_html_plot_save_path(self, trait_name: str) -> Path:
        return (
                self.get_gsMap3D_plot_save_dir(trait_name)
                / f"{self.sample_name}_{trait_name}_gsMap3D_plot.html"
        )


@dataclass
class CreateSliceMeanConfig:
    slice_mean_output_file: str | Path
    h5ad_yaml: str | dict | None = None
    sample_name_list: list | None = None
    h5ad_list: list | None = None
    homolog_file: str | None = None
    species: str | None = None
    data_layer: str = None

    def __post_init__(self):
        if self.h5ad_list is None and self.h5ad_yaml is None:
            raise ValueError("At least one of --h5ad_list or --h5ad_yaml must be provided.")
        if self.h5ad_yaml is not None:
            if isinstance(self.h5ad_yaml, str):
                logger.info(f"Reading h5ad yaml file: {self.h5ad_yaml}")
            h5ad_dict = (
                yaml.safe_load(open(self.h5ad_yaml))
                if isinstance(self.h5ad_yaml, str)
                else self.h5ad_yaml
            )
        elif self.sample_name_list and self.h5ad_list:
            logger.info("Reading sample name list and h5ad list")
            h5ad_dict = dict(zip(self.sample_name_list, self.h5ad_list, strict=False))
        else:
            raise ValueError(
                "Please provide either h5ad_yaml or both sample_name_list and h5ad_list."
            )

        # check if sample names is unique
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

        verify_homolog_file_format(self)


@dataclass
class FindLatentRepresentationsConfig(ConfigWithAutoPaths):
    # File paths and general settings
    spe_file_list: str
    workdir: str
    data_layer: str = "count"
    annotation: str = None
    spatial_key: str = "spatial"

    # Feature extraction parameters (LGCN)
    n_neighbors: int = 10
    K: int = 3
    feat_cell: int = 2000
    pearson_residual: bool = False

    # Model dimension parameters
    hidden_size: int = 128
    embedding_size: int = 32

    # Transformer module parameters
    use_tf: bool = False  # Changed to match args parser default (action="store_true" defaults to False)
    module_dim: int = 30
    hidden_gmf: int = 128
    n_modules: int = 16
    nhead: int = 4
    n_enc_layer: int = 2

    # Training parameters
    distribution: str = "nb"
    n_cell_training: int = 100000
    batch_size: int = 1024
    itermax: int = 100
    patience: int = 10
    two_stage: bool = True
    do_sampling: bool = True
    homolog_file: str = None

    def __post_init__(self):
        super().__post_init__()
        verify_homolog_file_format(self)
        if self.annotation and self.two_stage:
            logger.info(
                f"------Finding cell embeddings with reconstruction loss, followed by tuning based on annotation: {
                    self.annotation}."
            )
        else:
            logger.info(
                f"------Finding cell embeddings using only reconstruction loss."
            )

@dataclass
class LatentToGeneConfig(ConfigWithAutoPaths):
    # Required by parent class
    input_hdf5_path: str = None
    
    # Core parameters
    no_expression_fraction: bool = False
    latent_representation: str = "emb_gcn"  # Changed to match args parser default
    latent_representation_indv: str = "emb"  # Changed to match args parser default
    spatial_key: str = 'spatial'
    num_neighbour: int = 21
    num_anchor: int = 51
    num_neighbour_spatial: int = 201
    gM_slices: str = None
    annotation: str = None
    use_w: bool = False
    homolog_file: str = None
    species: str = None
    
    # GNN smoothing parameters
    use_gcn_smoothing: bool = False
    gcn_K: int = 1
    n_neighbors_gcn: int = 10
    zarr_group_path: str = None

    def __post_init__(self):
        super().__post_init__()
        verify_homolog_file_format(self)


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
class MaxPoolingConfig(ConfigWithAutoPaths):
    spe_file_list: str
    spatial_key: str = 'spatial'
    annotation: str = None
    sim_thresh: float = 0.5

    def __post_init__(self):
        super().__post_init__()


@dataclass
class ThreeDCombineConfig():
    workdir: str
    trait_name: str = None
    adata_3d: str = None
    project_name: str = None
    st_id: str = None
    annotation: str = None
    spatial_key: str = 'spatial'
    cmap: str = None
    point_size: float = 0.01
    background_color: str = 'white'
    n_snapshot: int = 200
    show_outline: bool = False
    save_mp4: bool = False
    save_gif: bool = False

    def __post_init__(self):
        if self.workdir is None:
            raise ValueError('workdir must be provided.')
        work_dir = Path(self.workdir)
        if self.project_name is not None:
            self.project_dir = work_dir / self.project_name
        else:
            self.project_dir = work_dir


@dataclass
class RunLinkModeConfig(ConfigWithAutoPaths):
    gsMap3D_resource_dir: str

    # == ST DATA PARAMETERS ==
    annotation: str = None
    spatial_key: str = 'spatial'

    # ==GWAS DATA PARAMETERS==
    trait_name: Optional[str] = None
    sumstats_file: Optional[str] = None
    sumstats_config_file: Optional[str] = None
    max_processes: int = 10
    use_pooling: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.gtffile = f"{
        self.gsMap3D_resource_dir}/genome_annotation/gtf/gencode.v46lift37.basic.annotation.gtf"
        self.bfile_root = f"{
        self.gsMap3D_resource_dir}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC"
        self.keep_snp_root = (
            f"{self.gsMap3D_resource_dir}/LDSC_resource/hapmap3_snps/hm"
        )
        self.w_file = (
            f"{self.gsMap3D_resource_dir}/LDSC_resource/weights_hm3_no_hla/weights."
        )
        self.snp_gene_weight_adata_path = (
            f"{self.gsMap3D_resource_dir}/quick_mode/snp_gene_weight_matrix.h5ad"
        )
        self.baseline_annotation_dir = Path(
            f"{self.gsMap3D_resource_dir}/quick_mode/baseline"
        ).resolve()
        self.SNP_gene_pair_dir = Path(
            f"{self.gsMap3D_resource_dir}/quick_mode/SNP_gene_pair"
        ).resolve()
        # check the existence of the input files and resources files
        for file in [self.gtffile]:
            if not Path(file).exists():
                raise FileNotFoundError(f"File {file} does not exist.")

        if self.sumstats_file is None and self.sumstats_config_file is None:
            raise ValueError(
                "One of sumstats_file and sumstats_config_file must be provided."
            )
        if self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError(
                "Only one of sumstats_file and sumstats_config_file must be provided."
            )
        if self.sumstats_file is not None and self.trait_name is None:
            raise ValueError(
                "trait_name must be provided if sumstats_file is provided."
            )
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
            for trait_name, sumstats_file in config.items():
                assert Path(sumstats_file).exists(
                ), f"{sumstats_file} does not exist."
                self.sumstats_config_dict[trait_name] = sumstats_file
        # load the sumstats file
        elif self.sumstats_file is not None and self.trait_name is not None:
            self.sumstats_config_dict[self.trait_name] = self.sumstats_file
        else:
            raise ValueError(
                "One of sumstats_file and sumstats_config_file must be provided."
            )

        for sumstats_file in self.sumstats_config_dict.values():
            assert Path(sumstats_file).exists(
            ), f"{sumstats_file} does not exist."


@dataclass
class GenerateLDScoreConfig(ConfigWithAutoPaths):
    chrom: int | str

    bfile_root: str

    # annotation by gene distance
    gtf_annotation_file: str
    gene_window_size: int = 50000
    keep_snp_root: str | None = None

    # annotation by enhancer
    enhancer_annotation_file: str = None
    snp_multiple_enhancer_strategy: Literal["max_mkscore", "nearest_TSS"] = "max_mkscore"
    gene_window_enhancer_priority: (
        Literal["gene_window_first", "enhancer_first", "enhancer_only"] | None
    ) = None

    # for calculating ld score
    additional_baseline_annotation: str = None
    spots_per_chunk: int = 1_000
    ld_wind: int = 1
    ld_unit: str = "CM"

    ldscore_save_format: Literal["feather", "quick_mode"] = "feather"

    # for pre calculating the SNP Gene ldscore Weight
    save_pre_calculate_snp_gene_weight_matrix: bool = False

    baseline_annotation_dir: str | None = None
    SNP_gene_pair_dir: str | None = None

    def __post_init__(self):
        # if self.mkscore_feather_file is None:
        #     self.mkscore_feather_file = self._get_mkscore_feather_path()

        if (
            self.enhancer_annotation_file is not None
            and self.gene_window_enhancer_priority is None
        ):
            logger.warning(
                "enhancer_annotation_file is provided but gene_window_enhancer_priority is not provided. "
                "by default, gene_window_enhancer_priority is set to 'enhancer_only', when enhancer_annotation_file is provided."
            )
            self.gene_window_enhancer_priority = "enhancer_only"
        if (
            self.enhancer_annotation_file is None
            and self.gene_window_enhancer_priority is not None
        ):
            logger.warning(
                "gene_window_enhancer_priority is provided but enhancer_annotation_file is not provided. "
                "by default, gene_window_enhancer_priority is set to None, when enhancer_annotation_file is not provided."
            )
            self.gene_window_enhancer_priority = None
        assert self.gene_window_enhancer_priority in [
            None,
            "gene_window_first",
            "enhancer_first",
            "enhancer_only",
        ], (
            f"gene_window_enhancer_priority must be one of None, 'gene_window_first', 'enhancer_first', 'enhancer_only', but got {self.gene_window_enhancer_priority}."
        )
        if self.gene_window_enhancer_priority in ["gene_window_first", "enhancer_first"]:
            logger.info(
                "Both gene_window and enhancer annotation will be used to calculate LD score. "
            )
            logger.info(
                f"SNP within +-{self.gene_window_size} bp of gene body will be used and enhancer annotation will be used to calculate LD score. If a snp maps to multiple enhancers, the strategy to choose by your select strategy: {self.snp_multiple_enhancer_strategy}."
            )
        elif self.gene_window_enhancer_priority == "enhancer_only":
            logger.info("Only enhancer annotation will be used to calculate LD score. ")
        else:
            logger.info(
                f"Only gene window annotation will be used to calculate LD score. SNP within +-{self.gene_window_size} bp of gene body will be used. "
            )

        # remind for baseline annotation
        if self.additional_baseline_annotation is None:
            logger.info(
                "------Baseline annotation is not provided. Default baseline annotation will be used."
            )
        else:
            logger.info(
                "------Baseline annotation is provided. Additional baseline annotation will be used with the default baseline annotation."
            )
            logger.info(
                f"------Baseline annotation directory: {self.additional_baseline_annotation}"
            )
            # check the existence of baseline annotation
            if self.chrom == "all":
                for chrom in range(1, 23):
                    chrom = str(chrom)
                    baseline_annotation_path = (
                        Path(self.additional_baseline_annotation) / f"baseline.{chrom}.annot.gz"
                    )
                    if not baseline_annotation_path.exists():
                        raise FileNotFoundError(
                            f"baseline.{chrom}.annot.gz is not found in {self.additional_baseline_annotation}."
                        )
            else:
                baseline_annotation_path = (
                    Path(self.additional_baseline_annotation) / f"baseline.{self.chrom}.annot.gz"
                )
                if not baseline_annotation_path.exists():
                    raise FileNotFoundError(
                        f"baseline.{self.chrom}.annot.gz is not found in {self.additional_baseline_annotation}."
                    )


@dataclass
class SpatialLDSCConfig(ConfigWithAutoPaths):
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
    all_chunk: int | None = None
    chunk_range: tuple[int, int] | None = None

    ldscore_save_format: Literal["feather", "quick_mode"] = "feather"

    spots_per_chunk_quick_mode: int = 1_000
    snp_gene_weight_adata_path: str | None = None
    
    # JAX acceleration option
    use_jax: bool = True  # Use JAX-accelerated implementation by default

    def __post_init__(self):
        super().__post_init__()
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
        # load the sumstats file
        elif self.sumstats_file is not None:
            self.sumstats_config_dict[self.trait_name] = self.sumstats_file
        else:
            raise ValueError("One of sumstats_file and sumstats_config_file must be provided.")

        for sumstats_file in self.sumstats_config_dict.values():
            assert Path(sumstats_file).exists(), f"{sumstats_file} does not exist."

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
        
        # Log the implementation choice
        if self.use_jax:
            logger.info("Using JAX-accelerated spatial LDSC implementation")
        else:
            logger.info("Using standard spatial LDSC implementation")

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
class CauchyCombinationConfig(ConfigWithAutoPaths):
    trait_name: str
    annotation: str
    sample_name_list: list[str] = dataclasses.field(default_factory=list)
    output_file: str | Path | None = None

    def __post_init__(self):
        if self.sample_name is not None:
            if self.sample_name_list and len(self.sample_name_list) > 0:
                raise ValueError("Only one of sample_name and sample_name_list must be provided.")
            else:
                self.sample_name_list = [self.sample_name]
                self.output_file = (
                    self.get_cauchy_result_file(self.trait_name)
                    if self.output_file is None
                    else self.output_file
                )
        else:
            assert len(self.sample_name_list) > 0, "At least one sample name must be provided."
            assert self.output_file is not None, (
                "Output_file must be provided if sample_name_list is provided."
            )


@dataclass
class VisualizeConfig(ConfigWithAutoPaths):
    trait_name: str

    annotation: str = None
    fig_title: str = None
    fig_height: int = 600
    fig_width: int = 800
    point_size: int = None
    fig_style: Literal["dark", "light"] = "light"


@dataclass
class DiagnosisConfig(ConfigWithAutoPaths):
    annotation: str
    # mkscore_feather_file: str

    trait_name: str
    sumstats_file: str
    plot_type: Literal["manhattan", "GSS", "gsMap", "all"] = "all"
    top_corr_genes: int = 50
    selected_genes: list[str] | None = None

    fig_width: int | None = None
    fig_height: int | None = None
    point_size: int | None = None
    fig_style: Literal["dark", "light"] = "light"

    def __post_init__(self):
        if any([self.fig_width, self.fig_height, self.point_size]):
            logger.info("Customizing the figure size and point size.")
            assert all([self.fig_width, self.fig_height, self.point_size]), (
                "All of fig_width, fig_height, and point_size must be provided."
            )
            self.customize_fig = True
        else:
            self.customize_fig = False


@dataclass
class ReportConfig(DiagnosisConfig):
    pass


@dataclass
class RunAllModeConfig(ConfigWithAutoPaths):
    gsMap_resource_dir: str

    # == ST DATA PARAMETERS ==
    hdf5_path: str
    annotation: str
    data_layer: str = "X"

    # == Find Latent Representation PARAMETERS ==
    n_comps: int = 300
    pearson_residuals: bool = False

    # == latent 2 Gene PARAMETERS ==
    gM_slices: str | None = None
    latent_representation: str = None
    num_neighbour: int = 21
    num_neighbour_spatial: int = 101

    # ==GWAS DATA PARAMETERS==
    trait_name: str | None = None
    sumstats_file: str | None = None
    sumstats_config_file: str | None = None

    # === homolog PARAMETERS ===
    homolog_file: str | None = None

    max_processes: int = 10
    
    # === spatial LDSC implementation ===
    use_jax: bool = True  # Use JAX-accelerated implementation by default

    def __post_init__(self):
        super().__post_init__()
        self.gtffile = f"{self.gsMap_resource_dir}/genome_annotation/gtf/gencode.v46lift37.basic.annotation.gtf"
        self.bfile_root = (
            f"{self.gsMap_resource_dir}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC"
        )
        self.keep_snp_root = f"{self.gsMap_resource_dir}/LDSC_resource/hapmap3_snps/hm"
        self.w_file = f"{self.gsMap_resource_dir}/LDSC_resource/weights_hm3_no_hla/weights."
        self.snp_gene_weight_adata_path = (
            f"{self.gsMap_resource_dir}/quick_mode/snp_gene_weight_matrix.h5ad"
        )
        self.baseline_annotation_dir = Path(
            f"{self.gsMap_resource_dir}/quick_mode/baseline"
        ).resolve()
        self.SNP_gene_pair_dir = Path(
            f"{self.gsMap_resource_dir}/quick_mode/SNP_gene_pair"
        ).resolve()
        # check the existence of the input files and resources files
        for file in [self.hdf5_path, self.gtffile]:
            if not Path(file).exists():
                raise FileNotFoundError(f"File {file} does not exist.")

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
            for trait_name, sumstats_file in config.items():
                assert Path(sumstats_file).exists(), f"{sumstats_file} does not exist."
                self.sumstats_config_dict[trait_name] = sumstats_file
        # load the sumstats file
        elif self.sumstats_file is not None and self.trait_name is not None:
            self.sumstats_config_dict[self.trait_name] = self.sumstats_file
        else:
            raise ValueError("One of sumstats_file and sumstats_config_file must be provided.")

        for sumstats_file in self.sumstats_config_dict.values():
            assert Path(sumstats_file).exists(), f"{sumstats_file} does not exist."


@dataclass
class FormatSumstatsConfig:
    sumstats: str
    out: str
    dbsnp: str
    snp: str = None
    a1: str = None
    a2: str = None
    info: str = None
    beta: str = None
    se: str = None
    p: str = None
    frq: str = None
    n: str | int = None
    z: str = None
    OR: str = None
    se_OR: str = None
    format: str = None
    chr: str = None
    pos: str = None
    chunksize: int = 1e7
    info_min: float = 0.9
    maf_min: float = 0.01
    keep_chr_pos: bool = False


@register_cli(
    name="quick_mode",
    description="Run the entire gsMap pipeline in quick mode, utilizing pre-computed weights for faster execution.",
    add_args_function=add_run_all_mode_args,
)
def run_all_mode_from_cli(args: argparse.Namespace):
    from gsMap.run_all_mode import run_pipeline

    config = get_dataclass_from_parser(args, RunAllModeConfig)
    run_pipeline(config)


@register_cli(
    name="run_find_latent_representations",
    description="Run Find_latent_representations \nFind the latent representations of each spot by running GNN",
    add_args_function=add_find_latent_representations_args,
)
def run_find_latent_representation_from_cli(args: argparse.Namespace):
    # Use the GNN implementation by default (gsMap version)
    from gsMap.find_latent_representation import run_find_latent_representation

    config = get_dataclass_from_parser(args, FindLatentRepresentationsConfig)
    
    # Ensure at least one input is provided
    if not config.input_hdf5_path and not config.spe_file_list:
        raise ValueError("Either --input_hdf5_path or --spe_file_list must be provided")
    
    run_find_latent_representation(config)


@register_cli(
    name="run_latent_to_gene",
    description="Run Latent_to_gene \nEstimate gene marker scores for each spot by using latent representations from nearby spots",
    add_args_function=add_latent_to_gene_args,
)
def run_latent_to_gene_from_cli(args: argparse.Namespace):
    from gsMap3D.latent_to_gene import run_latent_to_gene

    config = get_dataclass_from_parser(args, LatentToGeneConfig)
    run_latent_to_gene(config)


@register_cli(
    name="run_max_pooling",
    description="Run Max_pooling \nAdjust gene marker scores for each spot by using max pooling",
    add_args_function=add_max_pooling_args,
)
def run_gene_padding_from_cli(args: argparse.Namespace):
    from gsMap3D.max_pooling import run_max_pooling

    config = get_dataclass_from_parser(args, MaxPoolingConfig)
    run_max_pooling(config)


@register_cli(
    name="run_generate_ldscore",
    description="Run Generate_ldscore \nGenerate LD scores for each spot",
    add_args_function=add_generate_ldscore_args,
)
def run_generate_ldscore_from_cli(args: argparse.Namespace):
    from gsMap.generate_ldscore import run_generate_ldscore

    config = get_dataclass_from_parser(args, GenerateLDScoreConfig)
    run_generate_ldscore(config)


@register_cli(
    name="run_spatial_ldsc",
    description="Run Spatial_ldsc \nRun spatial LDSC for each spot",
    add_args_function=add_spatial_ldsc_args,
)
def run_spatial_ldsc_from_cli(args: argparse.Namespace):
    from gsMap.spatial_ldsc_multiple_sumstats import run_spatial_ldsc
    from gsMap.spatial_ldsc_jax_final import run_spatial_ldsc_jax

    config = get_dataclass_from_parser(args, SpatialLDSCConfig)
    
    # Dispatch to appropriate implementation based on use_jax flag
    if config.use_jax:
        logger.info("Using JAX-accelerated spatial LDSC implementation")
        run_spatial_ldsc_jax(config)
    else:
        logger.info("Using standard spatial LDSC implementation")
        run_spatial_ldsc(config)


@register_cli(
    name="run_cauchy_combination",
    description="Run Cauchy_combination for each annotation",
    add_args_function=add_Cauchy_combination_args,
)
def run_Cauchy_combination_from_cli(args: argparse.Namespace):
    from gsMap.cauchy_combination_test import run_Cauchy_combination

    config = get_dataclass_from_parser(args, CauchyCombinationConfig)
    run_Cauchy_combination(config)


@register_cli(
    name="run_report",
    description="Run Report to generate diagnostic plots and tables",
    add_args_function=add_report_args,
)
def run_Report_from_cli(args: argparse.Namespace):
    from gsMap.report import run_report

    config = get_dataclass_from_parser(args, ReportConfig)
    run_report(config)


@register_cli(
    name="format_sumstats",
    description="Format GWAS summary statistics",
    add_args_function=add_format_sumstats_args,
)
def gwas_format_from_cli(args: argparse.Namespace):
    from gsMap.format_sumstats import gwas_format

    config = get_dataclass_from_parser(args, FormatSumstatsConfig)
    gwas_format(config)


@register_cli(
    name="create_slice_mean",
    description="Create slice mean from multiple h5ad files",
    add_args_function=add_create_slice_mean_args,
)
def create_slice_mean_from_cli(args: argparse.Namespace):
    from gsMap.create_slice_mean import run_create_slice_mean

    config = get_dataclass_from_parser(args, CreateSliceMeanConfig)
    run_create_slice_mean(config)
