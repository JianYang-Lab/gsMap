import dataclasses
import sys
import argparse
import logging
from collections import OrderedDict, namedtuple
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Callable
from typing import Union, Literal, Tuple, Optional, List
from functools import wraps
import pyfiglet
import yaml

from gsMap.__init__ import __version__

# Global registry to hold functions
cli_function_registry = OrderedDict()
subcommand = namedtuple('subcommand', ['name', 'func', 'add_args_function', 'description'])


def get_gsMap_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '[{asctime}] {levelname:.5s} | {name} - {message}', style='{'))
    logger.addHandler(handler)
    return logger

logger = get_gsMap_logger('gsMap')

# Decorator to register functions for cli parsing
def register_cli(name: str, description: str, add_args_function: Callable) -> Callable:
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            name.replace('_', ' ')
            gsMap_main_logo = pyfiglet.figlet_format("gsMap", font='doom', width=80, justify='center', ).rstrip()
            print(gsMap_main_logo, flush=True)
            version_number = 'Version: ' + __version__
            print(version_number.center(80), flush=True)
            print('=' * 80, flush=True)
            logger.info(f"Running {name}...")
            func(*args, **kwargs)
            logger.info(f"Finished running {name}.")

        cli_function_registry[name] = subcommand(name=name, func=wrapper, add_args_function=add_args_function,
                                                 description=description)
        return wrapper

    return decorator

def add_shared_args(parser):
    parser.add_argument('--workdir', type=str, required=True, help='Path to the working directory.')
    parser.add_argument('--sample_name', type=str, required=True, help='Name of the sample.')

def add_find_latent_representations_args(parser):
    add_shared_args(parser)
    parser.add_argument('--input_hdf5_path', required=True, type=str, help='Path to the input HDF5 file.')
    parser.add_argument('--annotation', required=True, type=str, help='Name of the annotation in adata.obs to use.')
    parser.add_argument('--data_layer', type=str, default='counts', required=True,
                        help='Data layer for gene expression (e.g., "count", "counts", "log1p").')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--feat_hidden1', type=int, default=256, help='Neurons in the first hidden layer.')
    parser.add_argument('--feat_hidden2', type=int, default=128, help='Neurons in the second hidden layer.')
    parser.add_argument('--gat_hidden1', type=int, default=64, help='Units in the first GAT hidden layer.')
    parser.add_argument('--gat_hidden2', type=int, default=30, help='Units in the second GAT hidden layer.')
    parser.add_argument('--p_drop', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--gat_lr', type=float, default=0.001, help='Learning rate for the GAT.')
    parser.add_argument('--n_neighbors', type=int, default=11, help='Number of neighbors for GAT.')
    parser.add_argument('--n_comps', type=int, default=300, help='Number of principal components for PCA.')
    parser.add_argument('--weighted_adj', action='store_true', help='Use weighted adjacency in GAT.')
    parser.add_argument('--convergence_threshold', type=float, default=1e-4, help='Threshold for convergence.')
    parser.add_argument('--hierarchically', action='store_true', help='Enable hierarchical latent representation finding.')


def chrom_choice(value):
    if value.isdigit():
        ivalue = int(value)
        if 1 <= ivalue <= 22:
            return ivalue
    elif value.lower() == 'all':
        return value
    else:
        raise argparse.ArgumentTypeError(f"'{value}' is an invalid chromosome choice. Choose from 1-22 or 'all'.")


def filter_args_for_dataclass(args_dict, data_class: dataclass):
    return {k: v for k, v in args_dict.items() if k in data_class.__dataclass_fields__}


def get_dataclass_from_parser(args: argparse.Namespace, data_class: dataclass):
    remain_kwargs = filter_args_for_dataclass(vars(args), data_class)
    print(f'Using the following arguments for {data_class.__name__}:', flush=True)
    pprint(remain_kwargs, indent=4)
    sys.stdout.flush()
    return data_class(**remain_kwargs)


def add_latent_to_gene_args(parser):

    add_shared_args(parser)

    parser.add_argument(
        '--input_hdf5_path', type=str, default=None,
        help='Path to the input HDF5 file with latent representations, if --latent_representation is specified.'
    )
    parser.add_argument(
        '--no_expression_fraction', action='store_true',
        help='Skip expression fraction filtering.'
    )
    parser.add_argument(
        '--latent_representation', type=str, default=None,
        help='Type of latent representation. This should exist in the h5ad obsm.'
    )
    parser.add_argument(
        '--num_neighbour', type=int, default=21,
        help='Number of neighbors.'
    )
    parser.add_argument(
        '--num_neighbour_spatial', type=int, default=101,
        help='Number of spatial neighbors.'
    )
    parser.add_argument(
        '--homolog_file', type=str, default=None,
        help='Path to homologous gene conversion file (optional).'
    )
    parser.add_argument(
        '--gM_slices', type=str, default=None,
        help='Path to the slice mean file (optional).'
    )
    parser.add_argument(
        '--annotation', type=str, default=None,
        help='Name of the annotation in adata.obs to use (optional).'
    )


def add_generate_ldscore_args(parser):
    add_shared_args(parser)
    parser.add_argument('--chrom', type=str, required=True, help='Chromosome id (1-22) or "all".')
    parser.add_argument('--bfile_root', type=str, required=True, help='Root path for genotype plink bfiles (.bim, .bed, .fam).')
    parser.add_argument('--keep_snp_root', type=str, required=True, help='Root path for SNP files.')
    parser.add_argument('--gtf_annotation_file', type=str, required=True, help='Path to GTF annotation file.')
    parser.add_argument('--gene_window_size', type=int, default=50000, help='Gene window size in base pairs.')
    parser.add_argument('--enhancer_annotation_file', type=str, help='Path to enhancer annotation file (optional).')
    parser.add_argument('--snp_multiple_enhancer_strategy', type=str, choices=['max_mkscore', 'nearest_TSS'], default='max_mkscore',
                        help='Strategy for handling multiple enhancers per SNP.')
    parser.add_argument('--gene_window_enhancer_priority', type=str, choices=['gene_window_first', 'enhancer_first', 'enhancer_only'],
                        help='Priority between gene window and enhancer annotations.')
    parser.add_argument('--spots_per_chunk', type=int, default=1000, help='Number of spots per chunk.')
    parser.add_argument('--ld_wind', type=int, default=1, help='LD window size.')
    parser.add_argument('--ld_unit', type=str, choices=['SNP', 'KB', 'CM'], default='CM', help='Unit for LD window.')
    parser.add_argument('--additional_baseline_annotation', type=str, default=None, help='Path of additional baseline annotations')

def add_spatial_ldsc_args(parser):
    add_shared_args(parser)
    parser.add_argument('--sumstats_file', type=str, required=True, help='Path to GWAS summary statistics file.')
    parser.add_argument('--w_file', type=str, required=True, help='Path to regression weight file.')
    parser.add_argument('--trait_name', type=str, required=True, help='Name of the trait being analyzed.')
    parser.add_argument('--n_blocks', type=int, default=200, help='Number of blocks for jackknife resampling.')
    parser.add_argument('--chisq_max', type=int, help='Maximum chi-square value for filtering SNPs.')
    parser.add_argument('--num_processes', type=int, default=4, help='Number of processes for parallel computing.')
    parser.add_argument('--use_additional_baseline_annotation', type=bool, nargs='?', const=True, default=True, help='Use additional baseline annotations when provided')


def add_Cauchy_combination_args(parser):

    parser.add_argument('--workdir', type=str, required=True, help='Path to the working directory.')
    parser.add_argument('--sample_name', type=str, required=False, help='Name of the sample.')

    parser.add_argument('--trait_name', type=str, required=True, help='Name of the trait being analyzed.')
    parser.add_argument('--annotation', type=str, required=True, help='Name of the annotation in adata.obs to use.')

    parser.add_argument('--sample_name_list', type=str, nargs='+', required=False,
                        help='List of sample names to process. Provide as a space-separated list.')
    parser.add_argument('--output_file', type=str, required=False,
                        help='Path to save the combined Cauchy results. Required when using multiple samples.')

def add_report_args(parser):
    add_shared_args(parser)
    parser.add_argument('--trait_name', type=str, required=True, help='Name of the trait to generate the report for.')
    parser.add_argument('--annotation', type=str, required=True, help='Annotation layer name.')
    # parser.add_argument('--plot_type', type=str, choices=['manhattan', 'GSS', 'gsMap', 'all'], default='all',
    #                     help="Type of diagnostic plot to generate. Choose from 'manhattan', 'GSS', 'gsMap', or 'all'.")
    parser.add_argument('--top_corr_genes', type=int, default=50,
                        help='Number of top correlated genes to display.')
    parser.add_argument('--selected_genes', type=str, nargs='*',
                        help='List of specific genes to include in the report (optional).')
    parser.add_argument('--sumstats_file', type=str, required=True, help='Path to GWAS summary statistics file.')

    # Optional arguments for customization
    parser.add_argument('--fig_width', type=int, default=None, help='Width of the generated figures in pixels.')
    parser.add_argument('--fig_height', type=int, default=None, help='Height of the generated figures in pixels.')
    parser.add_argument('--point_size', type=int, default=None, help='Point size for the figures.')
    parser.add_argument('--fig_style', type=str, default='light', choices=['dark', 'light'],
                        help='Style of the generated figures.')

def add_create_slice_mean_args(parser):
    parser.add_argument('--sample_name_list', type=str, nargs='+', required=True,
                        help='List of sample names to process. Provide as a space-separated list.')

    parser.add_argument(
        '--h5ad_list', type=str, nargs='+',
        help="List of h5ad file paths corresponding to the sample names. Provide as a space-separated list."
    )
    parser.add_argument(
        '--h5ad_yaml', type=str, default=None,
        help="Path to the YAML file containing sample names and associated h5ad file paths"
    )
    parser.add_argument(
        '--slice_mean_output_file', type=str, required=True,
        help="Path to the output file for the slice mean"
    )
    parser.add_argument(
        '--homolog_file', type=str, 
        help='Path to homologous gene conversion file (optional).'
    )
    parser.add_argument(
        '--data_layer', type=str, default='counts', required=True,
        help='Data layer for gene expression (e.g., "count", "counts", "log1p").'
    )
    
def add_format_sumstats_args(parser):
    # Required arguments
    parser.add_argument('--sumstats', required=True, type=str,
                        help='Path to gwas summary data')
    parser.add_argument('--out', required=True, type=str,
                        help='Path to save the formatted gwas data')

    # Arguments for specify column name
    parser.add_argument('--snp', default=None, type=str,
                        help="Name of snp column (if not a name that gsMap understands)")
    parser.add_argument('--a1', default=None, type=str,
                        help="Name of effect allele column (if not a name that gsMap understands)")
    parser.add_argument('--a2', default=None, type=str,
                        help="Name of none-effect allele column (if not a name that gsMap understands)")
    parser.add_argument('--info', default=None, type=str,
                        help="Name of info column (if not a name that gsMap understands)")
    parser.add_argument('--beta', default=None, type=str,
                        help="Name of gwas beta column (if not a name that gsMap understands).")
    parser.add_argument('--se', default=None, type=str,
                        help="Name of gwas standar error of beta column (if not a name that gsMap understands)")
    parser.add_argument('--p', default=None, type=str,
                        help="Name of p-value column (if not a name that gsMap understands)")
    parser.add_argument('--frq', default=None, type=str,
                        help="Name of A1 ferquency column (if not a name that gsMap understands)")
    parser.add_argument('--n', default=None, type=str,
                        help="Name of sample size column (if not a name that gsMap understands)")
    parser.add_argument('--z', default=None, type=str,
                        help="Name of gwas Z-statistics column (if not a name that gsMap understands)")
    parser.add_argument('--OR', default=None, type=str,
                        help="Name of gwas OR column (if not a name that gsMap understands)")
    parser.add_argument('--se_OR', default=None, type=str,
                        help="Name of standar error of OR column (if not a name that gsMap understands)")

    # Arguments for convert SNP (chr, pos) to rsid
    parser.add_argument('--chr', default="Chr", type=str,
                        help="Name of SNP chromosome column (if not a name that gsMap understands)")
    parser.add_argument('--pos', default="Pos", type=str,
                        help="Name of SNP positions column (if not a name that gsMap understands)")
    parser.add_argument('--dbsnp', default=None, type=str,
                        help='Path to reference dnsnp file')
    parser.add_argument('--chunksize', default=1e+6, type=int,
                        help='Chunk size for loading dbsnp file')

    # Arguments for output format and quality
    parser.add_argument('--format', default='gsMap', type=str,
                        help='Format of output data', choices=['gsMap', 'COJO'])
    parser.add_argument('--info_min', default=0.9, type=float,
                        help='Minimum INFO score.')
    parser.add_argument('--maf_min', default=0.01, type=float,
                        help='Minimum MAF.')
    parser.add_argument('--keep_chr_pos', action='store_true', default=False,
                        help='Keep SNP chromosome and position columns in the output data')

def add_run_all_mode_args(parser):
    add_shared_args(parser)

    # Required paths and configurations
    parser.add_argument('--gsMap_resource_dir', type=str, required=True,
                        help='Directory containing gsMap resources (e.g., genome annotations, LD reference panel, etc.).')
    parser.add_argument('--hdf5_path', type=str, required=True,
                        help='Path to the input spatial transcriptomics data (H5AD format).')
    parser.add_argument('--annotation', type=str, required=True,
                        help='Name of the annotation in adata.obs to use.')
    parser.add_argument('--data_layer', type=str, default='counts', required=True,
                        help='Data layer for gene expression (e.g., "count", "counts", "log1p").')

    # GWAS Data Parameters
    parser.add_argument('--trait_name', type=str, help='Name of the trait for GWAS analysis (required if sumstats_file is provided).')
    parser.add_argument('--sumstats_file', type=str,
                        help='Path to GWAS summary statistics file. Either sumstats_file or sumstats_config_file is required.')
    parser.add_argument('--sumstats_config_file', type=str,
                        help='Path to GWAS summary statistics config file. Either sumstats_file or sumstats_config_file is required.')

    # Homolog Data Parameters
    parser.add_argument('--homolog_file', type=str,
                        help='Path to homologous gene for converting gene names from different species to human (optional, used for cross-species analysis).')

    # Maximum number of processes
    parser.add_argument('--max_processes', type=int, default=10,
                        help='Maximum number of processes for parallel execution.')


    parser.add_argument(
        '--latent_representation', type=str, default=None,
        help='Type of latent representation. This should exist in the h5ad obsm.'
    )
    parser.add_argument(
        '--num_neighbour', type=int, default=21,
        help='Number of neighbors.'
    )
    parser.add_argument(
        '--num_neighbour_spatial', type=int, default=101,
        help='Number of spatial neighbors.'
    )
    parser.add_argument(
        '--gM_slices', type=str, default=None,
        help='Path to the slice mean file (optional).'
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


@dataclass
class ConfigWithAutoPaths:
    workdir: str
    sample_name: str | None

    def __post_init__(self):
        if self.workdir is None:
            raise ValueError('workdir must be provided.')

    @property
    @ensure_path_exists
    def hdf5_with_latent_path(self) -> Path:
        return Path(f'{self.workdir}/{self.sample_name}/find_latent_representations/{self.sample_name}_add_latent.h5ad')

    @property
    @ensure_path_exists
    def mkscore_feather_path(self) -> Path:
        return Path(f'{self.workdir}/{self.sample_name}/latent_to_gene/{self.sample_name}_gene_marker_score.feather')

    @property
    @ensure_path_exists
    def ldscore_save_dir(self) -> Path:
        return Path(f'{self.workdir}/{self.sample_name}/generate_ldscore')

    @property
    @ensure_path_exists
    def ldsc_save_dir(self) -> Path:
        return Path(f'{self.workdir}/{self.sample_name}/spatial_ldsc')

    @property
    @ensure_path_exists
    def cauchy_save_dir(self) -> Path:
        return Path(f'{self.workdir}/{self.sample_name}/cauchy_combination')

    @ensure_path_exists
    def get_report_dir(self, trait_name: str) -> Path:
        return Path(f'{self.workdir}/{self.sample_name}/report/{trait_name}')

    def get_gsMap_report_file(self, trait_name: str) -> Path:
        return self.get_report_dir(trait_name) / f'{self.sample_name}_{trait_name}_gsMap_Report.html'

    @ensure_path_exists
    def get_manhattan_html_plot_path(self, trait_name: str) -> Path:
        return Path(
            f'{self.workdir}/{self.sample_name}/report/{trait_name}/manhattan_plot/{self.sample_name}_{trait_name}_Diagnostic_Manhattan_Plot.html')

    @ensure_path_exists
    def get_GSS_plot_dir(self, trait_name: str) -> Path:
        return Path(f'{self.workdir}/{self.sample_name}/report/{trait_name}/GSS_plot')

    def get_GSS_plot_select_gene_file(self, trait_name: str) -> Path:
        return self.get_GSS_plot_dir(trait_name) / 'plot_genes.csv'

    @ensure_path_exists
    def get_ldsc_result_file(self, trait_name: str) -> Path:
        return Path(f'{self.ldsc_save_dir}/{self.sample_name}_{trait_name}.csv.gz')

    @ensure_path_exists
    def get_cauchy_result_file(self, trait_name: str) -> Path:
        return Path(f'{self.cauchy_save_dir}/{self.sample_name}_{trait_name}.Cauchy.csv.gz')

    @ensure_path_exists
    def get_gene_diagnostic_info_save_path(self, trait_name: str) -> Path:
        return Path(
            f'{self.workdir}/{self.sample_name}/report/{trait_name}/{self.sample_name}_{trait_name}_Gene_Diagnostic_Info.csv')

    @ensure_path_exists
    def get_gsMap_plot_save_dir(self, trait_name: str) -> Path:
        return Path(f'{self.workdir}/{self.sample_name}/report/{trait_name}/gsMap_plot')

    def get_gsMap_html_plot_save_path(self, trait_name: str) -> Path:
        return self.get_gsMap_plot_save_dir(trait_name) / f'{self.sample_name}_{trait_name}_gsMap_plot.html'


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
            h5ad_dict = yaml.safe_load(open(self.h5ad_yaml)) if isinstance(self.h5ad_yaml, str) else self.h5ad_yaml
        elif self.sample_name_list and self.h5ad_list:
            logger.info(f"Reading sample name list and h5ad list")
            h5ad_dict = dict(zip(self.sample_name_list, self.h5ad_list))
        else:
            raise ValueError("Please provide either h5ad_yaml or both sample_name_list and h5ad_list.")

        # check if sample names is unique
        assert len(h5ad_dict) == len(set(h5ad_dict)), "Sample names must be unique."
        assert len(h5ad_dict) > 1, "At least two samples are required."

        logger.info(f'Input h5ad files: {h5ad_dict}')

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
    input_hdf5_path: str
    # output_hdf5_path: str
    annotation: str = None
    data_layer: str = None

    epochs: int = 300
    feat_hidden1: int = 256
    feat_hidden2: int = 128
    feat_cell: int = 3000
    gat_hidden1: int = 64
    gat_hidden2: int = 30
    p_drop: float = 0.1
    gat_lr: float = 0.001
    gcn_decay: float = 0.01
    n_neighbors: int = 11
    label_w: float = 1
    rec_w: float = 1
    input_pca: bool = True
    n_comps: int = 300
    weighted_adj: bool = False
    nheads: int = 3
    var: bool = False
    convergence_threshold: float = 1e-4
    hierarchically: bool = False

    def __post_init__(self):
        # self.output_hdf5_path = self.hdf5_with_latent_path
        if self.hierarchically:
            if self.annotation is None:
                raise ValueError('annotation must be provided if hierarchically is True.')
            logger.info(
                f'------Hierarchical mode is enabled. This will find the latent representations within each annotation.')

        # remind for not providing annotation
        if self.annotation is None:
            logger.warning(
                'annotation is not provided. This will find the latent representations for the whole dataset.')
        else:
            logger.info(f'------Find latent representations for {self.annotation}...')


@dataclass
class LatentToGeneConfig(ConfigWithAutoPaths):
    # input_hdf5_with_latent_path: str
    # output_feather_path: str
    input_hdf5_path: str | Path = None
    no_expression_fraction: bool = False
    latent_representation: str = None
    num_neighbour: int = 21
    num_neighbour_spatial: int = 101
    homolog_file: str = None
    gM_slices: str = None
    annotation: str = None
    species: str = None

    def __post_init__(self):
        if self.input_hdf5_path is None:
            self.input_hdf5_path = self.hdf5_with_latent_path
            assert self.input_hdf5_path.exists(), f"{self.input_hdf5_path} does not exist. Please run FindLatentRepresentations first."
        else:
            assert Path(self.input_hdf5_path).exists(), f"{self.input_hdf5_path} does not exist."
            # copy to self.hdf5_with_latent_path
            import shutil
            shutil.copy2(self.input_hdf5_path, self.hdf5_with_latent_path)

        if self.latent_representation is not None:
            logger.info(f"Using the provided latent representation: {self.latent_representation}")
        else:
            self.latent_representation = 'latent_GVAE'
            logger.info(f"Using default latent representation: {self.latent_representation}")

        if self.gM_slices is not None:
            assert Path(self.gM_slices).exists(), f"{self.gM_slices} does not exist."
            logger.info(f'Using the provided slice mean file: {self.gM_slices}.')

        verify_homolog_file_format(self)

def verify_homolog_file_format(config):

    if config.homolog_file is not None:
        logger.info(f"User provided homolog file to map gene names to human: {config.homolog_file}")
        # check the format of the homolog file
        with open(config.homolog_file, 'r') as f:
            first_line = f.readline().strip()
            _n_col = len(first_line.split())
            if _n_col != 2:
                raise ValueError(
                    f"Invalid homolog file format. Expected 2 columns, first column should be other species gene name, second column should be human gene name. "
                    f"Got {_n_col} columns in the first line.")
            else:
                first_col_name, second_col_name = first_line.split()
                config.species = first_col_name
                logger.info(
                    f"Homolog file provided and will map gene name from column1:{first_col_name} to column2:{second_col_name}")
    else:
        logger.info("No homolog file provided. Run in human mode.")


@dataclass
class GenerateLDScoreConfig(ConfigWithAutoPaths):
    chrom: Union[int, str]

    bfile_root: str
    keep_snp_root: Optional[str]

    # annotation by gene distance
    gtf_annotation_file: str
    gene_window_size: int = 50000

    # annotation by enhancer
    enhancer_annotation_file: str = None
    snp_multiple_enhancer_strategy: Literal['max_mkscore', 'nearest_TSS'] = 'max_mkscore'
    gene_window_enhancer_priority: Optional[Literal['gene_window_first', 'enhancer_first', 'enhancer_only',]] = None

    # for calculating ld score
    additional_baseline_annotation: str = None
    spots_per_chunk: int = 1_000
    ld_wind: int = 1
    ld_unit: str = 'CM'

    # zarr config
    ldscore_save_format: Literal['feather', 'zarr', 'quick_mode'] = 'feather'

    zarr_chunk_size: Tuple[int, int] = None

    # for pre calculating the SNP Gene ldscore Weight
    save_pre_calculate_snp_gene_weight_matrix: bool = False

    baseline_annotation_dir: Optional[str] = None
    SNP_gene_pair_dir: Optional[str] = None
    def __post_init__(self):
        # if self.mkscore_feather_file is None:
        #     self.mkscore_feather_file = self._get_mkscore_feather_path()

        if self.enhancer_annotation_file is not None and self.gene_window_enhancer_priority is None:
            logger.warning("enhancer_annotation_file is provided but gene_window_enhancer_priority is not provided. "
                           "by default, gene_window_enhancer_priority is set to 'enhancer_only', when enhancer_annotation_file is provided.")
            self.gene_window_enhancer_priority = 'enhancer_only'
        if self.enhancer_annotation_file is None and self.gene_window_enhancer_priority is not None:
            logger.warning("gene_window_enhancer_priority is provided but enhancer_annotation_file is not provided. "
                           "by default, gene_window_enhancer_priority is set to None, when enhancer_annotation_file is not provided.")
            self.gene_window_enhancer_priority = None
        assert self.gene_window_enhancer_priority in [None, 'gene_window_first', 'enhancer_first', 'enhancer_only', ], \
            f"gene_window_enhancer_priority must be one of None, 'gene_window_first', 'enhancer_first', 'enhancer_only', but got {self.gene_window_enhancer_priority}."
        if self.gene_window_enhancer_priority in ['gene_window_first', 'enhancer_first']:
            logger.info(f'Both gene_window and enhancer annotation will be used to calculate LD score. ')
            logger.info(
                f'SNP within +-{self.gene_window_size} bp of gene body will be used and enhancer annotation will be used to calculate LD score. If a snp maps to multiple enhancers, the strategy to choose by your select strategy: {self.snp_multiple_enhancer_strategy}.')
        elif self.gene_window_enhancer_priority == 'enhancer_only':
            logger.info(f'Only enhancer annotation will be used to calculate LD score. ')
        else:
            logger.info(
                f'Only gene window annotation will be used to calculate LD score. SNP within +-{self.gene_window_size} bp of gene body will be used. ')

        # remind for baseline annotation
        if self.additional_baseline_annotation is None:
            logger.info(f'------Baseline annotation is not provided. Default baseline annotation will be used.')
        else:
            logger.info(
                f'------Baseline annotation is provided. Additional baseline annotation will be used with the default baseline annotation.')
            logger.info(f'------Baseline annotation directory: {self.additional_baseline_annotation}')
            # check the existence of baseline annotation
            if self.chrom == 'all':
                for chrom in range(1, 23):
                    chrom = str(chrom)
                    baseline_annotation_path = Path(
                        self.additional_baseline_annotation) / f'baseline.{chrom}.annot.gz'
                    if not baseline_annotation_path.exists():
                        raise FileNotFoundError(
                            f'baseline.{chrom}.annot.gz is not found in {self.additional_baseline_annotation}.')
            else:
                baseline_annotation_path = Path(
                    self.additional_baseline_annotation) / f'baseline.{self.chrom}.annot.gz'
                if not baseline_annotation_path.exists():
                    raise FileNotFoundError(
                        f'baseline.{self.chrom}.annot.gz is not found in {self.additional_baseline_annotation}.')

        # set the default zarr chunk size
        if self.ldscore_save_format == 'zarr' and self.zarr_chunk_size is None:
            self.zarr_chunk_size = (10_000, self.spots_per_chunk)


@dataclass
class SpatialLDSCConfig(ConfigWithAutoPaths):
    w_file: str
    # ldscore_save_dir: str
    use_additional_baseline_annotation: bool = True
    trait_name: Optional[str] = None
    sumstats_file: Optional[str] = None
    sumstats_config_file: Optional[str] = None
    num_processes: int = 4
    not_M_5_50: bool = False
    n_blocks: int = 200
    chisq_max: Optional[int] = None
    all_chunk: Optional[int] = None
    chunk_range: Optional[Tuple[int, int]] = None

    ldscore_save_format: Literal['feather', 'zarr', 'quick_mode'] = 'feather'

    spots_per_chunk_quick_mode: int = 1_000
    snp_gene_weight_adata_path: Optional[str] = None

    def __post_init__(self):
        super().__post_init__()
        if self.sumstats_file is None and self.sumstats_config_file is None:
            raise ValueError('One of sumstats_file and sumstats_config_file must be provided.')
        if self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError('Only one of sumstats_file and sumstats_config_file must be provided.')
        if self.sumstats_file is not None and self.trait_name is None:
            raise ValueError('trait_name must be provided if sumstats_file is provided.')
        if self.sumstats_config_file is not None and self.trait_name is not None:
            raise ValueError('trait_name must not be provided if sumstats_config_file is provided.')
        self.sumstats_config_dict = {}
        # load the sumstats config file
        if self.sumstats_config_file is not None:
            import yaml
            with open(self.sumstats_config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for trait_name, sumstats_file in config.items():
                assert Path(sumstats_file).exists(), f'{sumstats_file} does not exist.'
        # load the sumstats file
        elif self.sumstats_file is not None:
            self.sumstats_config_dict[self.trait_name] = self.sumstats_file
        else:
            raise ValueError('One of sumstats_file and sumstats_config_file must be provided.')

        for sumstats_file in self.sumstats_config_dict.values():
            assert Path(sumstats_file).exists(), f'{sumstats_file} does not exist.'

        # check if additional baseline annotation is exist
        # self.use_additional_baseline_annotation = False
        
        if self.use_additional_baseline_annotation:
            self.process_additional_baseline_annotation()

    def process_additional_baseline_annotation(self):
        additional_baseline_annotation = Path(self.ldscore_save_dir) / 'additional_baseline'
        dir_exists = additional_baseline_annotation.exists()

        if not dir_exists:
            self.use_additional_baseline_annotation = False
            # if self.use_additional_baseline_annotation:
            #     logger.warning(f"additional_baseline directory is not found in {self.ldscore_save_dir}.")
            #     print('''\
            #         if you want to use additional baseline annotation, 
            #         please provide additional baseline annotation when calculating ld score.
            #         ''')
            #     raise FileNotFoundError(
            #         f'additional_baseline directory is not found.')
            # return
            # self.use_additional_baseline_annotation = self.use_additional_baseline_annotation or True
        else:
            logger.info(
                f'------Additional baseline annotation is provided. It will be used with the default baseline annotation.')
            logger.info(f'------Additional baseline annotation directory: {additional_baseline_annotation}')

            chrom_list = range(1, 23)
            for chrom in chrom_list:
                baseline_annotation_path = additional_baseline_annotation / f'baseline.{chrom}.l2.ldscore.feather'
                if not baseline_annotation_path.exists():
                    raise FileNotFoundError(
                        f'baseline.{chrom}.annot.gz is not found in {additional_baseline_annotation}.')
        return None


@dataclass
class CauchyCombinationConfig(ConfigWithAutoPaths):
    trait_name: str
    annotation: str
    sample_name_list: List[str] = dataclasses.field(default_factory=list)
    output_file: str | Path | None = None

    def __post_init__(self):
        if self.sample_name is not None:
            if len(self.sample_name_list) > 0:
                raise ValueError('Only one of sample_name and sample_name_list must be provided.')
            else:
                self.sample_name_list = [self.sample_name]
                self.output_file = self.get_cauchy_result_file(self.trait_name) if self.output_file is None else self.output_file
        else:
            assert len(self.sample_name_list) > 0, 'At least one sample name must be provided.'
            assert self.output_file is not None, 'Output_file must be provided if sample_name_list is provided.'


@dataclass
class VisualizeConfig(ConfigWithAutoPaths):
    trait_name: str

    annotation: str = None
    fig_title: str = None
    fig_height: int = 600
    fig_width: int = 800
    point_size: int = None
    fig_style: Literal['dark', 'light'] = 'light'


@dataclass
class DiagnosisConfig(ConfigWithAutoPaths):
    annotation: str
    # mkscore_feather_file: str

    trait_name: str
    sumstats_file: str
    plot_type: Literal['manhattan', 'GSS', 'gsMap', 'all'] = 'all'
    top_corr_genes: int = 50
    selected_genes: Optional[List[str]] = None

    fig_width: Optional[int] = None
    fig_height: Optional[int] = None
    point_size: Optional[int] = None
    fig_style: Literal['dark', 'light'] = 'light'

    def __post_init__(self):
        if any([self.fig_width, self.fig_height, self.point_size]):
            logger.info('Customizing the figure size and point size.')
            assert all([self.fig_width, self.fig_height, self.point_size]), 'All of fig_width, fig_height, and point_size must be provided.'
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
    data_layer: str = 'X'


    # == latent 2 Gene PARAMETERS ==
    gM_slices: str = None
    latent_representation: str = None
    num_neighbour: int = 21
    num_neighbour_spatial: int = 101

    # ==GWAS DATA PARAMETERS==
    trait_name: Optional[str] = None
    sumstats_file: Optional[str] = None
    sumstats_config_file: Optional[str] = None

    # === homolog PARAMETERS ===
    homolog_file: Optional[str] = None

    max_processes: int = 10

    def __post_init__(self):
        super().__post_init__()
        self.gtffile = f"{self.gsMap_resource_dir}/genome_annotation/gtf/gencode.v39lift37.annotation.gtf"
        self.bfile_root = f"{self.gsMap_resource_dir}/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC"
        self.keep_snp_root = f"{self.gsMap_resource_dir}/LDSC_resource/hapmap3_snps/hm"
        self.w_file = f"{self.gsMap_resource_dir}/LDSC_resource/weights_hm3_no_hla/weights."
        self.snp_gene_weight_adata_path = f"{self.gsMap_resource_dir}/quick_mode/snp_gene_weight_matrix.h5ad"
        self.baseline_annotation_dir = Path(f"{self.gsMap_resource_dir}/quick_mode/baseline").resolve()
        self.SNP_gene_pair_dir = Path(f"{self.gsMap_resource_dir}/quick_mode/SNP_gene_pair").resolve()
        # check the existence of the input files and resources files
        for file in [self.hdf5_path, self.gtffile]:
            if not Path(file).exists():
                raise FileNotFoundError(f"File {file} does not exist.")

        if self.sumstats_file is None and self.sumstats_config_file is None:
            raise ValueError('One of sumstats_file and sumstats_config_file must be provided.')
        if self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError('Only one of sumstats_file and sumstats_config_file must be provided.')
        if self.sumstats_file is not None and self.trait_name is None:
            raise ValueError('trait_name must be provided if sumstats_file is provided.')
        if self.sumstats_config_file is not None and self.trait_name is not None:
            raise ValueError('trait_name must not be provided if sumstats_config_file is provided.')
        self.sumstats_config_dict = {}
        # load the sumstats config file
        if self.sumstats_config_file is not None:
            import yaml
            with open(self.sumstats_config_file) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            for trait_name, sumstats_file in config.items():
                assert Path(sumstats_file).exists(), f'{sumstats_file} does not exist.'
                self.sumstats_config_dict[trait_name] = sumstats_file
        # load the sumstats file
        elif self.sumstats_file is not None and self.trait_name is not None:
            self.sumstats_config_dict[self.trait_name] = self.sumstats_file
        else:
            raise ValueError('One of sumstats_file and sumstats_config_file must be provided.')

        for sumstats_file in self.sumstats_config_dict.values():
            assert Path(sumstats_file).exists(), f'{sumstats_file} does not exist.'



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
    n: str = None
    z: str = None
    OR: str = None
    se_OR: str = None
    format: str = None
    chr: str = None
    pos: str = None
    chunksize: int = 1e+7
    info_min: float = 0.9
    maf_min: float = 0.01
    keep_chr_pos: bool = False


@register_cli(name='run_find_latent_representations',
              description='Run Find_latent_representations \nFind the latent representations of each spot by running GNN-VAE',
              add_args_function=add_find_latent_representations_args)
def run_find_latent_representation_from_cli(args: argparse.Namespace):
    from gsMap.find_latent_representation import run_find_latent_representation
    config = get_dataclass_from_parser(args, FindLatentRepresentationsConfig)
    run_find_latent_representation(config)


@register_cli(name='run_latent_to_gene',
              description='Run Latent_to_gene \nEstimate gene marker gene scores for each spot by using latent representations from nearby spots',
              add_args_function=add_latent_to_gene_args)
def run_latent_to_gene_from_cli(args: argparse.Namespace):
    from gsMap.latent_to_gene import run_latent_to_gene
    config = get_dataclass_from_parser(args, LatentToGeneConfig)
    run_latent_to_gene(config)


@register_cli(name='run_generate_ldscore',
              description='Run Generate_ldscore \nGenerate LD scores for each spot',
              add_args_function=add_generate_ldscore_args)
def run_generate_ldscore_from_cli(args: argparse.Namespace):
    from gsMap.generate_ldscore import run_generate_ldscore
    config = get_dataclass_from_parser(args, GenerateLDScoreConfig)
    run_generate_ldscore(config)


@register_cli(name='run_spatial_ldsc',
              description='Run Spatial_ldsc \nRun spatial LDSC for each spot',
              add_args_function=add_spatial_ldsc_args)
def run_spatial_ldsc_from_cli(args: argparse.Namespace):
    from gsMap.spatial_ldsc_multiple_sumstats import run_spatial_ldsc
    config = get_dataclass_from_parser(args, SpatialLDSCConfig)
    run_spatial_ldsc(config)


@register_cli(name='run_cauchy_combination',
              description='Run Cauchy_combination for each annotation',
              add_args_function=add_Cauchy_combination_args)
def run_Cauchy_combination_from_cli(args: argparse.Namespace):
    from gsMap.cauchy_combination_test import run_Cauchy_combination
    config = get_dataclass_from_parser(args, CauchyCombinationConfig)
    run_Cauchy_combination(config)


@register_cli(name='run_report',
              description='Run Report to generate diagnostic plots and tables',
              add_args_function=add_report_args)
def run_Report_from_cli(args: argparse.Namespace):
    from gsMap.report import run_report
    config = get_dataclass_from_parser(args, ReportConfig)
    run_report(config)


@register_cli(name='format_sumstats',
              description='Format gwas summary statistics',
              add_args_function=add_format_sumstats_args)
def gwas_format_from_cli(args: argparse.Namespace):
    from gsMap.format_sumstats import gwas_format
    config = get_dataclass_from_parser(args, FormatSumstatsConfig)
    gwas_format(config)

@register_cli(name='quick_mode',
                description='Run all the gsMap pipeline in quick mode',
                add_args_function=add_run_all_mode_args)
def run_all_mode_from_cli(args: argparse.Namespace):
    from gsMap.run_all_mode import run_pipeline
    config = get_dataclass_from_parser(args, RunAllModeConfig)
    run_pipeline(config)

@register_cli(name='create_slice_mean',
                description='Create slice mean from multiple h5ad files',
                add_args_function=add_create_slice_mean_args)
def create_slice_mean_from_cli(args: argparse.Namespace):
    from gsMap.create_slice_mean import run_create_slice_mean
    config = get_dataclass_from_parser(args, CreateSliceMeanConfig)
    run_create_slice_mean(config)