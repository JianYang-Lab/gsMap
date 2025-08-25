"""
Base configuration classes and utilities for gsMap.
"""
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Optional, Annotated, List
import typer


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

class ConfigWithAutoPaths:
    """Base configuration class with automatic path generation."""

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

    def __post_init__(self):
        if self.workdir is None:
            raise ValueError('workdir must be provided.')
        work_dir = Path(self.workdir)
        self.project_dir = work_dir / self.project_name

    ## ---- Find latent representation paths
    @property
    @ensure_path_exists
    def latent_dir(self) -> Path:
        return self.project_dir / "find_latent_representations"

    @property
    @ensure_path_exists
    def model_path(self) -> Path:
        return self.latent_dir / 'LGCN_model/gsMap_LGCN_.pt'

    ## ---- Latent to gene paths

    @property
    @ensure_path_exists
    def latent2gene_dir(self) -> Path:
        """Directory for latent to gene outputs"""
        return self.project_dir / "latent_to_gene"
    
    @property
    @ensure_path_exists
    def concatenated_latent_adata_path(self) -> Path:
        """Path to concatenated latent representations"""
        return self.latent2gene_dir / "concatenated_latent_adata.h5ad"
    
    @property
    @ensure_path_exists
    def rank_zarr_path(self) -> Path:
        """Path to rank zarr file"""
        return self.latent2gene_dir / "ranks.zarr"
    
    @property
    @ensure_path_exists
    def mean_frac_path(self) -> Path:
        """Path to mean expression fraction parquet"""
        return self.latent2gene_dir / "mean_frac.parquet"
    
    @property
    @ensure_path_exists
    def marker_scores_zarr_path(self) -> Path:
        """Path to marker scores zarr"""
        return self.latent2gene_dir / "marker_scores.zarr"
    
    @property
    @ensure_path_exists
    def latent2gene_metadata_path(self) -> Path:
        """Path to latent2gene metadata JSON"""
        return self.latent2gene_dir / "metadata.json"

    ## ---- LD score paths


    ## ---- Spatial LDSC paths

    @property
    @ensure_path_exists
    def ldsc_dir(self) -> Path:
        return self.project_dir / "spatial_ldsc"
    
    @property
    @ensure_path_exists
    def ldsc_save_dir(self) -> Path:
        """Directory for spatial LDSC results"""
        return self.project_dir / "spatial_ldsc"

    #
    #
    # @property
    # @ensure_path_exists
    # def mkscore_feather_path(self) -> Path:
    #     return Path(f'{self.project_dir}/latent_to_gene/mk_score/{self.sample_name}_gene_marker_score.feather')
    #
    # @property
    # @ensure_path_exists
    # def tuned_mkscore_feather_path(self) -> Path:
    #     return Path(f'{self.project_dir}/latent_to_gene/mk_score_pooling/{self.sample_name}_gene_marker_score.feather')
    #
    # @property
    # @ensure_path_exists
    # def ldscore_save_dir(self) -> Path:
    #     return Path(f'{self.project_dir}/generate_ldscore/{self.sample_name}')
    #
    # @property
    # @ensure_path_exists
    # def ldsc_save_dir(self) -> Path:
    #     return Path(f'{self.project_dir}/spatial_ldsc/{self.sample_name}')
    
    @property
    @ensure_path_exists
    def cauchy_save_dir(self) -> Path:
        return self.project_dir / "cauchy_combination"
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