"""
Base configuration classes and utilities for gsMap.
"""

from dataclasses import dataclass
from functools import wraps
from pathlib import Path


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