"""
Base configuration classes and utilities for gsMap.
"""

import inspect
from dataclasses import asdict, dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from gsMap.logging_config import is_interactive_output


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


@dataclass
class BaseConfig:
    """Base configuration class with display and conversion utility."""

    def to_dict_with_paths_as_strings(self) -> dict[str, Any]:
        """
        Convert the config object to a dictionary with all Path objects converted to strings.
        Also handles nested Path and Enum objects in dictionaries and lists.

        Returns:
            Dictionary representation of the config with all Path objects as strings
        """
        # Convert config to dict
        config_dict = asdict(self)

        # Convert all Path and Enum objects in config to strings/values
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            elif isinstance(value, Enum):
                config_dict[key] = value.value
            elif isinstance(value, dict):
                config_dict[key] = {
                    k: (str(v) if isinstance(v, Path) else (v.value if isinstance(v, Enum) else v))
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                config_dict[key] = [
                    (str(v) if isinstance(v, Path) else (v.value if isinstance(v, Enum) else v))
                    for v in value
                ]

        return config_dict

    def show_config(self, cls: type | None = None):
        """Show configuration in a nice way using rich."""
        if cls is not None and type(self) is not cls:
            return
        # Resolved configurations are persisted in the run manifest and debug
        # log. Printing a many-page Rich panel into SLURM output adds no value.
        if not is_interactive_output():
            return

        config_dict = self.to_dict_with_paths_as_strings()
        config_yaml = yaml.dump(config_dict, default_flow_style=False, sort_keys=False)

        # Get title from docstring (first line)
        doc = inspect.getdoc(type(self))
        title = doc.split("\n")[0] if doc else "Configuration"

        console = Console()
        console.print(
            Panel(
                Syntax(config_yaml, "yaml", theme="monokai", line_numbers=True),
                title=f"[bold]{title}[/bold]",
                expand=False,
            )
        )


@dataclass
class ConfigWithAutoPaths(BaseConfig):
    """Base configuration class with automatic path generation."""

    # Required from parent
    workdir: Annotated[
        Path,
        typer.Option(
            help="Path to the working directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ]

    project_name: Annotated[str, typer.Option(help="Name of the project")]

    @property
    @ensure_path_exists
    def project_dir(self) -> Path:
        """The main project directory, which is workdir / project_name."""
        return Path(self.workdir) / self.project_name

    def __post_init__(self):
        if self.workdir is None:
            raise ValueError("workdir must be provided.")

    ## ---- Find latent representation paths
    @property
    @ensure_path_exists
    def latent_dir(self) -> Path:
        return self.project_dir / "find_latent_representations"

    @property
    @ensure_path_exists
    def model_path(self) -> Path:
        return self.latent_dir / "LGCN_model" / "gsMap_LGCN_.pt"

    @property
    def find_latent_metadata_path(self) -> Path:
        return self.latent_dir / "find_latent_metadata.yaml"

    ## ---- Latent to gene paths

    @property
    @ensure_path_exists
    def latent2gene_dir(self) -> Path:
        """Directory for latent to gene outputs"""
        return self.project_dir / "latent_to_gene"

    @property
    def concatenated_latent_adata_path(self) -> Path:
        """Path to concatenated latent representations"""
        return self.latent2gene_dir / "concatenated_latent_adata.h5ad"

    @property
    def rank_memmap_path(self) -> Path:
        """Path to rank zarr file"""
        return self.latent2gene_dir / "ranks.dat"

    @property
    def mean_frac_path(self) -> Path:
        """Path to mean expression fraction parquet"""
        return self.latent2gene_dir / "mean_frac.parquet"

    @property
    def marker_scores_memmap_path(self) -> Path:
        """Path to marker scores zarr"""
        return self.latent2gene_dir / "marker_scores.dat"

    @property
    def latent2gene_metadata_path(self) -> Path:
        """Path to latent2gene metadata YAML"""
        return self.latent2gene_dir / "metadata.yaml"

    ## ---- Spatial LDSC paths

    @property
    @ensure_path_exists
    def ldsc_save_dir(self) -> Path:
        """Directory for spatial LDSC results"""
        return self.project_dir / "spatial_ldsc"

    def get_ldsc_result_file(self, trait_name: str) -> Path:
        return Path(f"{self.ldsc_save_dir}/{self.project_name}_{trait_name}.csv.gz")

    @property
    @ensure_path_exists
    def ldscore_save_dir(self) -> Path:
        """Directory for LD score generation results"""
        return self.project_dir / "generate_ldscore"

    @property
    @ensure_path_exists
    def cauchy_save_dir(self) -> Path:
        return self.project_dir / "cauchy_combination"

    @property
    @ensure_path_exists
    def report_data_dir(self) -> Path:
        """Directory for report data files (CSV, h5ad) - not needed for HTML viewing"""
        return self.project_dir / "report_data"

    @property
    @ensure_path_exists
    def web_report_dir(self) -> Path:
        """Directory for self-contained web report (HTML, JS, images)"""
        return self.project_dir / "gsmap_web_report"

    @property
    @ensure_path_exists
    def report_dir(self) -> Path:
        """Directory for gsMap report outputs - returns web_report_dir for backward compatibility"""
        return self.web_report_dir

    @property
    def ldsc_combined_parquet_path(self) -> Path:
        return self.cauchy_save_dir / f"{self.project_name}_combined_ldsc.parquet"

    def get_cauchy_result_file(
        self, trait_name: str, annotation: str | None = None, all_samples: bool = False
    ) -> Path:
        if annotation is None:
            annotation = getattr(self, "annotation", "unknown")
        if all_samples:
            return (
                self.cauchy_save_dir / f"{self.project_name}_{trait_name}.{annotation}.cauchy.csv"
            )
        else:
            return (
                self.cauchy_save_dir
                / f"{self.project_name}_{trait_name}.{annotation}.sample_cauchy.csv"
            )

    @ensure_path_exists
    def get_gene_diagnostic_info_save_path(self, trait_name: str) -> Path:
        """Path for gene diagnostic info CSV - uses trait prefix in gss_stats subfolder"""
        return self.report_data_dir / "gss_stats" / f"gene_trait_correlation_{trait_name}.csv"
