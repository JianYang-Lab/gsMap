"""
Configuration for generating reports.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from .base import ConfigWithAutoPaths, ensure_path_exists
from .cauchy_config import CauchyCombinationConfig
from .latent2gene_config import DatasetType

logger = logging.getLogger("gsMap.config")


def should_export_spatialvista(config) -> bool:
    """SpatialVista is a default report output for spatial3D datasets only."""
    return getattr(config, "dataset_type", None) == DatasetType.SPATIAL_3D and getattr(
        config, "export_spatialvista", True
    )


def spatialvista_output_is_current(config) -> bool:
    """Validate the export itself, including after an interrupted report rerun."""
    import h5py

    path = config.project_dir / f"{config.project_dir.name}.spatialvista.h5ad"
    if not path.is_file():
        return False
    try:
        with h5py.File(path, "r") as handle:
            return handle["uns/gsmap/schema_version"][()] == 1 and handle[
                "uns/gsmap/requested_max_spots"
            ][()] == (getattr(config, "spatialvista_max_spots", None) or 0)
    except (OSError, KeyError, TypeError, ValueError):
        return False


@dataclass
class ReportConfig(CauchyCombinationConfig, ConfigWithAutoPaths):
    """Report Generation Configuration"""

    export_spatialvista: Annotated[
        bool,
        typer.Option(
            help="Automatically export a SpatialVista H5AD for spatial3D reports; ignored for spatial2D and scRNA"
        ),
    ] = True

    spatialvista_max_spots: Annotated[
        int | None,
        typer.Option(
            min=1,
            help="Optional deterministic sampling limit for the SpatialVista H5AD; default exports all result spots, independently of HTML sampling",
        ),
    ] = None

    downsampling_n_spots_pcc: Annotated[
        int,
        typer.Option(
            help="Number of spots to downsample for PCC calculation if n_spots > this value",
            min=1000,
            max=100000,
        ),
    ] = 20000

    downsampling_n_spots_umap: Annotated[
        int,
        typer.Option(
            help=(
                "Maximum representative spots used to calculate and display each report UMAP. "
                "Valid precomputed UMAP coordinates are sampled to the same limit."
            ),
            min=1000,
            max=500000,
        ),
    ] = 50000

    downsampling_n_spots_3d: Annotated[
        int,
        typer.Option(
            help=(
                "Maximum representative spots rendered in the browser's 3D view. "
                "Larger datasets use deterministic annotation-aware sampling."
            ),
            min=1000,
            max=2000000,
        ),
    ] = 200000

    downsampling_n_spots_2d: Annotated[
        int,
        typer.Option(
            help=(
                "Maximum representative spots per sample in the browser's 2D view. "
                "Larger samples use deterministic annotation-aware sampling."
            ),
            min=10000,
            max=500000,
        ),
    ] = 300000

    top_corr_genes: Annotated[
        int,
        typer.Option(
            help="Number of top correlated genes per trait for expression/GSS diagnostic plots and Manhattan highlights",
            min=1,
            max=500,
        ),
    ] = 10

    # Advanced visualization parameters
    single_sample_multi_trait_max_cols: int = 5
    subsample_n_points: int | None = None
    single_sample_multi_trait_subplot_width_inches: float = 4.0
    single_sample_multi_trait_dpi: int = 300
    enable_pdf_output: bool = True
    hover_text_list: list | None = None
    single_trait_multi_sample_max_cols: int = 8
    single_trait_multi_sample_subplot_width_inches: float = 4.0
    single_trait_multi_sample_scaling_factor: float = 1.0
    single_trait_multi_sample_dpi: int = 300
    share_coords: bool = False

    # Generate both static overview matrices: traits across samples and samples across traits.
    generate_multi_sample_plots: bool = False

    # Plot origin for spatial plots ('upper' or 'lower')
    plot_origin: Annotated[
        str,
        typer.Option(
            help="Plot origin for spatial plots ('upper' or 'lower'). 'upper' will flip the y-axis (standard for images)."
        ),
    ] = "upper"

    # Legend marker size for categorical plots
    legend_marker_size: float = 10.0

    # Force re-run of report generation even if results exist
    force_report_re_run: bool = False

    # Compatibility properties for visualization paths
    @property
    @ensure_path_exists
    def visualization_result_dir(self) -> Path:
        return self.project_dir / "report" / self.project_name / (self.trait_name or "multi_trait")

    def __post_init__(self):
        CauchyCombinationConfig.__post_init__(self)
        self.show_config(ReportConfig)
