import logging
import shlex
import time
from datetime import datetime
from pathlib import Path

from jinja2 import Template

import gsMap
from gsMap.config import QuickModeConfig
from gsMap.config.report_config import should_export_spatialvista
from gsMap.logging_config import logged_stage

from .report_data import ReportDataManager, _atomic_write_text

logger = logging.getLogger(__name__)


def _render_report_html(project_name: str) -> str:
    template_path = Path(__file__).parent / "static" / "template.html"
    template = Template(template_path.read_text(encoding="utf-8"), autoescape=True)
    return template.render(
        title=f"gsMap Report - {project_name}",
        project_name=project_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        gsmap_version=getattr(gsMap, "__version__", "unknown"),
    )


@logged_stage("report")
def run_report(config: QuickModeConfig, run_parameters: dict = None):
    """
    Main entry point for report generation.
    Prepares data and saves the interactive report as a standalone Alpine+Tailwind HTML folder.

    Output structure:
        project_dir/
        ├── {project_name}.spatialvista.h5ad  # spatial3D coordinates, annotations and traits
        ├── report_data/                    # Prepared data and export provenance
        │   ├── spot_metadata.csv
        │   ├── cauchy_results.csv
        │   ├── umap_data.csv
        │   ├── gene_list.csv
        │   ├── spatial_3d_coordinate_info.json
        │   ├── gss_stats/
        │   │   └── gene_trait_correlation_{trait}.csv
        │   ├── manhattan_data/
        │   │   └── {trait}_manhattan.csv
        │   └── spatial_3d/
        │       ├── spatial_3d_manifest.json
        │       └── spatial_3d_display_spots.parquet
        │
        └── gsmap_web_report/               # Web report (self-contained)
            ├── index.html
            ├── report_meta.json
            ├── execution_summary.yaml
            ├── spatial_plots/
            │   └── ldsc_{trait}.png
            ├── gene_diagnostic_plots/
            ├── annotation_plots/
            ├── js_lib/
            └── js_data/
                ├── gss_stats/
                ├── annotation_schemas.js     # Shared categorical colors
                ├── sample_index.js
                ├── spatial_2d/             # Sample bases + lazy trait arrays
                ├── spatial_3d/             # Shared geometry + lazy feature arrays
                └── ... (other JS modules)
    """
    logger.info("Running gsMap Report Module")
    maximum = getattr(config, "spatialvista_max_spots", None)
    if should_export_spatialvista(config) and maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
            raise ValueError("spatialvista_max_spots must be a positive integer or None")
    # Validate the template before doing expensive data preparation.
    rendered_html = _render_report_html(config.project_name)

    # 1. Use ReportDataManager to prepare all data and JS assets
    manager = ReportDataManager(config)
    web_report_dir = manager.run()

    exported = None
    if should_export_spatialvista(config):
        from .spatialvista import export_spatialvista

        export_started = time.perf_counter()
        logger.info(
            "Preparing SpatialVista H5AD from report coordinates, annotations and traits..."
        )
        exported = export_spatialvista(
            config.project_dir,
            metadata=manager.metadata,
            max_spots=maximum,
            overwrite=True,
        )
        logger.info(
            f"SpatialVista export completed in {time.perf_counter() - export_started:.2f}s"
        )

    # 2. Save run_parameters for future reference
    if run_parameters:
        import yaml

        with open(web_report_dir / "execution_summary.yaml", "w") as f:
            yaml.dump(run_parameters, f)

    # Publish the entry point atomically, only after its data files are ready.
    report_file = web_report_dir / "index.html"
    _atomic_write_text(report_file, rendered_html)

    logger.info("Report generated successfully!")
    logger.info(f"Web report directory: {web_report_dir}")
    logger.info(f"Data files directory: {config.report_data_dir}")
    logger.info(
        f"Open gsMap report: gsmap report-view {shlex.quote(str(web_report_dir))} --port 8080 --no-browser"
    )
    logger.info(
        f"On your computer, copy the complete {web_report_dir.name} directory and open index.html."
    )
    if exported is not None:
        logger.info(f"SpatialVista H5AD: {exported}")
        logger.info(
            f"Open 3D traits and annotations with gsMap-enabled SpatialVista: spatialvista --input {shlex.quote(str(exported))}"
        )
        logger.info(
            "For remote SpatialVista, add --no-browser --port 8765 and forward that port to your computer."
        )
