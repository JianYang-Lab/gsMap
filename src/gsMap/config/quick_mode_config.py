"""
Configuration for Quick Mode pipeline.
"""

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Annotated

import typer

from gsMap.config.base import ConfigWithAutoPaths
from gsMap.config.gwas import trait_file_token, trait_labels
from gsMap.config.spatial import (
    is_thick_tissue,
    report_geometry,
    reuse_spatial_validation,
    spatial_layout,
    validate_spatial_config,
)
from gsMap.config.utils import process_h5ad_inputs

from .cauchy_config import CauchyCombinationConfig

# Use relative imports to avoid circular dependency
from .find_latent_config import FindLatentRepresentationsConfig
from .latent2gene_config import DatasetType, LatentToGeneConfig, validate_expression_contract
from .report_config import (
    ReportConfig,
    should_export_spatialvista,
    spatialvista_output_is_current,
)
from .spatial_ldsc_config import SpatialLDSCConfig

logger = logging.getLogger("gsMap.config")


@dataclass
class QuickModeConfig(
    ReportConfig,
    SpatialLDSCConfig,
    LatentToGeneConfig,
    FindLatentRepresentationsConfig,
    ConfigWithAutoPaths,
):
    """Quick Mode Pipeline Configuration"""

    __core_only__ = True

    # ------------------------------------------------------------------------
    # Pipeline Control
    # ------------------------------------------------------------------------
    start_step: Annotated[
        str,
        typer.Option(
            help="Step to start execution from (find_latent, latent2gene, spatial_ldsc, cauchy, report)",
            case_sensitive=False,
        ),
    ] = "find_latent"

    stop_step: Annotated[
        str | None,
        typer.Option(help="Step to stop execution at (inclusive)", case_sensitive=False),
    ] = None

    def __post_init__(self):
        ConfigWithAutoPaths.__post_init__(self)
        validate_expression_contract(self, start_step=self.start_step)

        self._init_sumstats()
        self._init_annotation_list()

        self.dataset_type = DatasetType(self.dataset_type)
        self.sample_h5ad_dict = process_h5ad_inputs(
            self,
            {
                "h5ad_yaml": ("h5ad_yaml", "yaml"),
                "h5ad_path": ("h5ad_path", "list"),
                "h5ad_list_file": ("h5ad_list_file", "file"),
            },
        )
        if not self.sample_h5ad_dict:
            self._auto_detect_h5ad_files()

        if self.is_both_latent_and_gene_running or self._resumes_from_find_latent_outputs:
            # A run resumed at latent2gene from gsMap's own find_latent outputs uses the
            # same dual-embedding settings as the full run that produced them.
            self.high_quality_neighbor_filter = self.high_quality_cell_qc

            # Use dual embeddings if both steps are running
            if self.latent_representation_niche is None:
                self.latent_representation_niche = "emb_niche"
            if self.latent_representation_cell is None:
                self.latent_representation_cell = "emb_cell"

        # LDSC/Cauchy-only runs consume existing scores, not spatial coordinates.
        if self.requires_spatial_inputs:
            validate_spatial_config(self)
            self._configure_dataset_parameters()

        # For scRNA, clear spatial_key and niche key so downstream steps (e.g. report)
        # skip spatial-coordinate-specific logic
        if self.dataset_type == DatasetType.SCRNA_SEQ:
            self.spatial_key = None
            self.latent_representation_niche = None
        self.show_config(QuickModeConfig)

    @property
    def requires_spatial_inputs(self) -> bool:
        return not (
            self.start_step in ("spatial_ldsc", "cauchy")
            and self.stop_step in ("spatial_ldsc", "cauchy")
        )

    @property
    def _resumes_from_find_latent_outputs(self) -> bool:
        """Starting at latent2gene with find_latent outputs present in this project."""
        stops_before = self.stop_step in ("find_latent",)
        return (
            self.start_step == "latent2gene"
            and not stops_before
            and self.dataset_type != DatasetType.SCRNA_SEQ
            and Path(self.find_latent_metadata_path).is_file()
        )

    @property
    def is_both_latent_and_gene_running(self) -> bool:
        """Check if both find_latent and latent2gene are in the execution range."""
        steps = ["find_latent", "latent2gene", "spatial_ldsc", "cauchy", "report"]
        try:
            start_idx = steps.index(self.start_step)
            stop_idx = steps.index(self.stop_step) if self.stop_step else len(steps) - 1
            return start_idx <= 0 and stop_idx >= 1
        except ValueError:
            return False

    @property
    def find_latent_config(self) -> FindLatentRepresentationsConfig:
        with reuse_spatial_validation(self):
            return FindLatentRepresentationsConfig(
                **{
                    f.name: getattr(self, f.name)
                    for f in fields(FindLatentRepresentationsConfig)
                    if f.init
                }
            )

    @property
    def latent2gene_config(self) -> LatentToGeneConfig:
        params = {f.name: getattr(self, f.name) for f in fields(LatentToGeneConfig) if f.init}
        with reuse_spatial_validation(self):
            return LatentToGeneConfig(**params)

    @property
    def spatial_ldsc_config(self) -> SpatialLDSCConfig:
        return SpatialLDSCConfig(
            **{f.name: getattr(self, f.name) for f in fields(SpatialLDSCConfig) if f.init}
        )

    @property
    def report_config(self) -> ReportConfig:
        return ReportConfig(
            **{
                f.name: getattr(self, f.name)
                for f in fields(ReportConfig)
                if f.init and hasattr(self, f.name)
            }
        )

    @property
    def cauchy_config(self) -> CauchyCombinationConfig:
        return CauchyCombinationConfig(
            **{
                f.name: getattr(self, f.name)
                for f in fields(CauchyCombinationConfig)
                if f.init and hasattr(self, f.name)
            }
        )


def check_report_done(config: QuickModeConfig, verbose: bool = False) -> bool:
    missing_data_files, missing_web_files = get_report_missing_files(config)
    missing_files = missing_data_files + missing_web_files

    if missing_files and verbose:
        logger.info(f"Report incomplete. Missing {len(missing_files)} files:")
        for f in missing_files[:10]:  # Show first 10
            logger.info(f"  - {f}")
        if len(missing_files) > 10:
            logger.info(f"  ... and {len(missing_files) - 10} more")

    return len(missing_files) == 0 and _report_visualization_config_matches(config, verbose)


def _report_visualization_config_matches(config: QuickModeConfig, verbose: bool = False) -> bool:
    """Return whether cached report payloads use the requested display limits."""
    report_meta_path = config.web_report_dir / "report_meta.json"
    if not report_meta_path.is_file():
        return False

    try:
        with report_meta_path.open() as handle:
            report_meta = json.load(handle)
    except (OSError, ValueError) as error:
        if verbose:
            logger.info(f"Report metadata could not be read: {error}")
        return False

    expected = {
        "spatial_layout": spatial_layout(config),
        "spatial_geometry": report_geometry(config),
        "downsampling_n_spots_umap": config.downsampling_n_spots_umap,
        "downsampling_n_spots_2d": config.downsampling_n_spots_2d,
        "downsampling_n_spots_3d": config.downsampling_n_spots_3d,
        "top_corr_genes": config.top_corr_genes,
        "trait_labels": trait_labels(config, config.trait_name_list),
        "trait_payload_version": 2,
    }
    if should_export_spatialvista(config):
        expected.update(
            export_spatialvista=True,
            spatialvista_max_spots=getattr(config, "spatialvista_max_spots", None),
        )
    mismatches = {
        key: (report_meta.get(key), value)
        for key, value in expected.items()
        if report_meta.get(key) != value
    }
    if mismatches and verbose:
        logger.info(f"Report visualization settings changed: {mismatches}")
    return not mismatches


def get_report_missing_files(config: QuickModeConfig) -> tuple[list[Path], list[Path]]:
    """
    Get lists of missing report files, categorized by type.

    Returns:
        Tuple of (missing_data_files, missing_web_files)
    """
    missing_data_files = []
    missing_web_files = []

    # Get dataset type (default to spatial2D if not found)
    dataset_type = config.dataset_type

    # === Web Report Files ===
    web_report_dir = config.web_report_dir
    js_data_dir = web_report_dir / "js_data"

    core_web_files = [
        web_report_dir / "index.html",
        web_report_dir / "report_meta.json",
        js_data_dir / "sample_index.js",
        js_data_dir / "report_meta.js",
        js_data_dir / "cauchy_index.js",
        js_data_dir / "association_overview_index.js",
        js_data_dir / "association_overview_manifest.json",
        js_data_dir / "cauchy_manifest.json",
    ]

    for f in core_web_files:
        if not f.exists():
            missing_web_files.append(f)

    # === Data Files ===
    report_data_dir = config.report_data_dir

    core_data_files = [
        report_data_dir / "spot_metadata.csv",
        report_data_dir / "gene_list.csv",
        report_data_dir / "cauchy_results.csv",
    ]
    if should_export_spatialvista(config):
        if not spatialvista_output_is_current(config):
            missing_data_files.append(
                config.project_dir / f"{config.project_dir.name}.spatialvista.h5ad"
            )

    for f in core_data_files:
        if not f.exists():
            missing_data_files.append(f)

    # Per-trait files
    traits = config.trait_name_list
    annotation_list = config.annotation_list

    for trait in traits:
        trait_gss_csv = report_data_dir / "gss_stats" / f"gene_trait_correlation_{trait}.csv"
        trait_manhattan_csv = report_data_dir / "manhattan_data" / f"{trait}_manhattan.csv"

        if not trait_gss_csv.exists():
            missing_data_files.append(trait_gss_csv)
        if not trait_manhattan_csv.exists():
            missing_data_files.append(trait_manhattan_csv)

        trait_gss_js = js_data_dir / "gss_stats" / f"gene_trait_correlation_{trait}.js"
        trait_manhattan_js = js_data_dir / f"manhattan_{trait}.js"

        if not trait_gss_js.exists():
            missing_web_files.append(trait_gss_js)
        if not trait_manhattan_js.exists():
            missing_web_files.append(trait_manhattan_js)

    # Per-sample spatial JS files (Spatial only)
    if dataset_type in (DatasetType.SPATIAL_2D, DatasetType.SPATIAL_3D) and not is_thick_tissue(
        config
    ):
        sample_payload_dir = js_data_dir / "spatial_2d"
        sample_manifest = js_data_dir / "sample_spatial_manifest.json"
        if not sample_manifest.exists():
            missing_web_files.append(sample_manifest)
        sample_h5ad_dict = config.sample_h5ad_dict
        if sample_h5ad_dict:
            for sample_name in sample_h5ad_dict.keys():
                safe_name = "".join(c if c.isalnum() else "_" for c in sample_name)
                sample_base = sample_payload_dir / f"sample_{safe_name}_base.js"
                if not sample_base.exists():
                    missing_web_files.append(sample_base)
                for trait in traits:
                    safe_trait = trait_file_token(trait)
                    trait_payload = (
                        sample_payload_dir / f"sample_{safe_name}_trait_{safe_trait}.js"
                    )
                    if not trait_payload.exists():
                        missing_web_files.append(trait_payload)

        # Gene diagnostic plots directory (Spatial only)
        gene_plot_dir = web_report_dir / "gene_diagnostic_plots"
        if not gene_plot_dir.exists():
            missing_web_files.append(gene_plot_dir)

    # 3D specific files (Spatial 3D only)
    if dataset_type == DatasetType.SPATIAL_3D:
        three_d_data_dir = report_data_dir / "spatial_3d"
        three_d_payload_dir = js_data_dir / "spatial_3d"
        for data_file in (
            report_data_dir / "spatial_3d_coordinate_info.json",
            three_d_data_dir / "spatial_3d_manifest.json",
            three_d_data_dir / "spatial_3d_display_spots.parquet",
        ):
            if not data_file.exists():
                missing_data_files.append(data_file)

        for web_file in (
            three_d_payload_dir / "spatial_3d_base.js",
            three_d_payload_dir / "spatial_3d_index.js",
        ):
            if not web_file.exists():
                missing_web_files.append(web_file)

        for trait in traits:
            safe_trait = trait_file_token(trait)
            trait_payload = three_d_payload_dir / f"trait_{safe_trait}.js"
            if not trait_payload.exists():
                missing_web_files.append(trait_payload)

        for anno in annotation_list:
            safe_anno = "".join(c if c.isalnum() else "_" for c in anno)
            anno_payload = three_d_payload_dir / f"annotation_{safe_anno}.js"
            if not anno_payload.exists():
                missing_web_files.append(anno_payload)

    # Multi-sample plots (Spatial only, and if enabled)
    if dataset_type != DatasetType.SCRNA_SEQ and config.generate_multi_sample_plots:
        spatial_plot_dir = web_report_dir / "spatial_plots"

        for trait in traits:
            safe_trait = trait_file_token(trait)
            plot_path = spatial_plot_dir / "by_trait" / f"ldsc_{safe_trait}.png"
            if not plot_path.exists():
                missing_web_files.append(plot_path)

        for sample_name in config.sample_h5ad_dict or {}:
            safe_sample = "".join(c if c.isalnum() else "_" for c in sample_name)
            plot_path = spatial_plot_dir / "by_sample" / f"sample_{safe_sample}.png"
            if not plot_path.exists():
                missing_web_files.append(plot_path)

    # UMAP data (optional)
    concat_adata_path = config.concatenated_latent_adata_path
    if concat_adata_path and concat_adata_path.exists():
        umap_file = report_data_dir / "umap_data.csv"
        umap_js = js_data_dir / "umap_data.js"
        if not umap_file.exists():
            missing_data_files.append(umap_file)
        if not umap_js.exists():
            missing_web_files.append(umap_js)

    return missing_data_files, missing_web_files
