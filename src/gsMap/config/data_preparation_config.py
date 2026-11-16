"""Initialize the data preparation stages without loading GWAS resources.

This configuration deliberately inherits only the two data-stage configurations.
Construction validates the selected source inputs; it never runs either stage.
"""

from collections import OrderedDict
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Annotated

import typer

from gsMap.config.base import ConfigWithAutoPaths
from gsMap.config.find_latent_config import FindLatentRepresentationsConfig
from gsMap.config.latent2gene_config import (
    CrossSliceSelectionMode,
    DatasetType,
    HomogeneousSelectionMode,
    LatentToGeneConfig,
    MarkerScoreCrossSliceStrategy,
    validate_expression_contract,
)
from gsMap.config.spatial import reuse_spatial_validation, validate_spatial_config
from gsMap.config.utils import process_h5ad_inputs, validate_h5ad_structure


@dataclass
class DataPreparationConfig(
    LatentToGeneConfig, FindLatentRepresentationsConfig, ConfigWithAutoPaths
):
    """GWAS-independent find_latent / latent2gene preparation configuration."""

    start_step: Annotated[
        str, typer.Option(help="First data stage: find_latent or latent2gene")
    ] = "find_latent"
    stop_step: Annotated[
        str, typer.Option(help="Last data stage (inclusive): find_latent or latent2gene")
    ] = "latent2gene"
    latent_representation_cell: Annotated[
        str | None,
        typer.Option(help="Existing cell embedding for latent2gene, or generated embedding key"),
    ] = None

    def __post_init__(self):
        ConfigWithAutoPaths.__post_init__(self)
        steps = ("find_latent", "latent2gene")
        if self.start_step not in steps or self.stop_step not in steps:
            raise ValueError("Data preparation supports only find_latent and latent2gene")
        if steps.index(self.start_step) > steps.index(self.stop_step):
            raise ValueError("start_step must not follow stop_step")
        validate_expression_contract(self, start_step=self.start_step)
        if self.expression_data_type == "noncounts" and (
            self.species is not None or self.homolog_file is not None
        ):
            raise ValueError(
                "Noncounts latent2gene requires a pre-mapped gene axis; omit species/homolog_file"
            )
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
            raise ValueError("Explicit source H5AD inputs are required for data preparation")
        self.sample_h5ad_dict = OrderedDict(
            (name, Path(path)) for name, path in self.sample_h5ad_dict.items()
        )

        if self.start_step == "find_latent":
            self.latent_representation_niche = self.latent_representation_niche or "emb_niche"
            self.latent_representation_cell = self.latent_representation_cell or "emb_cell"
        if self.is_both_latent_and_gene_running:
            self.high_quality_neighbor_filter = self.high_quality_cell_qc

        self.cross_slice_marker_score_strategy = MarkerScoreCrossSliceStrategy(
            self.cross_slice_marker_score_strategy
        )
        self.cross_slice_selection_mode = CrossSliceSelectionMode(self.cross_slice_selection_mode)
        self.homogeneous_selection_mode = HomogeneousSelectionMode(self.homogeneous_selection_mode)
        validate_spatial_config(self)
        self._configure_dataset_parameters()
        if self.dataset_type == DatasetType.SCRNA_SEQ:
            self.spatial_key = None
            self.latent_representation_niche = None

        # Source counts do not yet have the representations produced by the
        # first stage. Validate them with the actual first-stage initializer.
        if self.start_step == "find_latent":
            first = self.find_latent_config
            self.graph_mode = first.graph_mode
            self.species = first.species
        else:
            if (
                not isinstance(self.latent_representation_cell, str)
                or not self.latent_representation_cell.strip()
            ):
                raise ValueError("latent2gene requires an existing named cell representation")
            # Use the native validation fields without auto-detecting outputs
            # from an earlier analysis or initializing JAX during preflight.
            self._setup_and_validate_fields()
            validate_h5ad_structure(
                self.sample_h5ad_dict, {"data_layer": ("layers", self.data_layer, "Data layer")}
            )
        self.show_config(DataPreparationConfig)

    @property
    def is_both_latent_and_gene_running(self):
        return self.start_step == "find_latent" and self.stop_step == "latent2gene"

    @property
    def find_latent_config(self):
        with reuse_spatial_validation(self):
            return FindLatentRepresentationsConfig(
                **{
                    field.name: getattr(self, field.name)
                    for field in fields(FindLatentRepresentationsConfig)
                    if field.init
                }
            )

    @property
    def latent2gene_config(self):
        """Construct the native second-stage config when its inputs exist.

        For a pipeline beginning at find_latent, this is called after that stage
        has produced its output H5AD files, just as in QuickModeConfig.
        """
        with reuse_spatial_validation(self):
            return LatentToGeneConfig(
                **{
                    field.name: getattr(self, field.name)
                    for field in fields(LatentToGeneConfig)
                    if field.init
                }
            )
