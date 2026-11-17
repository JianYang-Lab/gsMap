"""
Configuration for latent to gene mapping.
"""

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
import yaml

from gsMap.config.base import ConfigWithAutoPaths
from gsMap.config.spatial import DatasetType, validate_spatial_config
from gsMap.config.utils import configure_jax_platform, process_h5ad_inputs, validate_h5ad_structure

logger = logging.getLogger("gsMap.config")


class MarkerScoreCrossSliceStrategy(StrEnum):
    GLOBAL_POOL = "global_pool"
    PER_SLICE_POOL = "per_slice_pool"
    HIERARCHICAL_POOL = "hierarchical_pool"


class CrossSliceSelectionMode(StrEnum):
    """Rule used to choose which serial sections contribute candidates."""

    ORDINAL = "ordinal"
    PHYSICAL_WINDOW = "physical_window"


class HomogeneousSelectionMode(StrEnum):
    """How niche similarity constrains homogeneous-neighbor selection."""

    SOFT_PRIORITY = "soft_priority"
    HARD_NICHE_FILTER = "hard_niche_filter"


class ExpressionDataType(StrEnum):
    """Explicit expression semantics for downstream rank-based analysis."""

    COUNTS = "counts"
    NONCOUNTS = "noncounts"


def validate_expression_contract(config, *, start_step=None):
    """Keep count QC strict and require explicit rank-only noncount semantics."""
    data_type = ExpressionDataType(getattr(config, "expression_data_type", "counts"))
    config.expression_data_type = data_type
    if data_type == ExpressionDataType.NONCOUNTS:
        if start_step == "find_latent":
            raise ValueError("Noncounts expression must skip find_latent and start at latent2gene")
        if not getattr(config, "no_expression_fraction", False):
            raise ValueError("Noncounts expression requires no_expression_fraction=True")
        if getattr(config, "high_quality_neighbor_filter", False) or getattr(
            config, "high_quality_cell_qc", False
        ):
            raise ValueError(
                "Noncounts expression cannot use count-based high-quality cell filtering"
            )
        if getattr(config, "cell_qc_min_counts", None) is not None:
            raise ValueError("Noncounts expression cannot use cell_qc_min_counts")
    return data_type


@dataclass
class LatentToGeneComputeConfig:
    """Compute configuration for latent-to-gene step.

    Rank and marker-score batch sizes use zero for deterministic workload-based
    sizing. Hardware concurrency uses stable defaults and explicit overrides.
    """

    __display_in_quick_mode_cli__ = True

    use_gpu: Annotated[
        bool,
        typer.Option(
            "--use-gpu/--no-gpu",
            help="Use GPU for JAX computations (requires sufficient GPU memory)",
        ),
    ] = True

    memmap_tmp_dir: Annotated[
        Path | None,
        typer.Option(
            help="Optional fast scratch directory for memory-mapped files. A file is staged only "
            "when the full logical memmap plus safety headroom fits; otherwise gsMap uses the "
            "project storage directly.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None

    # Workload-dependent controls: zero selects deterministic data-based sizing.
    rank_batch_size: int = 0

    mkscore_batch_size: Annotated[
        int,
        typer.Option(
            help="Cells per marker-score batch; 0 selects automatically",
            min=0,
            max=1000,
        ),
    ] = 0

    # Hardware-dependent controls: stable defaults, never inferred from the host.
    find_homogeneous_batch_size: int = 100
    rank_write_interval: int = 0

    rank_read_workers: Annotated[
        int,
        typer.Option(
            help="Threads used to materialize distinct rank rows for each cache window",
            min=1,
            max=128,
        ),
    ] = 120

    mkscore_compute_workers: Annotated[
        int,
        typer.Option(
            help="Marker-score compute workers",
            min=1,
            max=16,
        ),
    ] = 4

    mkscore_write_workers: Annotated[
        int,
        typer.Option(
            help="Marker-score and rank write workers",
            min=1,
            max=50,
        ),
    ] = 2

    compute_input_queue_size: Annotated[
        int,
        typer.Option(
            help="Maximum pending marker-score batches per compute worker",
            min=1,
            max=10,
        ),
    ] = 1

    writer_queue_size: Annotated[
        int,
        typer.Option(
            help="Writer-input queue capacity",
            min=10,
            max=500,
        ),
    ] = 20


@dataclass
class LatentToGeneCoreConfig:
    dataset_type: Annotated[
        DatasetType,
        typer.Option(
            help="Type of dataset: scRNA, spatial2D (XY), or spatial3D (XYZ sections or a single thick-tissue volume)",
            case_sensitive=False,
        ),
    ] = "spatial2D"

    # --------input h5ad file paths which have the latent representations
    h5ad_path: Annotated[
        list[Path] | None,
        typer.Option(
            help="Space-separated list of h5ad file paths. Sample names are derived from file names without suffix.",
            exists=True,
            file_okay=True,
        ),
    ] = None

    h5ad_yaml: Annotated[
        Path | None,
        typer.Option(
            help="YAML file with sample names and h5ad paths",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None

    h5ad_list_file: Annotated[
        Path | None,
        typer.Option(
            help="Each row is a h5ad file path, sample name is the file name without suffix",
            exists=True,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None

    sample_h5ad_dict: OrderedDict | None = None

    # --------input h5ad obs, obsm, layers keys

    annotation: Annotated[
        str | None,
        typer.Option(
            help="Cell type annotation in adata.obs to use. This would constrain finding homogeneous spots within each cell type"
        ),
    ] = None

    data_layer: Annotated[
        str,
        typer.Option(
            help="Expression data layer in h5ad layers, e.g., 'count', 'counts'; use 'X' for adata.X"
        ),
    ] = "X"

    expression_data_type: Annotated[
        ExpressionDataType,
        typer.Option(
            help=(
                "Expression QC contract: counts requires finite nonnegative values; "
                "noncounts permits finite signed values with external embeddings, "
                "counts distinct nonzero features for min_genes, and requires "
                "--no-expression-fraction."
            ),
            case_sensitive=False,
        ),
    ] = ExpressionDataType.COUNTS

    latent_representation_niche: Annotated[
        str | None, typer.Option(help="Key for spatial niche embedding in obsm")
    ] = None

    latent_representation_cell: Annotated[
        str, typer.Option(help="Key for cell identity embedding in obsm")
    ] = "emb_cell"

    spatial_key: Annotated[str, typer.Option(help="Spatial key in adata.obsm")] = "spatial"

    # --------parameters for finding homogeneous spots

    spatial_neighbors: Annotated[
        int,
        typer.Option(
            help="k1: Number of spatial neighbors in it's own slice for spatial dataset",
            min=10,
            max=5000,
        ),
    ] = 301

    homogeneous_neighbors: Annotated[
        int,
        typer.Option(
            help="k3: Number of homogeneous neighbors per cell (for spatial) or KNN neighbors (for scRNA-seq)",
            min=1,
            max=200,
        ),
    ] = 21

    cell_embedding_similarity_threshold: Annotated[
        float,
        typer.Option(help="Minimum similarity threshold for cell embedding.", min=0.0, max=1.0),
    ] = 0.0

    spatial_domain_similarity_threshold: Annotated[
        float,
        typer.Option(
            help="Minimum similarity threshold for spatial domain embedding.", min=0.0, max=1.0
        ),
    ] = 0.6

    homogeneous_selection_mode: Annotated[
        HomogeneousSelectionMode,
        typer.Option(
            help=(
                "How niche similarity is used: soft_priority ranks passing "
                "neighbors first and fills remaining slots; hard_niche_filter "
                "reproduces the historical rule by assigning zero marker-score "
                "weight below the niche threshold."
            ),
            case_sensitive=False,
        ),
    ] = HomogeneousSelectionMode.SOFT_PRIORITY

    no_expression_fraction: Annotated[
        bool, typer.Option("--no-expression-fraction", help="Skip expression fraction filtering")
    ] = False

    # --------3D slice-aware neighbor search parameters
    adjacent_slice_spatial_neighbors: Annotated[
        int,
        typer.Option(
            help="Number of spatial neighbors to find on each adjacent slice for 3D data",
            min=10,
            max=2000,
        ),
    ] = 200

    n_adjacent_slices: Annotated[
        int,
        typer.Option(
            help="Adjacent sections on each side (default 2). Set 0 to disable cross-section homogeneous matching while retaining XYZ distances. Automatically 0 for a single spatial3D volume. Does not disable an explicit section-aware embedding graph.",
            min=0,
            max=100,
        ),
    ] = 2

    cross_slice_selection_mode: Annotated[
        CrossSliceSelectionMode,
        typer.Option(
            help="How cross-section candidates are chosen: by ordinal input order (legacy) or by a measured physical z-distance window.",
            case_sensitive=False,
        ),
    ] = CrossSliceSelectionMode.ORDINAL

    cross_slice_max_distance: Annotated[
        float | None,
        typer.Option(
            help="Maximum absolute physical z distance for physical-window section selection. Required when cross_slice_selection_mode=physical_window.",
            min=0.0,
        ),
    ] = None

    cross_slice_z_scale: Annotated[
        float,
        typer.Option(
            help="Physical units per registered z-coordinate unit. For example, use 4 for the E11.5 z_correction coordinates to express the window in micrometres.",
            min=0.0,
        ),
    ] = 1.0

    cross_slice_marker_score_strategy: Annotated[
        MarkerScoreCrossSliceStrategy,
        typer.Option(
            help="Strategy for computing marker scores across slices in spatial3D datasets. "
            "'global_pool': Select the top K most similar neighbors globally across all slices combined. "
            "'per_slice_pool': Select a fixed number of neighbors (K) from each slice independently, then compute a single weighted average score from all selected neighbors. "
            "'hierarchical_pool': Compute an independent marker score for each slice using its top K neighbors, then take the average of these per-slice scores.",
            case_sensitive=False,
        ),
    ] = MarkerScoreCrossSliceStrategy.HIERARCHICAL_POOL

    high_quality_neighbor_filter: Annotated[
        bool,
        typer.Option(
            "--high-quality-neighbor-filter/--no-high-quality-filter",
            help="Only find neighbors within high quality cells (requires High_quality column in obs)",
        ),
    ] = False

    fix_cross_slice_homogenous_neighbors: bool = False

    # Performance options
    # Minimum number of cells per cell type in that dataset to be used for finding homogeneous neighbors
    min_cells_per_type: int | None = None


@dataclass
class LatentToGeneConfig(LatentToGeneComputeConfig, LatentToGeneCoreConfig, ConfigWithAutoPaths):
    """Latent to Gene Configuration"""

    @property
    def total_homogeneous_neighbor_per_cell(self):
        return self.homogeneous_neighbors * (1 + 2 * self.n_adjacent_slices)

    def __post_init__(self):
        """Initialize and validate configuration"""
        super().__post_init__()
        validate_expression_contract(self)

        # Dataclass construction does not coerce programmatic string values to
        # StrEnum members (Typer does). Normalize both API and CLI calls so all
        # downstream comparisons, logging, and metadata use one representation.
        self.dataset_type = DatasetType(self.dataset_type)
        self.cross_slice_marker_score_strategy = MarkerScoreCrossSliceStrategy(
            self.cross_slice_marker_score_strategy
        )
        self.cross_slice_selection_mode = CrossSliceSelectionMode(self.cross_slice_selection_mode)
        self.homogeneous_selection_mode = HomogeneousSelectionMode(self.homogeneous_selection_mode)

        # Step 2: Process and validate h5ad inputs
        self._resolve_h5ad_inputs()
        validate_spatial_config(self)

        # Step 3: Configure dataset-specific parameters first
        self._configure_dataset_parameters()

        # Step 4: Set up validation fields and validate structure (after dataset config)
        self._setup_and_validate_fields()

        # Do not initialize an accelerator before input/configuration errors are reported.
        configure_jax_platform(self.use_gpu)

        self.show_config(LatentToGeneConfig)

    def _resolve_h5ad_inputs(self):
        """Resolve h5ad inputs from various sources, prioritizing auto-detection."""

        # Step 1: Try auto-detection first
        self._auto_detect_h5ad_files()
        if self.sample_h5ad_dict:
            self._warn_if_explicit_inputs_unused()

        # Step 2: If auto-detection didn't find anything, try explicit inputs
        if not self.sample_h5ad_dict:
            # Define input options
            input_options = {
                "h5ad_yaml": ("h5ad_yaml", "yaml"),
                "h5ad_path": ("h5ad_path", "list"),
                "h5ad_list_file": ("h5ad_list_file", "file"),
            }
            self.sample_h5ad_dict = process_h5ad_inputs(self, input_options)

        # Step 3: Validate at least one sample exists
        if not self.sample_h5ad_dict or len(self.sample_h5ad_dict) == 0:
            raise ValueError(
                "No valid h5ad files found. Please provide one of: h5ad_yaml, h5ad_path, or h5ad_list_file, "
                "or ensure find_latent representation has been run to allow auto-detection."
            )

        logger.info(f"Loaded and validated {len(self.sample_h5ad_dict)} samples")

    def _warn_if_explicit_inputs_unused(self):
        """Explicit --h5ad-* inputs are replaced by this project's find_latent outputs."""
        options = {
            "h5ad_yaml": ("h5ad_yaml", "yaml"),
            "h5ad_path": ("h5ad_path", "list"),
            "h5ad_list_file": ("h5ad_list_file", "file"),
        }
        if not any(getattr(self, name, None) for name in options):
            return
        try:
            import yaml

            explicit = process_h5ad_inputs(self, options) or {}
            with open(self.find_latent_metadata_path) as handle:
                recorded = (yaml.safe_load(handle).get("config") or {}).get(
                    "sample_h5ad_dict"
                ) or {}
        except Exception:  # noqa: BLE001 - only a hint; never block resolution
            return
        known = {
            Path(path).resolve() for path in [*recorded.values(), *self.sample_h5ad_dict.values()]
        }
        unused = sorted(
            str(path) for path in explicit.values() if Path(path).resolve() not in known
        )
        if unused:
            logger.warning(
                "Using the find_latent outputs of this project "
                f"({self.find_latent_metadata_path}); the explicitly given inputs "
                f"{', '.join(unused[:3])}{' ...' if len(unused) > 3 else ''} are not used. "
                "To use other embeddings, run with a new --workdir or --project-name."
            )

    def _auto_detect_h5ad_files(self):
        """Auto-detect h5ad files from latent directory"""
        if self.find_latent_metadata_path.exists():
            import yaml

            with open(self.find_latent_metadata_path) as f:
                find_latent_metadata = yaml.safe_load(f)
            self.sample_h5ad_dict = OrderedDict(
                {
                    sample_name: Path(latent_file)
                    for sample_name, latent_file in find_latent_metadata["outputs"][
                        "latent_files"
                    ].items()
                }
            )
            # assert all files exist
            for sample_name, latent_file in self.sample_h5ad_dict.items():
                if not latent_file.exists():
                    raise FileNotFoundError(
                        f"Latent file not found for sample '{sample_name}': {latent_file}"
                    )
            logger.info(
                f"Auto-detected {len(self.sample_h5ad_dict)} samples from find_latent_metadata_path: {self.find_latent_metadata_path}"
            )

    def _setup_and_validate_fields(self):
        """Set up required/optional fields and validate h5ad structure"""
        # Define required fields
        required_fields = {
            "latent_representation_cell": (
                "obsm",
                self.latent_representation_cell,
                "Latent representation of cell identity",
            ),
            "spatial_key": ("obsm", self.spatial_key, "Spatial key"),
        }

        # Add annotation as required if provided
        if self.annotation:
            required_fields["annotation"] = ("obs", self.annotation, "Annotation")

        # Add niche representation as required if provided
        if self.latent_representation_niche:
            required_fields["latent_representation_niche"] = (
                "obsm",
                self.latent_representation_niche,
                "Latent representation of spatial niche",
            )

        # Add High_quality as required if find_neighbor_within_high_quality is enabled
        if self.high_quality_neighbor_filter:
            required_fields["High_quality"] = (
                "obs",
                "High_quality",
                "High quality cell indicator",
            )

        # Validate h5ad structure
        validate_h5ad_structure(self.sample_h5ad_dict, required_fields)

    def _configure_dataset_parameters(self):
        """Configure parameters based on dataset type"""
        self.min_cells_per_type = (
            self.homogeneous_neighbors
            if self.min_cells_per_type is None
            else min(self.min_cells_per_type, self.homogeneous_neighbors)
        )

        if self.dataset_type == DatasetType.SPATIAL_2D:
            self._configure_spatial_2d()
        elif self.dataset_type == DatasetType.SPATIAL_3D:
            self._configure_spatial_3d()
        elif self.dataset_type == DatasetType.SCRNA_SEQ:
            self._configure_scrna_seq()

    def _configure_spatial_2d(self):
        """Configure parameters for spatial 2D datasets"""
        # spatial2D can have multiple slices but doesn't search across them
        if self.n_adjacent_slices != 0:
            self.n_adjacent_slices = 0
            self.adjacent_slice_spatial_neighbors = 0
            logger.info(
                "Dataset type is spatial2D. This will only search homogeneous neighbors within each 2D slice (no cross-slice search). Setting adjacent_slices=0."
            )
        self.adjacent_slice_spatial_neighbors = 0
        self.fix_cross_slice_homogenous_neighbors = False

        if self.latent_representation_niche is None:
            logger.warning(
                "latent_representation_niche is not provided. Spatial domain similarity will not be used."
            )

        if self.homogeneous_neighbors > self.spatial_neighbors:
            raise ValueError(
                f"homogeneous_neighbors ({self.homogeneous_neighbors}) must be <= "
                f"spatial_neighbors ({self.spatial_neighbors}) for spatial2D datasets"
            )

    def _configure_spatial_3d(self):
        """Configure parameters for spatial 3D datasets"""
        if self.n_adjacent_slices < 0:
            raise ValueError("n_adjacent_slices must be nonnegative")
        if self.n_adjacent_slices == 0:
            self.adjacent_slice_spatial_neighbors = 0
            self.fix_cross_slice_homogenous_neighbors = False
            if self.homogeneous_neighbors > self.spatial_neighbors:
                raise ValueError("homogeneous_neighbors must be <= spatial_neighbors")
            return

        if self.latent_representation_niche is None:
            logger.warning(
                "latent_representation_niche is not provided. Spatial domain similarity will not be used."
            )

        if self.adjacent_slice_spatial_neighbors > self.spatial_neighbors:
            raise ValueError(
                f"adjacent_slice_neighbors ({self.adjacent_slice_spatial_neighbors}) must be "
                f"<= spatial_neighbors ({self.spatial_neighbors})"
            )
        if self.homogeneous_neighbors > self.adjacent_slice_spatial_neighbors:
            raise ValueError(
                f"homogeneous_neighbors ({self.homogeneous_neighbors}) must be <= "
                f"adjacent_slice_neighbors ({self.adjacent_slice_spatial_neighbors})"
            )

        if not math.isfinite(self.cross_slice_z_scale) or self.cross_slice_z_scale <= 0:
            raise ValueError("cross_slice_z_scale must be positive for spatial3D datasets")

        if self.cross_slice_selection_mode == CrossSliceSelectionMode.PHYSICAL_WINDOW:
            if (
                self.cross_slice_max_distance is None
                or not math.isfinite(self.cross_slice_max_distance)
                or self.cross_slice_max_distance <= 0
            ):
                raise ValueError(
                    "cross_slice_max_distance must be positive when "
                    "cross_slice_selection_mode='physical_window'"
                )
        n_slices = 1 + self.n_adjacent_slices  # only focal + above slices
        if n_slices > len(self.sample_h5ad_dict):
            raise ValueError(
                f"3D cross-slice search requires at least {n_slices} slices (1 focal + "
                f"{self.n_adjacent_slices} above or below). Only "
                f"{len(self.sample_h5ad_dict)} samples were provided."
            )

        if self.cross_slice_selection_mode == CrossSliceSelectionMode.PHYSICAL_WINDOW:
            logger.info(
                "Dataset type is spatial3D, selecting at most %s cross-sections within "
                "a physical z distance of %s (z scale=%s)",
                2 * self.n_adjacent_slices,
                self.cross_slice_max_distance,
                self.cross_slice_z_scale,
            )
        else:
            logger.info(
                f"Dataset type is spatial3D, using adjacent_slices={self.n_adjacent_slices} for cross-slice search"
            )
        logger.info(
            "The Z axis order of slices is determined by the h5ad input order. Currently, the order is: "
        )
        logger.info(f"{' -> '.join(list(self.sample_h5ad_dict.keys()))}")

        homogeneous_neighbors = self.homogeneous_neighbors
        n_adjacent_slices = self.n_adjacent_slices
        # Check if we should use fix number of homogeneous neighbors per slice
        if self.cross_slice_marker_score_strategy in [
            MarkerScoreCrossSliceStrategy.PER_SLICE_POOL,
            MarkerScoreCrossSliceStrategy.HIERARCHICAL_POOL,
        ]:
            self.fix_cross_slice_homogenous_neighbors = True
            logger.info(
                f"Using {self.cross_slice_marker_score_strategy.value} strategy with fixed number of homogeneous neighbors per adjacent slice: {self.homogeneous_neighbors} per slice."
            )

        elif self.cross_slice_marker_score_strategy == MarkerScoreCrossSliceStrategy.GLOBAL_POOL:
            logger.info(
                "Using global_pool strategy, will select top homogeneous neighbors from all adjacent slices based on similarity scores. Each adjacent slice can contribute variable number of homogeneous neighbors."
            )

        logger.info(
            f"Each focal cell will select {homogeneous_neighbors * (1 + 2 * n_adjacent_slices) = } total homogeneous neighbors across {(1 + 2 * n_adjacent_slices) = } slices."
        )

    def _configure_scrna_seq(self):
        """Configure parameters for scRNA-seq datasets"""
        self.n_adjacent_slices = 0
        self.spatial_key = None
        self.latent_representation_niche = None


def rank_checkpoint_present(rank_memmap_path: Path) -> bool:
    """Whether any file of the rank checkpoint (data or metadata) exists."""
    rank_memmap_path = Path(rank_memmap_path)
    return rank_memmap_path.exists() or rank_memmap_path.with_suffix(".meta.json").exists()


def check_latent2gene_done(config: LatentToGeneConfig) -> bool:
    """
    Check if latent2gene step is done by verifying validity of metadata and output files.
    """
    from gsMap.latent2gene.memmap_io import MemMapDense

    config._resolve_h5ad_inputs()

    expected_outputs = {
        "concatenated_latent_adata": Path(config.concatenated_latent_adata_path),
        "rank_memmap": Path(config.rank_memmap_path),
        "mean_frac": Path(config.mean_frac_path),
        "marker_scores": Path(config.marker_scores_memmap_path),
        "metadata": Path(config.latent2gene_metadata_path),
    }

    # The rank memmap is an intermediate checkpoint (as large as the marker scores) that
    # nothing downstream reads. A removed checkpoint does not invalidate final outputs;
    # an existing but incomplete one means a rerun is in progress.
    rank_removed = not rank_checkpoint_present(expected_outputs["rank_memmap"])
    if not all(
        p.exists()
        for key, p in expected_outputs.items()
        if not (key == "rank_memmap" and rank_removed)
    ):
        return False

    try:
        # Check rank memmap completion
        if not rank_removed:
            rank_memmap_complete, _ = MemMapDense.check_complete(expected_outputs["rank_memmap"])
            if not rank_memmap_complete:
                return False

        # Check marker scores memmap completion
        marker_scores_complete, _ = MemMapDense.check_complete(expected_outputs["marker_scores"])
        if not marker_scores_complete:
            return False

        # Check metadata
        with open(expected_outputs["metadata"]) as f:
            metadata = yaml.safe_load(f)

        if not isinstance(metadata, dict) or "outputs" not in metadata:
            return False

        return True
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Error checking latent2gene results: {e}")
        return False
