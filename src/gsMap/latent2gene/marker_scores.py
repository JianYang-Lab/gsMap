"""
Marker score calculation using homogeneous neighbors
Implements weighted geometric mean calculation in log space with JAX acceleration
"""

import gc
import json
import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from jax import jit

# Progress bar imports
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from gsMap.config import DatasetType, MarkerScoreCrossSliceStrategy
from gsMap.logging_config import is_interactive_output
from gsMap.utils.matrix_io import MatrixRowReader

from .connectivity import ConnectivityMatrixBuilder
from .marker_layout import MARKER_SCORE_LAYOUT_VERSION, SequentialMarkerRowWriter
from .memmap_io import (
    MemMapDense,
    ParallelMarkerScoreWriter,
    set_memmap_advice,
)
from .row_ordering_jax import optimize_row_order_jax, optimize_spatial_row_order
from .window_cache import (
    AdjacentRankWindowIndex,
    RankWindow,
    RankWindowMapping,
    independent_rank_window_mapping,
    plan_rank_windows,
)

# Marker scores are persisted as float16. Saturate the transformed result before
# that cast so strongly enriched genes remain finite instead of becoming +inf.
_FLOAT16_MARKER_SCORE_MAX = float(np.finfo(np.float16).max)
_MIB = 1024**2
_MARKER_BATCH_SIZE_CHOICES = (10, 25, 50, 100, 200)
_MAX_CELL_TYPE_RANK_CACHE_BYTES = 2 * 1024**3
_CELL_TYPE_CACHE_READ_CHUNK_ROWS = 500
_RANK_WINDOW_UPDATE_BUCKET_ROWS = 4096
_TARGET_CACHED_BATCH_BYTES = 128 * _MIB
_TARGET_A100_CACHED_BATCH_BYTES = 512 * _MIB


logger = logging.getLogger(__name__)


def _automatic_cached_marker_batch_size(
    genes: int,
    rank_dtype: np.dtype,
    neighbors_per_cell: int,
    device_kind: str,
) -> int:
    """Select a fixed-shape cached batch from measured device throughput."""
    if genes < 1 or neighbors_per_cell < 1:
        raise ValueError("Gene and neighbor counts must be positive")
    target_bytes = (
        _TARGET_A100_CACHED_BATCH_BYTES
        if "a100" in device_kind.casefold()
        else _TARGET_CACHED_BATCH_BYTES
    )
    bytes_per_cell = genes * np.dtype(rank_dtype).itemsize * neighbors_per_cell
    unconstrained = target_bytes // max(1, int(bytes_per_cell))
    eligible = [size for size in _MARKER_BATCH_SIZE_CHOICES if size <= unconstrained]
    return eligible[-1] if eligible else _MARKER_BATCH_SIZE_CHOICES[0]


def normalize_float16_rows(embedding: np.ndarray, source: np.ndarray) -> np.ndarray:
    """L2-normalize float16 rows; rows whose float16 norm overflows use float32.

    A float16 norm overflows above ~256, which would turn those rows into zeros. Every
    other row is normalized exactly as before.
    """
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = embedding / (norm + 1e-8)
    overflow = ~np.isfinite(norm[:, 0])
    if overflow.any():
        logger.warning(
            f"{int(overflow.sum()):,} cell embeddings have an L2 norm beyond float16 range; "
            "normalizing them in float32."
        )
        rows = np.asarray(source[overflow], dtype=np.float32)
        normalized[overflow] = (
            rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-8)
        ).astype(normalized.dtype)
    return normalized


def assign_rows_without_overflow(stored: np.ndarray, rows, values: np.ndarray) -> np.ndarray:
    """Assign ``values`` to ``stored[rows]``, widening the integer dtype if they do not fit.

    Cell types choose their own (possibly narrow) index dtype; assigning wider indices
    into a narrower shared array would wrap silently.
    """
    if (
        values.size
        and np.issubdtype(stored.dtype, np.integer)
        and int(values.max()) > np.iinfo(stored.dtype).max
    ):
        stored = stored.astype(np.result_type(stored.dtype, values.dtype))
    stored[rows] = values
    return stored


@dataclass
class PreparedRankWindow:
    """Host-side payload ready to update one bounded device rank cache."""

    window_index: int
    bounds: RankWindow
    mapping: RankWindowMapping
    host_ranks: np.ndarray | None
    update_slots: np.ndarray
    bucket_rows: int
    weights: np.ndarray
    remap_seconds: float
    read_seconds: float
    read_bytes: int


@partial(jit, donate_argnums=(0,))
def _update_rank_cache_prefix_jax(
    rank_cache: jnp.ndarray,
    rank_rows: jnp.ndarray,
) -> jnp.ndarray:
    """Replace a compact prefix while allowing JAX to reuse the cache buffer."""
    return jax.lax.dynamic_update_slice(rank_cache, rank_rows, (0, 0))


@partial(jit, donate_argnums=(0,))
def _update_rank_cache_slots_jax(
    rank_cache: jnp.ndarray,
    update_slots: jnp.ndarray,
    rank_rows: jnp.ndarray,
) -> jnp.ndarray:
    """Replace only newly required slots while retaining adjacent-window rows."""
    return rank_cache.at[update_slots].set(rank_rows)


@partial(jit, static_argnums=(2, 3, 6))
def compute_marker_scores_jax(
    log_ranks: jnp.ndarray,  # (B*N) × G matrix
    weights: jnp.ndarray,  # B × N weight matrix
    batch_size: int,
    num_neighbors: int,
    global_log_gmean: jnp.ndarray,  # G-dimensional vector
    global_expr_frac: jnp.ndarray,  # G-dimensional vector
    no_expression_fraction: bool = False,  # Skip expression fraction filtering if True
) -> jnp.ndarray:
    """
    JAX-accelerated marker score computation

    Returns:
        B × G marker scores
    """
    n_genes = log_ranks.shape[1]

    # Reshape to batch format
    log_ranks_3d = log_ranks.reshape(batch_size, num_neighbors, n_genes)

    # Compute weighted geometric mean in log space
    weight_sums = weights.sum(axis=1, keepdims=True)
    valid_neighborhood = weight_sums > 0
    weights = weights / jnp.where(valid_neighborhood, weight_sums, 1.0)
    weighted_log_mean = jnp.einsum("bn,bng->bg", weights, log_ranks_3d)

    # Calculate marker score
    marker_score = jnp.exp(weighted_log_mean - global_log_gmean)
    marker_score = jnp.where(marker_score < 1.0, 0.0, marker_score)

    # Apply expression fraction filter (only if not disabled)
    if not no_expression_fraction:
        # Compute expression fraction (mean of is_expressed across neighbors)
        # Treat min log rank as non-expressed
        is_expressed = log_ranks_3d != log_ranks_3d.min(axis=-1, keepdims=True)

        # Create mask for valid neighbors (where weights > 0)
        valid_mask = weights > 0  # Shape: (batch_size, num_neighbors)

        # Apply mask and compute mean only for valid neighbors
        is_expressed_masked = jnp.where(valid_mask[:, :, None], is_expressed, 0)
        valid_counts = valid_mask.sum(axis=1, keepdims=True)  # Count of valid neighbors per cell

        # Compute mean only over valid neighbors (avoid division by zero)
        expr_frac = jnp.where(
            valid_counts > 0,
            is_expressed_masked.astype(jnp.float16).sum(axis=1) / valid_counts,
            0.0,
        )

        frac_mask = expr_frac > global_expr_frac
        marker_score = jnp.where(frac_mask, marker_score, 0.0)

    marker_score = jnp.minimum(jnp.exp(marker_score**1.5) - 1.0, _FLOAT16_MARKER_SCORE_MAX)
    # A fully filtered neighborhood carries no evidence. In particular, do not
    # let an all-zero weight vector become NaN or a non-zero score when the
    # expression-fraction filter is disabled.
    marker_score = jnp.where(valid_neighborhood, marker_score, 0.0)

    # Return as float16 for memory efficiency
    return marker_score.astype(jnp.float16)


@partial(jit, static_argnums=(2, 3, 4, 7))
def compute_marker_scores_3d_hierarchical_pool_jax(
    log_ranks: jnp.ndarray,  # (B*N) × G matrix where N = n_slices * num_homogeneous_per_slice
    weights: jnp.ndarray,  # B × N weight matrix
    batch_size: int,
    n_slices: int,
    num_homogeneous_per_slice: int,
    global_log_gmean: jnp.ndarray,  # G-dimensional vector
    global_expr_frac: jnp.ndarray,  # G-dimensional vector
    no_expression_fraction: bool = False,  # Skip expression fraction filtering if True
) -> jnp.ndarray:
    """
    JAX-accelerated marker score computation with hierarchical pooling for 3D spatial data.
    Computes marker scores independently for each slice and takes the average.

    Args:
        log_ranks: Flattened log ranks (batch_size * total_neighbors, n_genes)
        weights: Flattened weights (batch_size, total_neighbors)
        batch_size: Number of cells in batch
        n_slices: Number of slices (1 + 2 * n_adjacent_slices)
        num_homogeneous_per_slice: Number of homogeneous neighbors per slice
        global_log_gmean: Global log geometric mean
        global_expr_frac: Global expression fraction

    Returns:
        (batch_size, n_genes) marker scores using average cross slices
    """
    n_genes = log_ranks.shape[1]
    # Reshape to separate slices: (batch_size, n_slices, num_homogeneous_per_slice, n_genes)
    log_ranks_4d = log_ranks.reshape(batch_size, n_slices, num_homogeneous_per_slice, n_genes)

    # Reshape weights: (batch_size, n_slices, num_homogeneous_per_slice)
    weights_3d = weights.reshape(batch_size, n_slices, num_homogeneous_per_slice)

    # Normalize weights within each slice (sum to 1 along num_homogeneous_per_slice axis)
    weights_sum = weights_3d.sum(axis=2, keepdims=True)  # Shape: (batch_size, n_slices, 1)
    weights_normalized = weights_3d / jnp.where(weights_sum > 0, weights_sum, 1.0)

    # Compute weighted geometric mean in log space for each slice
    # Result: (batch_size, n_slices, n_genes)
    weighted_log_mean = jnp.einsum("bsn,bsng->bsg", weights_normalized, log_ranks_4d)

    # Calculate marker score for each slice
    marker_score_per_slice = jnp.exp(weighted_log_mean - global_log_gmean[None, None, :])
    marker_score_per_slice = jnp.where(marker_score_per_slice < 1.0, 0.0, marker_score_per_slice)

    # Apply expression fraction filter for each slice (only if not disabled)
    if not no_expression_fraction:
        # Compute expression fraction for each slice
        # Treat min log rank as non-expressed
        min_log_rank = log_ranks_4d.min(axis=-1, keepdims=True)
        is_expressed = log_ranks_4d != min_log_rank

        # Create mask for valid neighbors within each slice (where weights > 0)
        valid_mask = weights_3d > 0  # Shape: (batch_size, n_slices, num_homogeneous_per_slice)

        # Apply mask and compute mean only for valid neighbors within each slice
        is_expressed_masked = jnp.where(valid_mask[:, :, :, None], is_expressed, 0)
        valid_counts = valid_mask.sum(axis=2, keepdims=True)  # Count of valid neighbors per slice

        # Compute mean only over valid neighbors (avoid division by zero)
        # Result: (batch_size, n_slices, n_genes)
        # valid_counts has shape (batch_size, n_slices, 1), need to broadcast properly
        expr_frac = jnp.where(
            valid_counts > 0,
            is_expressed_masked.astype(jnp.float16).sum(axis=2) / valid_counts,
            0.0,
        )

        frac_mask = expr_frac > global_expr_frac[None, None, :]
        marker_score_per_slice = jnp.where(frac_mask, marker_score_per_slice, 0.0)

    marker_score_per_slice = jnp.exp(marker_score_per_slice**1.5) - 1.0

    # Calculate median across slices instead of max pooling to reduce noise
    # Result: (batch_size, n_genes)

    # Get the invalid slice mask before calculating median.
    # These slices are invalid could be:
    # 1. no neighbors from this slice for the outermost slices
    # 2. the adjacent slices that do not have enough high quality cells which has been filtered out
    invalid_slice_mask = (weights_sum == 0).squeeze(-1)  # Shape: (batch_size, n_slices)

    # Set invalid slice scores to NaN, then calculate the median
    # NaN values will be ignored by jnp.nanmedian
    marker_score_per_slice = jnp.where(
        invalid_slice_mask[:, :, None],  # Broadcast to (batch_size, n_slices, n_genes)
        jnp.nan,
        marker_score_per_slice,
    )

    # Calculate median across slices (axis=1), ignoring NaN values
    marker_score = jnp.nanmean(marker_score_per_slice, axis=1)
    # marker_score = jnp.nanmedian(marker_score_per_slice, axis=1)

    # Handle cases where all slices are invalid (all NaN) - set to 0
    marker_score = jnp.where(jnp.isnan(marker_score), 0.0, marker_score)
    # Pool in float32 before saturation: an individual slice may exceed float16
    # while its cross-slice mean remains representable.
    marker_score = jnp.minimum(marker_score, _FLOAT16_MARKER_SCORE_MAX)

    # Return as float16 for memory efficiency
    return marker_score.astype(jnp.float16)


@partial(jit, static_argnums=(3, 4, 7))
def compute_marker_scores_from_cache_jax(
    rank_cache: jnp.ndarray,
    local_neighbor_indices: jnp.ndarray,
    weights: jnp.ndarray,
    batch_size: int,
    num_neighbors: int,
    global_log_gmean: jnp.ndarray,
    global_expr_frac: jnp.ndarray,
    no_expression_fraction: bool = False,
) -> jnp.ndarray:
    """Gather cached rows and calculate standard marker scores in one JIT."""
    log_ranks = rank_cache[local_neighbor_indices.reshape(-1)]
    return compute_marker_scores_jax(
        log_ranks,
        weights,
        batch_size,
        num_neighbors,
        global_log_gmean,
        global_expr_frac,
        no_expression_fraction,
    )


@partial(jit, static_argnums=(3, 4, 5, 8))
def compute_marker_scores_3d_from_cache_jax(
    rank_cache: jnp.ndarray,
    local_neighbor_indices: jnp.ndarray,
    weights: jnp.ndarray,
    batch_size: int,
    n_slices: int,
    num_homogeneous_per_slice: int,
    global_log_gmean: jnp.ndarray,
    global_expr_frac: jnp.ndarray,
    no_expression_fraction: bool = False,
) -> jnp.ndarray:
    """Gather cached rows and calculate hierarchical 3D scores in one JIT."""
    log_ranks = rank_cache[local_neighbor_indices.reshape(-1)]
    return compute_marker_scores_3d_hierarchical_pool_jax(
        log_ranks,
        weights,
        batch_size,
        n_slices,
        num_homogeneous_per_slice,
        global_log_gmean,
        global_expr_frac,
        no_expression_fraction,
    )


class MarkerScoreCalculator:
    """Main class for calculating marker scores"""

    def __init__(self, config):
        """
        Initialize with configuration

        Args:
            config: LatentToGeneConfig object
        """
        self.config = config
        self.connectivity_builder = ConnectivityMatrixBuilder(config)
        self.console = Console()
        self.show_rich_output = is_interactive_output()

    def load_global_stats(self, mean_frac_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load pre-calculated global geometric mean and expression fraction from parquet"""

        logger.info("Loading global statistics from parquet...")
        parquet_path = Path(mean_frac_path)

        if not parquet_path.exists():
            raise FileNotFoundError(f"Global stats file not found: {parquet_path}")

        # Load the dataframe
        mean_frac_df = pd.read_parquet(parquet_path)

        # Extract global log geometric mean and expression fraction
        global_log_gmean = mean_frac_df["G_Mean"].values.astype(np.float32)
        global_expr_frac = mean_frac_df["frac"].values.astype(np.float32)

        logger.info(f"Loaded global stats for {len(global_log_gmean)} genes")

        return global_log_gmean, global_expr_frac

    def _display_input_summary(self, adata, cell_types, n_cells, n_genes):
        """Display summary of input data and cell types"""
        dataset_type = getattr(self.config.dataset_type, "value", self.config.dataset_type)
        ct_counts = (
            adata.obs[self.config.annotation].value_counts()
            if self.config.annotation is not None
            else pd.Series({"all": adata.n_obs})
        )
        if not self.show_rich_output:
            logger.info(
                f"Marker score input: {n_cells:,} cells, {n_genes:,} genes, "
                f"{len(cell_types):,} cell types, dataset={dataset_type}, "
                f"annotation={self.config.annotation}"
            )
            logger.debug(f"Cell type counts: {ct_counts.to_dict()}")
            return

        table = Table(
            title="[bold cyan]Marker Score Input Summary[/bold cyan]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Property", style="dim")
        table.add_column("Value", style="green")

        table.add_row("Total Cells", str(n_cells))
        table.add_row("Total Genes", str(n_genes))
        table.add_row("Cell Types", str(len(cell_types)))
        table.add_row("Dataset Type", str(dataset_type))
        table.add_row("Annotation Key", str(self.config.annotation))

        self.console.print(table)

        # Display cell type breakdown
        ct_table = Table(
            title="[bold cyan]Cell Type Breakdown[/bold cyan]",
            show_header=True,
            header_style="bold blue",
        )
        ct_table.add_column("Cell Type", style="dim")
        ct_table.add_column("Count", justify="right")

        for ct in cell_types:
            count = ct_counts.get(ct, 0)
            style = "green" if count >= self.config.min_cells_per_type else "red"
            ct_table.add_row(ct, f"[{style}]{count}[/{style}]")

        self.console.print(ct_table)

    def _load_input_data(
        self,
        adata_path: str,
        rank_memmap_path: str | Path | MemMapDense,
        mean_frac_path: str,
    ) -> tuple[ad.AnnData, MemMapDense, np.ndarray, np.ndarray, int, int, np.ndarray]:
        """Load input data: AnnData, rank memory map, global statistics, and high quality mask

        Returns:
            Tuple of (adata, rank_memmap, global_log_gmean, global_expr_frac, n_cells, n_genes, high_quality_mask)
        """
        # Load concatenated AnnData
        logger.info(f"Loading concatenated AnnData from {adata_path}")
        if not Path(adata_path).exists():
            raise FileNotFoundError(f"Concatenated AnnData not found: {adata_path}")
        adata = sc.read_h5ad(adata_path)

        # Load pre-calculated global statistics
        global_log_gmean, global_expr_frac = self.load_global_stats(mean_frac_path)

        # Open rank memory map and get dimensions
        if isinstance(rank_memmap_path, MemMapDense):
            rank_memmap = rank_memmap_path
            logger.info(f"Reusing generated rank memory map at {rank_memmap.data_path}")
        else:
            rank_memmap_path = Path(rank_memmap_path)
            meta_path = rank_memmap_path.with_suffix(".meta.json")
            with open(meta_path) as f:
                meta = json.load(f)

            rank_memmap = MemMapDense(
                path=rank_memmap_path,
                shape=tuple(meta["shape"]),
                dtype=np.dtype(meta["dtype"]),
                mode="r",
                tmp_dir=self.config.memmap_tmp_dir,
            )
            logger.info(f"Opened rank memory map from {rank_memmap_path}")
        try:
            n_cells = adata.n_obs
            n_cells_rank = rank_memmap.shape[0]
            n_genes = rank_memmap.shape[1]

            logger.info(f"AnnData dimensions: {n_cells} cells × {adata.n_vars} genes")
            logger.info(f"Rank MemMap dimensions: {n_cells_rank} cells × {n_genes} genes")

            # Cells and genes must match exactly because rank rows/columns are
            # positional. Use runtime checks rather than assertions, which disappear
            # when Python is invoked with optimization enabled.
            if n_cells != n_cells_rank:
                raise ValueError(
                    f"Cell count mismatch: AnnData has {n_cells} cells, Rank MemMap has "
                    f"{n_cells_rank} cells. This indicates the filtering was not applied "
                    "consistently during rank calculation."
                )
            if adata.n_vars != n_genes:
                raise ValueError(
                    f"Gene count mismatch: AnnData has {adata.n_vars} genes, Rank MemMap has "
                    f"{n_genes} genes."
                )
            if global_log_gmean.shape != (n_genes,) or global_expr_frac.shape != (n_genes,):
                raise ValueError(
                    "Global-statistic dimensions do not match the rank matrix: "
                    f"G_Mean={global_log_gmean.shape}, frac={global_expr_frac.shape}, "
                    f"expected=({n_genes},)"
                )

            # Load high quality mask based on configuration
            if self.config.high_quality_neighbor_filter:
                if "High_quality" not in adata.obs.columns:
                    raise ValueError(
                        "High_quality column not found in AnnData obs. Please ensure QC was "
                        "applied during find_latent_representation step."
                    )
                high_quality_mask = adata.obs["High_quality"].values.astype(bool)
                logger.info(
                    f"Loaded high quality mask: {high_quality_mask.sum()}/"
                    f"{len(high_quality_mask)} cells marked as high quality"
                )
            else:
                # Create all-True mask when high quality filtering is disabled
                high_quality_mask = np.ones(n_cells, dtype=bool)
                logger.info("High quality filtering disabled - using all cells")

            return (
                adata,
                rank_memmap,
                global_log_gmean,
                global_expr_frac,
                n_cells,
                n_genes,
                high_quality_mask,
            )
        except BaseException:
            # Tuple assignment in calculate_marker_scores() has not completed yet,
            # so its outer finally block cannot see this handle. Close it here to
            # release the mapping and any staged scratch copy on validation errors.
            try:
                rank_memmap.close()
            except Exception as cleanup_error:  # noqa: BLE001
                logger.error(f"Failed to close invalid rank memmap: {cleanup_error}")
            raise

    def _prepare_embeddings(
        self, adata: ad.AnnData
    ) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray | None]:
        """Prepare and normalize embeddings based on dataset type

        Returns:
            Tuple of (coords, emb_niche, emb_indv, slice_ids)
        """
        logger.info("Loading shared data structures...")

        coords = None
        emb_niche = None
        slice_ids = None

        if self.config.dataset_type in {DatasetType.SPATIAL_2D, DatasetType.SPATIAL_3D}:
            spatial_key = self.config.spatial_key
            if spatial_key not in adata.obsm:
                raise KeyError(f"Spatial coordinates '{spatial_key}' are missing from adata.obsm")
            from gsMap.config.spatial import spatial_coordinates

            coords = spatial_coordinates(
                adata.obsm[spatial_key], self.config.dataset_type, n_obs=adata.n_obs
            )

            if "slice_id" not in adata.obs.columns:
                raise KeyError("Spatial marker-score calculation requires adata.obs['slice_id']")
            if adata.obs["slice_id"].isna().any():
                raise ValueError("adata.obs['slice_id'] contains missing values")
            try:
                slice_ids = adata.obs["slice_id"].to_numpy().astype(np.int32)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "adata.obs['slice_id'] must contain integer section IDs"
                ) from error

            # Try to load niche embeddings if they exist
            if self.config.latent_representation_niche in adata.obsm:
                emb_niche = adata.obsm[self.config.latent_representation_niche]

        # Load cell embeddings for all dataset types
        emb_indv = adata.obsm[self.config.latent_representation_cell].astype(np.float16)

        # If the niche embedding is omitted, a normalized constant vector makes
        # niche cosine similarity neutral without a separate execution path.
        if emb_niche is None:
            logger.info("No niche embeddings found. Using dummy embeddings (all ones).")
            emb_niche = np.ones((emb_indv.shape[0], 1), dtype=np.float32)

        # Normalize embeddings
        logger.info("Normalizing embeddings...")

        # L2 normalize niche embeddings (always exists now)
        emb_niche_norm = np.linalg.norm(emb_niche, axis=1, keepdims=True)
        emb_niche = emb_niche / (emb_niche_norm + 1e-8)

        # L2 normalize individual embeddings
        emb_indv = normalize_float16_rows(
            emb_indv, adata.obsm[self.config.latent_representation_cell]
        )

        return coords, emb_niche, emb_indv, slice_ids

    def _get_cell_types(self, adata: ad.AnnData) -> np.ndarray:
        """Get cell types from annotation key

        Returns:
            Array of unique cell types
        """
        annotation_key = self.config.annotation

        if annotation_key is not None:
            if annotation_key not in adata.obs.columns:
                raise KeyError(f"Annotation key '{annotation_key}' not found in adata.obs")
            # Get unique cell types, excluding NaN values
            cell_types = adata.obs[annotation_key].dropna().unique()

            # Check if there are any NaN values and handle them
            nan_count = adata.obs[annotation_key].isna().sum()
            if nan_count > 0:
                logger.warning(
                    f"Found {nan_count} cells with NaN annotation in '{annotation_key}', these will be skipped"
                )
        else:
            logger.info("No annotation configured; processing all cells as one type")
            cell_types = ["all"]

        logger.info(f"Processing {len(cell_types)} cell types")
        return cell_types

    @staticmethod
    def _clear_homogeneous_outputs(adata: ad.AnnData) -> None:
        """Remove match outputs that must not survive a recalculation.

        Rows with a missing annotation are intentionally skipped by marker-score
        calculation. Clearing the complete output fields first prevents those
        rows from retaining match indices from a previous ordinal/physical run.
        """

        obsm_keys = (
            "gsMap_homo_indices",
            "gsMap_homo_cell_sims",
            "gsMap_homo_niche_sims",
        )
        obs_keys = (
            "gsMap_homo_neighbor_count",
            "gsMap_homo_cell_sims_median",
            "gsMap_homo_niche_sims_median",
        )
        removed = []
        for key in obsm_keys:
            if key in adata.obsm:
                del adata.obsm[key]
                removed.append(f"obsm/{key}")
        for key in obs_keys:
            if key in adata.obs:
                del adata.obs[key]
                removed.append(f"obs/{key}")
        if removed:
            logger.info(
                f"Cleared stale homogeneous-match outputs before recalculation: "
                f"{', '.join(removed)}"
            )

    def _initialize_pipeline(
        self,
        rank_memmap: MemMapDense,
        output_memmap: MemMapDense,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray,
    ) -> None:
        """Resolve the single bounded-cache execution engine."""
        self.cross_slice_strategy = None
        self.n_slices = 1
        self.num_homogeneous_per_slice = self.config.homogeneous_neighbors

        if (
            self.config.dataset_type == DatasetType.SPATIAL_3D
            and self.config.n_adjacent_slices > 0
            and self.config.cross_slice_marker_score_strategy
            in [
                MarkerScoreCrossSliceStrategy.PER_SLICE_POOL,
                MarkerScoreCrossSliceStrategy.HIERARCHICAL_POOL,
            ]
        ):
            self.cross_slice_strategy = self.config.cross_slice_marker_score_strategy
            self.n_slices = 1 + 2 * self.config.n_adjacent_slices

        neighbors_per_cell = self.n_slices * self.num_homogeneous_per_slice
        device = jax.devices()[0]
        self.marker_batch_size = self.config.mkscore_batch_size or (
            _automatic_cached_marker_batch_size(
                rank_memmap.shape[1],
                rank_memmap.dtype,
                neighbors_per_cell,
                getattr(device, "device_kind", ""),
            )
        )
        self.rank_read_workers = self.config.rank_read_workers
        self.compute_workers = self.config.mkscore_compute_workers
        self.compute_pending_limit = max(
            1,
            self.compute_workers * self.config.compute_input_queue_size,
        )
        self.rank_values = rank_memmap.memmap
        self.rank_shape = rank_memmap.shape
        self.rank_dtype = np.dtype(rank_memmap.dtype)
        set_memmap_advice(self.rank_values, "random")
        self.global_log_gmean = jax.device_put(global_log_gmean)
        self.global_expr_frac = jax.device_put(global_expr_frac)
        self.global_log_gmean.block_until_ready()
        self.global_expr_frac.block_until_ready()
        self.no_expression_fraction = self.config.no_expression_fraction
        self.cell_type_rank_cache_limit_bytes = _MAX_CELL_TYPE_RANK_CACHE_BYTES
        self.rank_window_prefetch = getattr(device, "platform", "cpu") != "cpu"
        self.rank_window_adjacent_retention = self.rank_window_prefetch

        self.writer = ParallelMarkerScoreWriter(
            output_memmap,
            num_workers=self.config.mkscore_write_workers,
            queue_size=self.config.writer_queue_size,
        )
        cache_gib = self.cell_type_rank_cache_limit_bytes / 1024**3
        logger.info(
            f"Resolved bounded rank-cache engine: batch={self.marker_batch_size}, "
            f"rank readers={self.rank_read_workers}, compute={self.compute_workers}, "
            f"writers={self.config.mkscore_write_workers}, "
            f"pending={self.compute_pending_limit}, "
            f"writer queue={self.config.writer_queue_size}, advice=random, "
            f"cap={cache_gib:.1f} GiB, "
            f"multi-window prefetch={self.rank_window_prefetch}, "
            f"retention={self.rank_window_adjacent_retention}"
        )

    def _calculate_marker_scores_with_rank_cache(
        self,
        cell_type: str,
        neighbor_indices: np.ndarray,
        neighbor_weights: np.ndarray,
        output_rows: slice | np.ndarray,
    ) -> None:
        """Process every cell type through one bounded rank-cache engine.

        A fitting type becomes one right-sized cache window. A larger type is
        split into capacity-maximal windows; GPU execution then retains rows
        shared by adjacent windows and prefetches the next host slab. CPU uses
        independent sequential windows because real-data prefetch improved its
        throughput by only about one percent.
        """
        n_focal_cells = neighbor_indices.shape[0]
        if isinstance(output_rows, slice):
            output_start = 0 if output_rows.start is None else output_rows.start
            output_stop = output_rows.stop
            output_step = 1 if output_rows.step is None else output_rows.step
            if (
                output_stop is None
                or output_step != 1
                or output_stop - output_start != n_focal_cells
            ):
                raise ValueError("Contiguous marker output slice does not match focal cells")
        else:
            output_rows = np.asarray(output_rows)
            if output_rows.ndim != 1 or output_rows.size != n_focal_cells:
                raise ValueError("Marker output rows must match the number of focal cells")

        row_bytes = int(self.rank_shape[1]) * self.rank_dtype.itemsize
        max_cache_rows = self.cell_type_rank_cache_limit_bytes // row_bytes

        planning_started = time.perf_counter()
        windows = plan_rank_windows(
            neighbor_indices,
            max_cache_rows=max_cache_rows,
            global_rows=self.rank_shape[0],
        )
        planning_seconds = time.perf_counter() - planning_started
        multi_window = len(windows) > 1
        cache_rows = max_cache_rows if multi_window else windows[0].unique_rows
        retain_adjacent = self.rank_window_adjacent_retention and multi_window
        prefetch = self.rank_window_prefetch and multi_window
        rank_index = (
            AdjacentRankWindowIndex(cache_rows, self.rank_shape[0]) if retain_adjacent else None
        )
        total_batches = sum(
            (window.end - window.start + self.marker_batch_size - 1) // self.marker_batch_size
            for window in windows
        )
        window_suffix = "s" if multi_window else ""
        cache_gib = cache_rows * row_bytes / 1024**3
        logger.info(
            f"Using {len(windows)} bounded rank-cache window{window_suffix} for "
            f"{cell_type} (device cache {cache_gib:.2f} GiB, planned in "
            f"{planning_seconds:.2f}s); prefetch={prefetch}, "
            f"adjacent retention={retain_adjacent}"
        )

        self.writer.reset_for_cell_type(cell_type)

        device_cache = None
        if multi_window:
            device_cache = jnp.zeros(
                (cache_rows, self.rank_shape[1]),
                dtype=self.rank_dtype,
            )
            device_cache.block_until_ready()
        batch_size = self.marker_batch_size
        read_workers = self.rank_read_workers
        read_memmap = self.rank_values
        next_batch_id = 0
        total_unique_rows = 0
        total_new_rows = 0
        total_retained_rows = 0
        total_read_seconds = 0.0
        total_read_bytes = 0
        total_remap_seconds = 0.0
        total_update_seconds = 0.0
        compute_future_wait_seconds = 0.0
        writer_queue_wait_seconds = 0.0

        def prepare_window(
            window_index: int,
            bounds: RankWindow,
            read_executor: ThreadPoolExecutor,
        ) -> PreparedRankWindow:
            remap_started = time.perf_counter()
            block_neighbors = neighbor_indices[bounds.start : bounds.end]
            mapping = (
                rank_index.map_window(block_neighbors)
                if rank_index is not None
                else independent_rank_window_mapping(block_neighbors)
            )
            remap_seconds = time.perf_counter() - remap_started
            if mapping.unique_rows.size != bounds.unique_rows:
                raise RuntimeError(
                    f"Rank-window planner/remapper disagreement in window {window_index}: "
                    f"{bounds.unique_rows} != {mapping.unique_rows.size}"
                )

            new_rows = int(mapping.rows_to_read.size)
            bucket_rows = new_rows
            if multi_window and new_rows:
                bucket_rows = min(
                    cache_rows,
                    (
                        (new_rows + _RANK_WINDOW_UPDATE_BUCKET_ROWS - 1)
                        // _RANK_WINDOW_UPDATE_BUCKET_ROWS
                        * _RANK_WINDOW_UPDATE_BUCKET_ROWS
                    ),
                )
            host_ranks = (
                np.empty(
                    (bucket_rows, self.rank_shape[1]),
                    dtype=self.rank_dtype,
                )
                if bucket_rows
                else None
            )
            chunks = [
                (
                    start,
                    min(
                        start + _CELL_TYPE_CACHE_READ_CHUNK_ROWS,
                        new_rows,
                    ),
                )
                for start in range(0, new_rows, _CELL_TYPE_CACHE_READ_CHUNK_ROWS)
            ]

            def read_chunk(chunk: tuple[int, int]) -> int:
                if host_ranks is None:
                    raise RuntimeError("Missing host rank-window buffer")
                start, end = chunk
                return rank_reader.read_into(
                    mapping.rows_to_read[start:end], host_ranks[start:end]
                )

            read_started = time.perf_counter()
            read_bytes = 0
            if chunks:
                read_bytes = sum(read_executor.map(read_chunk, chunks))
                if host_ranks is None:
                    raise RuntimeError("Missing populated host rank-window buffer")
                if bucket_rows > new_rows:
                    host_ranks[new_rows:] = host_ranks[new_rows - 1]
            read_seconds = time.perf_counter() - read_started

            update_slots = mapping.update_slots
            if bucket_rows > update_slots.size:
                padded_slots = np.empty(bucket_rows, dtype=np.int32)
                padded_slots[: update_slots.size] = update_slots
                padded_slots[update_slots.size :] = update_slots[-1]
                update_slots = padded_slots

            return PreparedRankWindow(
                window_index=window_index,
                bounds=bounds,
                mapping=mapping,
                host_ranks=host_ranks,
                update_slots=update_slots,
                bucket_rows=bucket_rows,
                weights=np.asarray(neighbor_weights[bounds.start : bounds.end]),
                remap_seconds=remap_seconds,
                read_seconds=read_seconds,
                read_bytes=read_bytes,
            )

        def update_device_cache(prepared: PreparedRankWindow) -> float:
            nonlocal device_cache
            if not prepared.bucket_rows:
                return 0.0
            if prepared.host_ranks is None:
                raise RuntimeError("Host rank-window buffer was already released")
            update_started = time.perf_counter()
            device_rows = jax.device_put(prepared.host_ranks)
            if device_cache is None:
                device_cache = device_rows
            elif retain_adjacent:
                device_slots = jax.device_put(prepared.update_slots)
                device_cache = _update_rank_cache_slots_jax(
                    device_cache,
                    device_slots,
                    device_rows,
                )
                del device_slots
            else:
                device_cache = _update_rank_cache_prefix_jax(device_cache, device_rows)
            device_cache.block_until_ready()
            update_seconds = time.perf_counter() - update_started
            prepared.host_ranks = None
            del device_rows
            return update_seconds

        def padded_batch(
            prepared: PreparedRankWindow,
            start: int,
            end: int,
        ) -> tuple[np.ndarray, np.ndarray, slice | np.ndarray, int]:
            actual_size = end - start
            local_neighbors = prepared.mapping.local_neighbor_indices[start:end]
            weights = prepared.weights[start:end]
            relative_start = prepared.bounds.start + start
            relative_end = prepared.bounds.start + end
            if isinstance(output_rows, slice):
                marker_start = output_start + relative_start
                output_indices: slice | np.ndarray = slice(
                    marker_start,
                    marker_start + actual_size,
                )
            else:
                output_indices = output_rows[relative_start:relative_end]
            if actual_size < batch_size:
                pad = batch_size - actual_size
                local_neighbors = np.concatenate(
                    (local_neighbors, np.repeat(local_neighbors[-1:], pad, axis=0))
                )
                weights = np.concatenate((weights, np.repeat(weights[-1:], pad, axis=0)))
            return local_neighbors, weights, output_indices, actual_size

        def compute_window(
            prepared: PreparedRankWindow,
            compute_executor: ThreadPoolExecutor,
            progress: Progress,
            progress_task: int,
        ) -> None:
            nonlocal next_batch_id
            cache_for_window = device_cache
            if cache_for_window is None:
                raise RuntimeError("Rank cache was not transferred to the device")
            focal_cells = prepared.bounds.end - prepared.bounds.start

            def calculate_batch(
                payload: tuple[int, int, int],
            ) -> tuple[int, np.ndarray, slice | np.ndarray]:
                batch_id, start, end = payload
                local_neighbors, weights, output_indices, actual_size = padded_batch(
                    prepared,
                    start,
                    end,
                )
                device_neighbors = jax.device_put(local_neighbors)
                device_weights = jax.device_put(weights)
                if self.cross_slice_strategy == MarkerScoreCrossSliceStrategy.HIERARCHICAL_POOL:
                    marker_scores = compute_marker_scores_3d_from_cache_jax(
                        cache_for_window,
                        device_neighbors,
                        device_weights,
                        batch_size,
                        self.n_slices,
                        self.num_homogeneous_per_slice,
                        self.global_log_gmean,
                        self.global_expr_frac,
                        self.no_expression_fraction,
                    )
                else:
                    marker_scores = compute_marker_scores_from_cache_jax(
                        cache_for_window,
                        device_neighbors,
                        device_weights,
                        batch_size,
                        neighbor_indices.shape[1],
                        self.global_log_gmean,
                        self.global_expr_frac,
                        self.no_expression_fraction,
                    )
                output = np.asarray(marker_scores, dtype=np.float16)[:actual_size]
                return batch_id, output, output_indices

            pending = deque()

            def finish_one() -> None:
                nonlocal compute_future_wait_seconds, writer_queue_wait_seconds
                wait_started = time.perf_counter()
                batch_id, marker_scores, output_indices = pending.popleft().result()
                compute_future_wait_seconds += time.perf_counter() - wait_started
                self.writer.check_errors()
                wait_started = time.perf_counter()
                self.writer.write_queue.put((batch_id, marker_scores, output_indices))
                writer_queue_wait_seconds += time.perf_counter() - wait_started
                progress.advance(progress_task)

            for start in range(0, focal_cells, batch_size):
                end = min(start + batch_size, focal_cells)
                payload = (next_batch_id, start, end)
                next_batch_id += 1
                pending.append(compute_executor.submit(calculate_batch, payload))
                if len(pending) >= self.compute_pending_limit:
                    finish_one()
            while pending:
                finish_one()

        started = time.perf_counter()
        # Some low-level callers construct this calculator without invoking
        # __init__; keep the cache engine independently usable in that case.
        progress_console = getattr(self, "console", None) or Console()
        show_rich_output = getattr(self, "show_rich_output", is_interactive_output())
        with (
            MatrixRowReader(read_memmap) as rank_reader,
            Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=progress_console,
                disable=not show_rich_output,
                refresh_per_second=10,
            ) as progress,
        ):
            task = progress.add_task(
                f"[bold]{cell_type} (rank cache)[/bold]",
                total=total_batches,
            )
            with (
                ThreadPoolExecutor(max_workers=read_workers) as read_executor,
                ThreadPoolExecutor(max_workers=self.compute_workers) as compute_executor,
            ):
                if prefetch:
                    with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
                        current = prepare_window(0, windows[0], read_executor)
                        next_future = None
                        for window_index, bounds in enumerate(windows):
                            if window_index:
                                if next_future is None:
                                    raise RuntimeError("Missing prefetched rank window")
                                current = next_future.result()
                            update_seconds = update_device_cache(current)
                            next_future = (
                                prefetch_executor.submit(
                                    prepare_window,
                                    window_index + 1,
                                    windows[window_index + 1],
                                    read_executor,
                                )
                                if window_index + 1 < len(windows)
                                else None
                            )
                            compute_window(current, compute_executor, progress, task)
                            total_unique_rows += current.mapping.unique_rows.size
                            total_new_rows += current.mapping.rows_to_read.size
                            total_retained_rows += current.mapping.retained_rows
                            total_read_seconds += current.read_seconds
                            total_read_bytes += current.read_bytes
                            total_remap_seconds += current.remap_seconds
                            total_update_seconds += update_seconds
                else:
                    for window_index, bounds in enumerate(windows):
                        current = prepare_window(window_index, bounds, read_executor)
                        update_seconds = update_device_cache(current)
                        compute_window(current, compute_executor, progress, task)
                        total_unique_rows += current.mapping.unique_rows.size
                        total_new_rows += current.mapping.rows_to_read.size
                        total_retained_rows += current.mapping.retained_rows
                        total_read_seconds += current.read_seconds
                        total_read_bytes += current.read_bytes
                        total_remap_seconds += current.remap_seconds
                        total_update_seconds += update_seconds

        drain_started = time.perf_counter()
        self.writer.write_queue.join()
        writer_drain_seconds = time.perf_counter() - drain_started
        self.writer.check_errors()
        completed = self.writer.get_completed_count()
        if completed != total_batches or next_batch_id != total_batches:
            raise RuntimeError(
                "Rank-cache marker-score path wrote/submitted "
                f"{completed}/{next_batch_id}/{total_batches} batches"
            )
        elapsed = time.perf_counter() - started
        cells_per_second = n_focal_cells / elapsed
        logger.info(
            f"Completed rank-cache marker scores for {cell_type}: {elapsed:.2f}s, "
            f"{cells_per_second:.0f} cells/s; unique/new/retained rows="
            f"{total_unique_rows}/{total_new_rows}/{total_retained_rows}, "
            f"remap/read/update={total_remap_seconds:.2f}/"
            f"{total_read_seconds:.2f}/{total_update_seconds:.2f}s"
        )
        logger.info(
            f"Marker pipeline waits for {cell_type}: "
            f"compute futures={compute_future_wait_seconds:.2f}s, "
            f"writer queue={writer_queue_wait_seconds:.2f}s, "
            f"writer drain={writer_drain_seconds:.2f}s; "
            f"output backend={getattr(self.writer, 'io_backend', 'in-memory')}. "
            "These are host wait times, not GPU kernel durations."
        )
        logger.info(
            f"Rank input for {cell_type}: {total_read_bytes / 1024**3:.2f} GiB read for "
            f"{total_new_rows * row_bytes / 1024**3:.2f} GiB selected; "
            f"{total_read_bytes / max(total_read_seconds, 1e-9) / 1024**2:.0f} MiB/s "
            "during cache loading (includes any page-cache hits)."
        )

        device_cache = None
        gc.collect()

    def _find_homogeneous_spots(
        self,
        adata: ad.AnnData,
        cell_type: str,
        annotation_key: str | None,
        coords: np.ndarray | None,
        emb_niche: np.ndarray,
        emb_indv: np.ndarray,
        slice_ids: np.ndarray | None,
        rank_shape: tuple[int, int],
        high_quality_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, int] | None:
        """
        Prepare batch data for a cell type

        Returns:
            Tuple of (neighbor_indices, cell_sims, niche_sims, cell_indices_sorted, n_cells)
            or None if cell type should be skipped
        """
        # Get cells of this type, excluding NaN values
        if annotation_key is None:
            cell_mask = np.ones(adata.n_obs, dtype=bool)
        else:
            cell_mask = (adata.obs[annotation_key] == cell_type) & (
                adata.obs[annotation_key].notna()
            )
        cell_indices = np.where(cell_mask)[0]
        n_cells = len(cell_indices)

        # Check minimum cells
        min_cells = self.config.min_cells_per_type
        if n_cells < min_cells:
            logger.warning(f"Skipping {cell_type}: only {n_cells} cells (min: {min_cells})")
            return None

        logger.info(f"Processing {cell_type}: {n_cells} cells")

        # Build connectivity matrix
        logger.info("Building connectivity matrix...")
        neighbor_indices, cell_sims, niche_sims = (
            self.connectivity_builder.build_connectivity_matrix(
                coords=coords,
                emb_niche=emb_niche,
                emb_indv=emb_indv,
                cell_mask=cell_mask,
                high_quality_mask=high_quality_mask,
                slice_ids=slice_ids,
                k_central=self.config.spatial_neighbors,
                k_adjacent=self.config.adjacent_slice_spatial_neighbors,
                n_adjacent_slices=self.config.n_adjacent_slices,
            )
        )
        gc.collect()

        # Validate neighbor indices
        if neighbor_indices.size == 0:
            raise ValueError(f"No homogeneous-neighbor slots were generated for {cell_type}")
        max_valid_idx = rank_shape[0] - 1
        min_neighbor_idx = int(neighbor_indices.min())
        max_neighbor_idx = int(neighbor_indices.max())
        if min_neighbor_idx < -1 or max_neighbor_idx > max_valid_idx:
            raise IndexError(
                "Homogeneous-neighbor indices are outside the rank matrix: "
                f"observed=[{min_neighbor_idx}, {max_neighbor_idx}], "
                f"allowed=[-1, {max_valid_idx}]"
            )

        # A missing cross-section slot is represented by index -1 and weight 0.
        # Substitute the focal row before rank access; its rank is multiplied by
        # zero, so this only makes the rank-read index valid and does not change
        # either marker-score strategy. Preserve -1 in the AnnData diagnostics.
        neighbor_indices_for_storage = neighbor_indices
        invalid_neighbors = neighbor_indices < 0
        if np.any(invalid_neighbors):
            if np.any(cell_sims[invalid_neighbors] != 0):
                raise ValueError(
                    "Invalid homogeneous neighbors must have zero marker-score weight"
                )
            neighbor_indices = neighbor_indices.copy()
            focal_rows = np.broadcast_to(cell_indices[:, None], neighbor_indices.shape)
            neighbor_indices[invalid_neighbors] = focal_rows[invalid_neighbors]
            missing_count = int(invalid_neighbors.sum())
            logger.info(
                f"Replaced {missing_count} zero-weight missing-neighbor slots with focal rank rows"
            )

        # Spatial locality gives better rank-row reuse than the sequential
        # central-neighbor walk and is orders of magnitude cheaper to construct.
        if coords is not None:
            logger.info("Ordering rows by spatial Morton signature...")
            if slice_ids is None:
                raise RuntimeError("Spatial ordering requires slice IDs")
            row_order = optimize_spatial_row_order(
                coords[cell_indices],
                slice_ids=slice_ids[cell_indices],
                dataset_type=self.config.dataset_type,
                z_scale=self.config.cross_slice_z_scale
                if self.config.n_adjacent_slices > 0
                else 1.0,
            )
        else:
            logger.info("Optimizing non-spatial row order for cache efficiency...")
            row_order = optimize_row_order_jax(
                neighbor_indices=neighbor_indices[:, : self.config.homogeneous_neighbors],
                cell_indices=cell_indices,
                neighbor_weights=cell_sims[:, : self.config.homogeneous_neighbors],
            )

        neighbor_indices = neighbor_indices[row_order]
        neighbor_indices_for_storage = neighbor_indices_for_storage[row_order]
        cell_sims = cell_sims[row_order]
        if niche_sims is not None:
            niche_sims = niche_sims[row_order]
        cell_indices_sorted = cell_indices[row_order]

        # Save homogeneous neighbor data to adata
        has_real_niche_embedding = self.config.latent_representation_niche is not None
        self._save_homogeneous_data_to_adata(
            adata=adata,
            neighbor_indices=neighbor_indices_for_storage,
            cell_sims=cell_sims,
            niche_sims=niche_sims,
            cell_indices_sorted=cell_indices_sorted,
            has_real_niche_embedding=has_real_niche_embedding,
        )

        # warning for cells not find homo neighbors
        homo_neighbor_count = np.count_nonzero(cell_sims > 0, axis=1)
        zero_homo_neighbor_mask = homo_neighbor_count <= 5
        if np.any(zero_homo_neighbor_mask):
            logger.warning(
                f"Cell type {cell_type}: {zero_homo_neighbor_mask.sum()} cells can't find enough homogeneous neighbors"
            )

        return neighbor_indices, cell_sims, niche_sims, cell_indices_sorted, n_cells

    def _save_homogeneous_data_to_adata(
        self,
        adata: ad.AnnData,
        neighbor_indices: np.ndarray,
        cell_sims: np.ndarray,
        niche_sims: np.ndarray | None,
        cell_indices_sorted: np.ndarray,
        has_real_niche_embedding: bool,
    ):
        """
        Save homogeneous neighbor data to adata obsm and obs.

        Args:
            adata: AnnData object to save data to
            neighbor_indices: (n_cells, k) array of neighbor indices
            cell_sims: (n_cells, k) array of cell similarity scores
            niche_sims: (n_cells, k) array of niche similarity scores or None
            cell_indices_sorted: (n_cells,) array of cell indices in sorted order
            has_real_niche_embedding: Whether real niche embedding was provided
        """
        # Initialize obsm matrices if they don't exist
        if "gsMap_homo_indices" not in adata.obsm.keys() or (
            adata.obsm["gsMap_homo_indices"].shape[1] != neighbor_indices.shape[1]
        ):
            adata.obsm["gsMap_homo_indices"] = np.full(
                (adata.n_obs, neighbor_indices.shape[1]),
                -1,
                dtype=neighbor_indices.dtype,
            )
        if "gsMap_homo_cell_sims" not in adata.obsm.keys() or (
            adata.obsm["gsMap_homo_cell_sims"].shape[1] != neighbor_indices.shape[1]
        ):
            adata.obsm["gsMap_homo_cell_sims"] = np.zeros(
                (adata.n_obs, cell_sims.shape[1]), dtype=cell_sims.dtype
            )

        # Store the neighbor indices and cell_sims for this cell type
        adata.obsm["gsMap_homo_indices"] = assign_rows_without_overflow(
            adata.obsm["gsMap_homo_indices"], cell_indices_sorted, neighbor_indices
        )
        adata.obsm["gsMap_homo_cell_sims"][cell_indices_sorted] = cell_sims

        # Only store niche_sims if niche embedding was provided (not dummy)
        if has_real_niche_embedding and niche_sims is not None:
            if "gsMap_homo_niche_sims" not in adata.obsm.keys() or (
                adata.obsm["gsMap_homo_niche_sims"].shape[1] != neighbor_indices.shape[1]
            ):
                adata.obsm["gsMap_homo_niche_sims"] = np.zeros(
                    (adata.n_obs, niche_sims.shape[1]), dtype=niche_sims.dtype
                )
            adata.obsm["gsMap_homo_niche_sims"][cell_indices_sorted] = niche_sims

        # Initialize obs columns if they don't exist
        if "gsMap_homo_neighbor_count" not in adata.obs.columns:
            adata.obs["gsMap_homo_neighbor_count"] = 0
        if "gsMap_homo_cell_sims_median" not in adata.obs.columns:
            adata.obs["gsMap_homo_cell_sims_median"] = np.nan
        if has_real_niche_embedding and "gsMap_homo_niche_sims_median" not in adata.obs.columns:
            adata.obs["gsMap_homo_niche_sims_median"] = np.nan

        # Calculate statistics for each cell (only consider valid neighbors where cell_sims > 0)
        valid_mask = cell_sims > 0  # (n_cells, k) boolean mask

        # Count of valid homogeneous neighbors
        homo_neighbor_count = valid_mask.sum(axis=1)
        adata.obs.loc[adata.obs.index[cell_indices_sorted], "gsMap_homo_neighbor_count"] = (
            homo_neighbor_count
        )

        # Median of cell similarity (only for valid neighbors)
        cell_sims_masked = np.where(valid_mask, cell_sims, np.nan)
        cell_sims_median = np.nanmedian(cell_sims_masked, axis=1)
        adata.obs.loc[adata.obs.index[cell_indices_sorted], "gsMap_homo_cell_sims_median"] = (
            cell_sims_median
        )

        # Median of niche similarity (only for valid neighbors, only if niche embedding provided)
        if has_real_niche_embedding and niche_sims is not None:
            niche_sims_masked = np.where(valid_mask, niche_sims, np.nan)
            niche_sims_median = np.nanmedian(niche_sims_masked, axis=1)
            adata.obs.loc[adata.obs.index[cell_indices_sorted], "gsMap_homo_niche_sims_median"] = (
                niche_sims_median
            )

    def _calculate_marker_scores_by_cell_type(
        self,
        adata: ad.AnnData,
        cell_type: str,
        coords: np.ndarray | None,
        emb_niche: np.ndarray,
        emb_indv: np.ndarray,
        annotation_key: str | None,
        slice_ids: np.ndarray | None = None,
        high_quality_mask: np.ndarray = None,
    ) -> None:
        """Find homogeneous neighbors and score one cell type."""

        # Find homogeneous spots
        prepared = self._find_homogeneous_spots(
            adata,
            cell_type,
            annotation_key,
            coords,
            emb_niche,
            emb_indv,
            slice_ids,
            self.rank_shape,
            high_quality_mask,
        )
        if prepared is None:
            return

        neighbor_indices, cell_sims, _niche_sims, cell_indices_sorted, _n_cells = prepared
        marker_rows = self.marker_row_writer.append(cell_indices_sorted)
        self._calculate_marker_scores_with_rank_cache(
            cell_type,
            neighbor_indices,
            cell_sims,
            marker_rows,
        )

    def calculate_marker_scores(
        self,
        adata_path: str,
        rank_memmap_path: str | Path | MemMapDense,
        mean_frac_path: str,
        output_path: str | Path | None = None,
    ) -> str | Path:
        """
        Main execution function for marker score calculation

        Args:
            adata_path: Path to concatenated latent adata
            rank_memmap_path: Path to rank memory map
            mean_frac_path: Path to mean expression fraction parquet
            output_path: Optional output path for marker scores

        Returns:
            Path to output marker score memory map file
        """
        logger.info("Starting marker score calculation...")
        self.console = Console()
        self.show_rich_output = is_interactive_output()

        if self.show_rich_output:
            self.console.print(
                Panel.fit(
                    "[bold cyan]Marker Score Calculation Stage[/bold cyan]",
                    subtitle="gsMap Stage 2",
                    border_style="cyan",
                )
            )

        # Use config path if not specified
        if output_path is None:
            output_path = Path(self.config.marker_scores_memmap_path)
        else:
            output_path = Path(output_path)

        rank_memmap = None
        output_memmap = None
        self.marker_row_writer = None
        self.writer = None
        completed = False
        try:
            # Load all input data
            (
                adata,
                rank_memmap,
                global_log_gmean,
                global_expr_frac,
                n_cells,
                n_genes,
                high_quality_mask,
            ) = self._load_input_data(adata_path, rank_memmap_path, mean_frac_path)

            # Initialize output memory map
            output_memmap = MemMapDense(
                output_path,
                shape=(n_cells, n_genes),
                dtype=np.float16,  # Use float16 to save memory
                mode="w",
                # ParallelMarkerScoreWriter owns output writes. Starting the
                # MemMapDense asynchronous pool here creates idle duplicate threads.
                num_write_workers=0,
                tmp_dir=self.config.memmap_tmp_dir,
            )
            self.marker_row_writer = SequentialMarkerRowWriter(output_path, n_cells)

            # Get cell types to process
            cell_types = self._get_cell_types(adata)
            annotation_key = self.config.annotation
            self._clear_homogeneous_outputs(adata)

            # Display summary
            self._display_input_summary(adata, cell_types, n_cells, n_genes)

            # Prepare embeddings and initialize the bounded processing pipeline.
            coords, emb_niche, emb_indv, slice_ids = self._prepare_embeddings(adata)
            self._initialize_pipeline(
                rank_memmap,
                output_memmap,
                global_log_gmean,
                global_expr_frac,
            )

            # Process each cell type. Marker output is contiguous in this
            # processing order; the companion permutation preserves alignment
            # to the unchanged concatenated AnnData.
            for cell_type in cell_types:
                self._calculate_marker_scores_by_cell_type(
                    adata,
                    cell_type,
                    coords,
                    emb_niche,
                    emb_indv,
                    annotation_key,
                    slice_ids,
                    high_quality_mask,
                )

            # Surface background failures before publishing either the row
            # permutation or the completed marker memmap.
            logger.info("Closing the marker-score writer...")
            self.writer.close()

            logger.info(f"Saving updated AnnData with homogeneous spot matrices to {adata_path}")
            # This file was already published by the rank step; replace it atomically so
            # an interrupted write cannot leave a truncated file that later runs trust.
            from gsMap.utils.atomic_io import atomic_output

            with atomic_output(adata_path) as temporary:
                adata.write_h5ad(temporary)
            row_layout = self.marker_row_writer.finalize()
            output_memmap.close(
                metadata_updates={
                    "marker_score_layout_version": MARKER_SCORE_LAYOUT_VERSION,
                    "row_layout": row_layout,
                }
            )
            completed = True
        finally:
            # Cleanup must also run for JAX errors, asynchronous writer errors,
            # failed AnnData writes, and user interrupts. Never publish a partial
            # marker matrix or leave scratch mappings/background threads alive.
            if not completed:
                cleanup_actions = []
                if self.writer is not None:
                    cleanup_actions.append(("marker writer", self.writer.abort))
                if self.marker_row_writer is not None:
                    cleanup_actions.append(("row-layout writer", self.marker_row_writer.abort))
                    if self.marker_row_writer.finalized:
                        cleanup_actions.append(
                            (
                                "published row layout",
                                lambda: self.marker_row_writer.final_path.unlink(missing_ok=True),
                            )
                        )
                if output_memmap is not None:
                    cleanup_actions.append(("marker memmap", output_memmap.abort))
                for label, cleanup in cleanup_actions:
                    try:
                        cleanup()
                    except Exception as cleanup_error:  # noqa: BLE001
                        logger.error(f"Failed to clean up {label}: {cleanup_error}")
            if rank_memmap is not None:
                try:
                    rank_memmap.close()
                except Exception as cleanup_error:
                    if completed:
                        raise
                    logger.error(f"Failed to close rank memmap: {cleanup_error}")

        if self.show_rich_output:
            self.console.print(
                Panel.fit(
                    "[bold green]✓ Marker score calculation complete![/bold green]",
                    border_style="green",
                )
            )

        logger.info(f"Results saved to {output_path}")

        return str(output_path)
