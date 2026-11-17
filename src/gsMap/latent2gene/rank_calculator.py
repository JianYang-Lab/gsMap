"""
Rank calculation from latent representations
Extracts and processes the rank calculation logic from find_latent_representation.py
"""

import gc
import logging
import os
import time
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import h5py
import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
import pandas as pd
from anndata._io.specs import read_elem
from jax import jit
from numba import get_num_threads, get_thread_id, njit, prange, set_num_threads
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from scipy.sparse import csr_matrix

from gsMap.config import LatentToGeneConfig
from gsMap.config.latent2gene_config import ExpressionDataType, validate_expression_contract
from gsMap.logging_config import is_interactive_output
from gsMap.utils.expression_qc import expression_row_metrics, h5_expression_row_metrics

from .memmap_io import MemMapDense

logger = logging.getLogger(__name__)

_MIB = 1024**2
_TARGET_SPARSE_RANK_BATCH_BYTES = 64 * _MIB
_SPARSE_RANK_BATCH_SIZE_CHOICES = (250, 500, 1000, 2000)
_MAX_HISTOGRAM_RANK_VALUE = 65_536
_RANK_LOG_PERCENT_STEP = 10
_RANK_LOG_INTERVAL_SECONDS = 60.0
_GENE_INDEX_REMAP_CHUNK = 8_000_000


class _RankProgressReporter:
    """Persist coarse rank progress without copying Rich terminal output into logs."""

    def __init__(
        self,
        total_cells: int,
        *,
        percent_step: int = _RANK_LOG_PERCENT_STEP,
        log_interval_seconds: float = _RANK_LOG_INTERVAL_SECONDS,
    ) -> None:
        if total_cells < 1:
            raise ValueError("Rank progress requires at least one cell")
        if not 1 <= percent_step <= 100:
            raise ValueError("Rank progress percent_step must be between 1 and 100")
        if log_interval_seconds <= 0:
            raise ValueError("Rank progress log_interval_seconds must be positive")

        self.total_cells = total_cells
        self.percent_step = percent_step
        self.log_interval_seconds = log_interval_seconds
        self.processed_cells = 0
        self.active_seconds = 0.0
        self._next_percent = percent_step
        self._last_log_at = time.perf_counter()
        self._finished = False

    @property
    def throughput(self) -> float:
        if self.active_seconds <= 0:
            return 0.0
        return self.processed_cells / self.active_seconds

    def update(
        self,
        *,
        processed_cells: int,
        active_seconds: float,
        sample_name: str,
        sample_processed_cells: int,
        sample_total_cells: int,
    ) -> None:
        """Record one completed chunk and log at 10% or 60-second boundaries."""
        if self._finished:
            return
        if not self.processed_cells <= processed_cells <= self.total_cells:
            raise ValueError("Rank progress must be monotonic and cannot exceed total cells")
        if active_seconds < 0:
            raise ValueError("Rank active time cannot be negative")

        self.processed_cells = processed_cells
        self.active_seconds += active_seconds
        percent_complete = 100.0 * processed_cells / self.total_cells
        now = time.perf_counter()
        percent_due = percent_complete >= self._next_percent
        interval_due = now - self._last_log_at >= self.log_interval_seconds
        complete = processed_cells == self.total_cells
        if not (percent_due or interval_due or complete):
            return

        while self._next_percent <= percent_complete:
            self._next_percent += self.percent_step
        logger.info(
            f"Rank progress: {processed_cells:,}/{self.total_cells:,} cells "
            f"({percent_complete:.1f}%); active throughput={self.throughput:,.0f} cells/s; "
            f"sample={sample_name} ({sample_processed_cells:,}/{sample_total_cells:,})"
        )
        self._last_log_at = now
        self._finished = complete


def _automatic_sparse_rank_batch_size(genes: int) -> int:
    """Select a batch near 64 MiB for the encoded float16 rank output."""
    if genes < 1:
        raise ValueError("Gene count must be positive")
    unconstrained = _TARGET_SPARSE_RANK_BATCH_BYTES // (genes * np.dtype("float16").itemsize)
    eligible = [size for size in _SPARSE_RANK_BATCH_SIZE_CHOICES if size <= unconstrained]
    return eligible[-1] if eligible else _SPARSE_RANK_BATCH_SIZE_CHOICES[0]


def _configure_rank_numba_threads() -> int:
    """Use the process CPU affinity, which reflects a SLURM job's cgroup allocation."""
    try:
        available_cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        available_cpus = os.cpu_count() or 1
    resolved_threads = max(1, min(available_cpus, get_num_threads()))
    set_num_threads(resolved_threads)
    return resolved_threads


def _build_log_rank_lookup(n_genes: int) -> tuple[np.ndarray, np.ndarray]:
    """Build float32 log ranks and their final float16 storage bit patterns."""
    doubled_ranks = np.arange(2 * n_genes + 1, dtype=np.float32)
    doubled_ranks[0] = 2.0
    log_rank_float32 = np.log(doubled_ranks * np.float32(0.5)).astype(np.float32)
    log_rank_float16_bits = log_rank_float32.astype(np.float16).view(np.uint16)
    return log_rank_float32, log_rank_float16_bits


@njit
def _all_finite(values: np.ndarray) -> bool:
    for value in values:
        if not np.isfinite(value):
            return False
    return True


@njit(parallel=True)
def _sparse_log_rank_bits(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    n_genes: int,
    log_rank_float32: np.ndarray,
    log_rank_float16_bits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exactly rank sparse rows while directly encoding float16 output bits."""
    n_rows = indptr.size - 1
    n_threads = get_num_threads()
    output = np.empty((n_rows, n_genes), dtype=np.uint16)
    thread_sums = np.zeros((n_threads, n_genes), dtype=np.float32)
    thread_fractions = np.zeros((n_threads, n_genes), dtype=np.int32)

    for row in prange(n_rows):
        thread = get_thread_id()
        start = indptr[row]
        end = indptr[row + 1]
        nonzero_count = end - start
        zero_count = n_genes - nonzero_count

        use_histogram = True
        max_value = 0
        for position in range(start, end):
            value = data[position]
            integer_value = int(value)
            if value <= 0 or value != integer_value or integer_value > _MAX_HISTOGRAM_RANK_VALUE:
                use_histogram = False
                break
            if integer_value > max_value:
                max_value = integer_value

        if use_histogram:
            zero_rank_code = zero_count + 1
        else:
            values = data[start:end]
            order = np.argsort(values)
            negative_count = 0
            while negative_count < nonzero_count and values[order[negative_count]] < 0:
                negative_count += 1
            zero_rank_code = 2 * negative_count + zero_count + 1

        zero_log_rank = log_rank_float32[zero_rank_code]
        zero_log_rank_bits = log_rank_float16_bits[zero_rank_code]
        for gene in range(n_genes):
            output[row, gene] = zero_log_rank_bits
            thread_sums[thread, gene] += zero_log_rank

        if use_histogram:
            counts = np.zeros(max_value + 1, dtype=np.int32)
            for position in range(start, end):
                counts[int(data[position])] += 1

            cumulative = zero_count
            rank_codes = np.empty(max_value + 1, dtype=np.int32)
            for value in range(1, max_value + 1):
                count = counts[value]
                if count:
                    rank_codes[value] = 2 * cumulative + count + 1
                    cumulative += count
            for position in range(start, end):
                gene = indices[position]
                rank_code = rank_codes[int(data[position])]
                output[row, gene] = log_rank_float16_bits[rank_code]
                thread_sums[thread, gene] += log_rank_float32[rank_code] - zero_log_rank
                thread_fractions[thread, gene] += 1
        else:
            group_start = 0
            while group_start < nonzero_count:
                group_end = group_start + 1
                value = values[order[group_start]]
                while group_end < nonzero_count and values[order[group_end]] == value:
                    group_end += 1
                offset = zero_count if value > 0 else 0
                rank_code = 2 * offset + group_start + 1 + group_end
                for sorted_position in range(group_start, group_end):
                    source_position = start + order[sorted_position]
                    gene = indices[source_position]
                    output[row, gene] = log_rank_float16_bits[rank_code]
                    thread_sums[thread, gene] += log_rank_float32[rank_code] - zero_log_rank
                    thread_fractions[thread, gene] += 1
                group_start = group_end

    sum_log_ranks = np.zeros(n_genes, dtype=np.float32)
    sum_fractions = np.zeros(n_genes, dtype=np.float32)
    for thread in range(n_threads):
        for gene in range(n_genes):
            sum_log_ranks[gene] += thread_sums[thread, gene]
            sum_fractions[gene] += thread_fractions[thread, gene]
    return output, sum_log_ranks, sum_fractions


def filter_cells(
    obs_df,
    X=None,
    annotation_key=None,
    min_cells_per_type=None,
    min_genes=20,
    precomputed_gene_counts=None,
):
    """
    Apply cell filtering based on annotation and gene expression criteria.

    Args:
        obs_df: DataFrame with cell metadata
        X: Optional expression matrix (sparse or dense) for gene count filtering
        annotation_key: Optional annotation column to filter by
        min_cells_per_type: Minimum cells required per annotation group
        min_genes: Minimum number of genes expressed per cell (default: 20)
        precomputed_gene_counts: Optional precomputed gene counts per cell (e.g., from CSR indptr)

    Returns:
        Tuple of (filtered_obs_df, filtering_stats_dict, keep_mask)
    """
    n_cells_before = obs_df.shape[0]
    stats = {
        "input_cells": n_cells_before,
        "nan_removed": 0,
        "small_group_removed": 0,
        "low_gene_count_removed": 0,
        "final_cells": 0,
    }

    # Create a mask for cells to keep (start with all True)
    keep_mask = pd.Series(True, index=obs_df.index)

    # Filter 1: Remove cells with NaN in annotation_key
    if annotation_key is not None and annotation_key in obs_df.columns:
        nan_mask = obs_df[annotation_key].notna()
        stats["nan_removed"] = (~nan_mask).sum()
        keep_mask &= nan_mask

    # Filter 2: Remove cells from small annotation groups
    if (
        annotation_key is not None
        and annotation_key in obs_df.columns
        and min_cells_per_type is not None
    ):
        # Apply on the current filtered data
        temp_obs = obs_df[keep_mask]
        annotation_counts = temp_obs[annotation_key].value_counts()
        valid_annotations = annotation_counts[annotation_counts >= min_cells_per_type].index

        if len(valid_annotations) < len(annotation_counts):
            small_group_mask = obs_df[annotation_key].isin(valid_annotations)
            stats["small_group_removed"] = keep_mask.sum() - (keep_mask & small_group_mask).sum()
            keep_mask &= small_group_mask

    # Filter 3: Remove cells with too few genes
    if min_genes is not None:
        if precomputed_gene_counts is not None:
            # Use precomputed gene counts (efficient for CSR format from indptr)
            nfeature_mask = precomputed_gene_counts >= min_genes
            nfeature_mask = pd.Series(nfeature_mask, index=obs_df.index)
        elif X is not None:
            n_genes_per_cell, _ = expression_row_metrics(X)
            nfeature_mask = n_genes_per_cell >= min_genes
            # Convert to pandas Series for consistent indexing
            nfeature_mask = pd.Series(nfeature_mask, index=obs_df.index)
        else:
            # No gene count information available, skip this filter
            nfeature_mask = pd.Series(True, index=obs_df.index)

        stats["low_gene_count_removed"] = keep_mask.sum() - (keep_mask & nfeature_mask).sum()
        keep_mask &= nfeature_mask

    # Apply the combined filter
    filtered_obs = obs_df[keep_mask].copy()
    stats["final_cells"] = filtered_obs.shape[0]

    return filtered_obs, stats, keep_mask


@dataclass(frozen=True, slots=True)
class _RankInputRequest:
    """Everything needed to selectively load one section for ranking."""

    sample_name: str
    h5ad_path: Path
    slice_id: int
    keep_mask: pd.Series
    data_layer: str
    expected_var_names: tuple[str, ...] | None = None


@dataclass(slots=True)
class _RankInput:
    """Materialized subset needed by rank calculation and latent concatenation."""

    sample_name: str
    slice_id: int
    X: csr_matrix
    obs: pd.DataFrame
    var_names: list[str]
    obsm: dict[str, Any]
    matrix_source: str
    metadata_read_seconds: float
    matrix_read_seconds: float
    obsm_read_seconds: float
    prepare_seconds: float
    gene_order_aligned: bool

    @property
    def input_seconds(self) -> float:
        return (
            self.metadata_read_seconds
            + self.matrix_read_seconds
            + self.obsm_read_seconds
            + self.prepare_seconds
        )


def _subset_obsm_rows(
    obsm: dict[str, Any], keep: np.ndarray, obs_names: pd.Index
) -> dict[str, Any]:
    """Subset obsm positionally and keep DataFrame indices aligned with obs."""
    filtered: dict[str, Any] = {}
    for key, value in obsm.items():
        if value.shape[0] != keep.size:
            raise ValueError(f"obsm/{key} has {value.shape[0]} rows but obs has {keep.size} rows")
        if isinstance(value, pd.DataFrame):
            subset = value.iloc[keep].copy()
            subset.index = obs_names
        else:
            subset = value[keep].copy()
        filtered[key] = subset
    return filtered


def _gene_column_order(
    current_var_names: list[str] | tuple[str, ...],
    expected_var_names: tuple[str, ...],
    *,
    sample_name: str,
) -> np.ndarray | None:
    """Return canonical-to-current columns or fail on a true gene-set mismatch."""
    current_index = pd.Index(current_var_names)
    expected_index = pd.Index(expected_var_names)
    if not current_index.is_unique or not expected_index.is_unique:
        current_duplicates = current_index[current_index.duplicated()].unique().tolist()
        expected_duplicates = expected_index[expected_index.duplicated()].unique().tolist()
        raise ValueError(
            f"Gene names must be unique for {sample_name}: "
            f"current_duplicates={current_duplicates[:10]}, "
            f"expected_duplicates={expected_duplicates[:10]}"
        )

    if tuple(current_var_names) == expected_var_names:
        return None

    canonical_to_current = current_index.get_indexer(expected_index)
    missing = expected_index[canonical_to_current < 0].tolist()
    extra = current_index.difference(expected_index, sort=False).tolist()
    if missing or extra or len(current_index) != len(expected_index):
        raise ValueError(
            f"Gene set mismatch for {sample_name}: "
            f"missing={missing[:10]}, extra={extra[:10]}, "
            f"current={len(current_index)}, expected={len(expected_index)}"
        )
    return canonical_to_current


def _align_csr_gene_columns(X: csr_matrix, canonical_to_current: np.ndarray) -> csr_matrix:
    """Relabel CSR column indices in bounded chunks without copying expression data."""
    current_to_canonical = np.empty(X.shape[1], dtype=X.indices.dtype)
    current_to_canonical[canonical_to_current] = np.arange(X.shape[1], dtype=X.indices.dtype)
    for start in range(0, X.indices.size, _GENE_INDEX_REMAP_CHUNK):
        end = min(start + _GENE_INDEX_REMAP_CHUNK, X.indices.size)
        X.indices[start:end] = current_to_canonical[X.indices[start:end]]
    X.has_sorted_indices = False
    return X


def _select_rank_matrix_element(
    handle: h5py.File,
    data_layer: str,
) -> tuple[h5py.Dataset | h5py.Group, str]:
    """Resolve the expression element identically in both rank passes."""
    if data_layer != "X" and "layers" in handle and data_layer in handle["layers"]:
        return handle["layers"][data_layer], f"layers/{data_layer}"
    return handle["X"], "X"


def _read_rank_input(request: _RankInputRequest) -> _RankInput:
    """Read only obs, var names, the selected expression matrix, and obsm.

    In particular, this deliberately does not materialize ``raw`` or ``X`` when
    the configured expression layer exists.
    """
    keep = np.asarray(request.keep_mask, dtype=bool)

    metadata_started_at = time.perf_counter()
    with h5py.File(request.h5ad_path, "r") as handle:
        obs = read_elem(handle["obs"])
        var_names = read_elem(handle["var"]).index.astype(str).tolist()
        metadata_read_seconds = time.perf_counter() - metadata_started_at

        matrix_started_at = time.perf_counter()
        matrix_element, matrix_source = _select_rank_matrix_element(handle, request.data_layer)
        X = read_elem(matrix_element)
        matrix_read_seconds = time.perf_counter() - matrix_started_at

        obsm_started_at = time.perf_counter()
        obsm = read_elem(handle["obsm"]) if "obsm" in handle else {}
        obsm_read_seconds = time.perf_counter() - obsm_started_at

    prepare_started_at = time.perf_counter()
    if keep.ndim != 1 or keep.size != obs.shape[0]:
        raise ValueError(
            f"Filter mask for {request.sample_name} has shape {keep.shape}; "
            f"expected ({obs.shape[0]},)"
        )
    if X.shape != (obs.shape[0], len(var_names)):
        raise ValueError(
            f"Expression matrix for {request.sample_name} has shape {X.shape}; "
            f"expected ({obs.shape[0]}, {len(var_names)})"
        )
    canonical_to_current = (
        _gene_column_order(
            var_names,
            request.expected_var_names,
            sample_name=request.sample_name,
        )
        if request.expected_var_names is not None
        else None
    )

    # Match AnnData.obs_names_make_unique() before filtering so observation
    # identities remain exactly the same when duplicate rows are removed.
    obs.index = ad.utils.make_index_unique(obs.index)
    obs["slice_id"] = request.slice_id
    obs["slice_name"] = request.sample_name
    obs["sample_name"] = request.sample_name
    obs.index = obs.index.astype(str) + "|" + obs["slice_name"].astype(str)
    obs = obs.iloc[keep].copy()
    for column in obs.select_dtypes(include="category"):
        obs[column] = obs[column].cat.remove_unused_categories()

    if not keep.all():
        X = X[keep]
    if not hasattr(X, "tocsr"):
        X = csr_matrix(X, dtype=np.float32)
    else:
        X = X.tocsr()
        if X.dtype != np.float32:
            X = X.astype(np.float32)
    gene_order_aligned = canonical_to_current is not None
    if canonical_to_current is not None:
        X = _align_csr_gene_columns(X, canonical_to_current)
        var_names = list(request.expected_var_names)
    X.sum_duplicates()
    X.eliminate_zeros()
    X.sort_indices()

    obsm = _subset_obsm_rows(obsm, keep, obs.index)
    prepare_seconds = time.perf_counter() - prepare_started_at

    return _RankInput(
        sample_name=request.sample_name,
        slice_id=request.slice_id,
        X=X,
        obs=obs,
        var_names=var_names,
        obsm=obsm,
        matrix_source=matrix_source,
        metadata_read_seconds=metadata_read_seconds,
        matrix_read_seconds=matrix_read_seconds,
        obsm_read_seconds=obsm_read_seconds,
        prepare_seconds=prepare_seconds,
        gene_order_aligned=gene_order_aligned,
    )


@contextmanager
def _prefetched_rank_inputs(
    requests: Iterable[_RankInputRequest],
    loader: Callable[[_RankInputRequest], _RankInput] = _read_rank_input,
) -> Iterator[Iterator[tuple[_RankInput, float]]]:
    """Yield ordered inputs while loading at most one future section ahead.

    The yielded float is the input latency exposed to the rank loop. Loading
    time beyond that wait was hidden behind computation of the prior section.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rank-input")
    request_iterator = iter(requests)

    def iterate() -> Iterator[tuple[_RankInput, float]]:
        try:
            first_request = next(request_iterator)
        except StopIteration:
            return

        future = executor.submit(loader, first_request)
        while future is not None:
            wait_started_at = time.perf_counter()
            rank_input = future.result()
            exposed_wait_seconds = time.perf_counter() - wait_started_at

            try:
                next_request = next(request_iterator)
            except StopIteration:
                next_future = None
            else:
                next_future = executor.submit(loader, next_request)

            yield rank_input, exposed_wait_seconds
            future = next_future

    try:
        yield iterate()
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


@jit
def jax_process_chunk(dense_matrix, valid_rows):
    nonzero_mask = dense_matrix != 0

    ranks = jax.scipy.stats.rankdata(dense_matrix, method="average", axis=1)
    log_ranks = jnp.log(ranks)
    valid_mask = (jnp.arange(dense_matrix.shape[0]) < valid_rows)[:, None]
    # Preserve float32 accumulators while transferring only the final float16 payload.
    sum_log_ranks = jnp.where(valid_mask, log_ranks, 0).sum(axis=0)
    sum_frac = jnp.where(valid_mask, nonzero_mask, False).sum(axis=0)

    return log_ranks.astype(jnp.float16), sum_log_ranks, sum_frac


def rank_data_jax(
    X: csr_matrix,
    n_genes,
    memmap_dense=None,
    metadata: dict[str, Any] | None = None,
    chunk_size: int = 1000,
    write_interval: int = 10,
    current_row_offset: int = 0,
    progress=None,
    progress_task=None,
    progress_reporter: _RankProgressReporter | None = None,
):
    """Exact sparse rank calculation with a fixed-shape dense-JAX fallback.

    Args:
        X: Input sparse matrix
        n_genes: Total number of genes
        memmap_dense: Optional MemMapDense instance for writing
        metadata: Optional metadata dictionary
        chunk_size: Size of chunks for processing
        write_interval: How often to write chunks to memory map
        current_row_offset: Offset for writing to memory map (for multiple sections)
        progress: Progress instance for updates
        progress_task: Task ID for progress updates
        progress_reporter: Optional durable logger for global rank progress and throughput

    Returns:
        Tuple of (sum_log_ranks, sum_frac) as numpy arrays
    """
    n_rows, n_cols = X.shape
    if n_rows < 1:
        raise ValueError("Input matrix must contain at least one cell")
    if n_cols != n_genes:
        raise ValueError(
            f"Input matrix has {n_cols} genes but rank calculation expected {n_genes}"
        )
    if X.nnz == 0:
        raise ValueError("Input matrix must contain at least one non-zero expression value")
    if chunk_size < 1:
        raise ValueError("Rank chunk_size must be positive")
    if write_interval < 1:
        raise ValueError("Rank write_interval must be positive")
    if current_row_offset < 0:
        raise ValueError("Rank current_row_offset cannot be negative")

    # Initialize accumulators (use float32 for accumulators to avoid precision loss)
    sum_log_ranks = np.zeros(n_genes, dtype=np.float32)
    sum_frac = np.zeros(n_genes, dtype=np.float32)

    use_sparse_backend = _all_finite(X.data)
    if use_sparse_backend:
        rank_threads = _configure_rank_numba_threads()
        log_rank_float32, log_rank_float16_bits = _build_log_rank_lookup(n_genes)
        backend_name = "sparse-numba"
        logger.debug(
            f"Rank backend for {metadata.get('name', 'unknown') if metadata else 'unknown'}: "
            f"sparse-numba with {rank_threads} threads"
        )
    else:
        log_rank_float32 = None
        log_rank_float16_bits = None
        backend_name = "dense-jax"
        logger.warning("Non-finite expression values detected; using dense-JAX rank fallback")

    # Process in chunks to manage memory
    chunk_size = min(chunk_size, n_rows)
    pending_chunks = []  # Buffer for batching writes
    pending_indices = []  # Track global indices for writing
    chunks_processed = 0

    # Track speed at chunk level
    sample_started_at = time.perf_counter()
    processed_cells = 0
    sample_name = str(metadata.get("name", "unknown")) if metadata else "unknown"
    compute_seconds = 0.0
    accumulation_seconds = 0.0
    output_enqueue_seconds = 0.0

    for start_idx in range(0, n_rows, chunk_size):
        chunk_started_at = time.perf_counter()
        end_idx = min(start_idx + chunk_size, n_rows)

        chunk_X = X[start_idx:end_idx]
        valid_rows = end_idx - start_idx
        compute_started_at = time.perf_counter()
        if use_sparse_backend:
            chunk_log_rank_bits, chunk_sum_log_ranks, chunk_sum_frac = _sparse_log_rank_bits(
                chunk_X.indptr,
                chunk_X.indices,
                chunk_X.data,
                n_genes,
                log_rank_float32,
                log_rank_float16_bits,
            )
            chunk_log_ranks_np = chunk_log_rank_bits.view(np.float16)
        else:
            chunk_dense = chunk_X.toarray().astype(np.float32, copy=False)
            if valid_rows < chunk_size:
                padded = np.zeros((chunk_size, n_cols), dtype=np.float32)
                padded[:valid_rows] = chunk_dense
                chunk_dense = padded
            chunk_log_ranks, chunk_sum_log_ranks, chunk_sum_frac = jax_process_chunk(
                jnp.asarray(chunk_dense), valid_rows
            )
            chunk_log_ranks_np = np.asarray(chunk_log_ranks[:valid_rows])
        compute_seconds += time.perf_counter() - compute_started_at

        # Update global accumulators
        accumulation_started_at = time.perf_counter()
        sum_log_ranks += np.asarray(chunk_sum_log_ranks, dtype=np.float32)
        sum_frac += np.asarray(chunk_sum_frac, dtype=np.float32)
        accumulation_seconds += time.perf_counter() - accumulation_started_at

        pending_chunks.append(chunk_log_ranks_np)
        # Calculate global indices for this chunk
        global_start = current_row_offset + start_idx
        global_end = current_row_offset + end_idx
        pending_indices.append((global_start, global_end))
        chunks_processed += 1

        # Write to memory map periodically
        if memmap_dense and (chunks_processed % write_interval == 0 or end_idx == n_rows):
            output_started_at = time.perf_counter()
            # Combine pending chunks for batch write
            combined_data = (
                pending_chunks[0] if len(pending_chunks) == 1 else np.vstack(pending_chunks)
            )
            # Calculate row indices for batch write
            start_row = pending_indices[0][0]
            end_row = pending_indices[-1][1]
            # Write as a contiguous block
            memmap_dense.write_batch(combined_data, row_indices=slice(start_row, end_row))
            pending_chunks.clear()
            pending_indices.clear()
            output_enqueue_seconds += time.perf_counter() - output_started_at

        chunk_cells = end_idx - start_idx
        processed_cells += chunk_cells
        chunk_elapsed = time.perf_counter() - chunk_started_at
        if progress_reporter is not None:
            progress_reporter.update(
                processed_cells=current_row_offset + end_idx,
                active_seconds=chunk_elapsed,
                sample_name=sample_name,
                sample_processed_cells=end_idx,
                sample_total_cells=n_rows,
            )

        # Update progress bar with speed calculation
        if progress and progress_task is not None:
            elapsed_time = time.perf_counter() - sample_started_at
            speed = processed_cells / elapsed_time if elapsed_time > 0 else 0
            progress.update(progress_task, advance=chunk_cells, speed=f"{speed:.0f}")

    elapsed_seconds = time.perf_counter() - sample_started_at
    logger.debug(
        f"Rank backend summary for {sample_name}: backend={backend_name}, "
        f"cells={n_rows:,}, elapsed={elapsed_seconds:.2f}s "
        f"({n_rows / elapsed_seconds:,.0f} cells/s); "
        f"compute={compute_seconds:.2f}s, accumulate={accumulation_seconds:.2f}s, "
        f"output enqueue={output_enqueue_seconds:.2f}s"
    )
    return sum_log_ranks, sum_frac


class RankCalculator:
    """Calculate gene expression ranks and create concatenated latent representations"""

    def __init__(self, config: LatentToGeneConfig):
        """
        Initialize RankCalculator with configuration

        Args:
            config: LatentToGeneConfig object with all necessary parameters
        """
        self.config = config
        self.latent_dir = Path(config.latent_dir)
        self.output_dir = Path(config.latent2gene_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.console = Console()
        self.show_rich_progress = is_interactive_output()

    def calculate_ranks_and_concatenate(
        self,
        sample_h5ad_dict: dict[str, Path] | None = None,
        annotation_key: str | None = None,
        data_layer: str = "counts",
    ) -> dict[str, Any]:
        """Run rank construction and abort any unpublished mapping on failure."""
        self._active_rank_memmap: MemMapDense | None = None
        try:
            return self._calculate_ranks_and_concatenate(
                sample_h5ad_dict=sample_h5ad_dict,
                annotation_key=annotation_key,
                data_layer=data_layer,
            )
        except BaseException:
            if self._active_rank_memmap is not None:
                try:
                    self._active_rank_memmap.abort()
                except Exception as cleanup_error:  # noqa: BLE001 preserve primary failure
                    logger.error(f"Failed to abort incomplete rank memmap: {cleanup_error}")
            raise
        finally:
            self._active_rank_memmap = None

    def _calculate_ranks_and_concatenate(
        self,
        sample_h5ad_dict: dict[str, Path] | None = None,
        annotation_key: str | None = None,
        data_layer: str = "counts",
    ) -> dict[str, Any]:
        """
        Calculate expression ranks and create concatenated latent representation

        This combines the rank calculation and concatenation logic from find_latent_representation.py

        Args:
            sample_h5ad_dict: Optional dict of sample_name -> h5ad path. If None, uses config.sample_h5ad_dict
            annotation_key: Optional annotation to filter cells. If None, uses config.annotation
            data_layer: Data layer to use for expression

        Returns:
            Dictionary with paths to:
                - concatenated_latent_adata: Path to concatenated latent representations
                - rank_zarr: Path to rank zarr file
                - mean_frac: Path to mean expression fraction parquet
        """

        expression_data_type = validate_expression_contract(self.config)
        allow_negative = expression_data_type == ExpressionDataType.NONCOUNTS

        # Use provided sample_h5ad_dict or get from config
        if sample_h5ad_dict is None:
            sample_h5ad_dict = self.config.sample_h5ad_dict

        # Use provided annotation_key or get from config
        if annotation_key is None:
            annotation_key = self.config.annotation

        # Output paths from config
        concat_adata_path = Path(self.config.concatenated_latent_adata_path)
        rank_memmap_path = Path(self.config.rank_memmap_path)
        mean_frac_path = Path(self.config.mean_frac_path)

        # Check if outputs already exist
        if (
            concat_adata_path.exists()
            and mean_frac_path.exists()
            and MemMapDense.check_complete(rank_memmap_path)[0]
        ):
            logger.info(f"Rank outputs already exist in {self.output_dir}")
            return {
                "concatenated_latent_adata": str(concat_adata_path),
                "rank_memmap": str(rank_memmap_path),
                "mean_frac": str(mean_frac_path),
            }

        logger.info("Starting rank calculation and concatenation...")
        logger.info(f"Processing {len(sample_h5ad_dict)} samples")

        # Process each section
        adata_list = []
        n_total_cells = 0
        gene_list = None

        # Initialize global accumulators
        sum_log_ranks = None
        sum_frac = None
        total_cells = 0
        rank_memmap = None
        current_row_offset = 0  # Track current position in rank memory map

        # First pass: count total cells and determine which cells to keep
        logger.info("Counting total cells across all sections...")
        total_cells_expected = 0
        filtering_stats = []
        sample_keep_indices = {}  # Store cell indices to keep for each sample
        expected_var_names: tuple[str, ...] | None = None
        reordered_gene_samples = []

        for sample_name, h5ad_path in sample_h5ad_dict.items():
            # Apply same filtering logic as in main loop
            with h5py.File(h5ad_path, "r") as f:
                adata_temp_obs = read_elem(f["obs"])
                current_var_names = tuple(read_elem(f["var"]).index.astype(str).tolist())
                if expected_var_names is None:
                    _gene_column_order(
                        current_var_names,
                        current_var_names,
                        sample_name=sample_name,
                    )
                    expected_var_names = current_var_names
                elif (
                    _gene_column_order(
                        current_var_names,
                        expected_var_names,
                        sample_name=sample_name,
                    )
                    is not None
                ):
                    reordered_gene_samples.append(sample_name)

                # Independent downstream QC: inputs may never have passed find_latent.
                # Do not trust upstream metrics after gene/homolog conversion, and
                # do not confuse explicit zeros/duplicate CSR entries with genes.
                X_group, _matrix_source = _select_rank_matrix_element(f, data_layer)
                # Noncounts use distinct nonzero features, including signed
                # standardized values. Row sums are never used as UMI depth.
                n_genes_per_cell, _ = h5_expression_row_metrics(
                    X_group, allow_negative=allow_negative
                )

                if annotation_key is not None:
                    if annotation_key not in adata_temp_obs.columns:
                        raise KeyError(
                            f"Annotation key '{annotation_key}' not found in the obs of "
                            f"{sample_name}"
                        )

                # Apply unified filtering function
                filtered_obs, stats, keep_mask = filter_cells(
                    adata_temp_obs,
                    X=None,
                    annotation_key=annotation_key,
                    min_cells_per_type=self.config.min_cells_per_type,
                    min_genes=20,
                    precomputed_gene_counts=n_genes_per_cell,
                )

                # Store the indices of cells to keep for this sample
                sample_keep_indices[sample_name] = keep_mask

                total_cells_expected += stats["final_cells"]

                filtering_stats.append({"Sample": sample_name, **stats})

        if expected_var_names is None:
            raise ValueError("At least one sample is required for rank calculation")
        if total_cells_expected == 0:
            raise ValueError(
                "No cells remain after annotation, cell-type-size, and gene-count filtering"
            )
        if reordered_gene_samples:
            logger.warning(
                "Detected identical genes in a different column order for "
                f"{len(reordered_gene_samples)} samples; columns will be aligned to "
                f"the first section: {reordered_gene_samples}"
            )

        # Display filtering summary table
        table = Table(
            title="[bold]Cell Filtering Summary[/bold]",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Sample", style="dim")
        table.add_column("Input Cells", justify="right")
        table.add_column("NaN Removed", justify="right", style="red")
        table.add_column("Small Group Removed", justify="right", style="red")
        table.add_column("Low Gene Count Removed", justify="right", style="red")
        table.add_column("Final Cells", justify="right", style="green")

        for stat in filtering_stats:
            table.add_row(
                stat["Sample"],
                str(stat["input_cells"]),
                str(stat["nan_removed"]),
                str(stat["small_group_removed"]),
                str(stat["low_gene_count_removed"]),
                str(stat["final_cells"]),
            )

        # Add a total row
        table.add_section()
        table.add_row(
            "[bold]Total[/bold]",
            str(sum(s["input_cells"] for s in filtering_stats)),
            str(sum(s["nan_removed"] for s in filtering_stats)),
            str(sum(s["small_group_removed"] for s in filtering_stats)),
            str(sum(s["low_gene_count_removed"] for s in filtering_stats)),
            f"[bold green]{total_cells_expected}[/bold green]",
        )

        if self.show_rich_progress:
            self.console.print(table)
        else:
            logger.info(
                "Cell filtering summary: "
                f"input={sum(s['input_cells'] for s in filtering_stats):,}; "
                f"final={total_cells_expected:,}; "
                f"removed_nan={sum(s['nan_removed'] for s in filtering_stats):,}; "
                "removed_small_group="
                f"{sum(s['small_group_removed'] for s in filtering_stats):,}; "
                "removed_low_gene_count="
                f"{sum(s['low_gene_count_removed'] for s in filtering_stats):,}"
            )
        logger.info(f"Expected total cells after filtering: {total_cells_expected}")
        rank_progress_reporter = (
            _RankProgressReporter(total_cells_expected) if total_cells_expected > 0 else None
        )
        rank_section_started_at = time.perf_counter()
        logger.info(
            f"Rank progress will be logged every {_RANK_LOG_PERCENT_STEP}% or "
            f"{_RANK_LOG_INTERVAL_SECONDS:.0f}s for {total_cells_expected:,} cells"
        )
        rank_input_requests = [
            _RankInputRequest(
                sample_name=sample_name,
                h5ad_path=Path(h5ad_path),
                slice_id=st_id,
                keep_mask=sample_keep_indices[sample_name],
                data_layer=data_layer,
                expected_var_names=expected_var_names,
            )
            for st_id, (sample_name, h5ad_path) in enumerate(sample_h5ad_dict.items())
            if int(sample_keep_indices[sample_name].sum()) > 0
        ]
        cumulative_input_seconds = 0.0
        exposed_input_wait_seconds = 0.0
        logger.info("Rank input plan: selective HDF5 components with one-section prefetch")

        # Create overall section progress tracking
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
            disable=not self.show_rich_progress,
            refresh_per_second=1,
        ) as section_progress:
            # Overall section progress task
            section_task = section_progress.add_task(
                "Processing sections", total=len(rank_input_requests)
            )

            processed_cells_total = 0

            with _prefetched_rank_inputs(rank_input_requests) as prefetched_inputs:
                for rank_input, prefetch_wait_seconds in prefetched_inputs:
                    st_id = rank_input.slice_id
                    sample_name = rank_input.sample_name
                    X = rank_input.X
                    cumulative_input_seconds += rank_input.input_seconds
                    exposed_input_wait_seconds += prefetch_wait_seconds

                    input_message = (
                        f"Rank input ready: {sample_name} "
                        f"({st_id + 1}/{len(sample_h5ad_dict)}); "
                        f"source={rank_input.matrix_source}; "
                        f"metadata={rank_input.metadata_read_seconds:.2f}s, "
                        f"matrix={rank_input.matrix_read_seconds:.2f}s, "
                        f"obsm={rank_input.obsm_read_seconds:.2f}s, "
                        f"prepare={rank_input.prepare_seconds:.2f}s, "
                        "gene_order="
                        f"{'aligned' if rank_input.gene_order_aligned else 'canonical'}, "
                        f"exposed_wait={prefetch_wait_seconds:.2f}s"
                    )
                    logger.info(input_message)

                    # Get gene list (should be consistent across sections)
                    if gene_list is None:
                        gene_list = rank_input.var_names
                    elif rank_input.var_names != gene_list:
                        raise RuntimeError(
                            f"Internal gene alignment failure in section {st_id} ({sample_name})"
                        )

                    if rank_memmap is None:
                        n_genes = len(gene_list)
                        rank_batch_size = (
                            self.config.rank_batch_size
                            or _automatic_sparse_rank_batch_size(n_genes)
                        )
                        # Sparse batches are already tens of MiB and can be handed directly
                        # to the asynchronous writer without an extra np.vstack copy.
                        rank_write_interval = self.config.rank_write_interval or 1
                        rank_writer_workers = self.config.mkscore_write_workers
                        logger.info(
                            f"Resolved rank plan: backend=sparse-numba, "
                            f"batch={rank_batch_size}, "
                            f"write interval={rank_write_interval}, "
                            f"writers={rank_writer_workers}"
                        )
                        # Initialize the dense float16 rank matrix once its width is known.
                        # Each queued write holds up to batch x interval float16 rows;
                        # keep at most ~2 GiB of them waiting for the filesystem.
                        write_bytes = rank_batch_size * rank_write_interval * n_genes * 2
                        rank_memmap = MemMapDense(
                            str(rank_memmap_path),
                            shape=(total_cells_expected, n_genes),
                            dtype=np.float16,
                            mode="w",
                            num_write_workers=rank_writer_workers,
                            tmp_dir=self.config.memmap_tmp_dir,
                            queue_size=min(100, max(2, (2 * 1024**3) // max(1, write_bytes))),
                        )
                        self._active_rank_memmap = rank_memmap
                        sum_log_ranks = np.zeros(n_genes, dtype=np.float64)
                        sum_frac = np.zeros(n_genes, dtype=np.float64)

                    n_cells = X.shape[0]

                    with Progress(
                        SpinnerColumn(),
                        TextColumn(
                            f"[bold blue]Ranking {sample_name} ({{task.fields[cells]}} cells)"
                        ),
                        BarColumn(bar_width=None),
                        MofNCompleteColumn(),
                        TaskProgressColumn(),
                        TextColumn("[bold green]{task.fields[speed]} cells/s"),
                        TimeRemainingColumn(),
                        console=self.console,
                        disable=not self.show_rich_progress,
                        refresh_per_second=2,
                        transient=True,
                    ) as chunk_progress:
                        chunk_task = chunk_progress.add_task(
                            "Processing chunks", total=n_cells, cells=n_cells, speed="0"
                        )

                        metadata = {
                            "name": sample_name,
                            "cells": n_cells,
                            "study_id": st_id,
                        }
                        sample_rank_started_at = time.perf_counter()
                        batch_sum_log_ranks, batch_frac = rank_data_jax(
                            X,
                            n_genes,
                            memmap_dense=rank_memmap,
                            metadata=metadata,
                            chunk_size=rank_batch_size,
                            write_interval=rank_write_interval,
                            current_row_offset=current_row_offset,
                            progress=chunk_progress,
                            progress_task=chunk_task,
                            progress_reporter=rank_progress_reporter,
                        )
                        sample_rank_seconds = time.perf_counter() - sample_rank_started_at

                    sample_rank_throughput = n_cells / sample_rank_seconds
                    logger.info(
                        f"Rank sample complete: {sample_name} "
                        f"({st_id + 1}/{len(sample_h5ad_dict)}); {n_cells:,} cells in "
                        f"{sample_rank_seconds:.2f}s "
                        f"({sample_rank_throughput:,.0f} cells/s)"
                    )

                    sum_log_ranks += batch_sum_log_ranks
                    sum_frac += batch_frac
                    total_cells += n_cells
                    current_row_offset += n_cells
                    processed_cells_total += n_cells
                    section_progress.update(section_task, advance=1)

                    # The output only needs cell metadata and latent arrays. The
                    # expression matrix is released at the end of this iteration.
                    minimal_adata = ad.AnnData(
                        X=csr_matrix((n_cells, n_genes), dtype=np.float32),
                        obs=rank_input.obs,
                        var=pd.DataFrame(index=gene_list),
                        obsm=rank_input.obsm,
                    )
                    adata_list.append(minimal_adata)
                    n_total_cells += n_cells

                    del rank_input, X, minimal_adata
                    gc.collect()

            hidden_input_seconds = max(0.0, cumulative_input_seconds - exposed_input_wait_seconds)
            hidden_input_fraction = (
                hidden_input_seconds / cumulative_input_seconds
                if cumulative_input_seconds > 0
                else 0.0
            )
            logger.info(
                "Rank input pipeline summary: "
                f"selective_load={cumulative_input_seconds:.2f}s; "
                f"exposed_wait={exposed_input_wait_seconds:.2f}s; "
                f"hidden_by_prefetch={hidden_input_seconds:.2f}s "
                f"({hidden_input_fraction:.1%})"
            )

            rank_section_seconds = time.perf_counter() - rank_section_started_at
            active_rank_seconds = (
                rank_progress_reporter.active_seconds if rank_progress_reporter else 0.0
            )
            active_rank_throughput = (
                rank_progress_reporter.throughput if rank_progress_reporter else 0.0
            )
            wall_rank_throughput = (
                processed_cells_total / rank_section_seconds if rank_section_seconds > 0 else 0.0
            )
            logger.info(
                f"Rank calculation summary: {processed_cells_total:,} cells; "
                f"active={active_rank_seconds:.2f}s "
                f"({active_rank_throughput:,.0f} cells/s); "
                f"section wall={rank_section_seconds:.2f}s "
                f"({wall_rank_throughput:,.0f} cells/s)"
            )

        # Calculate mean log ranks and mean fraction
        mean_log_ranks = sum_log_ranks / total_cells
        mean_frac = sum_frac / total_cells

        # Save mean and fraction to parquet file
        mean_frac_df = pd.DataFrame(
            data=dict(
                G_Mean=mean_log_ranks,
                frac=mean_frac,
                gene_name=gene_list,
            ),
            index=gene_list,
        )
        # Save outside the progress context
        mean_frac_df.to_parquet(
            mean_frac_path,
            index=True,
            compression="gzip",
        )
        logger.info(f"Mean fraction data saved to {mean_frac_path}")

        # Concatenate all sections
        if adata_list:
            status = (
                self.console.status(
                    "[bold blue]Concatenating and saving latent representations..."
                )
                if self.show_rich_progress
                else nullcontext()
            )
            if not self.show_rich_progress:
                logger.info("Concatenating and saving latent representations")
            with status:
                concatenated_adata = ad.concat(adata_list, axis=0, join="outer", merge="same")

                # Ensure the var_names are the common genes
                concatenated_adata.var_names = gene_list

                # Save concatenated adata
                concatenated_adata.write_h5ad(concat_adata_path)
                logger.info(f"Saved concatenated latent representations to {concat_adata_path}")
                logger.info(f"  - Total cells: {concatenated_adata.n_obs}")
                logger.info(f"  - Total genes: {concatenated_adata.n_vars}")
                logger.info(
                    f"  - Latent representations in obsm: {list(concatenated_adata.obsm.keys())}"
                )
                if "slice_id" in concatenated_adata.obs.columns:
                    logger.info(
                        f"  - Number of slices: {concatenated_adata.obs['slice_id'].nunique()}"
                    )

                # Clean up
                del adata_list, concatenated_adata
                gc.collect()

        # Publish the rank checkpoint only after both sidecar outputs are
        # durable. Otherwise a failed parquet/H5AD write leaves a completed rank
        # file that refuses safe recreation on the next run.
        if rank_memmap is None:
            raise RuntimeError("Rank memmap was not initialized")
        if current_row_offset != total_cells_expected:
            raise RuntimeError(
                f"Rank matrix has {current_row_offset:,} written rows but "
                f"{total_cells_expected:,} cells passed filtering; refusing to publish it"
            )
        publish_started_at = time.perf_counter()
        rank_memmap.publish()
        publish_seconds = time.perf_counter() - publish_started_at
        rank_output_bytes = total_cells_expected * n_genes * np.dtype(np.float16).itemsize
        publish_mib_per_second = (
            rank_output_bytes / _MIB / publish_seconds if publish_seconds > 0 else 0.0
        )
        logger.info(
            f"Published rank matrix to {rank_memmap_path}; "
            f"marker scoring will read {rank_memmap.data_path}"
        )
        logger.info(
            f"Rank output finalization: {rank_output_bytes / 1024**3:.2f} GiB in "
            f"{publish_seconds:.2f}s ({publish_mib_per_second:,.0f} MiB/s); "
            f"scratch_used={rank_memmap.using_tmp}"
        )

        # Final completion message
        logger.info("Rank calculation and concatenation completed successfully")

        return {
            "concatenated_latent_adata": str(concat_adata_path),
            "rank_memmap": str(rank_memmap_path),
            "rank_memmap_handle": rank_memmap,
            "mean_frac": str(mean_frac_path),
        }
