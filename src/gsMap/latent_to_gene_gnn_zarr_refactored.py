#!/usr/bin/env python3
"""
Refactored latent_to_gene_gnn_zarr.py
Complete implementation with JAX acceleration and parallel I/O
"""

import logging
import queue
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Union
from functools import partial
import warnings

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import jit, vmap
import zarr
from numba import njit
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
from tqdm import tqdm
# Removed numba import - using JAX instead for acceleration
import anndata as ad
from gsMap.find_latent_representation import ZarrBackedCSR
# Configure JAX
jax.config.update("jax_enable_x64", False)  # Use float32 for speed
# jax.config.update("jax_platform_name", "cpu")  # or "gpu" if available

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# # ============================================================================
# # Configuration
# # ============================================================================

@dataclass
class LatentToGeneConfig:
    """Configuration for marker score calculation"""
    # Input paths
    latent_dir: str  # Directory containing concatenated_latent_adata.h5ad and mean_frac.parquet
    rank_zarr_path: str
    output_path: str
    
    # Latent representation keys
    latent_representation: str = "X_morpho_gcn"  # Spatial niche embedding
    latent_representation_indv: str = "X_morpho_indv"  # Cell identity embedding
    spatial_key: str = "spatial"
    annotation_key: str = "cell_type"
    
    # Connectivity matrix parameters
    num_neighbour_spatial: int = 201  # k1: spatial neighbors
    num_anchor: int = 51  # k2: spatial anchors
    num_neighbour: int = 21  # k3: homogeneous spots
    
    # Processing parameters
    batch_size: int = 1000
    num_read_workers: int = 4
    num_write_workers: int = 4  # Increased default write workers
    
    # GPU memory management
    gpu_batch_size: int = 500  # Batch size for GPU processing to avoid OOM
    
    # Score calculation parameters
    min_cells_per_type: int = 10
    
    # Zarr parameters
    chunks_cells: Optional[int] = None  # None means use optimal chunking (1, n_genes)
    chunks_genes: Optional[int] = None  # None means use full gene dimension
    
    # Performance
    cache_size_mb: int = 1000
    
    def __post_init__(self):
        """Validate configuration"""
        assert self.num_neighbour <= self.num_anchor
        assert self.num_anchor <= self.num_neighbour_spatial


# ============================================================================
# ZarrBackedDense: Dense matrix storage with async writing
# ============================================================================

class ZarrBackedDense:
    """Dense version of ZarrBackedCSR for cell × gene matrix storage"""
    
    def __init__(
        self,
        path: Union[str, Path],
        shape: Tuple[int, int],
        dtype=np.float32,
        chunks: Optional[Tuple[int, int]] = None,
        mode: str = 'w',
        num_write_workers: int = 4
    ):
        self.path = Path(path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.num_write_workers = num_write_workers
        
        # Default chunks: (1, n_genes) for optimal row-wise writing
        if chunks is None:
            chunks = (1, shape[1])
        self.chunks = chunks
        
        # Initialize Zarr array with integrity checking
        if mode == 'w':
            if self.path.exists():
                # Check if it's already complete
                try:
                    existing = zarr.open(str(self.path), mode='r')
                    if 'integrity_mark' in existing.attrs and existing.attrs['integrity_mark'] == 'complete':
                        raise ValueError(
                            f"ZarrBackedDense at {self.path} already exists and is marked as complete. "
                            f"Please delete it manually if you want to overwrite: rm -rf {self.path}"
                        )
                    else:
                        logger.warning(f"ZarrBackedDense at {self.path} exists but is incomplete. Deleting and recreating.")
                        import time
                        time.sleep(0.1)  # Brief pause to ensure files are released
                        shutil.rmtree(self.path, ignore_errors=True)
                        # Double-check deletion
                        if self.path.exists():
                            import os
                            os.system(f"rm -rf {self.path}")
                except ValueError:
                    # Re-raise the complete file error
                    raise
                except Exception as e:
                    logger.warning(f"Could not read existing Zarr at {self.path}: {e}. Deleting and recreating.")
                    import time
                    time.sleep(0.1)  # Brief pause to ensure files are released
                    shutil.rmtree(self.path, ignore_errors=True)
                    # Double-check deletion
                    if self.path.exists():
                        import os
                        os.system(f"rm -rf {self.path}")

            store = zarr.DirectoryStore(str(self.path))
            self.zarr_array = zarr.open(
                store,
                mode='w',
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                compressor=zarr.Blosc(cname='lz4', clevel=3)
            )
            # Mark as incomplete initially
            self.zarr_array.attrs['integrity_mark'] = 'incomplete'
        else:
            # Read mode - validate integrity
            self.zarr_array = zarr.open(str(self.path), mode='r')
            if 'integrity_mark' not in self.zarr_array.attrs:
                raise ValueError(f"ZarrBackedDense at {self.path} is incomplete or corrupted (no integrity mark).")
            if self.zarr_array.attrs['integrity_mark'] != 'complete':
                raise ValueError(f"ZarrBackedDense at {self.path} is incomplete (marked as {self.zarr_array.attrs['integrity_mark']}).")
        
        # Async writing setup (only if still in write mode)
        self.write_queue = queue.Queue(maxsize=100)
        self.writer_threads = []
        self.stop_writer = threading.Event()
        
        if mode == 'w' and self.mode == 'w':  # Only start writer if we're actually writing
            self._start_writer_threads()
    
    def _start_writer_threads(self):
        """Start multiple background writer threads"""
        def writer_worker(worker_id):
            while not self.stop_writer.is_set():
                try:
                    item = self.write_queue.get(timeout=0.1)
                    if item is None:
                        break
                    data, row_slice, col_slice = item
                    self.zarr_array[row_slice, col_slice] = data
                    self.write_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Writer thread {worker_id} error: {e}")
                    raise
        
        # Start multiple writer threads
        for i in range(self.num_write_workers):
            thread = threading.Thread(target=writer_worker, args=(i,), daemon=True)
            thread.start()
            self.writer_threads.append(thread)
        logger.info(f"Started {self.num_write_workers} writer threads for ZarrBackedDense")
    
    def write_batch(self, data: np.ndarray, row_indices: Union[int, np.ndarray], col_slice=slice(None)):
        """Queue batch for async writing
        
        Args:
            data: Data to write
            row_indices: Either a single row index or array of row indices
            col_slice: Column slice (default: all columns)
        """
        if self.mode != 'w':
            logger.warning("Cannot write to read-only ZarrBackedDense")
            return
        
        # Handle both single index and array of indices
        if isinstance(row_indices, (int, np.integer)):
            row_slice = slice(row_indices, row_indices + data.shape[0])
        else:
            row_slice = row_indices
        
        self.write_queue.put((data, row_slice, col_slice))
    
    def mark_complete(self):
        """Mark the zarr array as complete by setting integrity mark."""
        if self.mode == 'w':
            self.zarr_array.attrs['integrity_mark'] = 'complete'
            logger.info(f"Marked ZarrBackedDense at {self.path} as complete")
    
    def close(self):
        """Clean up resources"""
        if self.writer_threads:
            logger.info("Closing ZarrBackedDense, waiting for writes...")
            self.write_queue.join()
            self.stop_writer.set()
            # Send stop signal to all threads
            for _ in self.writer_threads:
                self.write_queue.put(None)
            # Wait for all threads to finish
            for thread in self.writer_threads:
                thread.join(timeout=5.0)
        
        # Mark as complete when closing in write mode
        if self.mode == 'w':
            self.mark_complete()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# Connectivity Matrix Builder with JAX
# ============================================================================

# Define the JIT-compiled function outside the class
@partial(jit, static_argnums=(5, 6))
def _find_anchors_and_homogeneous_batch_jit(
    emb_gcn_batch_norm: jnp.ndarray,      # (batch_size, d1) - pre-normalized
    emb_indv_batch_norm: jnp.ndarray,      # (batch_size, d2) - pre-normalized
    spatial_neighbors: jnp.ndarray,   # (batch_size, k1)
    all_emb_gcn_norm: jnp.ndarray,         # (n_all, d1) - pre-normalized
    all_emb_indv_norm: jnp.ndarray,        # (n_all, d2) - pre-normalized
    num_anchor: int,
    num_neighbour: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled function to find anchors and homogeneous neighbors.
    Processes a batch of cells to manage GPU memory.
    Expects pre-normalized embeddings for efficiency.
    """
    batch_size = emb_gcn_batch_norm.shape[0]
    
    # Step 1: Extract spatial neighbors' embeddings (already normalized)
    spatial_emb_gcn_norm = all_emb_gcn_norm[spatial_neighbors]  # (batch_size, k1, d1)
    
    # Step 2: Find spatial anchors via cosine similarity
    # Compute similarities (embeddings are already normalized)
    anchor_sims = jnp.einsum('bd,bkd->bk', emb_gcn_batch_norm, spatial_emb_gcn_norm)
    
    # Select top anchors
    top_anchor_idx = jnp.argsort(-anchor_sims, axis=1)[:, :num_anchor]
    batch_idx = jnp.arange(batch_size)[:, None]
    spatial_anchors = spatial_neighbors[batch_idx, top_anchor_idx]  # (batch_size, num_anchor)
    
    # Step 3: Find homogeneous neighbors from anchors
    # Extract anchor embeddings (already normalized)
    anchor_emb_indv_norm = all_emb_indv_norm[spatial_anchors]  # (batch_size, num_anchor, d2)
    
    # Compute similarities (embeddings are already normalized)
    homo_sims = jnp.einsum('bd,bkd->bk', emb_indv_batch_norm, anchor_emb_indv_norm)
    
    # Select top homogeneous neighbors
    top_homo_idx = jnp.argsort(-homo_sims, axis=1)[:, :num_neighbour]
    homogeneous_neighbors = spatial_anchors[batch_idx, top_homo_idx]  # (batch_size, num_neighbour)
    homogeneous_weights = homo_sims[batch_idx, top_homo_idx]
    
    # Use softmax to normalize weights
    homogeneous_weights = jax.nn.softmax(homogeneous_weights, axis=1)
    
    return homogeneous_neighbors, homogeneous_weights


class ConnectivityMatrixBuilder:
    """Build connectivity matrix using JAX-accelerated computation with GPU memory optimization"""
    
    def __init__(self, config: LatentToGeneConfig):
        self.config = config
        # Use configured batch size for GPU processing
        self.gpu_batch_size = config.gpu_batch_size
    
    def build_connectivity_matrix(
        self,
        coords: np.ndarray,
        emb_gcn: np.ndarray,
        emb_indv: np.ndarray,
        cell_mask: Optional[np.ndarray] = None,
        return_dense: bool = True
    ) -> Union[csr_matrix, np.ndarray]:
        """
        Build connectivity matrix for a group of cells with GPU memory optimization
        
        Args:
            coords: Spatial coordinates (n_cells, 2 or 3)
            emb_gcn: Spatial niche embeddings (n_cells, d1)
            emb_indv: Cell identity embeddings (n_cells, d2)
            cell_mask: Boolean mask for cells to process
            return_dense: If True, return dense (n_cells, k) array
        
        Returns:
            Connectivity matrix (sparse or dense format)
        """
        
        n_cells = len(coords)
        if cell_mask is None:
            cell_mask = np.ones(n_cells, dtype=bool)
        
        cell_indices = np.where(cell_mask)[0]
        n_masked = len(cell_indices)
        
        # Step 1: Find spatial neighbors using sklearn (CPU-optimized)
        logger.info(f"Finding {self.config.num_neighbour_spatial} spatial neighbors...")
        nbrs_spatial = NearestNeighbors(
            n_neighbors=min(self.config.num_neighbour_spatial, n_masked),
            metric='euclidean',
            algorithm='kd_tree'
        )
        nbrs_spatial.fit(coords[cell_mask])
        _, spatial_neighbors = nbrs_spatial.kneighbors(coords[cell_mask])
        
        # Convert to global indices
        spatial_neighbors = cell_indices[spatial_neighbors]
        
        # Step 2 & 3: Find anchors and homogeneous neighbors in batches
        logger.info(f"Finding anchors and homogeneous neighbors (batch size: {self.gpu_batch_size})...")
        
        # Pre-normalize embeddings once for all batches
        logger.debug("Pre-normalizing embeddings...")
        # L2 normalize for cosine similarity
        emb_gcn_norm = emb_gcn / np.linalg.norm(emb_gcn, axis=1, keepdims=True)
        emb_indv_norm = emb_indv / np.linalg.norm(emb_indv, axis=1, keepdims=True)
        
        # Handle any NaN from zero vectors
        emb_gcn_norm = np.nan_to_num(emb_gcn_norm, nan=0.0)
        emb_indv_norm = np.nan_to_num(emb_indv_norm, nan=0.0)
        
        # Convert to JAX arrays once
        all_emb_gcn_norm_jax = jnp.array(emb_gcn_norm)
        all_emb_indv_norm_jax = jnp.array(emb_indv_norm)
        
        # Process in batches to avoid GPU OOM
        homogeneous_neighbors_list = []
        homogeneous_weights_list = []
        
        for batch_start in range(0, n_masked, self.gpu_batch_size):
            batch_end = min(batch_start + self.gpu_batch_size, n_masked)
            batch_indices = slice(batch_start, batch_end)
            
            # Get batch data (already normalized)
            emb_gcn_batch_norm = emb_gcn_norm[cell_mask][batch_indices]
            emb_indv_batch_norm = emb_indv_norm[cell_mask][batch_indices]
            spatial_neighbors_batch = spatial_neighbors[batch_indices]
            
            # Process batch with single JIT-compiled function
            homo_neighbors_batch, homo_weights_batch = _find_anchors_and_homogeneous_batch_jit(
                jnp.array(emb_gcn_batch_norm),
                jnp.array(emb_indv_batch_norm),
                jnp.array(spatial_neighbors_batch),
                all_emb_gcn_norm_jax,
                all_emb_indv_norm_jax,
                self.config.num_anchor,
                self.config.num_neighbour
            )
            
            # Convert back to numpy and append
            homogeneous_neighbors_list.append(np.array(homo_neighbors_batch))
            homogeneous_weights_list.append(np.array(homo_weights_batch))
        
        # Concatenate all batches
        homogeneous_neighbors = np.vstack(homogeneous_neighbors_list)
        homogeneous_weights = np.vstack(homogeneous_weights_list)
        
        if return_dense:
            # Return dense format: (n_masked, num_neighbour) arrays
            return homogeneous_neighbors, homogeneous_weights
        else:
            # Build sparse matrix
            rows = np.repeat(cell_indices, self.config.num_neighbour)
            cols = homogeneous_neighbors.flatten()
            data = homogeneous_weights.flatten()
            
            connectivity = csr_matrix(
                (data, (rows, cols)),
                shape=(n_cells, n_cells)
            )
            return connectivity

# ============================================================================
# Row Sorting for Cache Optimization
# ============================================================================

@njit()
def compute_jaccard_similarity(neighbors_a: np.ndarray, neighbors_b: np.ndarray) -> float:
    """Compute Jaccard similarity between two neighbor sets"""
    set_a = set(neighbors_a)
    set_b = set(neighbors_b)
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0

def optimize_row_order(
    neighbor_indices: np.ndarray,
    method: str = 'greedy'
) -> np.ndarray:
    """
    Sort rows by shared neighbors to improve cache locality
    
    Args:
        neighbor_indices: (n_cells, k) array of neighbor indices
        method: 'greedy' or 'spectral'
    
    Returns:
        Reordered row indices
    """
    n_cells = len(neighbor_indices)
    
    if method == 'greedy':
        # Greedy approach: iteratively select most similar rows
        remaining = set(range(n_cells))
        ordered = []
        
        # Start with random row
        current = np.random.choice(list(remaining))
        ordered.append(current)
        remaining.remove(current)

        while remaining:
            # Find most similar remaining row
            max_sim = -1
            next_row = None
            
            for candidate in remaining:
                sim = compute_jaccard_similarity(
                    neighbor_indices[current],
                    neighbor_indices[candidate]
                )
                if sim > max_sim:
                    max_sim = sim
                    next_row = candidate
            
            ordered.append(next_row)
            remaining.remove(next_row)
            current = next_row
        
        return np.array(ordered)
    
    else:
        # Simple sequential order as fallback
        return np.arange(n_cells)


# ============================================================================
# Parallel Reading System
# ============================================================================

class ParallelRankReader:
    """Multi-threaded reader for log-rank data"""
    
    def __init__(
        self,
        rank_zarr_path: str,
        num_workers: int = 4,
        cache_size_mb: int = 1000
    ):

        self.rank_zarr = ZarrBackedCSR.open(rank_zarr_path, mode='r')
        self.num_workers = num_workers
        
        # Queues for communication
        self.read_queue = queue.Queue(maxsize=100)
        self.result_queue = queue.Queue(maxsize=100)
        
        # Start worker threads
        self.workers = []
        self.stop_workers = threading.Event()
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker(self, worker_id: int):
        """Worker thread for reading batches"""
        logger.info(f"Reader worker {worker_id} started")
        
        while not self.stop_workers.is_set():
            try:
                # Get batch request
                item = self.read_queue.get(timeout=0.1)
                if item is None:
                    break
                
                batch_id, neighbor_indices = item
                
                # Flatten and deduplicate indices for efficient reading
                flat_indices = np.unique(neighbor_indices.flatten())
                
                # Validate indices are within bounds
                max_idx = self.rank_zarr.shape[0] - 1
                assert flat_indices.max() <= max_idx, \
                    f"Worker {worker_id}: Indices exceed bounds (max: {flat_indices.max()}, limit: {max_idx})"
                
                # Read from Zarr (batch access bug fixed)
                rank_data = self.rank_zarr[flat_indices]
                
                # Convert to dense array if sparse
                if hasattr(rank_data, 'toarray'):
                    rank_data = rank_data.toarray()
                
                # Create mapping for reconstruction
                idx_map = {idx: i for i, idx in enumerate(flat_indices)}
                
                # Map neighbor indices to rank_data indices
                flat_neighbors = neighbor_indices.flatten()
                rank_indices = np.array([idx_map[neighbor_idx] for neighbor_idx in flat_neighbors])
                
                # Put result - send rank_data and rank_indices for main thread to combine
                self.result_queue.put((batch_id, rank_data, rank_indices, neighbor_indices.shape))
                self.read_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Reader worker {worker_id} error: {e}")
                raise
    
    def submit_batch(self, batch_id: int, neighbor_indices: np.ndarray):
        """Submit batch for reading"""
        self.read_queue.put((batch_id, neighbor_indices))
    
    def get_result(self):
        """Get next completed batch"""
        return self.result_queue.get()
    
    def close(self):
        """Clean up resources"""
        self.stop_workers.set()
        for _ in range(self.num_workers):
            self.read_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5)


# ============================================================================
# JAX Marker Score Computation
# ============================================================================

@partial(jit, static_argnums=(2, 3))
def compute_marker_scores_jax(
    log_ranks: jnp.ndarray,  # (B*N) × G matrix
    weights: jnp.ndarray,  # B × N weight matrix
    batch_size: int,
    num_neighbors: int,
    global_log_gmean: jnp.ndarray,  # G-dimensional vector
    global_expr_frac: jnp.ndarray  # G-dimensional vector
) -> jnp.ndarray:
    """
    JAX-accelerated marker score computation
    
    Returns:
        B × G marker scores
    """
    n_genes = log_ranks.shape[1]
    
    # Reshape to batch format
    log_ranks_3d = log_ranks.reshape(batch_size, num_neighbors, n_genes)
    
    # Handle zeros: fill with background log rank (cell-specific)
    is_zero = (log_ranks_3d == 0)
    # Sum zeros along neighbor dimension (axis=1) for each cell
    zero_counts_per_cell = is_zero.sum(axis=1, keepdims=True)  # Shape: (B, 1, G)
    background_log_rank = jnp.log((zero_counts_per_cell + 1) / 2)
    log_ranks_filled = jnp.where(is_zero, background_log_rank, log_ranks_3d)
    
    # Normalize weights
    weights_sum = weights.sum(axis=1, keepdims=True)
    weights_normalized = weights / weights_sum
    
    # Compute weighted geometric mean in log space
    weighted_log_mean = jnp.einsum('bn,bng->bg', weights_normalized, log_ranks_filled)
    
    # Compute expression fraction (mean of is_expressed across neighbors)
    is_expressed = (log_ranks_3d != 0)
    expr_frac = is_expressed.astype(jnp.float32).mean(axis=1)  # Mean across neighbors
    
    # Calculate marker score
    marker_score = jnp.exp(weighted_log_mean - global_log_gmean)
    
    # Apply expression fraction filter
    frac_mask = expr_frac > global_expr_frac
    marker_score = jnp.where(frac_mask, marker_score, 0.0)
    
    return marker_score


# ============================================================================
# Main Marker Score Calculator
# ============================================================================

class MarkerScoreCalculator:
    """Main class for calculating marker scores"""
    
    def __init__(self, config: LatentToGeneConfig):
        self.config = config
        self.connectivity_builder = ConnectivityMatrixBuilder(config)
        
    def load_global_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-calculated global geometric mean and expression fraction from parquet"""

        logger.info("Loading global statistics from parquet...")
        parquet_path = Path(self.config.latent_dir) / "mean_frac.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Global stats file not found: {parquet_path}")
        
        # Load the dataframe
        mean_frac_df = pd.read_parquet(parquet_path)
        
        # Extract global log geometric mean and expression fraction
        global_log_gmean = mean_frac_df['G_Mean'].values.astype(np.float32)
        global_expr_frac = mean_frac_df['frac'].values.astype(np.float32)
        
        logger.info(f"Loaded global stats for {len(global_log_gmean)} genes")
        
        return global_log_gmean, global_expr_frac
    
    def process_cell_type(
        self,
        adata: ad.AnnData,
        cell_type: str,
        output_zarr: ZarrBackedDense,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray,
        rank_zarr_shape: Tuple[int, int]
    ):
        """Process a single cell type"""
        
        # Get cells of this type
        cell_mask = adata.obs[self.config.annotation_key] == cell_type
        cell_indices = np.where(cell_mask)[0]
        n_cells = len(cell_indices)
        
        if n_cells < self.config.min_cells_per_type:
            logger.warning(f"Skipping {cell_type}: only {n_cells} cells")
            return
        
        logger.info(f"Processing {cell_type}: {n_cells} cells")
        
        # Validate dimensions
        n_cells_total = adata.n_obs
        n_cells_rank = rank_zarr_shape[0]
        
        assert n_cells_total == n_cells_rank, \
            f"Dimension mismatch: AnnData has {n_cells_total} cells, rank zarr has {n_cells_rank} cells"
        
        # Extract embeddings and coordinates
        coords = adata.obsm[self.config.spatial_key][cell_mask]
        emb_gcn = adata.obsm[self.config.latent_representation][cell_mask].astype(np.float32)
        emb_indv = adata.obsm[self.config.latent_representation_indv][cell_mask].astype(np.float32)
        
        # Build connectivity matrix
        logger.info("Building connectivity matrix...")
        neighbor_indices, neighbor_weights = self.connectivity_builder.build_connectivity_matrix(
            coords=adata.obsm[self.config.spatial_key],
            emb_gcn=adata.obsm[self.config.latent_representation].astype(np.float32),
            emb_indv=adata.obsm[self.config.latent_representation_indv].astype(np.float32),
            cell_mask=cell_mask,
            return_dense=True
        )
        
        # Validate neighbor indices are within bounds
        max_valid_idx = rank_zarr_shape[0] - 1
        assert neighbor_indices.max() <= max_valid_idx, \
            f"Neighbor indices exceed bounds (max: {neighbor_indices.max()}, limit: {max_valid_idx})"
        assert neighbor_indices.min() >= 0, \
            f"Found negative neighbor indices (min: {neighbor_indices.min()})"
        
        # Optimize row order
        logger.info("Optimizing row order for cache efficiency...")
        row_order = optimize_row_order(neighbor_indices)
        neighbor_indices = neighbor_indices[row_order]
        neighbor_weights = neighbor_weights[row_order]
        cell_indices_sorted = cell_indices[row_order]
        
        # Initialize parallel reader
        reader = ParallelRankReader(
            self.config.rank_zarr_path,
            num_workers=self.config.num_read_workers
        )
        
        # Process in batches
        n_batches = (n_cells + self.config.batch_size - 1) // self.config.batch_size
        
        # Submit all read requests
        logger.info(f"Submitting {n_batches} batches for reading...")
        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, n_cells)
            
            batch_neighbors = neighbor_indices[batch_start:batch_end]
            reader.submit_batch(batch_idx, batch_neighbors)
        
        # Process results as they complete
        logger.info("Processing batches...")
        pbar = tqdm(total=n_batches, desc=f"Processing {cell_type}")
        
        for _ in range(n_batches):
            # Get completed batch - rank_data and indices from worker
            batch_idx, rank_data, rank_indices, original_shape = reader.get_result()
            
            batch_start = batch_idx * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, n_cells)
            batch_size = batch_end - batch_start
            
            # Verify shape
            assert original_shape == (batch_size, self.config.num_neighbour), \
                f"Shape mismatch: expected {(batch_size, self.config.num_neighbour)}, got {original_shape}"
            
            # Use fancy indexing in main thread to save memory
            batch_ranks = rank_data[rank_indices]
            
            # Get batch weights
            batch_weights = neighbor_weights[batch_start:batch_end]
            
            # Compute marker scores using JAX (always use JAX)
            marker_scores = compute_marker_scores_jax(
                jnp.array(batch_ranks),
                jnp.array(batch_weights),
                batch_size,
                self.config.num_neighbour,
                jnp.array(global_log_gmean),
                jnp.array(global_expr_frac)
            )
            marker_scores = np.array(marker_scores)
            
            # Write results (async)
            global_indices = cell_indices_sorted[batch_start:batch_end]
            output_zarr.write_batch(marker_scores, global_indices)
            
            pbar.update(1)
        
        pbar.close()
        reader.close()
    
    
    def run(self):
        """Main execution function"""
        logger.info("Starting marker score calculation...")
        
        # Load concatenated AnnData
        adata_path = Path(self.config.latent_dir) / "concatenated_latent_adata.h5ad"
        logger.info(f"Loading concatenated AnnData from {adata_path}")
        
        if not adata_path.exists():
            raise FileNotFoundError(f"Concatenated AnnData not found: {adata_path}")
        
        adata = sc.read_h5ad(adata_path)
        
        # Load pre-calculated global statistics
        global_log_gmean, global_expr_frac = self.load_global_stats()
        
        # Validate annotation groups have enough cells
        if self.config.annotation_key and self.config.annotation_key in adata.obs.columns:
            annotation_counts = adata.obs[self.config.annotation_key].value_counts()
            min_required = self.config.num_neighbour  # Need at least as many cells as neighbors
            
            # Check for groups with too few cells
            small_groups = annotation_counts[annotation_counts < min_required]
            assert len(small_groups) == 0, \
                f"Found {len(small_groups)} annotation groups with fewer than {min_required} cells. " \
                f"Groups: {small_groups.to_dict()}. These should have been filtered in find_latent_representation."
            
            logger.info(f"All annotation groups have at least {min_required} cells")
        
        # Get dimensions
        n_cells = adata.n_obs
        rank_zarr = ZarrBackedCSR.open(self.config.rank_zarr_path, mode='r')
        n_cells_rank = rank_zarr.shape[0]
        n_genes = rank_zarr.shape[1]
        
        logger.info(f"AnnData dimensions: {n_cells} cells × {adata.n_vars} genes")
        logger.info(f"Rank Zarr dimensions: {n_cells_rank} cells × {n_genes} genes")
        
        # Cells should match exactly since filtering is done before rank zarr creation
        assert n_cells == n_cells_rank, \
            f"Cell count mismatch: AnnData has {n_cells} cells, Rank Zarr has {n_cells_rank} cells. " \
            f"This indicates the filtering was not applied consistently during find_latent_representation."
        
        # Initialize output with proper chunking
        chunks = None
        if self.config.chunks_cells is not None or self.config.chunks_genes is not None:
            # Use provided chunks
            chunks = (
                self.config.chunks_cells if self.config.chunks_cells is not None else 1,
                self.config.chunks_genes if self.config.chunks_genes is not None else n_genes
            )
        # If chunks is None, ZarrBackedDense will use default (1, n_genes)
        
        output_zarr = ZarrBackedDense(
            self.config.output_path,
            shape=(n_cells, n_genes),
            chunks=chunks,
            mode='w',
            num_write_workers=self.config.num_write_workers
        )
        
        # Process each cell type
        cell_types = adata.obs[self.config.annotation_key].unique()
        logger.info(f"Processing {len(cell_types)} cell types")
        
        with output_zarr:
            for cell_type in cell_types:
                self.process_cell_type(
                    adata,
                    cell_type,
                    output_zarr,
                    global_log_gmean,
                    global_expr_frac,
                    rank_zarr_shape=(n_cells_rank, n_genes)
                )
        
        logger.info("Marker score calculation complete!")
        
        # Save metadata
        metadata = {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'config': self.config.__dict__,
            'global_log_gmean': global_log_gmean.tolist(),
            'global_expr_frac': global_expr_frac.tolist()
        }
        
        import json
        metadata_path = Path(self.config.output_path).parent / 'marker_scores_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Results saved to {self.config.output_path}")
        logger.info(f"Metadata saved to {metadata_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate marker scores from latent representations")
    parser.add_argument("--latent-dir", required=True, help="Directory containing concatenated_latent_adata.h5ad and mean_frac.parquet")
    parser.add_argument("--rank-zarr", required=True, help="Path to rank Zarr")
    parser.add_argument("--output", required=True, help="Output path for marker scores")
    parser.add_argument("--config", help="Path to config JSON file")
    parser.add_argument("--annotation-key", default="cell_type", help="Annotation key for cell types")
    parser.add_argument("--spatial-key", default="spatial", help="Key for spatial coordinates")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of read workers")
    parser.add_argument("--no-jax", action="store_true", help="Disable JAX acceleration")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        import json
        with open(args.config) as f:
            config_dict = json.load(f)
        config = LatentToGeneConfig(**config_dict)
    else:
        config = LatentToGeneConfig(
            latent_dir=args.latent_dir,
            rank_zarr_path=args.rank_zarr,
            output_path=args.output,
            annotation_key=args.annotation_key,
            spatial_key=args.spatial_key,
            batch_size=args.batch_size,
            num_read_workers=args.num_workers,
            use_jax=not args.no_jax
        )
    
    # Run calculation
    calculator = MarkerScoreCalculator(config)
    calculator.run()


if __name__ == "__main__":
    main()