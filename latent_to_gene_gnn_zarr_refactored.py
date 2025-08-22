#!/usr/bin/env python3
"""
Refactored latent_to_gene_gnn_zarr.py
Complete implementation with JAX acceleration and parallel I/O
"""

import logging
import queue
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
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import NearestNeighbors
import scanpy as sc
from tqdm import tqdm
from numba import njit, prange
import anndata as ad
from gsMap.find_latent_representation import ZarrBackedCSR

# Configure JAX
jax.config.update("jax_enable_x64", False)  # Use float32 for speed
jax.config.update("jax_platform_name", "cpu")  # or "gpu" if available

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ============================================================================
# Configuration
# ============================================================================

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
    num_write_workers: int = 2
    
    # Score calculation parameters
    expr_frac_threshold: float = 0.1
    min_cells_per_type: int = 10
    
    # Zarr parameters
    chunks_cells: int = 10000
    chunks_genes: int = 1000
    
    # Performance
    use_jax: bool = True
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
        mode: str = 'w'
    ):
        self.path = Path(path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        
        if chunks is None:
            chunks = (min(1000, shape[0]), min(1000, shape[1]))
        self.chunks = chunks
        
        # Initialize Zarr array
        if mode == 'w':
            store = zarr.DirectoryStore(str(self.path))
            self.zarr_array = zarr.open(
                store,
                mode='w',
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                compressor=zarr.Blosc(cname='lz4', clevel=3)
            )
        else:
            self.zarr_array = zarr.open(str(self.path), mode='r')
        
        # Async writing setup
        self.write_queue = queue.Queue(maxsize=100)
        self.writer_thread = None
        self.stop_writer = threading.Event()
        
        if mode == 'w':
            self._start_writer_thread()
    
    def _start_writer_thread(self):
        """Start background writer thread"""
        def writer_worker():
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
                    logger.error(f"Writer thread error: {e}")
                    raise
        
        self.writer_thread = threading.Thread(target=writer_worker, daemon=True)
        self.writer_thread.start()
    
    def write_batch(self, data: np.ndarray, row_start: int, col_slice=slice(None)):
        """Queue batch for async writing"""
        row_end = row_start + data.shape[0]
        row_slice = slice(row_start, row_end)
        self.write_queue.put((data, row_slice, col_slice))
    
    def close(self):
        """Clean up resources"""
        if self.writer_thread and self.writer_thread.is_alive():
            logger.info("Closing ZarrBackedDense, waiting for writes...")
            self.write_queue.join()
            self.write_queue.put(None)
            self.writer_thread.join(timeout=5)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# Connectivity Matrix Builder with JAX
# ============================================================================

class ConnectivityMatrixBuilder:
    """Build connectivity matrix using JAX-accelerated computation"""
    
    def __init__(self, config: LatentToGeneConfig):
        self.config = config
        
    @partial(jit, static_argnums=(0, 4, 5, 6))
    def _compute_similarities_jax(
        self,
        query_emb: jnp.ndarray,
        target_emb: jnp.ndarray,
        indices: jnp.ndarray,
        k: int,
        metric: str = 'cosine',
        return_distances: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """JAX-accelerated similarity computation"""
        
        # Extract neighbors' embeddings
        neighbor_emb = target_emb[indices]  # shape: (n_query, k_candidates, d)
        
        if metric == 'cosine':
            # Normalize embeddings
            query_norm = query_emb / jnp.linalg.norm(query_emb, axis=1, keepdims=True)
            neighbor_norm = neighbor_emb / jnp.linalg.norm(
                neighbor_emb, axis=2, keepdims=True
            )
            
            # Compute cosine similarities
            similarities = jnp.einsum('nd,nkd->nk', query_norm, neighbor_norm)
            
            # Select top k
            top_k_idx = jnp.argsort(-similarities, axis=1)[:, :k]
            
            # Gather top k indices and similarities
            batch_idx = jnp.arange(len(query_emb))[:, None]
            top_k_neighbors = indices[batch_idx, top_k_idx]
            
            if return_distances:
                top_k_sims = similarities[batch_idx, top_k_idx]
                return top_k_neighbors, top_k_sims
            else:
                return top_k_neighbors, None
        
        else:
            raise NotImplementedError(f"Metric {metric} not implemented")
    
    def build_connectivity_matrix(
        self,
        coords: np.ndarray,
        emb_gcn: np.ndarray,
        emb_indv: np.ndarray,
        cell_mask: Optional[np.ndarray] = None,
        return_dense: bool = True
    ) -> Union[csr_matrix, np.ndarray]:
        """
        Build connectivity matrix for a group of cells
        
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
        
        # Step 2: Find spatial anchors using JAX
        logger.info(f"Finding {self.config.num_anchor} spatial anchors...")
        if self.config.use_jax:
            spatial_anchors, anchor_sims = self._compute_similarities_jax(
                jnp.array(emb_gcn[cell_mask]),
                jnp.array(emb_gcn),
                jnp.array(spatial_neighbors),
                self.config.num_anchor
            )
            spatial_anchors = np.array(spatial_anchors)
            anchor_sims = np.array(anchor_sims)
        else:
            spatial_anchors, anchor_sims = self._compute_similarities_numpy(
                emb_gcn[cell_mask],
                emb_gcn,
                spatial_neighbors,
                self.config.num_anchor
            )
        
        # Step 3: Find homogeneous spots using JAX
        logger.info(f"Finding {self.config.num_neighbour} homogeneous spots...")
        if self.config.use_jax:
            homogeneous_neighbors, homogeneous_weights = self._compute_similarities_jax(
                jnp.array(emb_indv[cell_mask]),
                jnp.array(emb_indv),
                jnp.array(spatial_anchors),
                self.config.num_neighbour
            )
            homogeneous_neighbors = np.array(homogeneous_neighbors)
            homogeneous_weights = np.array(homogeneous_weights)
        else:
            homogeneous_neighbors, homogeneous_weights = self._compute_similarities_numpy(
                emb_indv[cell_mask],
                emb_indv,
                spatial_anchors,
                self.config.num_neighbour
            )
        
        # Normalize weights to sum to 1 for each cell
        homogeneous_weights = homogeneous_weights / homogeneous_weights.sum(
            axis=1, keepdims=True
        )
        
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
    
    def _compute_similarities_numpy(
        self,
        query_emb: np.ndarray,
        target_emb: np.ndarray,
        indices: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Numpy fallback for similarity computation"""
        n_query = len(query_emb)
        top_k_neighbors = np.zeros((n_query, k), dtype=np.int64)
        top_k_sims = np.zeros((n_query, k), dtype=np.float32)
        
        for i in range(n_query):
            neighbor_emb = target_emb[indices[i]]
            
            # Cosine similarity
            query_norm = query_emb[i] / np.linalg.norm(query_emb[i])
            neighbor_norm = neighbor_emb / np.linalg.norm(
                neighbor_emb, axis=1, keepdims=True
            )
            similarities = neighbor_norm @ query_norm
            
            # Top k
            top_k_idx = np.argsort(-similarities)[:k]
            top_k_neighbors[i] = indices[i, top_k_idx]
            top_k_sims[i] = similarities[top_k_idx]
        
        return top_k_neighbors, top_k_sims


# ============================================================================
# Row Sorting for Cache Optimization
# ============================================================================

@njit(parallel=True)
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
                
                # Read from Zarr (benefits from chunk caching)
                rank_data = self.rank_zarr[flat_indices]
                
                # Convert to dense array if sparse
                if hasattr(rank_data, 'toarray'):
                    rank_data = rank_data.toarray()
                
                # Create mapping for reconstruction
                idx_map = {idx: i for i, idx in enumerate(flat_indices)}
                
                # Put result
                self.result_queue.put((batch_id, rank_data, idx_map, neighbor_indices))
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

@partial(jit, static_argnums=(3, 4))
def compute_marker_scores_jax(
    log_ranks: jnp.ndarray,  # (B*N) × G matrix
    weights: jnp.ndarray,  # B × N weight matrix
    neighbor_valid: jnp.ndarray,  # B × N validity mask
    batch_size: int,
    num_neighbors: int,
    global_log_gmean: jnp.ndarray,  # G-dimensional vector
    global_expr_frac: jnp.ndarray,  # G-dimensional vector
    expr_frac_threshold: float = 0.1
) -> jnp.ndarray:
    """
    JAX-accelerated marker score computation
    
    Returns:
        B × G marker scores
    """
    n_genes = log_ranks.shape[1]
    
    # Reshape to batch format
    log_ranks_3d = log_ranks.reshape(batch_size, num_neighbors, n_genes)
    
    # Mask invalid neighbors
    log_ranks_3d = jnp.where(
        neighbor_valid[:, :, None],
        log_ranks_3d,
        0.0
    )
    
    # Handle zeros: fill with background log rank
    is_zero = (log_ranks_3d == 0)
    zero_counts = is_zero.sum(axis=(0, 1))  # Per gene
    background_log_rank = jnp.log((zero_counts + 1) / 2)
    log_ranks_filled = jnp.where(is_zero, background_log_rank, log_ranks_3d)
    
    # Normalize weights accounting for valid neighbors
    weights_masked = weights * neighbor_valid
    weights_sum = weights_masked.sum(axis=1, keepdims=True)
    weights_normalized = jnp.where(
        weights_sum > 0,
        weights_masked / weights_sum,
        0.0
    )
    
    # Compute weighted geometric mean in log space
    weighted_log_mean = jnp.einsum('bn,bng->bg', weights_normalized, log_ranks_filled)
    
    # Compute expression fraction
    is_expressed = (log_ranks_3d != 0) & neighbor_valid[:, :, None]
    expr_frac = jnp.einsum('bn,bng->bg', weights_normalized, is_expressed.astype(jnp.float32))
    
    # Calculate marker score
    marker_score = jnp.exp(weighted_log_mean - global_log_gmean)
    
    # Apply expression fraction filter
    frac_mask = expr_frac > (global_expr_frac + expr_frac_threshold)
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
        global_expr_frac: np.ndarray
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
            # Get completed batch
            batch_idx, rank_data, idx_map, batch_neighbors = reader.get_result()
            
            batch_start = batch_idx * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, n_cells)
            batch_size = batch_end - batch_start
            
            # Reconstruct full rank matrix for batch
            n_genes = rank_data.shape[1]
            batch_ranks = np.zeros(
                (batch_size * self.config.num_neighbour, n_genes),
                dtype=np.float32
            )
            
            # Create validity mask
            neighbor_valid = np.ones(
                (batch_size, self.config.num_neighbour),
                dtype=bool
            )
            
            for i in range(batch_size):
                for j in range(self.config.num_neighbour):
                    neighbor_idx = batch_neighbors[i, j]
                    if neighbor_idx >= 0 and neighbor_idx in idx_map:
                        rank_idx = idx_map[neighbor_idx]
                        batch_ranks[i * self.config.num_neighbour + j] = rank_data[rank_idx]
                    else:
                        neighbor_valid[i, j] = False
            
            # Get batch weights
            batch_weights = neighbor_weights[batch_start:batch_end]
            
            # Compute marker scores using JAX
            if self.config.use_jax:
                marker_scores = compute_marker_scores_jax(
                    jnp.array(batch_ranks),
                    jnp.array(batch_weights),
                    jnp.array(neighbor_valid),
                    batch_size,
                    self.config.num_neighbour,
                    jnp.array(global_log_gmean),
                    jnp.array(global_expr_frac),
                    self.config.expr_frac_threshold
                )
                marker_scores = np.array(marker_scores)
            else:
                # Numpy fallback
                marker_scores = self._compute_marker_scores_numpy(
                    batch_ranks,
                    batch_weights,
                    neighbor_valid,
                    batch_size,
                    global_log_gmean,
                    global_expr_frac
                )
            
            # Write results (async)
            global_indices = cell_indices_sorted[batch_start:batch_end]
            output_zarr.write_batch(marker_scores, global_indices[0])
            
            pbar.update(1)
        
        pbar.close()
        reader.close()
    
    def _compute_marker_scores_numpy(
        self,
        log_ranks: np.ndarray,
        weights: np.ndarray,
        neighbor_valid: np.ndarray,
        batch_size: int,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray
    ) -> np.ndarray:
        """Numpy fallback for marker score computation"""
        n_genes = log_ranks.shape[1]
        marker_scores = np.zeros((batch_size, n_genes), dtype=np.float32)
        
        for i in range(batch_size):
            # Get this cell's data
            start_idx = i * self.config.num_neighbour
            end_idx = start_idx + self.config.num_neighbour
            cell_ranks = log_ranks[start_idx:end_idx]
            cell_weights = weights[i]
            cell_valid = neighbor_valid[i]
            
            # Apply validity mask
            cell_weights = cell_weights * cell_valid
            if cell_weights.sum() == 0:
                continue
            cell_weights = cell_weights / cell_weights.sum()
            
            # Compute weighted geometric mean
            for g in range(n_genes):
                gene_ranks = cell_ranks[:, g]
                
                # Handle zeros
                nonzero_mask = (gene_ranks > 0) & cell_valid
                if nonzero_mask.sum() == 0:
                    continue
                
                # Weighted geometric mean in log space
                log_values = np.log(gene_ranks[nonzero_mask])
                weights_subset = cell_weights[nonzero_mask]
                weights_subset = weights_subset / weights_subset.sum()
                
                weighted_log_mean = np.sum(log_values * weights_subset)
                
                # Expression fraction
                expr_frac = nonzero_mask.sum() / self.config.num_neighbour
                
                # Calculate marker score
                if expr_frac > global_expr_frac[g] + self.config.expr_frac_threshold:
                    marker_scores[i, g] = np.exp(weighted_log_mean - global_log_gmean[g])
        
        return marker_scores
    
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
        
        # Get dimensions
        n_cells = adata.n_obs
        rank_zarr = ZarrBackedCSR.open(self.config.rank_zarr_path, mode='r')
        n_genes = rank_zarr.shape[1]
        
        # Initialize output
        output_zarr = ZarrBackedDense(
            self.config.output_path,
            shape=(n_cells, n_genes),
            chunks=(self.config.chunks_cells, self.config.chunks_genes),
            mode='w'
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
                    global_expr_frac
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