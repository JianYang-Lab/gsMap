"""
Marker score calculation using homogeneous neighbors
Implements weighted geometric mean calculation in log space with JAX acceleration
"""

import logging
import queue
import threading
import json
from pathlib import Path
from typing import Optional, Tuple, Union
from functools import partial

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
from scipy.sparse import csr_matrix
import jax
import jax.numpy as jnp
from jax import jit

from .zarr_utils import ZarrBackedDense
from .connectivity import ConnectivityMatrixBuilder
from .row_ordering import optimize_row_order

logger = logging.getLogger(__name__)


class ParallelRankReader:
    """Multi-threaded reader for log-rank data from dense zarr"""
    
    def __init__(
        self,
        rank_zarr: Union[ZarrBackedDense, str],
        num_workers: int = 4,
        cache_size_mb: int = 1000
    ):
        if isinstance(rank_zarr, str):
            # Open as ZarrBackedDense in read mode
            import zarr
            z = zarr.open(str(rank_zarr), mode='r')
            self.rank_zarr = z  # Direct zarr array access for reading
            self.shape = z.shape
        else:
            self.rank_zarr = rank_zarr.zarr_array if hasattr(rank_zarr, 'zarr_array') else rank_zarr
            self.shape = rank_zarr.shape if hasattr(rank_zarr, 'shape') else self.rank_zarr.shape
        self.num_workers = num_workers
        
        # Queues for communication
        self.read_queue = queue.Queue()
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
        """Worker thread for reading batches from dense zarr"""
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
                max_idx = self.shape[0] - 1
                assert flat_indices.max() <= max_idx, \
                    f"Worker {worker_id}: Indices exceed bounds (max: {flat_indices.max()}, limit: {max_idx})"
                
                # Read from Dense Zarr (direct array access)
                # Dense zarr stores log-ranks directly
                rank_data = self.rank_zarr[flat_indices]
                
                # Ensure we have a numpy array
                if not isinstance(rank_data, np.ndarray):
                    rank_data = np.array(rank_data)
                
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
    
    # # Handle zeros: fill with background log rank (cell-specific)
    # is_zero = (log_ranks_3d == 0)
    # # Sum zeros along neighbor dimension (axis=1) for each cell
    # zero_counts_per_cell = is_zero.sum(axis=1, keepdims=True)  # Shape: (B, 1, G)
    # background_log_rank = jnp.log((zero_counts_per_cell + 1) / 2)
    # log_ranks_filled = jnp.where(is_zero, background_log_rank, log_ranks_3d)
    #
    # # Normalize weights (already softmax normalized, but ensure sum to 1)
    # weights_sum = weights.sum(axis=1, keepdims=True)
    # weights_normalized = weights / weights_sum
    
    # Compute weighted geometric mean in log space
    weighted_log_mean = jnp.einsum('bn,bng->bg', weights, log_ranks_3d)
    
    # Compute expression fraction (mean of is_expressed across neighbors)
    # is_expressed = (log_ranks_3d != 0)
    is_expressed = (log_ranks_3d != log_ranks_3d.min(axis=-1, keepdims=True))  # Treat min log rank as non-expressed
    expr_frac = is_expressed.astype(jnp.float32).mean(axis=1)  # Mean across neighbors
    
    # Calculate marker score
    marker_score = jnp.exp(weighted_log_mean - global_log_gmean)
    marker_score = jnp.where(marker_score < 1.0, 0.0, marker_score)

    # Apply expression fraction filter
    frac_mask = expr_frac > global_expr_frac
    marker_score = jnp.where(frac_mask, marker_score, 0.0)

    marker_score = jnp.exp(marker_score **1.5) - 1.0

    return marker_score


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
        
    def load_global_stats(self, mean_frac_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load pre-calculated global geometric mean and expression fraction from parquet"""
        
        logger.info("Loading global statistics from parquet...")
        parquet_path = Path(mean_frac_path)
        
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
        rank_zarr,  # Now a zarr array directly
        reader: ParallelRankReader,
        coords: np.ndarray,
        emb_gcn: np.ndarray,
        emb_indv: np.ndarray,
        annotation_key: str
    ):
        """Process a single cell type"""
        
        # Get cells of this type
        cell_mask = adata.obs[annotation_key] == cell_type
        cell_indices = np.where(cell_mask)[0]
        n_cells = len(cell_indices)
        
        min_cells = getattr(self.config, 'min_cells_per_type', 21)
        if n_cells < min_cells:
            logger.warning(f"Skipping {cell_type}: only {n_cells} cells (min: {min_cells})")
            return
        
        logger.info(f"Processing {cell_type}: {n_cells} cells")
        
        # Get rank zarr shape
        rank_zarr_shape = rank_zarr.shape if hasattr(rank_zarr, 'shape') else reader.shape
        
        # Build connectivity matrix
        logger.info("Building connectivity matrix...")
        neighbor_indices, neighbor_weights = self.connectivity_builder.build_connectivity_matrix(
            coords=coords,
            emb_gcn=emb_gcn,
            emb_indv=emb_indv,
            cell_mask=cell_mask,
            return_dense=True
        )
        
        # Validate neighbor indices are within bounds
        max_valid_idx = rank_zarr_shape[0] - 1
        assert neighbor_indices.max() <= max_valid_idx, \
            f"Neighbor indices exceed bounds (max: {neighbor_indices.max()}, limit: {max_valid_idx})"
        assert neighbor_indices.min() >= 0, \
            f"Found negative neighbor indices (min: {neighbor_indices.min()})"
        
        # Optimize row order (auto-selects best method)
        logger.info("Optimizing row order for cache efficiency...")
        row_order = optimize_row_order(
            neighbor_indices,
            cell_indices = cell_indices,
            method=None,  # Auto-select based on data
            neighbor_weights=neighbor_weights
        )
        neighbor_indices = neighbor_indices[row_order]
        neighbor_weights = neighbor_weights[row_order]
        cell_indices_sorted = cell_indices[row_order]
        
        # Process in batches
        batch_size = getattr(self.config, 'batch_size', 1000)
        n_batches = (n_cells + batch_size - 1) // batch_size
        
        # Submit all read requests
        logger.info(f"Submitting {n_batches} batches for reading...")
        for batch_idx in range(n_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_cells)
            
            batch_neighbors = neighbor_indices[batch_start:batch_end]
            reader.submit_batch(batch_idx, batch_neighbors)
        
        # Process results as they complete
        logger.info("Processing batches...")
        pbar = tqdm(total=n_batches, desc=f"Processing {cell_type}")
        
        for _ in range(n_batches):
            # Get completed batch - rank_data and indices from worker
            batch_idx, rank_data, rank_indices, original_shape = reader.get_result()
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n_cells)
            actual_batch_size = batch_end - batch_start
            
            # Verify shape
            assert original_shape == (actual_batch_size, self.config.num_neighbour), \
                f"Shape mismatch: expected {(actual_batch_size, self.config.num_neighbour)}, got {original_shape}"
            
            # Use fancy indexing in main thread to save memory
            batch_ranks = rank_data[rank_indices]
            
            # Get batch weights
            batch_weights = neighbor_weights[batch_start:batch_end]
            
            # Compute marker scores using JAX
            marker_scores = compute_marker_scores_jax(
                jnp.array(batch_ranks),
                jnp.array(batch_weights),
                actual_batch_size,
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
    
    def calculate_marker_scores(
        self,
        adata_path: str,
        rank_zarr_path: str,
        mean_frac_path: str,
        output_path: Optional[Union[str, Path]] = None
    ) -> Union[str, Path]:
        """
        Main execution function for marker score calculation
        
        Args:
            adata_path: Path to concatenated latent adata
            rank_zarr_path: Path to rank zarr file
            mean_frac_path: Path to mean expression fraction parquet
            output_path: Optional output path for marker scores
            
        Returns:
            Path to output marker score zarr file
        """
        logger.info("Starting marker score calculation...")
        
        # Use config path if not specified
        if output_path is None:
            output_path = Path(self.config.marker_scores_zarr_path)
        else:
            output_path = Path(output_path)
        
        # Load concatenated AnnData
        logger.info(f"Loading concatenated AnnData from {adata_path}")
        
        if not Path(adata_path).exists():
            raise FileNotFoundError(f"Concatenated AnnData not found: {adata_path}")
        
        adata = sc.read_h5ad(adata_path)
        
        # Load pre-calculated global statistics
        global_log_gmean, global_expr_frac = self.load_global_stats(mean_frac_path)
        
        # Get annotation key
        annotation_key = self.config.annotation
        
        # Open rank zarr and get dimensions
        import zarr
        rank_zarr = zarr.open(str(rank_zarr_path), mode='r')
        n_cells = adata.n_obs
        n_cells_rank = rank_zarr.shape[0]
        n_genes = rank_zarr.shape[1]
        
        logger.info(f"AnnData dimensions: {n_cells} cells × {adata.n_vars} genes")
        logger.info(f"Rank Zarr dimensions: {n_cells_rank} cells × {n_genes} genes")
        
        # Cells should match exactly since filtering is done before rank zarr creation
        assert n_cells == n_cells_rank, \
            f"Cell count mismatch: AnnData has {n_cells} cells, Rank Zarr has {n_cells_rank} cells. " \
            f"This indicates the filtering was not applied consistently during rank calculation."
        
        # Initialize output with proper chunking
        chunks = None
        if hasattr(self.config, 'chunks_cells') and hasattr(self.config, 'chunks_genes'):
            if self.config.chunks_cells is not None or self.config.chunks_genes is not None:
                # Use provided chunks
                chunks = (
                    self.config.chunks_cells if self.config.chunks_cells is not None else 1,
                    self.config.chunks_genes if self.config.chunks_genes is not None else n_genes
                )
        # If chunks is None, ZarrBackedDense will use default (1, n_genes)
        
        output_zarr = ZarrBackedDense(
            output_path,
            shape=(n_cells, n_genes),
            chunks=chunks,
            mode='w',
            num_write_workers=self.config.num_write_workers
        )
        
        # Process each cell type
        if annotation_key and annotation_key in adata.obs.columns:
            cell_types = adata.obs[annotation_key].unique()
        else:
            logger.warning(f"Annotation {annotation_key} not found, processing all cells as one type")
            cell_types = ["all"]
            adata.obs[annotation_key] = "all"
        
        logger.info(f"Processing {len(cell_types)} cell types")
        
        # Load shared data structures once
        logger.info("Loading shared data structures...")
        coords = adata.obsm[self.config.spatial_key]
        emb_gcn = adata.obsm[self.config.latent_representation_niche].astype(np.float32)
        emb_indv = adata.obsm[self.config.latent_representation_cell].astype(np.float32)
        
        # Normalize embeddings
        logger.info("Normalizing embeddings...")
        # L2 normalize niche embeddings
        emb_gcn_norm = np.linalg.norm(emb_gcn, axis=1, keepdims=True)
        emb_gcn = emb_gcn / (emb_gcn_norm + 1e-8)
        
        # L2 normalize individual embeddings
        emb_indv_norm = np.linalg.norm(emb_indv, axis=1, keepdims=True)
        emb_indv = emb_indv / (emb_indv_norm + 1e-8)
        
        # Initialize parallel reader once for all cell types
        logger.info("Initializing parallel reader...")
        reader = ParallelRankReader(
            rank_zarr,
            num_workers=self.config.num_read_workers
        )
        
        for cell_type in cell_types:
            self.process_cell_type(
                adata,
                cell_type,
                output_zarr,
                global_log_gmean,
                global_expr_frac,
                rank_zarr,
                reader,
                coords,
                emb_gcn,
                emb_indv,
                annotation_key
            )
        
        # Close the shared reader after all cell types are processed
        reader.close()
        
        output_zarr.close()
        logger.info("Marker score calculation complete!")
        
        # Save metadata
        metadata = {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'config': {
                'num_neighbour_spatial': self.config.num_neighbour_spatial,
                'num_anchor': self.config.num_anchor,
                'num_neighbour': self.config.num_neighbour,
                'batch_size': getattr(self.config, 'batch_size', 1000),
                'num_read_workers': self.config.num_read_workers,
                'num_write_workers': self.config.num_write_workers
            },
            'global_log_gmean': global_log_gmean.tolist(),
            'global_expr_frac': global_expr_frac.tolist()
        }
        
        metadata_path = output_path.parent / f'{output_path.stem}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return str(output_path)