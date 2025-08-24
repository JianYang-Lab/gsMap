"""
Marker score calculation using homogeneous neighbors
Implements weighted geometric mean calculation in log space
"""

import logging
import queue
import threading
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import scanpy as sc
from tqdm import tqdm
from numba import njit

from .zarr_utils import ZarrBackedDense, ZarrBackedCSR
from .connectivity import ConnectivityMatrixBuilder
from .row_ordering import optimize_row_order

logger = logging.getLogger(__name__)


class ParallelRankReader:
    """Parallel reader for rank zarr data with caching"""
    
    def __init__(
        self,
        rank_zarr_path: str,
        num_workers: int = 4,
        cache_size_mb: int = 1000
    ):
        self.rank_zarr = ZarrBackedCSR.open(rank_zarr_path, mode='r')
        self.num_workers = num_workers
        
        # Queues for communication
        self.read_queue = queue.Queue()
        self.result_queue = queue.Queue(maxsize=100)
        
        # Start worker threads
        self.workers = []
        self.stop_workers = threading.Event()
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads for parallel reading"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self, worker_id: int):
        """Worker loop for reading rank data"""
        while not self.stop_workers.is_set():
            try:
                item = self.read_queue.get(timeout=0.1)
                if item is None:
                    break
                
                batch_idx, neighbor_indices, neighbor_weights = item
                batch_size, k = neighbor_indices.shape
                
                # Pre-allocate output
                rank_data = np.zeros((batch_size, k), dtype=np.float32)
                rank_indices = np.zeros((batch_size, k), dtype=np.int32)
                
                # Read rank data for each neighbor
                for i in range(batch_size):
                    for j in range(k):
                        neighbor_idx = neighbor_indices[i, j]
                        # Get rank data from zarr
                        indices, values = self.rank_zarr[neighbor_idx]
                        if len(indices) > 0:
                            # Store gene indices and their ranks
                            rank_indices[i, j] = indices[0]  # Top ranked gene
                            rank_data[i, j] = values[0]  # Its rank value
                
                # Put result in queue
                self.result_queue.put((batch_idx, rank_data, rank_indices, (batch_size, k)))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                raise
    
    def submit_batch(self, batch_idx: int, neighbor_indices: np.ndarray, neighbor_weights: np.ndarray):
        """Submit batch for reading"""
        self.read_queue.put((batch_idx, neighbor_indices, neighbor_weights))
    
    def get_result(self):
        """Get completed result"""
        return self.result_queue.get()
    
    def close(self):
        """Close reader and cleanup"""
        self.stop_workers.set()
        for _ in range(self.num_workers):
            self.read_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=1.0)


@njit
def calculate_weighted_geometric_mean(
    rank_values: np.ndarray,
    weights: np.ndarray,
    zero_fill: float = 1e-10
) -> float:
    """
    Calculate weighted geometric mean in log space
    
    Args:
        rank_values: Rank values for neighbors
        weights: Softmax weights (sum to 1)
        zero_fill: Small value to replace zeros
        
    Returns:
        Weighted geometric mean
    """
    # Replace zeros with small value
    rank_values = np.where(rank_values > 0, rank_values, zero_fill)
    
    # Calculate in log space for numerical stability
    log_values = np.log(rank_values)
    weighted_log_mean = np.sum(weights * log_values)
    
    return np.exp(weighted_log_mean)


class MarkerScoreCalculator:
    """Calculate marker scores for each cell type"""
    
    def __init__(self, config):
        """
        Initialize with configuration
        
        Args:
            config: LatentToGeneConfig object
        """
        self.config = config
        self.connectivity_builder = ConnectivityMatrixBuilder(config)
        self.output_dir = Path(config.workdir) / "latent_to_gene"
        if config.project_name:
            self.output_dir = Path(config.workdir) / config.project_name / "latent_to_gene"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_marker_scores(
        self,
        adata_path: str,
        rank_zarr_path: str,
        mean_frac_path: str,
        output_path: Optional[str] = None
    ) -> str:
        """
        Calculate marker scores for all cell types
        
        Args:
            adata_path: Path to concatenated latent adata
            rank_zarr_path: Path to rank zarr file
            mean_frac_path: Path to mean expression fraction parquet
            output_path: Optional output path for marker scores
            
        Returns:
            Path to output marker score zarr file
        """
        if output_path is None:
            output_path = self.output_dir / "marker_scores.zarr"
        else:
            output_path = Path(output_path)
        
        # Load data
        logger.info("Loading concatenated latent representations...")
        adata = sc.read_h5ad(adata_path)
        
        # Load mean expression fractions
        mean_frac_df = pd.read_parquet(mean_frac_path)
        global_expr_frac = mean_frac_df['mean_expr_frac'].values
        
        # Calculate global log geometric mean for baseline
        global_log_gmean = np.log(global_expr_frac + 1e-10)
        
        # Get dimensions
        n_cells = adata.n_obs
        n_genes = len(mean_frac_df)
        
        # Open rank zarr
        rank_zarr = ZarrBackedCSR.open(rank_zarr_path, mode='r')
        n_cells_rank = rank_zarr.shape[0]
        
        assert n_cells == n_cells_rank, \
            f"Cell count mismatch: AnnData has {n_cells} cells, Rank Zarr has {n_cells_rank} cells."
        
        # Initialize output zarr
        chunks = None
        if self.config.chunks_cells is not None or self.config.chunks_genes is not None:
            chunks = (
                self.config.chunks_cells if self.config.chunks_cells is not None else 1,
                self.config.chunks_genes if self.config.chunks_genes is not None else n_genes
            )
        
        output_zarr = ZarrBackedDense(
            output_path,
            shape=(n_cells, n_genes),
            chunks=chunks,
            mode='w',
            num_write_workers=self.config.num_write_workers
        )
        
        # Process each cell type
        annotation_key = self.config.annotation or "cell_type"
        if annotation_key in adata.obs.columns:
            cell_types = adata.obs[annotation_key].unique()
        else:
            logger.warning(f"Annotation {annotation_key} not found, processing all cells as one type")
            cell_types = ["all"]
            adata.obs[annotation_key] = "all"
        
        logger.info(f"Processing {len(cell_types)} cell types")
        
        for cell_type in cell_types:
            self._process_cell_type(
                adata,
                cell_type,
                output_zarr,
                global_log_gmean,
                global_expr_frac,
                rank_zarr_shape=(n_cells_rank, n_genes)
            )
        
        output_zarr.close()
        logger.info(f"Marker score calculation complete! Saved to {output_path}")
        
        # Save metadata
        metadata = {
            'n_cells': n_cells,
            'n_genes': n_genes,
            'global_log_gmean': global_log_gmean.tolist(),
            'global_expr_frac': global_expr_frac.tolist()
        }
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return str(output_path)
    
    def _process_cell_type(
        self,
        adata,
        cell_type: str,
        output_zarr: ZarrBackedDense,
        global_log_gmean: np.ndarray,
        global_expr_frac: np.ndarray,
        rank_zarr_shape: Tuple[int, int]
    ):
        """Process a single cell type"""
        
        # Get cells of this type
        annotation_key = self.config.annotation or "cell_type"
        cell_mask = adata.obs[annotation_key] == cell_type
        cell_indices = np.where(cell_mask)[0]
        n_cells = len(cell_indices)
        
        if n_cells < self.config.min_cells_per_type:
            logger.warning(f"Skipping {cell_type}: only {n_cells} cells")
            return
        
        logger.info(f"Processing {cell_type}: {n_cells} cells")
        
        # Build connectivity matrix
        logger.info("Building connectivity matrix...")
        neighbor_indices, neighbor_weights = self.connectivity_builder.build_connectivity_matrix(
            coords=adata.obsm[self.config.spatial_key],
            emb_gcn=adata.obsm[self.config.latent_representation].astype(np.float32),
            emb_indv=adata.obsm[self.config.latent_representation_indv].astype(np.float32),
            cell_mask=cell_mask,
            return_dense=True
        )
        
        # Optimize row order for cache efficiency
        logger.info("Optimizing row order for cache efficiency...")
        row_order = optimize_row_order(
            neighbor_indices,
            method=None,  # Auto-select
            neighbor_weights=neighbor_weights
        )
        neighbor_indices = neighbor_indices[row_order]
        neighbor_weights = neighbor_weights[row_order]
        cell_indices_sorted = cell_indices[row_order]
        
        # Initialize parallel reader
        reader = ParallelRankReader(
            self.config.rank_zarr_path or rank_zarr_shape,
            num_workers=self.config.num_read_workers
        )
        
        # Process in batches
        n_batches = (n_cells + self.config.batch_size - 1) // self.config.batch_size
        
        # Submit all batches for reading
        for batch_idx in range(n_batches):
            batch_start = batch_idx * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, n_cells)
            batch_indices = neighbor_indices[batch_start:batch_end]
            batch_weights = neighbor_weights[batch_start:batch_end]
            
            reader.submit_batch(batch_idx, batch_indices, batch_weights)
        
        # Process results as they complete
        logger.info("Processing batches...")
        pbar = tqdm(total=n_batches, desc=f"Processing {cell_type}")
        
        for _ in range(n_batches):
            # Get completed batch
            batch_idx, rank_data, rank_indices, original_shape = reader.get_result()
            
            batch_start = batch_idx * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, n_cells)
            batch_size = batch_end - batch_start
            
            # Calculate marker scores
            batch_scores = np.zeros((batch_size, rank_zarr_shape[1]), dtype=np.float32)
            
            for i in range(batch_size):
                cell_weights = neighbor_weights[batch_start + i]
                cell_rank_data = rank_data[i]
                
                # Calculate weighted geometric mean for each gene
                for gene_idx in range(rank_zarr_shape[1]):
                    # Get ranks for this gene across neighbors
                    gene_ranks = cell_rank_data  # Simplified for this example
                    
                    # Calculate weighted geometric mean
                    score = calculate_weighted_geometric_mean(
                        gene_ranks,
                        cell_weights,
                        zero_fill=global_expr_frac[gene_idx] + 1e-10
                    )
                    batch_scores[i, gene_idx] = score
            
            # Write batch to zarr
            global_indices = cell_indices_sorted[batch_start:batch_end]
            output_zarr.write_batch(batch_scores, global_indices)
            
            pbar.update(1)
        
        pbar.close()
        reader.close()
        logger.info(f"Completed processing {cell_type}")