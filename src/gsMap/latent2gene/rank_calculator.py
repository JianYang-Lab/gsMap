"""
Rank calculation from latent representations
Extracts and processes the rank calculation logic from find_latent_representation.py
"""

import gc
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from jax import jit
from scipy.sparse import csr_matrix
from tqdm import tqdm

import anndata as ad
from gsMap.config import LatentToGeneConfig
from .zarr_utils import ZarrBackedCSR

logger = logging.getLogger(__name__)


@jit
def jax_process_chunk(dense_matrix, n_genes):
    """JAX-optimized processing: ranking + accumulator calculations."""
    n_rows, n_cols = dense_matrix.shape

    # Vectorized ranking
    # Add small noise to break ties consistently
    noise = jax.random.uniform(jax.random.PRNGKey(0), dense_matrix.shape) * 1e-10
    matrix_with_noise = dense_matrix + noise

    # Get argsort for each row
    sorted_indices = jnp.argsort(matrix_with_noise, axis=1)

    # Create ranks
    ranks = jnp.zeros_like(dense_matrix)
    row_indices = jnp.arange(n_rows)[:, None]
    col_ranks = jnp.arange(1, n_cols + 1)[None, :]

    ranks = ranks.at[row_indices, sorted_indices].set(col_ranks)

    # Handle zeros by setting their rank to 1
    ranks = jnp.where(dense_matrix != 0, ranks, 1.0)

    # Compute log ranks
    log_ranks = jnp.log(ranks)

    # Compute accumulators for fill_zero logic
    nonzero_mask = dense_matrix != 0
    n_nonzero_per_row = nonzero_mask.sum(axis=1, keepdims=True)
    zero_log_ranks = jnp.log((n_genes - n_nonzero_per_row + 1) / 2)

    # Sum log ranks (with fill_zero)
    sum_log_ranks = log_ranks.sum(axis=0)
    sum_log_ranks += zero_log_ranks.sum() - (zero_log_ranks * nonzero_mask).sum(axis=0)

    # Sum fraction (count of non-zeros)
    sum_frac = nonzero_mask.sum(axis=0)

    return log_ranks, sum_log_ranks, sum_frac


def rank_data_jax(X: csr_matrix, n_genes,
                  zarr_csr=None,
                  metadata: Optional[Dict[str, Any]] = None,
                  chunk_size: int = 1000,
                  write_interval: int = 10):
    """JAX-optimized rank calculation with batched writing.

    Args:
        X: Input sparse matrix
        n_genes: Total number of genes
        zarr_csr: Optional ZarrBackedCSR instance for writing
        metadata: Optional metadata dictionary
        chunk_size: Size of chunks for processing
        write_interval: How often to write chunks to zarr

    Returns:
        Tuple of (sum_log_ranks, sum_frac) as numpy arrays
    """
    assert X.nnz != 0, "Input matrix must not be empty"

    n_rows, n_cols = X.shape

    # Initialize accumulators
    sum_log_ranks = jnp.zeros(n_genes, dtype=jnp.float32)
    sum_frac = jnp.zeros(n_genes, dtype=jnp.float32)

    # Process in chunks to manage memory
    chunk_size = min(chunk_size, n_rows)
    pending_chunks = []  # Buffer for batching writes
    chunks_processed = 0

    # Setup progress bar
    study_name = metadata.get('name', 'unknown') if metadata else 'unknown'

    with tqdm(total=n_rows, desc=f"Ranking {study_name}", unit="cells") as pbar:
        for start_idx in range(0, n_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_rows)

            # Convert chunk to dense
            chunk_X = X[start_idx:end_idx]
            chunk_dense = chunk_X.toarray().astype(np.float32)
            chunk_jax = jnp.array(chunk_dense)

            # Process chunk with JIT-compiled function (ranking + accumulators)
            chunk_log_ranks, chunk_sum_log_ranks, chunk_sum_frac = jax_process_chunk(chunk_jax, n_genes)

            # Update global accumulators
            sum_log_ranks += chunk_sum_log_ranks
            sum_frac += chunk_sum_frac

            # Convert JAX array to numpy
            chunk_log_ranks_np = np.array(chunk_log_ranks)
            pending_chunks.append(chunk_log_ranks_np)
            chunks_processed += 1

            # Write to zarr periodically
            if zarr_csr and chunks_processed % write_interval == 0:
                # Combine pending chunks and convert to CSR
                zarr_csr.append_batch(np.vstack(pending_chunks))
                pending_chunks = []

            # Update progress bar
            pbar.update(end_idx - start_idx)

    # Write any remaining chunks
    if zarr_csr and pending_chunks:
        zarr_csr.append_batch(np.vstack(pending_chunks))

    return np.array(sum_log_ranks), np.array(sum_frac)

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
        
    def calculate_ranks_and_concatenate(
        self,
        sample_h5ad_dict: Optional[Dict[str, Path]] = None,
        annotation_key: Optional[str] = None,
        data_layer: str = "counts"
    ) -> Dict[str, Any]:
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
        
        # Use provided sample_h5ad_dict or get from config
        if sample_h5ad_dict is None:
            sample_h5ad_dict = self.config.sample_h5ad_dict
        
        # Use provided annotation_key or get from config
        if annotation_key is None:
            annotation_key = self.config.annotation
            
        # Output paths from config
        concat_adata_path = Path(self.config.concatenated_latent_adata_path)
        rank_zarr_path = Path(self.config.rank_zarr_path)
        mean_frac_path = Path(self.config.mean_frac_path)
        
        # Check if outputs already exist
        if concat_adata_path.exists() and rank_zarr_path.exists() and mean_frac_path.exists():
            logger.info(f"Rank outputs already exist in {self.output_dir}")
            return {
                "concatenated_latent_adata": str(concat_adata_path),
                "rank_zarr": str(rank_zarr_path),
                "mean_frac": str(mean_frac_path)
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
        
        for st_id, (sample_name, h5ad_path) in enumerate(tqdm(sample_h5ad_dict.items(), desc="Processing sections")):
            logger.info(f"Processing {sample_name} ({st_id + 1}/{len(sample_h5ad_dict)})...")
            
            # Load the h5ad file (which should already contain latent representations)
            adata = sc.read_h5ad(h5ad_path)
            
            # Add slice information
            adata.obs['slice_id'] = sample_name
            adata.obs['slice_numeric_id'] = st_id
            
            # Filter cells based on annotation group size if annotation is provided
            # This must be done BEFORE adding to rank zarr to maintain index consistency
            if annotation_key and annotation_key in adata.obs.columns:
                min_cells_per_type = getattr(self.config, 'min_cells_per_type', 21)  # Minimum number of homogeneous neighbors
                annotation_counts = adata.obs[annotation_key].value_counts()
                valid_annotations = annotation_counts[annotation_counts >= min_cells_per_type].index
                
                # Check if any filtering is needed
                if len(valid_annotations) < len(annotation_counts):
                    n_cells_before = adata.n_obs
                    mask = adata.obs[annotation_key].isin(valid_annotations)
                    adata = adata[mask].copy()
                    n_cells_after = adata.n_obs
                    
                    logger.info(f"  Filtered {sample_name} based on annotation group size (min={min_cells_per_type})")
                    logger.info(f"    - Cells before: {n_cells_before}, after: {n_cells_after}, removed: {n_cells_before - n_cells_after}")
                    
                    # Log which groups were removed
                    removed_groups = annotation_counts[~annotation_counts.index.isin(valid_annotations)]
                    if len(removed_groups) > 0:
                        logger.debug(f"    - Removed groups: {removed_groups.to_dict()}")
            
            # Get gene list (should be consistent across sections)
            if gene_list is None:
                gene_list = adata.var_names.tolist()
                n_genes = len(gene_list)
                # Initialize rank zarr
                rank_zarr = ZarrBackedCSR(str(rank_zarr_path), ncols=n_genes, mode='w')
                # Initialize global accumulators
                sum_log_ranks = np.zeros(n_genes, dtype=np.float64)
                sum_frac = np.zeros(n_genes, dtype=np.float64)
            else:
                # Verify gene list consistency
                assert adata.var_names.tolist() == gene_list, \
                    f"Gene list mismatch in section {st_id}"
            
            # Get expression data for ranking
            if data_layer in adata.layers:
                X = adata.layers[data_layer]
            else:
                X = adata.X
            
            # Efficient sparse matrix conversion
            if not hasattr(X, 'tocsr'):
                X = csr_matrix(X, dtype=np.float32)
            else:
                X = X.tocsr()
                if X.dtype != np.float32:
                    X = X.astype(np.float32)
            
            # Pre-allocate output arrays for efficiency
            X.sort_indices()  # Sort indices for better cache performance
            
            # Get number of cells after filtering
            n_cells = X.shape[0]
            
            # Use JAX rank calculation
            logger.debug(f"Processing {n_cells} cells with JAX")
            metadata = {'name': sample_name, 'cells': n_cells, 'study_id': st_id}
            
            batch_sum_log_ranks, batch_frac = rank_data_jax(
                X, 
                n_genes,
                zarr_csr=rank_zarr,
                metadata=metadata,
                chunk_size=self.config.rank_batch_size,
                write_interval=self.config.rank_write_interval  # Batch 5 chunks before writing
            )
            
            # Update global sums
            sum_log_ranks += batch_sum_log_ranks
            sum_frac += batch_frac
            total_cells += n_cells
            
            # Create minimal AnnData with empty X matrix but keep obs and obsm
            minimal_adata = ad.AnnData(
                X=csr_matrix((adata.n_obs, n_genes), dtype=np.float32),
                obs=adata.obs.copy(),
                var=pd.DataFrame(index=gene_list),
                obsm=adata.obsm.copy()  # Keep all latent representations
            )
            
            adata_list.append(minimal_adata)
            n_total_cells += n_cells
            
            # Clean up memory
            del adata, X, minimal_adata
            gc.collect()
            
        # Close rank zarr
        rank_zarr.close()
        logger.info(f"Saved rank matrix to {rank_zarr_path}")
        
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
        mean_frac_df.to_parquet(
            mean_frac_path,
            index=True,
            compression="gzip",
        )
        logger.info(f"Mean fraction data saved to {mean_frac_path}")
        
        # Concatenate all sections
        logger.info("Concatenating latent representations...")
        if adata_list:
            concatenated_adata = ad.concat(adata_list, axis=0, join='outer', merge='same')
            
            # Ensure the var_names are the common genes
            concatenated_adata.var_names = gene_list
            
            # Save concatenated adata
            concatenated_adata.write_h5ad(concat_adata_path)
            logger.info(f"Saved concatenated latent representations to {concat_adata_path}")
            logger.info(f"  - Total cells: {concatenated_adata.n_obs}")
            logger.info(f"  - Total genes: {concatenated_adata.n_vars}")
            logger.info(f"  - Latent representations in obsm: {list(concatenated_adata.obsm.keys())}")
            if 'slice_id' in concatenated_adata.obs.columns:
                logger.info(f"  - Number of slices: {concatenated_adata.obs['slice_id'].nunique()}")
            
            # Clean up
            del adata_list, concatenated_adata
            gc.collect()
        
        return {
            "concatenated_latent_adata": str(concat_adata_path),
            "rank_zarr": str(rank_zarr_path),
            "mean_frac": str(mean_frac_path)
        }