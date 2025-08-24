"""
Rank calculation from latent representations
Extracts and processes the rank calculation logic from find_latent_representation.py
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import csr_matrix
from tqdm import tqdm

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from scipy.sparse import csr_matrix
from tqdm import tqdm
from typing import Optional, Dict, Any

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
    
    def __init__(self, config):
        """
        Initialize RankCalculator with configuration
        
        Args:
            config: LatentToGeneConfig object with all necessary parameters
        """
        self.config = config
        self.latent_dir = Path(config.latent_dir)
        self.output_dir = Path(config.workdir) / "latent_to_gene"
        if config.project_name:
            self.output_dir = Path(config.workdir) / config.project_name / "latent_to_gene"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_ranks_and_concatenate(
        self,
        spe_file_list: List[str],
        annotation_key: Optional[str] = None,
        data_layer: str = "counts"
    ) -> Dict[str, Any]:
        """
        Calculate expression ranks and create concatenated latent representation
        
        This combines the rank calculation and concatenation logic from find_latent_representation.py
        
        Args:
            spe_file_list: List of spatial transcriptomics h5ad files
            annotation_key: Optional annotation to filter cells
            data_layer: Data layer to use for expression
            
        Returns:
            Dictionary with paths to:
                - concatenated_latent_adata: Path to concatenated latent representations
                - rank_zarr: Path to rank zarr file
                - mean_frac: Path to mean expression fraction parquet
        """
        
        # Output paths
        concat_adata_path = self.output_dir / "concatenated_latent_adata.h5ad"
        rank_zarr_path = self.output_dir / "ranks.zarr"
        mean_frac_path = self.output_dir / "mean_frac.parquet"
        
        # Check if outputs already exist
        if concat_adata_path.exists() and rank_zarr_path.exists() and mean_frac_path.exists():
            logger.info(f"Rank outputs already exist in {self.output_dir}")
            return {
                "concatenated_latent_adata": concat_adata_path,
                "rank_zarr": rank_zarr_path,
                "mean_frac": mean_frac_path
            }
        
        logger.info("Starting rank calculation and concatenation...")
        
        # Load latent representations
        latent_files = list(self.latent_dir.glob("*_latent_adata.h5ad"))
        if not latent_files:
            raise FileNotFoundError(f"No latent representation files found in {self.latent_dir}")
        
        # Process each section
        adata_list = []
        n_total_cells = 0
        gene_list = None
        
        for st_id, st_file in enumerate(tqdm(spe_file_list, desc="Processing sections")):
            # Load spatial data
            st_adata = sc.read_h5ad(st_file)
            
            # Load corresponding latent representation
            latent_file = self.latent_dir / f"{Path(st_file).stem}_latent_adata.h5ad"
            if not latent_file.exists():
                logger.warning(f"Latent file not found for {st_file}, skipping")
                continue
            
            latent_adata = sc.read_h5ad(latent_file)
            
            # Filter by annotation if provided
            if annotation_key and annotation_key in st_adata.obs.columns:
                mask = st_adata.obs[annotation_key].notna()
                st_adata = st_adata[mask].copy()
                latent_adata = latent_adata[mask].copy()
            
            # Get gene list (should be consistent across sections)
            if gene_list is None:
                gene_list = st_adata.var_names.tolist()
                n_genes = len(gene_list)
                # Initialize rank zarr
                rank_zarr = ZarrBackedCSR.create(rank_zarr_path, ncols=n_genes, mode='w')
            else:
                # Verify gene list consistency
                assert st_adata.var_names.tolist() == gene_list, \
                    f"Gene list mismatch in section {st_id}"
            
            # Add section ID to obs
            latent_adata.obs['section_id'] = st_id
            latent_adata.obs['section_name'] = Path(st_file).stem
            
            # Calculate ranks for this section using JAX
            X = st_adata.X
            if not hasattr(X, 'toarray'):
                X = csr_matrix(X)
            
            # Calculate expression fraction
            is_expressed = (X > 0).astype(np.float32)
            if hasattr(is_expressed, 'toarray'):
                is_expressed = is_expressed.toarray()
            expr_frac = is_expressed.mean(axis=0)
            
            # Use JAX rank calculation
            metadata = {'name': Path(st_file).stem}
            sum_log_ranks, sum_frac = rank_data_jax(
                X, 
                n_genes,
                zarr_csr=rank_zarr,
                metadata=metadata,
                chunk_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 1000,
                write_interval=10
            )
            
            # Store expression fraction in var
            if 'expr_frac' not in latent_adata.var.columns:
                latent_adata.var['expr_frac'] = expr_frac
            
            adata_list.append(latent_adata)
            n_total_cells += n_cells
            
        # Close rank zarr
        rank_zarr.close()
        logger.info(f"Saved rank matrix to {rank_zarr_path}")
        
        # Concatenate all sections
        logger.info("Concatenating latent representations...")
        concatenated_adata = ad.concat(adata_list, axis=0, merge='same')
        
        # Calculate and save mean expression fractions
        mean_expr_frac = pd.DataFrame({
            'gene': gene_list,
            'mean_expr_frac': np.mean([adata.var['expr_frac'].values for adata in adata_list], axis=0)
        })
        mean_expr_frac.to_parquet(mean_frac_path)
        logger.info(f"Saved mean expression fractions to {mean_frac_path}")
        
        # Save concatenated adata
        concatenated_adata.write_h5ad(concat_adata_path)
        logger.info(f"Saved concatenated latent representations to {concat_adata_path}")
        
        return {
            "concatenated_latent_adata": concat_adata_path,
            "rank_zarr": rank_zarr_path,
            "mean_frac": mean_frac_path
        }