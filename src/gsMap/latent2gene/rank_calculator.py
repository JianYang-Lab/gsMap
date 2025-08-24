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

from .zarr_utils import ZarrBackedCSR
from .jax_rank_processing import jax_process_chunk, rank_data_jax

logger = logging.getLogger(__name__)


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
        
        # Initialize global accumulators
        sum_log_ranks = None
        sum_frac = None
        total_cells = 0
        
        for st_id, st_file in enumerate(tqdm(spe_file_list, desc="Processing sections")):
            st_name = Path(st_file).stem
            logger.info(f"Processing {st_name} ({st_id + 1}/{len(spe_file_list)})...")
            
            # Load spatial data
            st_adata = sc.read_h5ad(st_file)
            
            # Load corresponding latent representation
            latent_file = self.latent_dir / f"{st_name}_latent_adata.h5ad"
            if not latent_file.exists():
                # Try alternative naming pattern
                latent_file = self.latent_dir / f"{st_name}_add_latent.h5ad"
                if not latent_file.exists():
                    logger.warning(f"Latent file not found for {st_file}, skipping")
                    continue
            
            latent_adata = sc.read_h5ad(latent_file)
            
            # Add slice information
            latent_adata.obs['slice_id'] = st_name
            latent_adata.obs['slice_numeric_id'] = st_id
            
            # Compute depth if count layer exists
            if data_layer in ["count", "counts"] and data_layer in st_adata.layers:
                latent_adata.obs['depth'] = np.array(st_adata.layers[data_layer].sum(axis=1)).flatten()
            
            # Filter cells based on annotation group size if annotation is provided
            # This must be done BEFORE adding to rank zarr to maintain index consistency
            if annotation_key and annotation_key in st_adata.obs.columns:
                min_cells_per_type = 21  # Minimum number of homogeneous neighbors
                annotation_counts = st_adata.obs[annotation_key].value_counts()
                valid_annotations = annotation_counts[annotation_counts >= min_cells_per_type].index
                
                # Check if any filtering is needed
                if len(valid_annotations) < len(annotation_counts):
                    n_cells_before = st_adata.n_obs
                    mask = st_adata.obs[annotation_key].isin(valid_annotations)
                    st_adata = st_adata[mask].copy()
                    latent_adata = latent_adata[mask].copy()
                    n_cells_after = st_adata.n_obs
                    
                    logger.info(f"  Filtered {st_name} based on annotation group size (min={min_cells_per_type})")
                    logger.info(f"    - Cells before: {n_cells_before}, after: {n_cells_after}, removed: {n_cells_before - n_cells_after}")
                    
                    # Log which groups were removed
                    removed_groups = annotation_counts[~annotation_counts.index.isin(valid_annotations)]
                    if len(removed_groups) > 0:
                        logger.debug(f"    - Removed groups: {removed_groups.to_dict()}")
            
            # Get gene list (should be consistent across sections)
            if gene_list is None:
                gene_list = st_adata.var_names.tolist()
                n_genes = len(gene_list)
                # Initialize rank zarr
                rank_zarr = ZarrBackedCSR(str(rank_zarr_path), ncols=n_genes, mode='w')
                # Initialize global accumulators
                sum_log_ranks = np.zeros(n_genes, dtype=np.float64)
                sum_frac = np.zeros(n_genes, dtype=np.float64)
            else:
                # Verify gene list consistency
                assert st_adata.var_names.tolist() == gene_list, \
                    f"Gene list mismatch in section {st_id}"
            
            # Get expression data for ranking
            if data_layer in ["count", "counts"]:
                X = st_adata.layers[data_layer]
            else:
                X = st_adata.X
            
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
            metadata = {'name': Path(st_file).stem, 'cells': n_cells, 'study_id': st_id}
            
            batch_sum_log_ranks, batch_frac = rank_data_jax(
                X, 
                n_genes,
                zarr_csr=rank_zarr,
                metadata=metadata,
                chunk_size=self.config.batch_size if hasattr(self.config, 'batch_size') else 1000,
                write_interval=5  # Batch 5 chunks before writing
            )
            
            # Update global sums
            sum_log_ranks += batch_sum_log_ranks
            sum_frac += batch_frac
            total_cells += n_cells
            
            # Create minimal AnnData with empty X matrix but keep obs and obsm
            minimal_adata = ad.AnnData(
                X=csr_matrix((latent_adata.n_obs, n_genes), dtype=np.float32),
                obs=latent_adata.obs.copy(),
                var=pd.DataFrame(index=gene_list),
                obsm=latent_adata.obsm.copy()  # Keep all latent representations
            )
            
            adata_list.append(minimal_adata)
            n_total_cells += n_cells
            
            # Clean up memory
            del st_adata, X, latent_adata, minimal_adata
            import gc
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
            "concatenated_latent_adata": concat_adata_path,
            "rank_zarr": rank_zarr_path,
            "mean_frac": mean_frac_path
        }