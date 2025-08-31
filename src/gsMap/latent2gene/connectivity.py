"""
Connectivity matrix building for homogeneous spot identification
Implements the spatial → anchor → homogeneous neighbor finding algorithm
"""

import logging
from typing import Optional, Tuple, Union
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from tqdm import tqdm, trange

from gsMap.config import LatentToGeneConfig

logger = logging.getLogger(__name__)

# Configure JAX
jax.config.update("jax_enable_x64", False)  # Use float32 for speed


@partial(jit, static_argnums=(5, 6, 7))
def _find_anchors_and_homogeneous_batch_jit(
    emb_gcn_batch_norm: jnp.ndarray,      # (batch_size, d1) - pre-normalized
    emb_indv_batch_norm: jnp.ndarray,      # (batch_size, d2) - pre-normalized
    spatial_neighbors: jnp.ndarray,   # (batch_size, k1)
    all_emb_gcn_norm: jnp.ndarray,         # (n_all, d1) - pre-normalized
    all_emb_indv_norm: jnp.ndarray,        # (n_all, d2) - pre-normalized
    num_anchor: int,
    num_neighbour: int,
    similarity_threshold: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    JIT-compiled function to find anchors and homogeneous neighbors.
    Processes a batch of cells to manage GPU memory.
    Expects pre-normalized embeddings for efficiency.
    
    Args:
        similarity_threshold: Minimum similarity threshold. Weights for similarities 
                            below this threshold will be set to 0 after softmax.
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
    
    # Apply similarity threshold: set similarities below threshold to -inf before softmax
    homogeneous_weights = jnp.where(
        homogeneous_weights >= similarity_threshold,
        homogeneous_weights,
        -jnp.inf
    )
    
    # Use softmax to normalize weights (values with -inf will become 0)
    homogeneous_weights = jax.nn.softmax(homogeneous_weights, axis=1)
    
    return homogeneous_neighbors, homogeneous_weights


class ConnectivityMatrixBuilder:
    """Build connectivity matrix using JAX-accelerated computation with GPU memory optimization"""
    
    def __init__(self, config: LatentToGeneConfig):
        """
        Initialize with configuration
        
        Args:
            config: LatentToGeneConfig object
        """
        self.config = config
        # Use configured batch size for GPU processing
        self.mkscore_batch_size = config.mkscore_batch_size
    
    def build_connectivity_matrix(
        self,
        coords: np.ndarray,
        emb_gcn: np.ndarray,
        emb_indv: np.ndarray,
        cell_mask: Optional[np.ndarray] = None,
        return_dense: bool = True
    ) -> Union[csr_matrix, Tuple[np.ndarray, np.ndarray]]:
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
        logger.info(f"Finding anchors and homogeneous neighbors (batch size: {self.mkscore_batch_size})...")

        # Convert to JAX arrays once
        all_emb_gcn_norm_jax = jnp.array(emb_gcn)
        all_emb_indv_norm_jax = jnp.array(emb_indv)
        
        # Process in batches to avoid GPU OOM
        homogeneous_neighbors_list = []
        homogeneous_weights_list = []
        
        for batch_start in trange(0, n_masked, self.mkscore_batch_size, desc =f"Finding homogeneous neighbors"):
            batch_end = min(batch_start + self.mkscore_batch_size, n_masked)
            batch_indices = slice(batch_start, batch_end)
            
            # Get batch data (already normalized)
            emb_gcn_batch_norm = emb_gcn[cell_mask][batch_indices]
            emb_indv_batch_norm = emb_indv[cell_mask][batch_indices]
            spatial_neighbors_batch = spatial_neighbors[batch_indices]
            
            # Process batch with single JIT-compiled function
            homo_neighbors_batch, homo_weights_batch = _find_anchors_and_homogeneous_batch_jit(
                jnp.array(emb_gcn_batch_norm),
                jnp.array(emb_indv_batch_norm),
                jnp.array(spatial_neighbors_batch),
                all_emb_gcn_norm_jax,
                all_emb_indv_norm_jax,
                self.config.num_anchor,
                self.config.num_neighbour,
                self.config.similarity_threshold
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