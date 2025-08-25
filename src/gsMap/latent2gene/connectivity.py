"""
Connectivity matrix building for homogeneous spot identification
Implements the spatial → anchor → homogeneous neighbor finding algorithm
"""

import logging
from typing import Optional, Tuple
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# Configure JAX
jax.config.update("jax_enable_x64", False)  # Use float32 for speed


# JIT-compiled function for finding anchors and homogeneous neighbors
@partial(jit, static_argnums=(5, 6))
def _find_anchors_and_homogeneous_batch_jit(
    emb_gcn_batch_norm: jnp.ndarray,      # (batch_size, d1) - pre-normalized
    emb_indv_batch: jnp.ndarray,           # (batch_size, d2)
    spatial_indices: jnp.ndarray,          # (batch_size, k1)
    emb_gcn_all_norm: jnp.ndarray,         # (n_total, d1) - pre-normalized
    emb_indv_all: jnp.ndarray,             # (n_total, d2)
    k2: int,                               # Number of anchors
    k3: int                                # Number of homogeneous neighbors
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Combined anchor and homogeneous neighbor finding using JAX
    
    Returns:
        homogeneous_indices: (batch_size, k3) indices of homogeneous neighbors
        homogeneous_weights: (batch_size, k3) softmax weights
    """
    batch_size = emb_gcn_batch_norm.shape[0]
    
    # Step 1: Find anchors - cells with similar spatial niche (using pre-normalized embeddings)
    # Since embeddings are pre-normalized, dot product = cosine similarity
    anchor_sims = jnp.dot(emb_gcn_batch_norm, emb_gcn_all_norm.T)  # (batch_size, n_total)
    
    # Mask out self-similarities and non-spatial neighbors
    mask = jnp.ones_like(anchor_sims, dtype=bool)
    for i in range(batch_size):
        # Mask self
        batch_idx = jnp.arange(batch_size)
        mask = mask.at[i, batch_idx[i]].set(False)
        
        # Keep only spatial neighbors
        spatial_mask = jnp.zeros(anchor_sims.shape[1], dtype=bool)
        spatial_mask = spatial_mask.at[spatial_indices[i]].set(True)
        mask = mask.at[i].set(mask[i] & spatial_mask)
    
    anchor_sims_masked = jnp.where(mask, anchor_sims, -jnp.inf)
    
    # Get top-k2 anchors
    anchor_indices = jnp.argsort(anchor_sims_masked, axis=1)[:, -k2:]  # (batch_size, k2)
    
    # Step 2: Find homogeneous neighbors from anchors
    homogeneous_sims = jnp.zeros((batch_size, anchor_sims.shape[1]))
    
    for i in range(batch_size):
        # Get individual embedding for this cell
        cell_emb_indv = emb_indv_batch[i:i+1]  # (1, d2)
        
        # Compute similarities with anchors' individual embeddings
        anchor_idx = anchor_indices[i]  # (k2,)
        anchor_emb_indv = emb_indv_all[anchor_idx]  # (k2, d2)
        
        # Cosine similarity
        anchor_emb_indv_norm = anchor_emb_indv / (jnp.linalg.norm(anchor_emb_indv, axis=1, keepdims=True) + 1e-8)
        cell_emb_indv_norm = cell_emb_indv / (jnp.linalg.norm(cell_emb_indv) + 1e-8)
        
        sims = jnp.dot(anchor_emb_indv_norm, cell_emb_indv_norm.T).squeeze()  # (k2,)
        
        # Place similarities back
        homogeneous_sims = homogeneous_sims.at[i, anchor_idx].set(sims)
    
    # Mask out non-anchors
    homo_mask = homogeneous_sims > -jnp.inf
    for i in range(batch_size):
        anchor_mask = jnp.zeros(homogeneous_sims.shape[1], dtype=bool)
        anchor_mask = anchor_mask.at[anchor_indices[i]].set(True)
        homo_mask = homo_mask.at[i].set(anchor_mask)
    
    homogeneous_sims_masked = jnp.where(homo_mask, homogeneous_sims, -jnp.inf)
    
    # Get top-k3 homogeneous neighbors
    homogeneous_indices = jnp.argsort(homogeneous_sims_masked, axis=1)[:, -k3:]  # (batch_size, k3)
    
    # Get corresponding similarities for softmax
    homogeneous_weights = jnp.array([
        homogeneous_sims_masked[i, homogeneous_indices[i]]
        for i in range(batch_size)
    ])
    
    # Apply softmax to get weights
    homogeneous_weights = jax.nn.softmax(homogeneous_weights, axis=1)
    
    return homogeneous_indices, homogeneous_weights


class ConnectivityMatrixBuilder:
    """Build connectivity matrix for finding homogeneous spots"""
    
    def __init__(self, config):
        """
        Initialize with configuration
        
        Args:
            config: LatentToGeneConfig object
        """
        self.config = config
        self.k1 = config.num_neighbour_spatial  # Spatial neighbors
        self.k2 = config.num_anchor  # Anchors
        self.k3 = config.num_neighbour  # Homogeneous spots
        self.gpu_batch_size = config.gpu_batch_size
        
    def build_connectivity_matrix(
        self,
        coords: np.ndarray,
        emb_gcn: np.ndarray,
        emb_indv: np.ndarray,
        cell_mask: Optional[np.ndarray] = None,
        return_dense: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build connectivity matrix: spatial → anchor → homogeneous
        
        Args:
            coords: Spatial coordinates (n_total, 2)
            emb_gcn: Spatial niche embeddings (n_total, d1)
            emb_indv: Individual embeddings (n_total, d2)
            cell_mask: Boolean mask for cells to process (n_total,)
            return_dense: If True, return dense arrays for masked cells only
            
        Returns:
            If return_dense=True and cell_mask is provided:
                neighbor_indices: (n_masked, k3) dense array
                neighbor_weights: (n_masked, k3) dense array
            Otherwise:
                connectivity_matrix: Sparse CSR matrix (n_total, n_total)
        """
        n_total = coords.shape[0]
        
        # Step 1: Find spatial neighbors using coordinates
        logger.info(f"Finding {self.k1} spatial neighbors...")
        nbrs = NearestNeighbors(n_neighbors=self.k1, algorithm='kd_tree')
        nbrs.fit(coords)
        distances, spatial_indices = nbrs.kneighbors(coords)
        
        # Prepare data for JAX processing
        if cell_mask is not None:
            cell_indices = np.where(cell_mask)[0]
            n_cells = len(cell_indices)
        else:
            cell_indices = np.arange(n_total)
            n_cells = n_total

        # Convert to JAX arrays
        emb_gcn_norm_jax = jnp.array(emb_gcn)
        emb_indv_jax = jnp.array(emb_indv)
        
        # Process in batches to avoid GPU OOM
        all_indices = []
        all_weights = []

        logger.info(f"Fining the homogeneous neighbors in batches of {self.gpu_batch_size}...")
        for batch_start in range(0, n_cells, self.gpu_batch_size):
            batch_end = min(batch_start + self.gpu_batch_size, n_cells)
            batch_cell_indices = cell_indices[batch_start:batch_end]
            batch_size = len(batch_cell_indices)
            
            # Get batch data
            batch_emb_gcn = emb_gcn_norm_jax[batch_cell_indices]
            batch_emb_indv = emb_indv_jax[batch_cell_indices]
            batch_spatial = jnp.array(spatial_indices[batch_cell_indices])
            
            # Find homogeneous neighbors using JAX
            homo_indices, homo_weights = _find_anchors_and_homogeneous_batch_jit(
                batch_emb_gcn,
                batch_emb_indv,
                batch_spatial,
                emb_gcn_norm_jax,
                emb_indv_jax,
                self.k2,
                self.k3
            )
            
            # Convert back to numpy
            homo_indices = np.array(homo_indices)
            homo_weights = np.array(homo_weights)
            
            all_indices.append(homo_indices)
            all_weights.append(homo_weights)
        
        # Concatenate results
        all_indices = np.vstack(all_indices)  # (n_cells, k3)
        all_weights = np.vstack(all_weights)  # (n_cells, k3)
        
        if return_dense and cell_mask is not None:
            # Return dense arrays for masked cells
            return all_indices, all_weights
        else:
            # Build sparse connectivity matrix
            row_indices = np.repeat(cell_indices, self.k3)
            col_indices = all_indices.flatten()
            data = all_weights.flatten()
            
            connectivity = csr_matrix(
                (data, (row_indices, col_indices)),
                shape=(n_total, n_total)
            )
            return connectivity