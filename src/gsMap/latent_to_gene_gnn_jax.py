"""
JAX-optimized implementation of latent to gene GNN.
GPU-accelerated version with batch processing for scalability.
"""

import gc
import logging
import os
import pickle
import uuid
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from functools import partial
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
from jax import jit, vmap, pmap
from jax.scipy.spatial.distance import cdist
from jax.scipy.special import logsumexp
from scipy.sparse import issparse
from tqdm import tqdm

from gsMap.config import LatentToGeneConfig
from gsMap.slice_mean import merge_zarr_means
from gsMap.find_latent_representation import ZarrBackedCSR

logger = logging.getLogger(__name__)

# Configure JAX for optimal performance
jax.config.update('jax_enable_x64', False)  # Use float32 for speed
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.75')


# ============================================================================
# JAX-optimized neighbor finding
# ============================================================================

@jit
def euclidean_distances_jax(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise Euclidean distances using JAX."""
    x_sqnorms = jnp.sum(x ** 2, axis=1, keepdims=True)
    y_sqnorms = jnp.sum(y ** 2, axis=1, keepdims=True)
    xy = jnp.dot(x, y.T)
    distances = x_sqnorms - 2 * xy + y_sqnorms.T
    return jnp.sqrt(jnp.maximum(distances, 0))


@partial(jit, static_argnums=(1,))
def find_k_nearest_neighbors_jax(distances: jnp.ndarray, k: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find k nearest neighbors for each point using JAX."""
    # Note: JAX doesn't have argsort with partial sorting, so we use full sort
    # For very large k, consider using jax.lax.top_k when available
    indices = jnp.argsort(distances, axis=1)[:, :k]
    
    # Gather the corresponding distances
    batch_indices = jnp.arange(distances.shape[0])[:, None]
    neighbor_distances = distances[batch_indices, indices]
    
    return indices, neighbor_distances


def find_neighbors_batch(coor: np.ndarray, num_neighbour: int, batch_size: int = 5000) -> pd.DataFrame:
    """
    Find neighbors using JAX with batching for memory efficiency.
    """
    n_cells = coor.shape[0]
    coor_jax = jnp.array(coor, dtype=jnp.float32)
    
    all_indices = []
    all_distances = []
    
    # Process in batches to manage memory
    for start_idx in tqdm(range(0, n_cells, batch_size), desc="Finding neighbors"):
        end_idx = min(start_idx + batch_size, n_cells)
        batch_coor = coor_jax[start_idx:end_idx]
        
        # Compute distances for this batch
        distances = euclidean_distances_jax(batch_coor, coor_jax)
        
        # Find k nearest neighbors
        k = min(num_neighbour, n_cells)
        indices, neighbor_distances = find_k_nearest_neighbors_jax(distances, k)
        
        all_indices.append(np.array(indices))
        all_distances.append(np.array(neighbor_distances))
    
    # Concatenate results
    all_indices = np.vstack(all_indices)
    all_distances = np.vstack(all_distances)
    
    # Create DataFrame
    cell_indices = np.arange(n_cells)
    cell1 = np.repeat(cell_indices, all_indices.shape[1])
    cell2 = all_indices.flatten()
    distance = all_distances.flatten()
    
    spatial_net = pd.DataFrame({
        "Cell1": cell1,
        "Cell2": cell2, 
        "Distance": distance
    })
    
    return spatial_net


# ============================================================================
# JAX-optimized similarity computations
# ============================================================================

@jit
def cosine_similarity_jax(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Compute cosine similarity between two sets of vectors using JAX."""
    x_norm = jnp.linalg.norm(x, axis=1, keepdims=True)
    y_norm = jnp.linalg.norm(y, axis=1, keepdims=True)
    
    x_normalized = x / (x_norm + 1e-8)
    y_normalized = y / (y_norm + 1e-8)
    
    return jnp.dot(x_normalized, y_normalized.T)


@jit
def softmax_jax(x: jnp.ndarray) -> jnp.ndarray:
    """Compute softmax using JAX with numerical stability."""
    x_max = jnp.max(x)
    exp_x = jnp.exp(x - x_max)
    return exp_x / jnp.sum(exp_x)


@partial(jit, static_argnums=(2,))
def find_anchors_jax(cell_latent: jnp.ndarray, 
                     neighbour_latent: jnp.ndarray,
                     num_anchor: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Find anchor cells based on latent similarity using JAX."""
    similarity = cosine_similarity_jax(cell_latent, neighbour_latent).flatten()
    
    # Get top k anchors
    if num_anchor >= len(similarity):
        top_indices = jnp.arange(len(similarity))
        top_similarities = similarity
    else:
        # Use partial sort for efficiency
        top_indices = jnp.argsort(-similarity)[:num_anchor]
        top_similarities = similarity[top_indices]
    
    return top_indices, softmax_jax(top_similarities)


# ============================================================================
# Load pre-computed ranks from Zarr
# ============================================================================

def load_precomputed_ranks(config: LatentToGeneConfig, adata, gM, slice_start_idx: int = 0) -> np.ndarray:
    """
    Load pre-computed ranks from Zarr storage.
    
    Args:
        config: Configuration object containing paths
        adata: AnnData object with cell information
        gM: Geometric mean values for normalization
        slice_start_idx: Starting index for this slice in the global rank matrix
        
    Returns:
        ranks: Array of shape (n_cells, n_genes) containing normalized ranks
    """
    # Construct path to ranks zarr file
    ranks_zarr_path = config.latent_dir / "ranks.zarr"
    
    if not ranks_zarr_path.exists():
        # Fallback: compute ranks if pre-computed ones don't exist
        logger.warning(
            f"Pre-computed ranks not found at {ranks_zarr_path}. "
            "Computing ranks on the fly (this will be slower)."
        )
        
        from scipy.stats import rankdata
        
        # Get expression data - note that X might be empty in concatenated adata
        if issparse(adata.X):
            data = adata.X.toarray()
        else:
            data = adata.X
        
        # Check if X is empty (all zeros)
        if data.max() == 0:
            logger.error("Expression matrix X is empty. Cannot compute ranks.")
            raise ValueError("Expression matrix is empty. Pre-computed ranks are required.")
        
        n_cells, n_genes = data.shape
        ranks = np.zeros((n_cells, n_genes), dtype=np.float32)
        
        for i in tqdm(range(n_cells), desc="Computing ranks"):
            ranks[i, :] = rankdata(data[i, :], method="average")
        
        # Normalize by geometric mean
        ranks = ranks / gM
        return ranks
    
    logger.info(f"Loading pre-computed ranks from {ranks_zarr_path}")
    
    # Load ranks using ZarrBackedCSR
    zarr_ranks = ZarrBackedCSR.open(str(ranks_zarr_path), mode="r")
    
    # Get the current sample's cell indices in the zarr file
    n_cells = adata.n_obs
    
    # Calculate the correct indices based on slice position
    cell_indices = np.arange(slice_start_idx, slice_start_idx + n_cells)
    
    logger.info(f"Loading ranks for cells {slice_start_idx} to {slice_start_idx + n_cells - 1}")
    
    # Extract ranks for the specified cells
    # The ranks are stored as log-ranks in the zarr file
    ranks_data = zarr_ranks[cell_indices]
    
    # Convert from sparse to dense if needed
    if hasattr(ranks_data, 'toarray'):
        ranks_data = ranks_data.toarray()
    
    # Convert log-ranks back to regular ranks
    ranks = np.exp(ranks_data)
    
    # The ranks in the zarr are already normalized by gM during computation
    # So we don't need to normalize again
    
    return ranks.astype(np.float32)


# ============================================================================
# JAX-optimized marker score computation
# ============================================================================

@partial(jit, static_argnums=(4,))
def compute_regional_mkscore_jax(
    cell_select_similarity: jnp.ndarray,
    ranks_tg: jnp.ndarray,
    adata_X_bool_region: jnp.ndarray,
    frac_whole: jnp.ndarray,
    use_expression_fraction: bool = True
) -> jnp.ndarray:
    """
    Compute marker scores for a region using JAX.
    """
    # Compute weighted geometric mean of ranks
    log_ranks = jnp.log(jnp.maximum(ranks_tg, 1.0))
    weighted_log_ranks = log_ranks * cell_select_similarity[:, None]
    gene_ranks_region = jnp.exp(jnp.sum(weighted_log_ranks, axis=0))
    gene_ranks_region = jnp.where(gene_ranks_region <= 1, 0, gene_ranks_region)
    
    if use_expression_fraction:
        # Compute expression fraction
        n_cells = adata_X_bool_region.shape[0]
        frac_focal = jnp.sum(adata_X_bool_region, axis=0) / n_cells
        frac_region = frac_focal - frac_whole
        frac_region = jnp.where(frac_region > 0, 1.0, 0.0)
        
        # Apply expression fraction filter
        gene_ranks_region = gene_ranks_region * frac_region
    
    # Compute final marker score
    mkscore = jnp.exp(jnp.power(gene_ranks_region, 1.5)) - 1
    return mkscore


def process_cells_batch_jax(
    cell_positions: np.ndarray,
    spatial_net_dict: Dict,
    coor_latent: np.ndarray,
    coor_latent_indv: np.ndarray,
    ranks: np.ndarray,
    adata_X_bool: np.ndarray,
    frac_whole: np.ndarray,
    config: LatentToGeneConfig
) -> np.ndarray:
    """
    Process a batch of cells to compute marker scores using JAX.
    """
    n_genes = ranks.shape[1]
    mk_scores = []
    
    # Convert to JAX arrays
    coor_latent_jax = jnp.array(coor_latent, dtype=jnp.float32)
    coor_latent_indv_jax = jnp.array(coor_latent_indv, dtype=jnp.float32)
    ranks_jax = jnp.array(ranks, dtype=jnp.float32)
    frac_whole_jax = jnp.array(frac_whole, dtype=jnp.float32)
    
    # Convert sparse to dense if needed
    if issparse(adata_X_bool):
        adata_X_bool_dense = adata_X_bool.toarray()
    else:
        adata_X_bool_dense = adata_X_bool
    adata_X_bool_jax = jnp.array(adata_X_bool_dense, dtype=jnp.bool_)
    
    for cell_pos in tqdm(cell_positions, desc="Computing marker scores"):
        # Get spatial neighbors
        cell_use_pos = spatial_net_dict.get(cell_pos, [])
        if len(cell_use_pos) == 0:
            mk_scores.append(np.zeros(n_genes, dtype=np.float32))
            continue
        
        cell_use_pos = np.array(cell_use_pos)
        num_neighbour = min(len(cell_use_pos), config.num_neighbour)
        num_anchor = min(len(cell_use_pos), config.num_anchor)
        
        # Find anchors using GCN smoothed embeddings
        cell_latent = coor_latent_jax[cell_pos:cell_pos+1]
        neighbour_latent = coor_latent_jax[cell_use_pos]
        
        anchors_idx, anchors_similarity = find_anchors_jax(
            cell_latent, neighbour_latent, num_anchor
        )
        anchors_pos = cell_use_pos[np.array(anchors_idx)]
        
        # Find homogeneous spots using expression embeddings
        cell_latent_indv = coor_latent_indv_jax[cell_pos:cell_pos+1]
        neighbour_latent_indv = coor_latent_indv_jax[anchors_pos]
        
        similarity = cosine_similarity_jax(cell_latent_indv, neighbour_latent_indv).flatten()
        top_indices = jnp.argsort(-similarity)[:num_neighbour]
        
        cell_select_pos = anchors_pos[np.array(top_indices)]
        cell_select_similarity = softmax_jax(similarity[top_indices])
        
        # Compute marker scores
        ranks_tg = ranks_jax[cell_select_pos]
        adata_X_bool_region = adata_X_bool_jax[cell_select_pos]
        
        mkscore = compute_regional_mkscore_jax(
            cell_select_similarity,
            ranks_tg,
            adata_X_bool_region,
            frac_whole_jax,
            not config.no_expression_fraction
        )
        
        mk_scores.append(np.array(mkscore))
    
    return np.vstack(mk_scores).T


# ============================================================================
# Main processing function
# ============================================================================

def run_latent_to_gene_jax(config: LatentToGeneConfig, *, layer_key=None, slice_id=None):
    """
    JAX-optimized version of latent to gene mapping with GPU acceleration.
    
    Args:
        config: Configuration object
        layer_key: Layer to use for expression data (if applicable)
        slice_id: Specific slice ID to process from concatenated data
    """
    logger.info(f"------Loading spatial data: {config.sample_name}...")
    logger.info(f"Using JAX backend: {jax.default_backend()}")
    
    # Check available devices
    devices = jax.devices()
    logger.info(f"Available JAX devices: {devices}")
    
    # Check if we're using concatenated data
    concatenated_path = config.latent_dir / "concatenated_latent_adata.h5ad"
    if concatenated_path.exists() and slice_id is not None:
        logger.info(f"Loading concatenated AnnData from {concatenated_path}")
        full_adata = sc.read_h5ad(concatenated_path)
        logger.info(f"Loaded concatenated data: {full_adata.n_obs} cells, {full_adata.n_vars} genes")
        
        # Filter by slice_id
        if 'slice_id' in full_adata.obs.columns:
            # Calculate starting index for this slice in the global rank matrix
            slice_masks = full_adata.obs['slice_id'] == slice_id
            slice_start_idx = np.where(slice_masks)[0][0] if slice_masks.any() else 0
            
            adata = full_adata[full_adata.obs['slice_id'] == slice_id].copy()
            logger.info(f"Filtered to slice '{slice_id}': {adata.n_obs} cells")
            
            if adata.n_obs == 0:
                raise ValueError(f"No cells found for slice_id '{slice_id}'")
        else:
            logger.warning("No 'slice_id' column found in concatenated data, using all cells")
            adata = full_adata
            slice_start_idx = 0
    else:
        # Fallback to loading individual file
        adata = sc.read_h5ad(config.hdf5_with_latent_path)
        logger.info(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")
        slice_start_idx = 0
    
    # Handle layer selection - note that X might be empty in concatenated data
    if layer_key is not None:
        if (counts := adata.layers.get(layer_key, None)) is not None:
            adata.X = counts.copy()
            logger.info(f"Using data from `adata.layers[{layer_key}]`")
        else:
            logger.warning(f"Invalid layer_key: {layer_key}, using `adata.X`")
    else:
        # Check if X is empty (as expected in concatenated data)
        if issparse(adata.X):
            if adata.X.nnz == 0:
                logger.info("X matrix is empty (as expected for concatenated data with pre-computed ranks)")
        elif np.all(adata.X == 0):
            logger.info("X matrix is empty (as expected for concatenated data with pre-computed ranks)")
        else:
            logger.info("Using `adata.X`")
    
    # Filter by annotation if provided
    if config.annotation is not None:
        logger.info(f"Filtering by annotation: {config.annotation}")
        adata = adata[~pd.isnull(adata.obs[config.annotation]), :]
        logger.info(f"Filtered to {adata.n_obs} cells")
    
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    
    # Build spatial graph with JAX optimization
    logger.info("------Building spatial graph...")
    coor = adata.obsm[config.spatial_key]
    
    if config.annotation is not None:
        spatial_net_list = []
        
        # Process each cell type
        for ct in adata.obs[config.annotation].dropna().unique():
            idx = np.where(adata.obs[config.annotation] == ct)[0]
            coor_temp = coor[idx, :]
            
            # Use JAX-optimized neighbor finding
            spatial_net_temp = find_neighbors_batch(
                coor_temp, 
                min(config.num_neighbour_spatial, coor_temp.shape[0]),
                batch_size=5000
            )
            
            # Map back to original indices
            spatial_net_temp["Cell1"] = idx[spatial_net_temp["Cell1"].values]
            spatial_net_temp["Cell2"] = idx[spatial_net_temp["Cell2"].values]
            spatial_net_list.append(spatial_net_temp)
            logger.info(f"{ct}: {coor_temp.shape[0]} cells")
        
        # Handle NaN cells
        if pd.isnull(adata.obs[config.annotation]).any():
            idx_nan = np.where(pd.isnull(adata.obs[config.annotation]))[0]
            logger.info(f"NaN: {len(idx_nan)} cells")
            
            spatial_net_temp = find_neighbors_batch(
                coor, config.num_neighbour_spatial, batch_size=5000
            )
            spatial_net_temp = spatial_net_temp[spatial_net_temp["Cell1"].isin(idx_nan)]
            spatial_net_list.append(spatial_net_temp)
        
        spatial_net = pd.concat(spatial_net_list, axis=0)
    else:
        spatial_net = find_neighbors_batch(
            coor, config.num_neighbour_spatial, batch_size=5000
        )
    
    spatial_net_dict = spatial_net.groupby("Cell1")["Cell2"].apply(np.array).to_dict()
    
    # Extract latent representations
    logger.info("------Extracting latent representations...")
    coor_latent = adata.obsm[config.latent_representation].astype(np.float32)
    coor_latent_indv = adata.obsm[config.latent_representation_indv].astype(np.float32)
    
    # Prepare expression data
    logger.info("------Preparing expression data...")
    if issparse(adata.X):
        if adata.X.nnz > 0:  # Only convert if non-empty
            logger.info("Converting sparse matrix to dense...")
            adata_X = adata.X.toarray()
        else:
            # X is empty, create zero matrix
            logger.info("X matrix is empty, creating zero matrix for boolean mask")
            adata_X = np.zeros((adata.n_obs, adata.n_vars), dtype=np.float32)
    else:
        adata_X = adata.X
    
    # For boolean matrix, we need actual expression data
    # If X is empty, we'll use the ranks as a proxy
    if np.all(adata_X == 0):
        logger.warning("Using ranks as proxy for expression presence")
        adata_X_bool = (ranks > 1).astype(bool)  # Rank > 1 indicates expression
    else:
        adata_X_bool = adata_X.astype(bool)
    
    # Load slice means first (needed for rank normalization)
    logger.info("------Loading slice means...")
    gM, frac_whole = merge_zarr_means(config.zarr_group_path, None)
    frac_whole = frac_whole + 1e-12
    
    # Load pre-computed ranks from Zarr (already normalized)
    logger.info("------Loading pre-computed ranks...")
    ranks = load_precomputed_ranks(config, adata, gM, slice_start_idx)
    
    # Process cells in batches with JAX
    logger.info("------Computing marker scores with JAX...")
    cell_positions = np.arange(n_cells)
    
    # Determine optimal batch size based on available memory
    try:
        # Try to use GPU memory info if available
        gpu_memory = jax.devices()[0].memory_stats()
        if gpu_memory:
            available_memory = gpu_memory.get('bytes_limit', 8e9)
            batch_size = min(n_cells, int(available_memory / (n_genes * 100)))
        else:
            batch_size = 10000
    except:
        batch_size = 10000
    
    logger.info(f"Processing {n_cells} cells in batches of {batch_size}")
    
    mk_score = process_cells_batch_jax(
        cell_positions,
        spatial_net_dict,
        coor_latent,
        coor_latent_indv,
        ranks,
        adata_X_bool,
        frac_whole,
        config
    )
    
    # Remove mitochondrial genes
    logger.info("------Filtering mitochondrial genes...")
    gene_names = adata.var_names.values.astype(str)
    mt_gene_mask = ~(
        np.char.startswith(gene_names, "MT-") | 
        np.char.startswith(gene_names, "mt-")
    )
    mk_score = mk_score[mt_gene_mask, :]
    gene_names = gene_names[mt_gene_mask]
    
    # Save results
    logger.info("------Saving marker scores...")
    mk_score_df = pd.DataFrame(mk_score, index=gene_names, columns=adata.obs_names)
    mk_score_df.reset_index(inplace=True)
    mk_score_df.rename(columns={"index": "HUMAN_GENE_SYM"}, inplace=True)
    
    # Atomic write with temporary file
    target_path = config.mkscore_feather_path
    rand_prefix = uuid.uuid4().hex[:8]
    tmp_path = target_path.with_name(f"{rand_prefix}_{target_path.name}")
    mk_score_df.to_feather(tmp_path)
    os.rename(tmp_path, target_path)
    
    logger.info(f"------Marker scores saved to {target_path}")
    
    # Clear JAX cache to free memory
    jax.clear_caches()
    gc.collect()
    
    return mk_score_df


# ============================================================================
# Process all slices from concatenated data
# ============================================================================

def process_all_slices_from_concatenated(config: LatentToGeneConfig, *, layer_key=None):
    """
    Process all slices from the concatenated AnnData file.
    
    Args:
        config: Configuration object  
        layer_key: Layer to use for expression data (if applicable)
        
    Returns:
        dict: Dictionary mapping slice_id to marker score DataFrame
    """
    concatenated_path = config.latent_dir / "concatenated_latent_adata.h5ad"
    
    if not concatenated_path.exists():
        raise FileNotFoundError(
            f"Concatenated AnnData not found at {concatenated_path}. "
            "Please run find_latent_representation.py first."
        )
    
    # Load concatenated data to get slice IDs
    logger.info(f"Loading concatenated AnnData from {concatenated_path}")
    full_adata = sc.read_h5ad(concatenated_path)
    
    if 'slice_id' not in full_adata.obs.columns:
        raise ValueError("No 'slice_id' column found in concatenated data")
    
    # Get unique slice IDs
    slice_ids = full_adata.obs['slice_id'].unique()
    logger.info(f"Found {len(slice_ids)} slices to process")
    
    # Process each slice
    results = {}
    for slice_id in tqdm(slice_ids, desc="Processing slices"):
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing slice: {slice_id}")
        logger.info(f"{'='*60}")
        
        # Update config for this specific slice
        # Create a new config with updated values
        slice_config = deepcopy(config)
        slice_config.sample_name = slice_id
        
        # Set output path for this slice
        slice_config.mkscore_feather_path = (
            config.latent_dir / f"mkscore_{slice_id}.feather"
        )
        
        # Process the slice
        mk_score_df = run_latent_to_gene_jax(
            slice_config, 
            layer_key=layer_key,
            slice_id=slice_id
        )
        
        results[slice_id] = mk_score_df
        
        # Clear memory between slices
        gc.collect()
        jax.clear_caches()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Completed processing all {len(slice_ids)} slices")
    logger.info(f"{'='*60}")
    
    return results


# ============================================================================
# Alternative: Multi-GPU version using pmap
# ============================================================================

def run_latent_to_gene_multi_gpu(config: LatentToGeneConfig, *, layer_key=None):
    """
    Multi-GPU version using JAX pmap for even larger datasets.
    """
    n_devices = jax.device_count()
    if n_devices == 1:
        logger.info("Single device detected, using standard JAX version")
        return run_latent_to_gene_jax(config, layer_key=layer_key)
    
    logger.info(f"Multi-GPU mode: {n_devices} devices available")
    
    # The multi-GPU implementation would shard data across devices
    # and use pmap for parallel processing
    # This is a placeholder for the full implementation
    
    raise NotImplementedError(
        "Multi-GPU version requires data sharding implementation. "
        "Use run_latent_to_gene_jax for single GPU."
    )