"""
Connectivity matrix building for homogeneous spot identification
Implements the spatial → anchor → homogeneous neighbor finding algorithm
"""

import logging
from functools import partial

import anndata as ad
import jax.numpy as jnp
import numpy as np
import scanpy as sc
from jax import jit
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    track,
)
from scipy.spatial import cKDTree

from gsMap.config import (
    CrossSliceSelectionMode,
    DatasetType,
    HomogeneousSelectionMode,
    LatentToGeneConfig,
)
from gsMap.logging_config import is_interactive_output

logger = logging.getLogger(__name__)


def find_spatial_neighbors_with_slices(
    coords: np.ndarray,
    slice_ids: np.ndarray | None = None,
    query_cell_mask: np.ndarray | None = None,
    high_quality_cell_mask: np.ndarray = None,
    k_central: int = 101,
    k_adjacent: int = 50,
    n_adjacent_slices: int = 2,
    cross_slice_selection_mode: CrossSliceSelectionMode | str = (CrossSliceSelectionMode.ORDINAL),
    cross_slice_max_distance: float | None = None,
    cross_slice_z_scale: float = 1.0,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """
    Find spatial neighbors with slice-aware search for 3D data.

    For 2D data (slice_ids is None), performs standard KNN.
    For 3D data, implements slice-aware neighbor search:
    - Finds k_central neighbors on the same slice
    - Finds k_adjacent neighbors on each of n_adjacent_slices above and below

    KDTrees are built using only high quality cells, and neighbors are searched
    within high quality cells only.

    Args:
        coords: Spatial coordinates in XY (2D) or XYZ (3D) column order.
        slice_ids: Slice/z-coordinate indices (n_cells,) - sequential integers
        query_cell_mask: Boolean mask for cells to find neighbors for
        high_quality_cell_mask: Boolean mask for high quality cells (defines the neighbor search pool)
        k_central: Number of neighbors to find on the central slice
        k_adjacent: Number of neighbors to find on each adjacent slice
        n_adjacent_slices: Number of slices to search above and below for ordinal
            selection, or half the maximum number of physical-window sections.
        cross_slice_selection_mode: Select sections by ordinal IDs (legacy) or
            measured physical z distance.
        cross_slice_max_distance: Maximum physical z distance when using
            physical-window selection; Z is always the third column.
        cross_slice_z_scale: Physical units per registered z-coordinate unit.

    Returns:
        Tuple of:
        - spatial_neighbors: Array of neighbor indices (n_query_cells, k_central + 2*n_adjacent_slices*k_adjacent)
        - neighbor_pool_per_slice: Dict mapping slice_id to local indices of neighbor pool cells on that slice
    """
    n_cells = len(coords)
    if query_cell_mask is None:
        query_cell_mask = np.ones(n_cells, dtype=bool)

    if high_quality_cell_mask is None:
        high_quality_cell_mask = np.ones(n_cells, dtype=bool)

    logger.debug(
        f"Finding neighbors: {high_quality_cell_mask.sum()}/{n_cells} cells are high quality"
    )

    query_cell_indices = np.where(query_cell_mask)[0]
    n_query_cells = len(query_cell_indices)

    # Neighbor pool: high quality cells that will be used to build KDTrees
    # (intersection of query_cell_mask and high_quality_cell_mask)
    neighbor_pool_mask = query_cell_mask & high_quality_cell_mask
    neighbor_pool_indices = np.where(neighbor_pool_mask)[0]

    total_neighbors_per_cell = k_central + (
        2 * n_adjacent_slices * k_adjacent if slice_ids is not None else 0
    )
    if n_query_cells == 0 or len(neighbor_pool_indices) == 0:
        return np.full((n_query_cells, total_neighbors_per_cell), -1, dtype=np.int32), {}

    # If no slice_ids provided, perform standard 2D KNN
    if slice_ids is None:
        logger.info(f"No slice IDs provided, performing standard 2D KNN with k={k_central}")
        # Build tree with neighbor pool cells only
        kdtree = cKDTree(coords[neighbor_pool_mask])
        # Query for all query cells (including non-HQ ones)
        _, neighbor_local_indices = kdtree.query(
            coords[query_cell_mask],
            k=list(range(1, k_central + 1)),
            workers=-1,  # Use all available cores
        )
        # Convert local indices to global indices
        spatial_neighbors = np.full(neighbor_local_indices.shape, -1, dtype=np.int32)
        valid = neighbor_local_indices < len(neighbor_pool_indices)
        spatial_neighbors[valid] = neighbor_pool_indices[neighbor_local_indices[valid]]
        return spatial_neighbors, {}

    # Slice-aware neighbor search with fixed-size arrays
    logger.info(
        f"Performing slice-aware neighbor search: k_central={k_central}, "
        f"k_adjacent={k_adjacent}, n_adjacent_slices={n_adjacent_slices}"
    )

    query_cell_slice_ids = slice_ids[query_cell_mask]
    query_cell_coords = coords[query_cell_mask]

    # Pre-allocate output with fixed size, initialized with -1 (invalid)
    total_neighbors_per_cell = k_central + 2 * n_adjacent_slices * k_adjacent
    # Always use int32 for spatial neighbors since they contain global cell indices
    # which can easily exceed int16 range in large spatial datasets
    spatial_neighbors = np.full((n_query_cells, total_neighbors_per_cell), -1, dtype=np.int32)

    # Get unique slices and create mapping for all query cells
    unique_slice_ids = np.unique(query_cell_slice_ids)
    slice_to_query_cell_indices = {
        s: np.where(query_cell_slice_ids == s)[0] for s in unique_slice_ids
    }

    # Create mapping for neighbor pool cells per slice (for building KDTrees)
    neighbor_pool_per_slice = {}

    # Get slice IDs and coordinates for neighbor pool cells
    neighbor_pool_slice_ids = slice_ids[neighbor_pool_mask]
    neighbor_pool_coords = coords[neighbor_pool_mask]

    for slice_id in unique_slice_ids:
        # Find neighbor pool cells on this slice (in the neighbor_pool array)
        slice_neighbor_pool_mask = neighbor_pool_slice_ids == slice_id
        slice_neighbor_pool_local_indices = np.where(slice_neighbor_pool_mask)[0]
        neighbor_pool_per_slice[slice_id] = slice_neighbor_pool_local_indices
        logger.debug(
            f"Slice {slice_id}: {len(slice_neighbor_pool_local_indices)} high quality cells"
        )

    # Pre-compute KDTree for each slice using neighbor pool cells only
    # Note: KDTree can handle cases where k > number of points in tree
    # It will return valid neighbors first, then fill remaining slots with invalid indices
    slice_kdtrees = {}
    for slice_id in unique_slice_ids:
        # Use neighbor pool cells to build tree
        slice_neighbor_pool_local = neighbor_pool_per_slice.get(slice_id, np.array([]))
        if len(slice_neighbor_pool_local) > 0:  # Build tree as long as there's at least 1 hq cell
            # Get coordinates of neighbor pool cells on this slice
            slice_neighbor_pool_coords = neighbor_pool_coords[slice_neighbor_pool_local]
            kdtree = cKDTree(slice_neighbor_pool_coords)
            # Store tree with global indices of neighbor pool cells
            slice_kdtrees[slice_id] = (kdtree, neighbor_pool_indices[slice_neighbor_pool_local])

            if len(slice_neighbor_pool_local) < max(k_central, k_adjacent):
                logger.warning(
                    f"Slice {slice_id} has only {len(slice_neighbor_pool_local)} high quality cells, "
                    f"which is less than required k={max(k_central, k_adjacent)}. "
                    # f"Some neighbors will be invalid (-1)."
                )

    selection_mode = (
        CrossSliceSelectionMode(cross_slice_selection_mode)
        if n_adjacent_slices > 0
        else CrossSliceSelectionMode.ORDINAL
    )

    # Build a mapping of slice_id -> list of adjacent slice_ids to search.
    # The fixed number of slots keeps downstream array shapes unchanged. Physical
    # selection pads missing slots instead of substituting a more distant plane.
    slice_adjacent_mapping = {}
    min_slice_id = min(unique_slice_ids)
    max_slice_id = max(unique_slice_ids)

    slice_z_positions = None
    if selection_mode == CrossSliceSelectionMode.PHYSICAL_WINDOW:
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(
                "physical_window cross-section selection requires a spatial key "
                "with exactly three coordinate columns [X, Y, Z]"
            )
        if (
            cross_slice_max_distance is None
            or not np.isfinite(cross_slice_max_distance)
            or cross_slice_max_distance <= 0
        ):
            raise ValueError(
                "cross_slice_max_distance must be positive for physical_window selection"
            )
        if not np.isfinite(cross_slice_z_scale) or cross_slice_z_scale <= 0:
            raise ValueError("cross_slice_z_scale must be positive for physical_window selection")

        slice_z_positions = {
            slice_id: float(np.median(coords[slice_ids == slice_id, 2]) * cross_slice_z_scale)
            for slice_id in unique_slice_ids
        }
        if not np.isfinite(list(slice_z_positions.values())).all():
            raise ValueError("Non-finite physical z position detected")

    for slice_id in unique_slice_ids:
        target_count = 2 * n_adjacent_slices
        if selection_mode == CrossSliceSelectionMode.PHYSICAL_WINDOW:
            if slice_z_positions is None:
                raise RuntimeError("Physical z positions were not initialized")
            focal_z = slice_z_positions[slice_id]
            adjacent_slices = []
            for candidate_slice in unique_slice_ids:
                if candidate_slice == slice_id or candidate_slice not in slice_kdtrees:
                    continue
                delta = slice_z_positions[candidate_slice] - focal_z
                distance = abs(delta)
                if distance <= cross_slice_max_distance + 1e-9:
                    adjacent_slices.append((candidate_slice, distance, -1 if delta < 0 else 1))
            adjacent_slices.sort(key=lambda item: (item[1], item[2], item[0]))
            adjacent_slices = adjacent_slices[:target_count]
        else:
            adjacent_slices = []

            # Collect slices in both directions up to n_adjacent_slices.
            for offset in range(1, n_adjacent_slices + 1):
                neg_slice = slice_id - offset
                pos_slice = slice_id + offset
                if neg_slice in slice_kdtrees:
                    adjacent_slices.append((neg_slice, offset, -1))
                if pos_slice in slice_kdtrees:
                    adjacent_slices.append((pos_slice, offset, 1))

            # Preserve legacy boundary compensation for ordinal mode only.
            if len(adjacent_slices) < target_count:
                extra_offset = n_adjacent_slices + 1
                while len(adjacent_slices) < target_count and extra_offset <= max(
                    max_slice_id - min_slice_id, 10
                ):
                    neg_slice = slice_id - extra_offset
                    pos_slice = slice_id + extra_offset

                    if neg_slice >= min_slice_id and neg_slice in slice_kdtrees:
                        if not any(s[0] == neg_slice for s in adjacent_slices):
                            adjacent_slices.append((neg_slice, extra_offset, -1))
                            if len(adjacent_slices) >= target_count:
                                break

                    if pos_slice <= max_slice_id and pos_slice in slice_kdtrees:
                        if not any(s[0] == pos_slice for s in adjacent_slices):
                            adjacent_slices.append((pos_slice, extra_offset, 1))
                            if len(adjacent_slices) >= target_count:
                                break

                    extra_offset += 1

            adjacent_slices.sort(key=lambda item: (item[1], item[2], item[0]))

        slice_adjacent_mapping[slice_id] = adjacent_slices

        if len(adjacent_slices) < target_count:
            logger.warning(
                f"Slice {slice_id} only has {len(adjacent_slices)} adjacent slices available "
                f"(target: {target_count}). Some neighbor slots will remain empty."
            )

    # Log the adjacent slice mapping for verification
    for slice_id in sorted(slice_adjacent_mapping.keys()):
        adj_slice_ids = [s[0] for s in slice_adjacent_mapping[slice_id]]
        logger.debug(f"Slice {slice_id} will search in adjacent slices: {adj_slice_ids}")

    # Batch process all query cells (including edge cases with few neighbor pool cells)
    for slice_id in unique_slice_ids:
        if slice_id in slice_kdtrees:
            # Get all query cells on this slice
            query_cells_on_slice = slice_to_query_cell_indices[slice_id]
            query_coords = query_cell_coords[query_cells_on_slice]

            # Central slice neighbors (fixed k_central)
            central_kdtree, central_neighbor_pool_global_indices = slice_kdtrees[slice_id]
            _, central_neighbor_local_indices = central_kdtree.query(
                query_coords, k=list(range(1, k_central + 1)), workers=-1
            )

            # Handle invalid indices returned by KDTree when k > number of points in tree
            # KDTree returns index >= tree_size for invalid slots
            n_central_pool_cells = len(central_neighbor_pool_global_indices)
            # Initialize with -1 (invalid)
            central_neighbor_global_indices = np.full_like(
                central_neighbor_local_indices, -1, dtype=np.int32
            )
            # Create mask for valid indices (< tree size)
            central_valid_mask = central_neighbor_local_indices < n_central_pool_cells
            # Map valid local indices to global indices
            central_neighbor_global_indices[central_valid_mask] = (
                central_neighbor_pool_global_indices[
                    central_neighbor_local_indices[central_valid_mask]
                ]
            )
            spatial_neighbors[query_cells_on_slice, :k_central] = central_neighbor_global_indices

            # Adjacent slices - use the pre-computed mapping
            # Get the list of adjacent slices to search for this slice_id
            adjacent_slices_to_search = slice_adjacent_mapping.get(slice_id, [])

            neighbor_column_offset = k_central

            # We need to fill exactly 2*n_adjacent_slices slots in the output array
            # The array structure is: [offset=1,dir=-1], [offset=1,dir=+1], [offset=2,dir=-1], [offset=2,dir=+1], ...
            for slot_idx in range(2 * n_adjacent_slices if k_adjacent > 0 else 0):
                if slot_idx < len(adjacent_slices_to_search):
                    # We have a slice to search for this slot
                    adjacent_slice_id, _, _ = adjacent_slices_to_search[slot_idx]

                    adjacent_kdtree, adjacent_neighbor_pool_global_indices = slice_kdtrees[
                        adjacent_slice_id
                    ]
                    _, adjacent_neighbor_local_indices = adjacent_kdtree.query(
                        query_coords, k=list(range(1, k_adjacent + 1)), workers=-1
                    )

                    # Handle invalid indices for adjacent slices
                    n_adjacent_pool_cells = len(adjacent_neighbor_pool_global_indices)
                    adjacent_neighbor_global_indices = np.full_like(
                        adjacent_neighbor_local_indices, -1, dtype=np.int32
                    )
                    adjacent_valid_mask = adjacent_neighbor_local_indices < n_adjacent_pool_cells
                    adjacent_neighbor_global_indices[adjacent_valid_mask] = (
                        adjacent_neighbor_pool_global_indices[
                            adjacent_neighbor_local_indices[adjacent_valid_mask]
                        ]
                    )
                    spatial_neighbors[
                        query_cells_on_slice,
                        neighbor_column_offset : neighbor_column_offset + k_adjacent,
                    ] = adjacent_neighbor_global_indices
                # else: slot remains as -1 (already initialized) if no slice available

                neighbor_column_offset += k_adjacent

    # Note: Slices with 0 neighbor pool cells will not be in slice_kdtrees
    # Query cells on such slices will have all neighbors remain as -1 (already initialized)

    return spatial_neighbors, neighbor_pool_per_slice


@partial(jit, static_argnums=(5, 6, 8))
def _find_homogeneous_spot_dual_embedding_batch_jit(
    emb_niche_batch_norm: jnp.ndarray,  # (batch_size, d1)
    emb_indv_batch_norm: jnp.ndarray,  # (batch_size, d2)
    spatial_neighbors: jnp.ndarray,  # (batch_size, k1)
    all_emb_niche_norm: jnp.ndarray,  # (n_all, d1)
    all_emb_indv_norm: jnp.ndarray,  # (n_all, d2)
    homogeneous_neighbors: int,
    cell_embedding_similarity_threshold: float = 0.0,
    spatial_domain_similarity_threshold: float = 0.5,
    hard_niche_filter: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Find homogeneous neighbors with soft-priority or historical hard filtering.

    Only valid candidates with positive thresholded cell similarity are ranked.
    Slots without an eligible candidate have index -1 and zero similarities.

    Returns:
        homogeneous_neighbors_array: Indices of selected neighbors (batch, homogeneous_neighbors)
        selected_cell_sims: Cell similarity scores for selected neighbors (batch, homogeneous_neighbors)
        selected_niche_sims: Niche/Spatial similarity scores for selected neighbors (batch, homogeneous_neighbors)
    """
    batch_size = emb_indv_batch_norm.shape[0]

    # Step 1: Extract spatial neighbors' embeddings
    # Handle padding: map -1 to 0 temporarily for extraction
    safe_neighbors = jnp.where(spatial_neighbors >= 0, spatial_neighbors, 0)
    spatial_emb_indv_norm = all_emb_indv_norm[safe_neighbors]
    spatial_emb_niche_norm = all_emb_niche_norm[safe_neighbors]

    # Step 2: Compute Similarities
    # 2a. Cell Similarity
    raw_cell_sims = jnp.einsum("bd,bkd->bk", emb_indv_batch_norm, spatial_emb_indv_norm)
    # Apply cell threshold: values below threshold become 0.0
    cell_sims = jnp.where(raw_cell_sims >= cell_embedding_similarity_threshold, raw_cell_sims, 0.0)

    # High-quality and annotation restrictions are already applied to the spatial
    # candidate pool. Enforce positive weight before niche prioritization: a
    # zero-weight niche match must not displace a usable soft-priority fallback.
    eligible_neighbor_mask = (spatial_neighbors >= 0) & (cell_sims > 0)

    # 2b. Niche (Spatial) Similarity
    niche_sims = jnp.einsum("bd,bkd->bk", emb_niche_batch_norm, spatial_emb_niche_norm)

    # Step 3: Compute the ranking score. The default soft-priority rule ranks
    # niche-passing candidates first and then fills remaining slots. The
    # historical rule ranks non-passing candidates at zero and, critically,
    # gives any selected fallback slot zero marker-score weight.
    BONUS = 2.0
    pass_spatial_mask = niche_sims >= spatial_domain_similarity_threshold
    if hard_niche_filter:
        ranking_score = jnp.where(pass_spatial_mask, cell_sims, 0.0)
    else:
        ranking_score = cell_sims + jnp.where(pass_spatial_mask, BONUS, 0.0)

    # Step 4: Rank only eligible neighbors; padding and zero-weight candidates
    # must sort after every eligible candidate, regardless of niche similarity.
    ranking_score = jnp.where(eligible_neighbor_mask, ranking_score, -jnp.inf)

    # Step 5: Sort and Select
    # Get indices of top scores
    top_homo_idx = jnp.argsort(-ranking_score, axis=1)[:, :homogeneous_neighbors]
    batch_idx = jnp.arange(batch_size)[:, None]

    # Gather the results
    homogeneous_neighbors_array = spatial_neighbors[batch_idx, top_homo_idx]
    selected_cell_sims = cell_sims[batch_idx, top_homo_idx]
    selected_niche_sims = niche_sims[batch_idx, top_homo_idx]

    if hard_niche_filter:
        selected_cell_sims = jnp.where(
            selected_niche_sims >= spatial_domain_similarity_threshold,
            selected_cell_sims,
            0.0,
        )

    # Step 6: Keep the fixed-size output, but represent exhausted/ineligible
    # slots explicitly as missing. Downstream rank reads handle -1 with weight 0.
    final_valid_mask = eligible_neighbor_mask[batch_idx, top_homo_idx]
    homogeneous_neighbors_array = jnp.where(final_valid_mask, homogeneous_neighbors_array, -1)
    selected_cell_sims = jnp.where(final_valid_mask, selected_cell_sims, 0.0)
    selected_niche_sims = jnp.where(final_valid_mask, selected_niche_sims, 0.0)

    return homogeneous_neighbors_array, selected_cell_sims, selected_niche_sims


def _find_homogeneous_3d_memory_efficient(
    emb_niche_masked_jax: jnp.ndarray,
    emb_indv_masked_jax: jnp.ndarray,
    spatial_neighbors: np.ndarray,
    all_emb_niche_norm_jax: jnp.ndarray,
    all_emb_indv_norm_jax: jnp.ndarray,
    num_homogeneous_per_slice: int,
    k_central: int,
    k_adjacent: int,
    n_adjacent_slices: int,
    cell_embedding_similarity_threshold: float,
    spatial_domain_similarity_threshold: float,
    hard_niche_filter: bool,
    find_homogeneous_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient version of 3D homogeneous neighbor finding.
    Processes slices separately to avoid large memory allocations.

    Args:
        emb_niche_masked_jax: Niche embeddings for masked cells (JAX array, float16, always exists, may be dummy ones)
        emb_indv_masked_jax: Individual embeddings for masked cells (JAX array, float16)
        spatial_neighbors: Spatial neighbors array with structure [central | adj1 | adj2 | ...] (numpy array)
        all_emb_niche_norm_jax: All normalized niche embeddings (JAX array, float16, always exists, may be dummy ones)
        all_emb_indv_norm_jax: All normalized individual embeddings (JAX array, float16)
        num_homogeneous_per_slice: Number of neighbors to select per slice
        k_central: Number of neighbors in central slice
        k_adjacent: Number of neighbors per adjacent slice
        n_adjacent_slices: Number of adjacent slices above and below
        cell_embedding_similarity_threshold: Minimum similarity threshold for cell embedding
        spatial_domain_similarity_threshold: Minimum similarity threshold for spatial domain embedding
        hard_niche_filter: Give candidates below the niche threshold zero weight.
        find_homogeneous_batch_size: Batch size for processing

    Returns:
        homogeneous_neighbors: Selected neighbors
        homogeneous_cell_sims: Cell similarity scores for selected neighbors
        homogeneous_niche_sims: Niche/Spatial similarity scores for selected neighbors
    """
    n_masked = emb_indv_masked_jax.shape[0]

    homogeneous_neighbors_all_slices = []
    homogeneous_cell_sims_all_slices = []
    homogeneous_niche_sims_all_slices = []

    # Process all slices (central + adjacent) in a single loop
    total_slices = 1 + 2 * n_adjacent_slices

    # Create overall slice progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        TimeElapsedColumn(),
        disable=not is_interactive_output(),
        refresh_per_second=1,
    ) as slice_progress:
        # Overall slice progress task
        slice_task = slice_progress.add_task(
            "Finding homogeneous neighbors (3D cross-slice)...", total=total_slices
        )

        for slice_num in range(total_slices):
            # Determine slice name and parameters
            if slice_num == 0:
                slice_name = "central slice"
                slice_start = 0
                slice_end = k_central
            else:
                # Adjacent slices
                adj_idx = slice_num - 1
                if adj_idx < n_adjacent_slices:
                    slice_name = f"adjacent slice -{n_adjacent_slices - adj_idx}"
                else:
                    slice_name = f"adjacent slice +{adj_idx - n_adjacent_slices + 1}"

                slice_start = k_central + adj_idx * k_adjacent
                slice_end = slice_start + k_adjacent

            # Convert slice neighbors to JAX array once per slice
            spatial_neighbors_slice = jnp.asarray(spatial_neighbors[:, slice_start:slice_end])

            homogeneous_neighbors_slice_list = []
            homogeneous_cell_sims_slice_list = []
            homogeneous_niche_sims_slice_list = []

            # Process batches for this slice using simple transient track
            for batch_start in track(
                range(0, n_masked, find_homogeneous_batch_size),
                description=f"Finding homogeneous neighbors ({slice_name})",
                transient=True,
                disable=not is_interactive_output(),
            ):
                batch_end = min(batch_start + find_homogeneous_batch_size, n_masked)
                batch_indices = slice(batch_start, batch_end)

                # Get batch data (emb_niche always exists now, may be dummy ones)
                emb_niche_batch_norm = emb_niche_masked_jax[batch_indices]
                emb_indv_batch_norm = emb_indv_masked_jax[batch_indices]

                # Extract batch of neighbors for this slice
                spatial_neighbors_slice_batch = spatial_neighbors_slice[batch_indices, :]

                # Process with 2D function
                homo_neighbors_batch, cell_sims_batch, niche_sims_batch = (
                    _find_homogeneous_spot_dual_embedding_batch_jit(
                        emb_niche_batch_norm=emb_niche_batch_norm,
                        emb_indv_batch_norm=emb_indv_batch_norm,
                        spatial_neighbors=spatial_neighbors_slice_batch,
                        all_emb_niche_norm=all_emb_niche_norm_jax,
                        all_emb_indv_norm=all_emb_indv_norm_jax,
                        homogeneous_neighbors=num_homogeneous_per_slice,
                        cell_embedding_similarity_threshold=cell_embedding_similarity_threshold,
                        spatial_domain_similarity_threshold=spatial_domain_similarity_threshold,
                        hard_niche_filter=hard_niche_filter,
                    )
                )

                homogeneous_neighbors_slice_list.append(np.array(homo_neighbors_batch))
                homogeneous_cell_sims_slice_list.append(np.array(cell_sims_batch))
                homogeneous_niche_sims_slice_list.append(np.array(niche_sims_batch))

            # Concatenate this slice's results
            homogeneous_neighbors_slice = np.vstack(homogeneous_neighbors_slice_list)
            homogeneous_cell_sims_slice = np.vstack(homogeneous_cell_sims_slice_list)
            homogeneous_niche_sims_slice = np.vstack(homogeneous_niche_sims_slice_list)
            homogeneous_neighbors_all_slices.append(homogeneous_neighbors_slice)
            homogeneous_cell_sims_all_slices.append(homogeneous_cell_sims_slice)
            homogeneous_niche_sims_all_slices.append(homogeneous_niche_sims_slice)

            # Update slice progress
            slice_progress.update(slice_task, advance=1)

    # Concatenate all slices along axis 1
    homogeneous_neighbors = np.concatenate(homogeneous_neighbors_all_slices, axis=1)
    homogeneous_cell_sims = np.concatenate(homogeneous_cell_sims_all_slices, axis=1)
    homogeneous_niche_sims = np.concatenate(homogeneous_niche_sims_all_slices, axis=1)

    return homogeneous_neighbors, homogeneous_cell_sims, homogeneous_niche_sims


def build_scrna_connectivity(
    emb_cell: np.ndarray,
    cell_mask: np.ndarray | None = None,
    n_neighbors: int = 21,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, None]:
    """
    Build connectivity for scRNA-seq data using KNN on cell embeddings.

    Args:
        emb_cell: Cell embeddings (n_cells, d)
        cell_mask: Boolean mask for cells to process
        n_neighbors: Number of nearest neighbors
        metric: Distance metric for KNN

    Returns:
        neighbor_indices: (n_masked, n_neighbors) array of neighbor indices
        neighbor_weights: (n_masked, n_neighbors) array of weights from KNN graph
        niche_sims: None (scRNA-seq has no niche embedding)
    """
    n_cells = len(emb_cell)
    if cell_mask is None:
        cell_mask = np.ones(n_cells, dtype=bool)

    cell_indices = np.where(cell_mask)[0]
    n_masked = len(cell_indices)

    logger.info(f"Building scRNA-seq connectivity using KNN with k={n_neighbors}")

    # Create temporary AnnData for using scanpy's neighbors function
    adata_temp = ad.AnnData(X=emb_cell[cell_mask])
    adata_temp.obsm["X_emb"] = emb_cell[cell_mask]

    # Compute neighbors using scanpy
    sc.pp.neighbors(
        adata_temp, n_neighbors=n_neighbors, use_rep="X_emb", metric=metric, method="umap"
    )

    # Extract connectivity matrix
    connectivities = adata_temp.obsp["connectivities"].tocsr()

    # Convert to dense format for consistency with spatial methods
    # Check the actual max cell index value to determine dtype
    max_cell_idx = cell_indices.max() if len(cell_indices) > 0 else 0
    idx_dtype = np.int16 if max_cell_idx < 32768 else np.int32
    neighbor_indices = np.zeros((n_masked, n_neighbors), dtype=idx_dtype)
    neighbor_weights = np.zeros((n_masked, n_neighbors), dtype=np.float16)

    for i in range(n_masked):
        row = connectivities.getrow(i)
        neighbors = row.indices
        weights = row.data

        # Sort by weight (descending)
        sorted_idx = np.argsort(-weights)[:n_neighbors]

        if len(sorted_idx) < n_neighbors:
            # Pad with self-index if needed
            n_found = len(sorted_idx)
            neighbor_indices[i, :n_found] = cell_indices[neighbors[sorted_idx]]
            neighbor_indices[i, n_found:] = cell_indices[i]
            neighbor_weights[i, :n_found] = weights[sorted_idx]
            neighbor_weights[i, n_found:] = 0.0
        else:
            neighbor_indices[i] = cell_indices[neighbors[sorted_idx]]
            neighbor_weights[i] = weights[sorted_idx]

    # Normalize weights to sum to 1
    weight_sums = neighbor_weights.sum(axis=1, keepdims=True)
    weight_sums = np.where(weight_sums > 0, weight_sums, 1.0)
    neighbor_weights = neighbor_weights / weight_sums

    logger.info(f"scRNA-seq connectivity built: {n_masked} cells × {n_neighbors} neighbors")

    return neighbor_indices, neighbor_weights, None


class ConnectivityMatrixBuilder:
    """Build connectivity matrix using JAX-accelerated computation with GPU memory optimization"""

    def __init__(self, config: LatentToGeneConfig):
        """
        Initialize with configuration

        Args:
            config: LatentToGeneConfig object
        """
        self.config = config
        self.find_homogeneous_batch_size = config.find_homogeneous_batch_size
        self.dataset_type = config.dataset_type

    def build_connectivity_matrix(
        self,
        coords: np.ndarray | None = None,
        emb_niche: np.ndarray = None,
        emb_indv: np.ndarray | None = None,
        cell_mask: np.ndarray | None = None,
        high_quality_mask: np.ndarray = None,
        slice_ids: np.ndarray | None = None,
        k_central: int | None = None,
        k_adjacent: int | None = None,
        n_adjacent_slices: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Build connectivity matrix for a group of cells based on dataset type.

        For scRNA-seq: Uses KNN on cell embeddings (emb_indv)
        For spatial2D: Uses spatial anchors and homogeneous neighbors
        For spatial3D: Uses slice-aware spatial anchors and homogeneous neighbors

        Args:
            coords: Spatial coordinates (n_cells, 2) - required for spatial datasets
            emb_niche: Niche embeddings (n_cells, d1) - always provided (may be dummy ones for scRNA-seq)
            emb_indv: Cell identity embeddings (n_cells, d2) - required for all datasets
            cell_mask: Boolean mask for cells to process
            high_quality_mask: Boolean mask for high quality cells (used for neighbor search in spatial data)
            slice_ids: Optional slice/z-coordinate indices (n_cells,) for spatial3D
            k_central: Number of neighbors on central slice (defaults to config settings)
            k_adjacent: Number of neighbors on adjacent slices for spatial3D
            n_adjacent_slices: Number of slices to search above/below for spatial3D

        Returns:
            Tuple of (neighbor_indices, cell_sims, niche_sims) arrays.
            niche_sims is None for scRNA-seq datasets.
        """
        # Check dataset type and call appropriate method
        if self.dataset_type == DatasetType.SCRNA_SEQ:
            logger.info("Building connectivity for scRNA-seq dataset")
            if emb_indv is None:
                raise ValueError("emb_indv (cell embeddings) required for scRNA-seq dataset")

            return build_scrna_connectivity(
                emb_cell=emb_indv,
                cell_mask=cell_mask,
                n_neighbors=self.config.homogeneous_neighbors,
                metric="euclidean",
            )

        elif self.dataset_type in ["spatial2D", "spatial3D"]:
            logger.info(f"Building connectivity for {self.dataset_type} dataset")

            # Validate required inputs for spatial datasets
            if coords is None or emb_indv is None:
                raise ValueError("coords and emb_indv required for spatial datasets")

            from gsMap.config.spatial import spatial_coordinates

            coords = spatial_coordinates(coords, self.dataset_type, n_obs=len(emb_indv))

            # Use config defaults if not provided
            if k_central is None:
                k_central = self.config.spatial_neighbors
            if k_adjacent is None:
                k_adjacent = self.config.adjacent_slice_spatial_neighbors
            if n_adjacent_slices is None:
                if self.dataset_type == DatasetType.SPATIAL_2D:
                    n_adjacent_slices = 0
                else:  # spatial3D
                    n_adjacent_slices = self.config.n_adjacent_slices

            # For spatial2D, ensure no cross-slice search (but can still have slice_ids)
            if self.dataset_type == DatasetType.SPATIAL_2D:
                n_adjacent_slices = 0  # No cross-slice search, but keep slice_ids if provided

            return self._build_spatial_connectivity(
                coords=coords,
                emb_niche=emb_niche,
                emb_indv=emb_indv,
                cell_mask=cell_mask,
                high_quality_mask=high_quality_mask,
                slice_ids=slice_ids,
                k_central=k_central,
                k_adjacent=k_adjacent,
                n_adjacent_slices=n_adjacent_slices,
                cross_slice_selection_mode=self.config.cross_slice_selection_mode,
                cross_slice_max_distance=self.config.cross_slice_max_distance,
                cross_slice_z_scale=self.config.cross_slice_z_scale,
            )

        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def _build_spatial_connectivity(
        self,
        coords: np.ndarray,
        emb_indv: np.ndarray,
        emb_niche: np.ndarray = None,
        cell_mask: np.ndarray | None = None,
        high_quality_mask: np.ndarray = None,
        slice_ids: np.ndarray | None = None,
        k_central: int = 101,
        k_adjacent: int = 50,
        n_adjacent_slices: int = 2,
        cross_slice_selection_mode: CrossSliceSelectionMode | str = (
            CrossSliceSelectionMode.ORDINAL
        ),
        cross_slice_max_distance: float | None = None,
        cross_slice_z_scale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Internal method for building spatial connectivity matrix.

        Args:
            coords: Spatial coordinates in XY (2D) or XYZ (3D) column order.
            emb_niche: Niche embeddings (n_cells, d1) - always provided (may be dummy ones)
            emb_indv: Cell identity embeddings (n_cells, d2)
            cell_mask: Boolean mask for cells to process
            high_quality_mask: Boolean mask for high quality cells (used for neighbor search)
            slice_ids: Optional slice/z-coordinate indices (n_cells,) for 3D data
            k_central: Number of neighbors on central slice
            k_adjacent: Number of neighbors on adjacent slices for 3D data
            n_adjacent_slices: Number of slices to search above/below for 3D data
            cross_slice_selection_mode: Ordinal or physical-window section selection.
            cross_slice_max_distance: Physical Z-distance cutoff using the third column.
            cross_slice_z_scale: Physical units per registered z unit.

        Returns:
            Tuple of (neighbor_indices, cell_sims, niche_sims) arrays
        """

        n_cells = len(coords)
        if cell_mask is None:
            cell_mask = np.ones(n_cells, dtype=bool)

        cell_indices = np.where(cell_mask)[0]
        n_masked = len(cell_indices)

        # Step 1: Find spatial neighbors (slice-aware if slice_ids provided)
        spatial_neighbors, neighbor_pool_per_slice = find_spatial_neighbors_with_slices(
            coords=coords,
            slice_ids=slice_ids,
            query_cell_mask=cell_mask,
            high_quality_cell_mask=high_quality_mask,
            k_central=k_central,
            k_adjacent=k_adjacent,
            n_adjacent_slices=n_adjacent_slices,
            cross_slice_selection_mode=cross_slice_selection_mode,
            cross_slice_max_distance=cross_slice_max_distance,
            cross_slice_z_scale=cross_slice_z_scale,
        )

        # Log statistics about neighbor pool cells per slice if available
        if neighbor_pool_per_slice:
            for slice_id, neighbor_pool_indices in neighbor_pool_per_slice.items():
                logger.debug(
                    f"Slice {slice_id}: {len(neighbor_pool_indices)} high quality cells available for neighbor search"
                )

        # Step 2 & 3: Find anchors and homogeneous neighbors in batches
        logger.info(
            f"Finding anchors and homogeneous neighbors (batch size: {self.find_homogeneous_batch_size})..."
        )

        # Convert embeddings to JAX arrays once (shared for both paths)
        # Note: float16 provides sufficient precision for normalized embeddings
        # emb_niche is guaranteed to exist now (may be dummy ones)
        # The full embeddings are the same for every cell type: upload them once.
        cache_key = (id(emb_niche), emb_niche.shape, id(emb_indv), emb_indv.shape)
        cached = getattr(self, "_embedding_cache", None)
        if cached is not None and cached[0] == cache_key:
            all_emb_niche_norm_jax, all_emb_indv_norm_jax = cached[1], cached[2]
        else:
            all_emb_niche_norm_jax = jnp.array(emb_niche, dtype=jnp.float16)
            all_emb_indv_norm_jax = jnp.array(emb_indv, dtype=jnp.float16)
            # Keep the source arrays referenced so their ids cannot be reused.
            self._embedding_cache = (
                cache_key,
                all_emb_niche_norm_jax,
                all_emb_indv_norm_jax,
                emb_niche,
                emb_indv,
            )

        # Get masked embeddings
        masked_cell_indices = np.where(cell_mask)[0]
        emb_niche_masked_jax = all_emb_niche_norm_jax[masked_cell_indices]
        emb_indv_masked_jax = all_emb_indv_norm_jax[masked_cell_indices]

        if self.config.fix_cross_slice_homogenous_neighbors:
            logger.info(
                f"Using 3D constrained selection (ensuring {self.config.homogeneous_neighbors} neighbors per slice)"
            )

            # Use memory-efficient version that processes slices separately
            homogeneous_neighbors, homogeneous_cell_sims, homogeneous_niche_sims = (
                _find_homogeneous_3d_memory_efficient(
                    emb_niche_masked_jax=emb_niche_masked_jax,
                    emb_indv_masked_jax=emb_indv_masked_jax,
                    spatial_neighbors=spatial_neighbors,
                    all_emb_niche_norm_jax=all_emb_niche_norm_jax,
                    all_emb_indv_norm_jax=all_emb_indv_norm_jax,
                    num_homogeneous_per_slice=self.config.homogeneous_neighbors,
                    k_central=k_central,
                    k_adjacent=k_adjacent,
                    n_adjacent_slices=n_adjacent_slices,
                    cell_embedding_similarity_threshold=self.config.cell_embedding_similarity_threshold,
                    spatial_domain_similarity_threshold=self.config.spatial_domain_similarity_threshold,
                    hard_niche_filter=(
                        self.config.homogeneous_selection_mode
                        == HomogeneousSelectionMode.HARD_NICHE_FILTER
                    ),
                    find_homogeneous_batch_size=self.find_homogeneous_batch_size,
                )
            )

        else:
            # Convert spatial_neighbors to JAX array for regular processing
            spatial_neighbors_jax = jnp.array(spatial_neighbors, dtype=jnp.int32)
            homogeneous_neighbors_list = []
            homogeneous_cell_sims_list = []
            homogeneous_niche_sims_list = []

            # Use the standard function (2D or 3D without fix_cross_slice_homogenous_neighbors)
            for batch_start in track(
                range(0, n_masked, self.find_homogeneous_batch_size),
                description="Finding homogeneous neighbors",
                transient=True,
                disable=not is_interactive_output(),
            ):
                batch_end = min(batch_start + self.find_homogeneous_batch_size, n_masked)
                batch_indices = slice(batch_start, batch_end)

                # Get batch data directly from JAX arrays (no GPU movement)
                emb_niche_batch_norm = emb_niche_masked_jax[batch_indices]
                emb_indv_batch_norm = emb_indv_masked_jax[batch_indices]
                spatial_neighbors_batch = spatial_neighbors_jax[batch_indices]

                # Process batch with single JIT-compiled function
                homo_neighbors_batch, cell_sims_batch, niche_sims_batch = (
                    _find_homogeneous_spot_dual_embedding_batch_jit(
                        emb_niche_batch_norm=emb_niche_batch_norm,
                        emb_indv_batch_norm=emb_indv_batch_norm,
                        spatial_neighbors=spatial_neighbors_batch,
                        all_emb_niche_norm=all_emb_niche_norm_jax,
                        all_emb_indv_norm=all_emb_indv_norm_jax,
                        homogeneous_neighbors=self.config.total_homogeneous_neighbor_per_cell,
                        cell_embedding_similarity_threshold=self.config.cell_embedding_similarity_threshold,
                        spatial_domain_similarity_threshold=self.config.spatial_domain_similarity_threshold,
                        hard_niche_filter=(
                            self.config.homogeneous_selection_mode
                            == HomogeneousSelectionMode.HARD_NICHE_FILTER
                        ),
                    )
                )

                # Convert back to numpy and append
                homogeneous_neighbors_list.append(np.array(homo_neighbors_batch))
                homogeneous_cell_sims_list.append(np.array(cell_sims_batch))
                homogeneous_niche_sims_list.append(np.array(niche_sims_batch))

            # Regular batched processing
            homogeneous_neighbors = np.vstack(homogeneous_neighbors_list)
            homogeneous_cell_sims = np.vstack(homogeneous_cell_sims_list)
            homogeneous_niche_sims = np.vstack(homogeneous_niche_sims_list)

        # Return dense format: (n_masked, num_homogeneous) arrays
        return homogeneous_neighbors, homogeneous_cell_sims, homogeneous_niche_sims
