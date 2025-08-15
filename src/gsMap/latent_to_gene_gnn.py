import logging
import os
import pickle
import numpy as np
import pandas as pd
import scanpy as sc
import uuid
from scipy.stats import rankdata
from scipy.sparse import issparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.special import softmax
from gsMap.config import LatentToGeneConfig
from gsMap.slice_mean import merge_zarr_means

logger = logging.getLogger(__name__)


def find_neighbors(coor, num_neighbour):
    """
    Find Neighbors of each cell (based on spatial coordinates).
    """
    nbrs = NearestNeighbors(n_neighbors=num_neighbour).fit(coor)
    distances, indices = nbrs.kneighbors(coor, return_distance=True)
    cell_indices = np.arange(coor.shape[0])
    cell1 = np.repeat(cell_indices, indices.shape[1])
    cell2 = indices.flatten()
    distance = distances.flatten()
    spatial_net = pd.DataFrame(
        {"Cell1": cell1, "Cell2": cell2, "Distance": distance})
    return spatial_net


def build_spatial_net(adata, annotation, num_neighbour, spatial_key):
    """
    Build spatial neighbourhood matrix for each spot (cell) based on the spatial coordinates.
    """
    logger.info(f"------Building spatial graph based on spatial coordinates...")

    coor = adata.obsm[spatial_key]
    if annotation is not None:
        logger.info(f"Cell annotations are provided...")
        spatial_net_list = []
        # Cells with annotations
        for ct in adata.obs[annotation].dropna().unique():
            idx = np.where(adata.obs[annotation] == ct)[0]
            coor_temp = coor[idx, :]
            spatial_net_temp = find_neighbors(
                coor_temp, min(num_neighbour, coor_temp.shape[0])
            )
            # Map back to original indices
            spatial_net_temp["Cell1"] = idx[spatial_net_temp["Cell1"].values]
            spatial_net_temp["Cell2"] = idx[spatial_net_temp["Cell2"].values]
            spatial_net_list.append(spatial_net_temp)
            logger.info(f"{ct}: {coor_temp.shape[0]} cells")

        # Cells labeled as nan
        if pd.isnull(adata.obs[annotation]).any():
            idx_nan = np.where(pd.isnull(adata.obs[annotation]))[0]
            logger.info(f"Nan: {len(idx_nan)} cells")
            spatial_net_temp = find_neighbors(coor, num_neighbour)
            spatial_net_temp = spatial_net_temp[spatial_net_temp["Cell1"].isin(
                idx_nan)]
            spatial_net_list.append(spatial_net_temp)
        spatial_net = pd.concat(spatial_net_list, axis=0)
    else:
        logger.info(f"Cell annotations are not provided...")
        spatial_net = find_neighbors(coor, num_neighbour)

    return spatial_net


def find_anchors(cell_latent, neighbour_latent, num_anchor, cell_use_pos):
    if len(cell_use_pos) < 2:
        return cell_use_pos, softmax(np.ones(len(cell_use_pos)))
    similarity = cosine_similarity(cell_latent, neighbour_latent).flatten()
    indices = np.argsort(-similarity)
    top_indices = indices[:num_anchor].flatten()
    return cell_use_pos[top_indices], similarity[top_indices]


def find_neighbors_regional(
        cell_pos, spatial_net_dict, coor_latent, coor_latent_indv, config
):
    num_neighbour = config.num_neighbour
    num_anchor = config.num_anchor

    cell_use_pos = np.array(spatial_net_dict.get(cell_pos, []))
    if len(cell_use_pos) == 0:
        return []

    num_neighbour = min(len(cell_use_pos), num_neighbour)
    num_anchor = min(len(cell_use_pos), num_anchor)

    # find spatial anchors based on GCN smoothed embeddings
    cell_latent = coor_latent[cell_pos, :].reshape(1, -1)
    neighbour_latent = coor_latent[cell_use_pos, :]
    anchors_pos, anchors_similarity = find_anchors(cell_latent, neighbour_latent, num_anchor, cell_use_pos)

    # find homogeneous spots based on expression embeddings
    # norm = np.linalg.norm(coor_latent_indv, axis=1, keepdims=True)
    # coor_latent_indv = coor_latent_indv / (norm + 1e-6)

    cell_latent_indv = coor_latent_indv[cell_pos, :].reshape(1, -1)
    neighbour_latent_indv = coor_latent_indv[anchors_pos, :]

    # similarity = cosine_similarity(cell_latent_indv, neighbour_latent_indv).flatten() * anchors_similarity.flatten()
    similarity = cosine_similarity(cell_latent_indv, neighbour_latent_indv).flatten()
    indices = np.argsort(-similarity)  # descending order
    top_indices = indices[:num_neighbour]

    cell_select_pos = anchors_pos[top_indices]
    cell_select_similarity = softmax(similarity[top_indices])

    return cell_select_pos, cell_select_similarity


def compute_regional_mkscore(cell_pos, spatial_net_dict, coor_latent, coor_latent_indv, config,
                             ranks, frac_whole, adata_X_bool):
    """
    Compute gmean ranks of a region.
    """
    cell_select_pos, cell_select_similarity = find_neighbors_regional(
        cell_pos,
        spatial_net_dict,
        coor_latent,
        coor_latent_indv,
        config,
    )

    if len(cell_select_pos) == 0:
        return np.zeros(ranks.shape[1], dtype=np.float16)

    # Ratio of expression ranks
    ranks_tg = ranks[cell_select_pos, :]
    gene_ranks_region = np.exp((np.log(ranks_tg) * cell_select_similarity[:, np.newaxis]).sum(axis=0))
    gene_ranks_region[gene_ranks_region <= 1] = 0

    if not config.no_expression_fraction:
        # Ratio of expression fractions
        sum_axis0 = adata_X_bool[cell_select_pos, :].sum(axis=0)
        if hasattr(sum_axis0, "A1"):
            frac_focal = sum_axis0.A1 / len(cell_select_pos)
        else:
            frac_focal = sum_axis0.ravel() / len(cell_select_pos)

        frac_region = frac_focal - frac_whole
        frac_region[frac_region <= 0] = 0
        frac_region[frac_region > 0] = 1

        # Simultaneously consider the ratio of expression fractions and ranks
        gene_ranks_region = gene_ranks_region * frac_region

    mkscore = np.exp(gene_ranks_region ** 1.5) - 1
    return mkscore


def run_latent_to_gene(config: LatentToGeneConfig, *, layer_key=None):
    logger.info(f"------Loading the spatial data: {config.sample_name}...")
    adata = sc.read_h5ad(config.hdf5_with_latent_path)
    print(config.hdf5_with_latent_path)

    if layer_key is not None:
        if (counts := adata.layers.get(layer_key, None)) is not None:
            adata.X = counts.copy()
            logger.info(
                f"------Using data from `adata.layers[{layer_key}]`...")
        else:
            logger.warn(
                f"------Invalid layer_key: {layer_key}" "falling back to `adata.X`..."
            )
    else:
        logger.info(f"------Using `adata.X`...")

    if config.annotation is not None:
        logger.info(
            f"------Cell annotations are provided as {config.annotation}...")
        adata = adata[~pd.isnull(adata.obs[config.annotation]), :]

    # Create mappings
    n_cells = adata.n_obs
    n_genes = adata.n_vars

    # Build the spatial graph
    spatial_net = build_spatial_net(
        adata,
        config.annotation,
        config.num_neighbour_spatial,
        config.spatial_key
    )
    spatial_net_dict = spatial_net.groupby("Cell1")["Cell2"].apply(np.array).to_dict()

    # Extract the latent representation
    coor_latent = adata.obsm[config.latent_representation]
    coor_latent = coor_latent.astype(np.float32)
    coor_latent_indv = adata.obsm[config.latent_representation_indv]
    coor_latent_indv = coor_latent_indv.astype(np.float32)

    # Compute ranks
    logger.info("------Ranking the spatial data...")
    adata_X = adata.X if isinstance(adata.X, np.ndarray) else adata.X.toarray()
    adata_X_bool = adata_X.astype(bool)

    # data = adata.layers['residuals'] if 'residuals' in adata.layers.keys() else adata.X
    data = adata.X
    if issparse(data):
        logger.info("Converting sparse matrix to dense...")
        data = data.toarray()

    ranks = np.zeros((n_cells, n_genes), dtype=np.float32)
    for i in tqdm(range(n_cells), desc="Computing ranks per cell"):
        ranks[i, :] = rankdata(data[i, :], method="average")

    # Compute the geometric mean of ranks across slices
    logger.info("Calculating the slices mean...")
    gM, frac_whole = merge_zarr_means(config.zarr_group_path, None)
    frac_whole += 1e-12

    # Normalize the ranks
    ranks = ranks / gM

    # Compute marker scores in parallel
    logger.info("------Computing marker scores...")

    def compute_mk_score_wrapper(cell_pos):
        return compute_regional_mkscore(
            cell_pos,
            spatial_net_dict,
            coor_latent,
            coor_latent_indv,
            config,
            ranks,
            frac_whole,
            adata_X_bool,
        )

    mk_scores = []
    for cell_pos in tqdm(range(n_cells), desc="Calculating marker scores"):
        mk_score_cell = compute_mk_score_wrapper(cell_pos)
        mk_scores.append(mk_score_cell)

    mk_score = np.vstack(mk_scores).T

    # Remove mitochondrial genes
    gene_names = adata.var_names.values.astype(str)
    mt_gene_mask = ~(
            np.char.startswith(
                gene_names, "MT-") | np.char.startswith(gene_names, "mt-")
    )
    mk_score = mk_score[mt_gene_mask, :]
    gene_names = gene_names[mt_gene_mask]

    logger.info(f"------Saving marker scores ...")

    # Save the marker scores
    mk_score_df = pd.DataFrame(mk_score, index=gene_names, columns=adata.obs_names)
    mk_score_df.reset_index(inplace=True)
    mk_score_df.rename(columns={"index": "HUMAN_GENE_SYM"}, inplace=True)

    target_path = config.mkscore_feather_path
    # Add # short random string to avoid overwriting
    rand_prefix = uuid.uuid4().hex[:8]
    tmp_path = target_path.with_name(f"{rand_prefix}_{target_path.name}")
    mk_score_df.to_feather(tmp_path)
    os.rename(tmp_path, target_path)
