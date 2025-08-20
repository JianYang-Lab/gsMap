import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy
from scipy.stats import gmean, rankdata
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm, trange

from gsMap.config import LatentToGeneConfig

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
    spatial_net = pd.DataFrame({"Cell1": cell1, "Cell2": cell2, "Distance": distance})
    return spatial_net


def build_spatial_net(adata, annotation, num_neighbour):
    """
    Build spatial neighbourhood matrix for each spot (cell) based on the spatial coordinates.
    """
    logger.info("------Building spatial graph based on spatial coordinates...")

    coor = adata.obsm["spatial"]
    if annotation is not None:
        logger.info("Cell annotations are provided...")
        spatial_net_list = []
        # Cells with annotations
        for ct in adata.obs[annotation].dropna().unique():
            idx = np.where(adata.obs[annotation] == ct)[0]
            coor_temp = coor[idx, :]
            spatial_net_temp = find_neighbors(coor_temp, min(num_neighbour, coor_temp.shape[0]))
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
            spatial_net_temp = spatial_net_temp[spatial_net_temp["Cell1"].isin(idx_nan)]
            spatial_net_list.append(spatial_net_temp)
        spatial_net = pd.concat(spatial_net_list, axis=0)
    else:
        logger.info("Cell annotations are not provided...")
        spatial_net = find_neighbors(coor, num_neighbour)

    return spatial_net.groupby("Cell1")["Cell2"].apply(np.array).to_dict()


def find_neighbors_regional(cell_pos, spatial_net_dict, coor_latent, config, cell_annotations):
    num_neighbour = config.num_neighbour
    annotations = config.annotation

    cell_use_pos = spatial_net_dict.get(cell_pos, [])
    if len(cell_use_pos) == 0:
        return []

    cell_latent = coor_latent[cell_pos, :].reshape(1, -1)
    neighbors_latent = coor_latent[cell_use_pos, :]
    similarity = cosine_similarity(cell_latent, neighbors_latent).reshape(-1)

    if annotations is not None:
        cell_annotation = cell_annotations[cell_pos]
        neighbor_annotations = cell_annotations[cell_use_pos]
        mask = neighbor_annotations == cell_annotation
        if not np.any(mask):
            return []
        similarity = similarity[mask]
        cell_use_pos = cell_use_pos[mask]

    if len(similarity) == 0:
        return []

    indices = np.argsort(-similarity)  # descending order
    top_indices = indices[:num_neighbour]
    cell_select_pos = cell_use_pos[top_indices]
    return cell_select_pos


def compute_regional_mkscore(
    cell_pos,
    spatial_net_dict,
    coor_latent,
    config,
    cell_annotations,
    ranks,
    frac_whole,
    adata_X_bool,
    pearson_residuals,
):
    """
    Compute gmean ranks of a region.
    """
    cell_select_pos = find_neighbors_regional(
        cell_pos, spatial_net_dict, coor_latent, config, cell_annotations
    )
    if len(cell_select_pos) == 0:
        return np.zeros(ranks.shape[1], dtype=np.float16)

    # Ratio of expression ranks
    ranks_tg = ranks[cell_select_pos, :]
    gene_ranks_region = gmean(ranks_tg, axis=0)
    gene_ranks_region[gene_ranks_region <= 1] = 0

    if not config.no_expression_fraction:
        # Ratio of expression fractions
        frac_focal = adata_X_bool[cell_select_pos, :].sum(axis=0).A1 / len(cell_select_pos)
        frac_region = frac_focal / frac_whole
        frac_region[frac_region <= 1] = 0
        frac_region[frac_region > 1] = 1

        # Simultaneously consider the ratio of expression fractions and ranks
        gene_ranks_region = gene_ranks_region * frac_region

    mkscore = np.exp(gene_ranks_region) - 1 if not pearson_residuals else gene_ranks_region

    return mkscore.astype(np.float16, copy=False)


def run_latent_to_gene(config: LatentToGeneConfig):
    logger.info("------Loading the spatial data...")
    adata = sc.read_h5ad(config.hdf5_with_latent_path)
    logger.info(f"Loaded spatial data with {adata.n_obs} cells and {adata.n_vars} genes.")

    if config.annotation is not None:
        logger.info(f"------Cell annotations are provided as {config.annotation}...")
        initial_cell_count = adata.n_obs
        adata = adata[~pd.isnull(adata.obs[config.annotation]), :]
        logger.info(
            f"Removed null annotations. Cells retained: {adata.n_obs} (initial: {initial_cell_count})."
        )

    # Homologs transformation
    if config.homolog_file is not None and config.species is not None:
        species_col_name = f"{config.species}_homolog"

        # Check if homolog conversion has already been performed
        if species_col_name in adata.var.columns:
            logger.warning(
                f"Column '{species_col_name}' already exists in adata.var. "
                f"It appears gene names have already been converted to human gene symbols. "
                f"Skipping homolog transformation."
            )
        else:
            logger.info(f"------Transforming the {config.species} to HUMAN_GENE_SYM...")
            homologs = pd.read_csv(config.homolog_file, sep="\t")
            if homologs.shape[1] != 2:
                raise ValueError(
                    "Homologs file must have two columns: one for the species and one for the human gene symbol."
                )

            homologs.columns = [config.species, "HUMAN_GENE_SYM"]
            homologs.set_index(config.species, inplace=True)

            # original_gene_names = adata.var_names.copy()

            # Filter genes present in homolog file
            adata = adata[:, adata.var_names.isin(homologs.index)]
            logger.info(f"{adata.shape[1]} genes retained after homolog transformation.")
            if adata.shape[1] < 100:
                raise ValueError("Too few genes retained in ST data (<100).")

            # Create mapping table of original to human gene names
            gene_mapping = pd.Series(
                homologs.loc[adata.var_names, "HUMAN_GENE_SYM"].values, index=adata.var_names
            )

            # Store original species gene names in var dataframe with the suffixed column name
            adata.var[species_col_name] = adata.var_names.values

            # Convert var_names to human gene symbols
            adata.var_names = gene_mapping.values
            adata.var.index.name = "HUMAN_GENE_SYM"

            # Remove duplicated genes after conversion
            adata = adata[:, ~adata.var_names.duplicated()]
            logger.info(f"{adata.shape[1]} genes retained after removing duplicates.")

    if config.annotation is not None:
        cell_annotations = adata.obs[config.annotation].values
        logger.info(f"Using cell annotations for {len(cell_annotations)} cells.")
    else:
        cell_annotations = None

    # Build the spatial graph
    logger.info("------Building the spatial graph...")
    spatial_net_dict = build_spatial_net(adata, config.annotation, config.num_neighbour_spatial)
    logger.info("Spatial graph built successfully.")

    # Extract the latent representation
    logger.info("------Extracting the latent representation...")
    coor_latent = adata.obsm[config.latent_representation]
    coor_latent = coor_latent.astype(np.float32)
    logger.info("Latent representation extracted.")

    # Geometric mean across slices
    gM = None
    frac_whole = None
    if config.gM_slices is not None:
        logger.info("Geometrical mean across multiple slices is provided.")
        gM_df = pd.read_parquet(config.gM_slices)
        if config.species is not None:
            homologs = pd.read_csv(config.homolog_file, sep="\t")
            if homologs.shape[1] < 2:
                raise ValueError(
                    "Homologs file must have at least two columns: one for the species and one for the human gene symbol."
                )
            homologs.columns = [config.species, "HUMAN_GENE_SYM"]
            homologs.set_index(config.species, inplace=True)
            gM_df = gM_df.loc[gM_df.index.isin(homologs.index)]
            gM_df.index = homologs.loc[gM_df.index, "HUMAN_GENE_SYM"].values
        common_genes = np.intersect1d(adata.var_names, gM_df.index)
        gM_df = gM_df.loc[common_genes]
        gM = gM_df["G_Mean"].values
        frac_whole = gM_df["frac"].values
        adata = adata[:, common_genes]
        logger.info(
            f"{len(common_genes)} common genes retained after loading the cross slice geometric mean."
        )

    # Compute ranks after taking common genes with gM_slices
    logger.info("------Ranking the spatial data...")
    if not scipy.sparse.issparse(adata.X):
        adata_X = scipy.sparse.csr_matrix(adata.X)
    elif isinstance(adata.X, scipy.sparse.csr_matrix):
        adata_X = adata.X  # Avoid copying if already CSR
    else:
        adata_X = adata.X.tocsr()

    # Create mappings
    n_cells = adata.n_obs
    n_genes = adata.n_vars
    pearson_residuals = True if "pearson_residuals" in adata.layers else False
    ranks = np.zeros((n_cells, adata.n_vars), dtype=np.float16)

    if pearson_residuals:
        logger.info("Using pearson residuals for ranking.")
        data = adata.layers["pearson_residuals"]
        for i in tqdm(range(n_cells), desc="Computing ranks per cell"):
            ranks[i, :] = rankdata(data[i, :], method="average")
    else:
        for i in tqdm(range(n_cells), desc="Computing ranks per cell"):
            data = adata_X[i, :].toarray().flatten()
            ranks[i, :] = rankdata(data, method="average")

    if gM is None:
        gM = gmean(ranks, axis=0)
        gM = gM.astype(np.float16)

    adata_X_bool = adata_X.astype(bool)
    if frac_whole is None:
        # Compute the fraction of each gene across cells
        frac_whole = np.asarray(adata_X_bool.sum(axis=0)).flatten() / n_cells
        logger.info("Gene expression proportion of each gene across cells computed.")
    else:
        logger.info(
            "Gene expression proportion of each gene across cells in all sections has been provided."
        )

    frac_whole += 1e-12  # Avoid division by zero
    # Normalize the ranks
    ranks /= gM

    def compute_mk_score_wrapper(cell_pos):
        return compute_regional_mkscore(
            cell_pos,
            spatial_net_dict,
            coor_latent,
            config,
            cell_annotations,
            ranks,
            frac_whole,
            adata_X_bool,
            pearson_residuals,
        )

    logger.info("------Computing marker scores...")
    mk_score = np.zeros((n_cells, n_genes), dtype=np.float16)
    for cell_pos in trange(n_cells, desc="Calculating marker scores"):
        mk_score[cell_pos, :] = compute_mk_score_wrapper(cell_pos)

    mk_score = mk_score.T
    logger.info("Marker scores computed.")

    # Remove mitochondrial genes
    gene_names = adata.var_names.values.astype(str)
    mt_gene_mask = ~(np.char.startswith(gene_names, "MT-") | np.char.startswith(gene_names, "mt-"))
    mk_score = mk_score[mt_gene_mask, :]
    gene_names = gene_names[mt_gene_mask]
    logger.info(f"Removed mitochondrial genes. Remaining genes: {len(gene_names)}.")

    # Save the marker scores
    logger.info("------Saving marker scores ...")
    output_file_path = Path(config.mkscore_feather_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
    mk_score_df = pd.DataFrame(mk_score, index=gene_names, columns=adata.obs_names)
    mk_score_df.reset_index(inplace=True)
    mk_score_df.rename(columns={"index": "HUMAN_GENE_SYM"}, inplace=True)
    mk_score_df.to_feather(output_file_path)
    logger.info(f"Marker scores saved to {output_file_path}.")

    # Save the modified adata object to disk
    adata.write(config.hdf5_with_latent_path)
    logger.info(f"Modified adata object saved to {config.hdf5_with_latent_path}.")
