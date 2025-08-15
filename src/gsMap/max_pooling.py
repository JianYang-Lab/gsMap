
import logging
import scanpy as sc 
import pandas as pd
import numpy as np
import os
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from sklearn.neighbors import NearestNeighbors
from gsMap.config import MaxPoolingConfig

logger = logging.getLogger(__name__)


# Define the MNN and pooling functions
def build_soft_mnn_scores(emb_a, 
                          emb_b, 
                          names_a, 
                          names_b, 
                          coords_a=None, 
                          coords_b=None,
                          k=5, 
                          sim_thresh = 0.95,
                          spatial_topk=100):
    
    sim = cosine_similarity(emb_a, emb_b)
    topk_ab = np.argsort(sim, axis=1)[:, -k:]
    topk_ba = np.argsort(sim.T, axis=1)[:, -k:]
    spatial_topk = min(spatial_topk, int(len(names_b)*0.5))
    
    if coords_a is not None and coords_b is not None:
        nn_spatial = NearestNeighbors(n_neighbors=spatial_topk, metric='euclidean').fit(coords_b)
        _, spatial_topk_idx = nn_spatial.kneighbors(coords_a)
    else:
        spatial_topk_idx = None
    
    soft_scores = {}
    for i, a_name in enumerate(names_a):
        scores = {}
        for j in topk_ab[i]:
            b_name = names_b[j]
            
            # Check if b is among top spatial_topk nearest neighbors of a
            if spatial_topk_idx is not None and j not in spatial_topk_idx[i]:
                continue
            
            # mutual nearest neighbor score
            mnn_score = 0.5 * sim[i, j]
            if i in topk_ba[j]: 
                mnn_score += 0.5 * sim[i, j]   
            if mnn_score < sim_thresh:
                continue
        
            scores[b_name] = mnn_score    
        if scores:
            soft_scores[a_name] = scores

    return soft_scores 


def compute_pooled_vector(gss_b, b_names, weights, strategy='weighted_mean'):
    if strategy == 'weighted_mean':
        weights /= np.sum(weights)
        return (gss_b[b_names] * weights).sum(axis=1)
    elif strategy == 'softmax':
        return (gss_b[b_names] * softmax(weights)).sum(axis=1)
    else: 
        return gss_b[b_names].median(axis=1)

    
def pool_soft_mnn_scores(gss_a, gss_b, soft_scores_dict, section_name=None):
    
    for a_name, neighbor_scores in tqdm(soft_scores_dict.items(), desc=f"Pooling soft MNN: {section_name}"):
        b_names = list(neighbor_scores.keys())
        weights = np.array([neighbor_scores[b] for b in b_names])
        
        gss_weighted = compute_pooled_vector(gss_b, b_names, weights, strategy='weighted_mean')
        gss_a[a_name] = np.maximum(gss_a[a_name].values,gss_weighted)
            

def run_max_pooling(config: MaxPoolingConfig):
    
    # Load the adata
    spe_file_list = pd.read_csv(config.spe_file_list,header=None).values[:,0].tolist()
    spe_file_list = np.array(
        [f'{st.split('/')[-1].split('.h5ad')[0]}_add_latent.h5ad' for st in spe_file_list]
    )

    # Set the window size for pooling
    focal_st_name = f'{config.sample_name}_add_latent.h5ad'
    
    st_idx = np.where(np.array(focal_st_name)==spe_file_list)[0][0]
    window_size = min(max(round(len(spe_file_list)*0.1),2),10)
    spe_file_list_focal = spe_file_list[
        np.arange(max(st_idx - window_size, 0),
                min(st_idx + window_size + 1, len(spe_file_list)))
    ]

    # Get the spatial coordinates and embeddings
    embs, annotations, cell_names = {},{},{}
    for st_name in tqdm(spe_file_list_focal, total=len(spe_file_list_focal), desc='Loading the data'):
        adata = sc.read_h5ad(config.hdf5_with_latent_path.parent / st_name, backed='r')
        name = st_name.split('_add_latent')[0]
        
        embs[name] = adata.obsm['emb_gcn']
        # spatials[name] = adata.obsm[config.spatial_key]
        cell_names[name] = adata.obs_names
        annotations[name] = adata.obs[config.annotation] if config.annotation is not None else None


    # Build the MNN
    focal = config.sample_name
    others = [name for name in embs if name != focal]
    cell_nn_sections = {}
    annotation_cats = annotations[focal].unique() if config.annotation is not None else None

    for ref in tqdm(others, total=len(others), desc=f'Building soft MNN: {focal}'):
        cell_nn = {}
        annotation_values = (annotation_cats if config.annotation is not None else [None])

        for current_annotation in annotation_values:
            if current_annotation is not None:
                # Apply masks for this annotation category
                mask_focal = (annotations[focal] == current_annotation).values
                mask_ref = (annotations[ref] == current_annotation).values

                # Skip if no cells in this category
                if np.sum(mask_focal) == 0 or np.sum(mask_ref) == 0:
                    continue

                emb_focal = embs[focal][mask_focal]
                emb_ref = embs[ref][mask_ref]
                names_focal = cell_names[focal][mask_focal].values
                names_ref = cell_names[ref][mask_ref].values

            else:
                # Use all data if no annotation
                emb_focal = embs[focal]
                emb_ref = embs[ref]
                names_focal = cell_names[focal].values
                names_ref = cell_names[ref].values

            # Compute soft MNN scores
            soft_scores = build_soft_mnn_scores(
                emb_a=emb_focal,
                emb_b=emb_ref,
                names_a=names_focal,
                names_b=names_ref,
                coords_a=None,
                coords_b=None,
                k=5,
                sim_thresh=config.sim_thresh,
                spatial_topk=100,
            )

            # Accumulate results
            cell_nn.update(soft_scores)

        # Save results if any MNNs were found
        if cell_nn:
            cell_nn_sections[ref] = cell_nn

    for key, value in cell_nn_sections.items():
        logger.info(f"{len(value)} cells in {focal} can found MNN from {key}")
        
        
    # Max pooling the MNN
    gss_focal = pd.read_feather(config.mkscore_feather_path).set_index('HUMAN_GENE_SYM')
    for ref_name in cell_nn_sections.keys():
        
        gss_ref = pd.read_feather(config.mkscore_feather_path.parent / f'{ref_name}_gene_marker_score.feather').set_index('HUMAN_GENE_SYM')
        pool_soft_mnn_scores(
            gss_a=gss_focal, 
            gss_b=gss_ref, 
            soft_scores_dict=cell_nn_sections[ref_name], 
            section_name=ref_name
        )
    
    gss_focal[gss_focal <= (np.exp(1) - 1)] = 0
    gss_focal = gss_focal.copy()
    gss_focal.reset_index(inplace=True)
    gss_focal.rename(columns={"index": "HUMAN_GENE_SYM"}, inplace=True)
    
    # Save pooled mk score
    target_path = config.tuned_mkscore_feather_path
    rand_prefix = uuid.uuid4().hex[:8]
    tmp_path = target_path.with_name(f"{rand_prefix}_{target_path.name}")
    gss_focal.to_feather(tmp_path)
    os.rename(tmp_path, target_path)
    
        
