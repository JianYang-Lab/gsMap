import numpy as np
import logging
import pickle
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)

def top_cells(similarity_row, cell_names, top_n=50):
    """
    Returns the top N most similar cells based on similarity row.
    
    Args:
        similarity_row: Array of similarity scores
        cell_names: List of cell names corresponding to the similarity scores
        top_n: Number of top cells to return
        
    Returns:
        Dictionary mapping cell names to similarity scores for top N cells
    """
    non_zero_indices = np.where(similarity_row > 0)[0]
    if len(non_zero_indices) == 0:
        return {}
    
    scores = similarity_row[non_zero_indices]
    names = [cell_names[i] for i in non_zero_indices]
    sorted_indices = np.argsort(-scores)
    top_indices = sorted_indices[:min(top_n, len(sorted_indices))]
    return {names[i]: float(scores[i]) for i in top_indices}



def build_across_section_net(
    emb_dict, 
    cell_name, 
    cell_net_path,
    batch_size = 1000,
    threshold = 0.95,
    cell_size = 10000,
    ):
    
    cell_net_across_section = dict()
    for key_v in emb_dict.keys():
        emb_focal = emb_dict[key_v]
        cell_name_focal = cell_name[key_v]

        # using mini-batch to avoid memory error
        batch_size_v = min(batch_size,len(emb_focal))
        for i in tqdm(range(0, len(emb_focal), batch_size_v),desc=f'Building cell-net across sections for {key_v}'):
            batch_focal = emb_focal[i:i + batch_size_v]
            batch_names = cell_name_focal[i:i + batch_size_v]
            
            similar_mat,cell_name_others = [],[]

            for key_q in emb_dict.keys():
                if key_v == key_q:
                    continue
                else:
                    emb_q = emb_dict[key_q]
                    cell_q = cell_name[key_q]
                    
                    # batch_size_q = min(batch_size,len(emb_q))
                    # for j in range(0, len(emb_q), batch_size_q):
                    #     emb_q_focal = emb_q[j:j + batch_size_q]
                    #     cell_q_focal = cell_q[j:j + batch_size_q]
                
                    # downsample the qurey section
                    cell_size = min(cell_size,emb_q.shape[0])
                    sample_idx = np.random.choice(np.arange(emb_q.shape[0]),cell_size,replace=False)
                    cell_name_q = cell_q[sample_idx].to_list()
            
                    sim = cosine_similarity(batch_focal, emb_q[sample_idx,:])
                    sim[sim < threshold] = 0
                    sim_sparse = sparse.csr_matrix(sim)
                    cell_name_others.extend(cell_name_q)
                    similar_mat.append(sim_sparse)
                
            similar_mat = sparse.hstack(similar_mat)
            for idx in range(len(batch_names)):
                row = similar_mat[idx].toarray().flatten()
                cell_net_across_section[batch_names[idx]] = top_cells(row, cell_name_others)
    
    # save the cell-net
    with open(cell_net_path, 'wb') as f:
        pickle.dump({'cell_net_across_section': cell_net_across_section}, f)
