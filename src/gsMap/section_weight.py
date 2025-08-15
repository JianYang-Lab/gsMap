import numpy as np
import logging
import pickle
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

def sigmoid_transform(x, alpha=10,cut_off=0.5):
    return 1 / (1 + np.exp(-alpha * (x - cut_off)))


def compute_weight(emb_dict,depth_dict,weight_path,cell_size=None):
    
    section_weight = {}    
    for key_v in tqdm(emb_dict.keys(),desc='Computing section weights'):
        weights = []
        for key_q in emb_dict.keys():
            if key_v == key_q:
                weight = 1.0
            else:
                emb_v = emb_dict[key_v]
                emb_q = emb_dict[key_q]
                
                cell_size = min(emb_q.shape[0],cell_size) if cell_size is not None else emb_q.shape[0]
                emb_q = emb_q[np.random.choice(np.arange(emb_q.shape[0]),cell_size,replace=False),:]
                
                d_v = np.median(depth_dict[key_v])
                d_q = np.median(depth_dict[key_q])
                
                # Compute embedding dissimilarity and depth similarity
                emb_dissimilar_q = 1 - np.quantile(np.max(cosine_similarity(emb_v, emb_q), axis=1), 0.1)
                d_similar_q = min(d_v / d_q, d_q / d_v)
    
                weight = sigmoid_transform(emb_dissimilar_q) * sigmoid_transform(d_similar_q)
            
            weights.append(weight)
        
        # Normalize weights to ensure none exceed 1
        weights = np.clip(weights, 0, 1)
        section_weight[key_v] = weights

    with open(weight_path, 'wb') as f:
        pickle.dump({'section_weight': section_weight}, f)
