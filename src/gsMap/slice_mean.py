import zarr
import numpy as np 
import scanpy as sc   
import logging
from scipy.stats import rankdata
from scipy.sparse import issparse
from tqdm import tqdm
import pandas as pd
import warnings

logger = logging.getLogger(__name__)


def calculate_one_slice_mean(st_name, adata, zarr_group_path):
    """
    Calculate the geometric mean (using log trick) of gene expressions for a single slice and store it in a Zarr group.
    """
    gmean_zarr_group = zarr.open(zarr_group_path, mode='a')
    n_cells = adata.shape[0]
    log_ranks = np.zeros((n_cells, adata.n_vars), dtype=np.float32)
    
    data = adata.layers['residuals'] if 'residuals' in adata.layers else adata.X
    if issparse(data):
        data = data.toarray()
    
    # Compute log of ranks to avoid overflow when computing geometric mean
    for i in range(n_cells):
        ranks = rankdata(data[i,:], method='average')
        log_ranks[i, :] = np.log(ranks)

    # Calculate geometric mean via log trick: exp(mean(log(values)))
    gmean = (np.exp(np.mean(log_ranks, axis=0))).reshape(-1, 1)

    # Calculate the expression fraction
    adata_X_bool = adata.X.astype(bool)
    frac = (np.asarray(adata_X_bool.sum(axis=0)).flatten()).reshape(-1, 1)
    
    # Save to zarr group
    gmean_frac = np.concatenate([gmean, frac], axis=1)
    s1_zarr = gmean_zarr_group.array(st_name, data=gmean_frac, chunks=None, dtype='f4')
    s1_zarr.attrs['spot_number'] = adata.shape[0]


def process_slice_mean(params, st_name, adata):
    
    # get the pearson_residuals
    if hasattr(params, 'pearson_residuals') and params.pearson_residuals:
        if params.data_layer in ["count", "counts"]:
            logger.info(f"Calculating pearson residuals for {st_name}...")
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            pearson_residuals = sc.experimental.pp.normalize_pearson_residuals(adata,inplace=False,clip=200)['X']
            pearson_residuals[np.isnan(pearson_residuals)] = -300
            adata.layers['residuals'] = pearson_residuals
            
    zarr_path = params.zarr_group_path
    zarr_path_str = zarr_path.as_posix()

    # Check if the Zarr group exists
    if zarr_path.exists():
        try:
            zarr_group = zarr.open(zarr_path_str, mode='r')
        except Exception as e:
            logger.error(f"Failed to open Zarr group at {zarr_path_str}: {e}")
            return
        # Check if the slice mean for this file already exists
        if st_name in zarr_group.array_keys():
            logger.info(f"Skipping {st_name}, already processed.")
            return
    
    # Calculate and store the slice mean
    calculate_one_slice_mean(st_name, adata, zarr_path)


def merge_zarr_means(zarr_group_path, w=None):
    """
    Merge all Zarr arrays into a weighted geometric mean and save to a Parquet file.
    Instead of calculating the mean, it sums the logs and applies the exponential.
    """
    gmean_zarr_group = zarr.open(zarr_group_path, mode='a')

    log_sum_mat, frac_mat, n_spot, section_name = [], [], [], []
    for key in tqdm(gmean_zarr_group.array_keys(), desc="Merging Zarr arrays"):
        s1 = gmean_zarr_group[key]
        s1_array_gmean = s1[:][:,0]
        s1_array_frac = s1[:][:,1]
        n = s1.attrs['spot_number']
        
        log_sum_mat.append(np.log(s1_array_gmean))
        frac_mat.append(s1_array_frac)
        n_spot.append(n)
        section_name.append(key)

    log_sum_mat = pd.DataFrame(log_sum_mat).T
    frac_mat = pd.DataFrame(frac_mat).T
    log_sum_mat.columns = gmean_zarr_group.array_keys()
    frac_mat.columns = gmean_zarr_group.array_keys()

    w_use = n_spot * w.loc[section_name].values if w is not None else n_spot
    w_use = w_use / np.sum(w_use)
    
    # Weighted average of log values
    weighted_log_mean = (log_sum_mat * w_use).sum(axis=1)
    gmean = np.exp(weighted_log_mean).values
    
    # Weighted average of fractions
    frac = (frac_mat * w_use).sum(axis=1).values
    
    return gmean, frac