import logging
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import zarr
from scipy.stats import rankdata
from tqdm import tqdm

from gsMap.config import CreateSliceMeanConfig

#%% Helper functions
logger = logging.getLogger(__name__)

def get_common_genes(h5ad_files):
    """
    Get common genes from a list of h5ad files.
    """
    common_genes = None
    for file in tqdm(h5ad_files, desc="Finding common genes"):
        sc_adatas = sc.read(file)
        if common_genes is None:
            common_genes = sc_adatas.var_names
        else:
            common_genes = common_genes.intersection(sc_adatas.var_names)
    # sort
    common_genes = sorted(list(common_genes))
    return common_genes

def calculate_one_slice_mean(sample_name, file_path: Path, common_genes, zarr_group_path):
    """
    Calculate the geometric mean (using log trick) of gene expressions for a single slice and store it in a Zarr group.
    """
    # file_name = file_path.name
    gmean_zarr_group = zarr.open(zarr_group_path, mode='a')
    sc_adatas = anndata.read_h5ad(file_path)
    sc_adatas = sc_adatas[:, common_genes].copy()

    n_cells = sc_adatas.shape[0]
    log_ranks = np.zeros((n_cells, sc_adatas.n_vars), dtype=np.float32)

    # Compute log of ranks to avoid overflow when computing geometric mean
    for i in tqdm(range(n_cells), desc=f"Computing log ranks for {sample_name}"):
        data = sc_adatas.X[i, :].toarray().flatten()
        ranks = rankdata(data, method='average')
        log_ranks[i, :] = np.log(ranks + 1e-6)  # Adding small value to avoid log(0)

    # Calculate geometric mean via log trick: exp(mean(log(values)))
    gmean = np.exp(np.mean(log_ranks, axis=0))

    # Save to zarr group
    s1_zarr = gmean_zarr_group.array(sample_name, data=gmean, chunks=None, dtype='f4')
    s1_zarr.attrs['spot_number'] = sc_adatas.shape[0]

def merge_zarr_means(zarr_group_path, output_file, common_genes):
    """
    Merge all Zarr arrays into a weighted geometric mean and save to a Parquet file.
    Instead of calculating the mean, it sums the logs and applies the exponential.
    """
    gmean_zarr_group = zarr.open(zarr_group_path, mode='a')
    log_sum = None
    total_spot_number = 0
    for key in tqdm(gmean_zarr_group.array_keys(), desc="Merging Zarr arrays"):
        s1 = gmean_zarr_group[key]
        s1_array = s1[:]
        n = s1.attrs['spot_number']
        if log_sum is None:
            log_sum = np.log(s1_array + 1e-6) * n  # Summing logs of values, avoid log(0)
        else:
            log_sum += np.log(s1_array + 1e-6) * n  # Summing logs of values, avoid log(0)
        total_spot_number += n

    # Apply the geometric mean via exponentiation of the averaged logs
    final_mean = np.exp(log_sum / total_spot_number)

    # Save the final mean to a Parquet file
    gene_names = common_genes
    final_mean_df = pd.DataFrame({'gene': gene_names, 'G_Mean': final_mean})
    final_mean_df.set_index('gene', inplace=True)
    final_mean_df.to_parquet(output_file)
    print(f"Final mean saved to {output_file}")
    return final_mean_df


def run_create_slice_mean(config: CreateSliceMeanConfig):
    """
    Main entrypoint to create slice means.
    Now works with a config that can accept either:
    1. An h5ad_yaml file.
    2. Direct lists of sample names and h5ad files.
    """
    h5ad_files = list(config.h5ad_dict.values())
    output_dir = config.slice_mean_output_file

    # Step 2: Get common genes from the h5ad files
    common_genes = get_common_genes(h5ad_files)

    # Step 3: Initialize the Zarr group
    zarr_group_path = config.slice_mean_output_file.with_suffix('.zarr')

    for sample_name, h5ad_file in config.h5ad_dict.items():
        # Step 4: Process each file to calculate the slice means
        if zarr_group_path.exists():
            zarr_group = zarr.open(zarr_group_path.as_posix(), mode='r')
            # Check if the slice mean for this file already exists
            if sample_name in zarr_group.array_keys():
                logger.info(f"Skipping {sample_name}, already processed.")
                continue

        calculate_one_slice_mean(sample_name, h5ad_file, common_genes, zarr_group_path)


    output_file = config.slice_mean_output_file
    final_mean_df = merge_zarr_means(zarr_group_path, output_file, common_genes)
    logger.info(f"Final slice mean saved to {output_file}")
