import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import zarr
import scipy
from scipy.stats import rankdata
from pathlib import Path
import os
from tqdm import tqdm
from dataclasses import dataclass

#%% Helper functions

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

def calculate_one_slice_mean(file_path: Path, common_genes, zarr_group_path):
    """
    Calculate the geometric mean (using log trick) of gene expressions for a single slice and store it in a Zarr group.
    """
    file_name = file_path.name
    gmean_zarr_group = zarr.open(zarr_group_path, mode='a')
    sc_adatas = anndata.read_h5ad(file_path)
    sc_adatas = sc_adatas[:, common_genes].copy()

    n_cells = sc_adatas.shape[0]
    log_ranks = np.zeros((n_cells, sc_adatas.n_vars), dtype=np.float32)

    # Compute log of ranks to avoid overflow when computing geometric mean
    for i in tqdm(range(n_cells), desc=f"Computing log ranks for {file_name}"):
        data = sc_adatas.X[i, :].toarray().flatten()
        ranks = rankdata(data, method='average')
        log_ranks[i, :] = np.log(ranks + 1e-6)  # Adding small value to avoid log(0)

    # Calculate geometric mean via log trick: exp(mean(log(values)))
    gmean = np.exp(np.mean(log_ranks, axis=0))

    # Save to zarr group
    zarr_array_name = file_name.replace('.h5ad', '')
    s1_zarr = gmean_zarr_group.array(zarr_array_name, data=gmean, chunks=None, dtype='f4')
    s1_zarr.attrs['spot_number'] = sc_adatas.shape[0]
    print(f"Processed {file_name} finished.")

    # Save h5ad for gsMap
    sc_adatas.layers['log1p'] = sc_adatas.X
    sc_adatas.write(Path("gsMap_output") / f'{file_name}')

def merge_zarr_means(zarr_group_path, output_file, common_genes):
    """
    Merge all Zarr arrays into a weighted geometric mean and save to a Parquet file.
    """
    gmean_zarr_group = zarr.open(zarr_group_path, mode='a')
    final_mean = None
    total_spot_number = 0
    for key in tqdm(gmean_zarr_group.array_keys(), desc="Merging Zarr arrays"):
        s1 = gmean_zarr_group[key]
        s1_array = s1[:]
        n = s1.attrs['spot_number']
        if final_mean is None:
            final_mean = s1_array * n
        else:
            final_mean += s1_array * n
        total_spot_number += n
    final_mean /= total_spot_number

    # Save the final mean to a Parquet file
    gene_names = common_genes
    final_mean_df = pd.DataFrame({'gene': gene_names, 'G_Mean': final_mean})
    final_mean_df.set_index('gene', inplace=True)
    final_mean_df.to_parquet(output_file)
    print(f"Final mean saved to {output_file}")
    return final_mean_df

#%% Main function
@dataclass
class CreateSliceMeanConfig:
    h5ad_yaml: str
    slice_mean_output_file: str

def run_create_slice_mean(config: CreateSliceMeanConfig):
    """
    Main entrypoint to create slice means.
    """
    h5ad_dir = Path(config.h5ad_yaml)
    output_dir = Path(config.slice_mean_output_file)
    sample_name = h5ad_dir.stem  # Assuming sample name is derived from directory name

    # Step 1: Get common genes
    h5ad_files = list(h5ad_dir.rglob('*.h5ad'))
    common_genes = get_common_genes(h5ad_files)

    # Step 2: Initialize the Zarr group path and log folder
    zarr_group_path = output_dir / f'{sample_name}_gmean.zarr'
    log_folder = output_dir / 'logs'
    log_folder.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f'{sample_name}_final_mean.parquet'

    # Step 3: Process each file to calculate the slice means
    for file_path in tqdm(h5ad_files, desc="Processing h5ad files"):
        if file_path.name.replace('.h5ad', '') in list(zarr.open(zarr_group_path, mode='a').array_keys()):
            print(f"Skip {file_path.name}")
            continue
        calculate_one_slice_mean(file_path, common_genes, zarr_group_path)

    # Step 4: Merge results into a final weighted geometric mean and save it to a Parquet file
    final_mean_df = merge_zarr_means(zarr_group_path, output_file, common_genes)

    # Step 5: Plot the final mean distribution
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(final_mean_df['G_Mean'], bins=100)
    ax.set_title('Final Mean Distribution')
    plt.tight_layout()
    plt.show()

