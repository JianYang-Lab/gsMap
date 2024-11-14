from dataclasses import dataclass
from pathlib import Path

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import zarr
from scipy.stats import rankdata
from tqdm import tqdm
import scipy
from concurrent.futures.process import ProcessPoolExecutor
import os

#%%
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
    Calculate the mean rank of gene expressions for a single slice and store it in a Zarr group.
    """
    file_name = file_path.name
    gmean_zarr_group = zarr.open(zarr_group_path, mode='a')
    sc_adatas = anndata.read_h5ad(file_path)
    sc_adatas = sc_adatas[:, common_genes].copy()
    # Compute ranks and mean ranks
    # ranks = rankdata(sc_adatas.X.toarray().astype(np.float32), axis=1).astype(np.float32)
    n_cells = sc_adatas.shape[0]
    ranks = np.zeros((n_cells, sc_adatas.n_vars), dtype=np.float32)
    # adata_X = scipy.sparse.csr_matrix(adata.X)
    for i in tqdm(range(n_cells), desc="Computing ranks per cell"):
        data = sc_adatas.X[i, :].toarray().flatten()
        ranks[i, :] = rankdata(data, method='average')


    gM = np.mean(ranks, axis=0)

    # Save to zarr group
    zarr_array_name = file_name.replace('.h5ad', '')
    s1_zarr = gmean_zarr_group.array(zarr_array_name, data=gM, chunks=None, dtype='f4')
    s1_zarr.attrs['spot_number'] = sc_adatas.shape[0]
    print(f"Processed {file_name} finished.")

    # save h5ad for gsMap
    sc_adatas.layers['log1p'] = sc_adatas.X
    sc_adatas.write(H5AD_OUTPUT_PATH_FOR_GSMAP_INPUT/f'{file_name}')

def submit_jobs(h5ad_files, common_genes, zarr_group_path, log_folder):
    """
    Submit jobs to process each h5ad file and calculate slice means.
    """
    executor = submitit.AutoExecutor(folder=log_folder)
    # executor = submitit.DebugExecutor(folder=log_folder)
    executor.update_parameters(slurm_partition='intel-sc3,amd-ep2,amd-ep2-short',
                               slurm_cpus_per_task=1,
                               slurm_mem='40G',
                               slurm_qos='huge',
                               slurm_time='1-00:00:00',
                               slurm_exclude='amdepyc06,intel042',

                               )
    # executor = ProcessPoolExecutor(max_workers=5)
    jobs = []
    gmean_zarr_group = zarr.open(zarr_group_path, mode='a')
    for file_path in h5ad_files:
        if file_path.name.replace('.h5ad', '') in list(gmean_zarr_group.array_keys()):
            print(f"Skip {file_path.name}")
            continue
        executor.update_parameters(slurm_job_name=file_path.name)
        job = executor.submit(calculate_one_slice_mean, file_path, common_genes, zarr_group_path)
        jobs.append(job)

    for job in jobs:
        job.result()
    print("Job submission complete.")


def merge_zarr_means(zarr_group_path, output_file, common_genes):
    """
    Merge all Zarr arrays into a weighted mean and save to a Parquet file.
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

def create_slice_mean_workflow(h5ad_dir, sample_name, output_dir,common_genes):
    """
    Main function to wrap the entire workflow.
    """
    # Define paths
    h5ad_files = list(Path(h5ad_dir).rglob(f'*.h5ad'))
    zarr_group_path = Path(output_dir) / f'{sample_name}_gmean.zarr'

    log_folder = Path(output_dir) / 'submitit_log'
    log_folder.mkdir(parents=True, exist_ok=True)

    output_file = Path(output_dir) / f'{sample_name}_final_mean.parquet'

    # Step 1: Get common genes
    # common_genes = get_common_genes(h5ad_files)

    # Step 2: Submit jobs to calculate slice mean for each file
    submit_jobs(h5ad_files, common_genes, zarr_group_path, log_folder)

    # Step 3: Merge the results to get the weighted mean
    final_mean_df = merge_zarr_means(zarr_group_path, output_file, common_genes)

    # plot the final mean
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.hist(final_mean_df, bins=100)
    plt.title('Final Mean Distribution')
    plt.tight_layout()
    plt.show()

@dataclass
class CreateSliceMeanConfig:
    h5ad_dir: str
    sample_name: str
    output_dir: str
    common_genes: pd

def run_create_slice_mean(config:CreateSliceMeanConfig):



# %% ======================================================================================================================
#  3. Run the workflow for  Marmoset1
# %% ======================================================================================================================
#%% get protein coding genes
protein_coding_genes_path='/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/resource/gsMap_resource/genome_annotation/gtf/gencode.v46lift37.basic.annotation.protein_coding.gene.csv'
protein_coding_genes = pd.read_csv(protein_coding_genes_path)
# adata = sc.read_h5ad('/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/ST_data_Collection/03_Stero-seq/02_hongtaoyu_human_embryo_brain/01_raw_h5ad/NB20241105101705317/4_C03627G1_WT202403110489.h5ad')
adata = sc.read_h5ad('/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/ST_data_Collection/03_Stero-seq/02_hongtaoyu_human_embryo_brain/01_raw_h5ad/NB20241105101705317/115_B03609F5G6_WT202403310081.h5ad')
common_genes = adata.var_names.intersection(protein_coding_genes['gene_name'])
#%%

h5ad_dir = '/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/ST_data_Collection/03_Stero-seq/02_hongtaoyu_human_embryo_brain/01_raw_h5ad/NB20241105101705317'
sample_name = 'fetal_brain_bin100'
output_dir = '/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/ST_data_Collection/03_Stero-seq/02_hongtaoyu_human_embryo_brain/02_gsmap_result/fetal_brain_bin100_slice_mean'
create_slice_mean_workflow(h5ad_dir, sample_name, output_dir,common_genes)
#%%
final_mean_df=pd.read_parquet('/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/ST_data_Collection/03_Stero-seq/02_hongtaoyu_human_embryo_brain/02_gsmap_result/fetal_brain_bin100_slice_mean/fetal_brain_bin100_final_mean.parquet')
# # %% ======