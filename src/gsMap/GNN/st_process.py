import torch
import scanpy as sc
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import re 
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from scipy.special import softmax
from gsMap.GNN.GCN import GCN, build_spatial_graph

logger = logging.getLogger(__name__)


def find_common_hvg(spe_file_list, params):
    """
    Identifies common highly variable genes (HVGs) across multiple ST datasets and calculates
    the number of cells to sample from each dataset.

    Args:
        spe_file_list (list): List of file paths to ST datasets.
        params (object): Parameter object containing attributes.
    """

    variances_list = []
    cell_number = []
    annotation_list = []

    logger.info("Finding highly variable genes (HVGs)...")
    for st_file in tqdm(spe_file_list, desc="Finding common genes"):
        adata_temp = sc.read_h5ad(st_file)
        
        # Filter out mitochondrial and hemoglobin genes
        gene_keep = ~adata_temp.var_names.str.match(re.compile(r'^(HB.-|MT-)', re.IGNORECASE))
        adata_temp = adata_temp[:,gene_keep].copy()
        
        # Set data layer
        if params.data_layer not in adata_temp.layers:
            if adata_temp.X is not None and np.issubdtype(
                adata_temp.X.dtype,
                np.integer
            ):
                logger.info(
                    f'Data layer {params.data_layer} not found or not integer'
                    f', falling back to adata.X'
                )
                adata_temp.layers[params.data_layer] = adata_temp.X.copy()
                params.data_layer = 'count'
            else:
                params.data_layer = None
        else:
            adata_temp.X = adata_temp.layers[params.data_layer]
        
        # Identify highly variable genes
        flavor = "seurat_v3" if params.data_layer in ["count", "counts", "impute_count"] else "seurat"
        try:
            sc.pp.highly_variable_genes(
                adata_temp, n_top_genes=params.feat_cell, subset=False, flavor=flavor
            )
            var_df = adata_temp.var
            var_df["gene"] = var_df.index.tolist()
            variances_list.append(var_df)
        except:
            logger.warning(f"Failed to find HVGs for {st_file}.")
        
        cell_number.append(adata_temp.n_obs)
        # Store the annotation
        if params.annotation is not None:
            annotation_list = (
                annotation_list + adata_temp.obs[params.annotation].to_list()
            )

    # Find the common genes across all datasets
    common_genes = np.array(
        list(set.intersection(
            *map(set, [st.index.to_list() for st in variances_list])))
    )

    # Aggregate variances and identify HVGs
    df = pd.concat(variances_list, axis=0)
    df["highly_variable"] = df["highly_variable"].astype(int)
    if flavor=='seurat_v3':
        df = df.groupby("gene", observed=True).agg(
            dict(
                variances_norm="median",
                highly_variable="sum",
            )
        )
        df = df.loc[common_genes]
        df["highly_variable_nbatches"] = df["highly_variable"]
        df.sort_values(
            ["highly_variable_nbatches", "variances_norm"],
            ascending=False,
            na_position="last",
            inplace=True,
        )
    else:
        df = df.groupby("gene", observed=True).agg(
            dict(
                dispersions_norm="median",
                highly_variable="sum",
            )
        )
        df = df.loc[common_genes]
        df["highly_variable_nbatches"] = df["highly_variable"]
        df.sort_values(
            ["highly_variable_nbatches", "dispersions_norm"],
            ascending=False,
            na_position="last",
            inplace=True,
        )
    
    hvg = df.iloc[: params.feat_cell,].index.tolist()
    
    # Find the number of sampling cells for each batch
    total_cell = np.sum(cell_number)
    total_cell_training = np.minimum(total_cell, params.n_cell_training)
    cell_proportion = cell_number / total_cell
    n_cell_used = [
        int(cell) for cell in (total_cell_training * cell_proportion).tolist()
    ]

    # Find the percentatges of each annotation
    if params.annotation is not None:
        percent_annotation = (
            pd.DataFrame(
                np.unique(annotation_list, return_counts=True),
                index=["annotation", "cell_num"],
            )
            .T.assign(cell_num=lambda x: x.cell_num / x.cell_num.sum())
            .set_index("annotation")
        )
    else:
        percent_annotation = None

    # Only use the common genes that can be transformed to human genes
    if params.species is not None:
        homologs = pd.read_csv(params.homolog_file, sep='\t')
        if homologs.shape[1] < 2:
            raise ValueError("Homologs file must have at least two columns: one for the species and one for the human gene symbol.")
        homologs.columns = [params.species, 'HUMAN_GENE_SYM']
        homologs.set_index(params.species, inplace=True)
        hvg = [gene for gene in hvg if gene in homologs.index]
        logger.info(f"Retained {len(hvg)} HVGs after homolog filtering.")

    return hvg, n_cell_used, percent_annotation


class TrainingData(object):
    """
    Managing and processing training data for graph-based models.

    Attributes:
        params (dict): A dictionary of parameters used for data processing and training.
    """

    def __init__(self, params):
        self.params = params
        self.gcov = GCN(self.params.K)
        self.expression_merge = []
        self.expression_gcn_merge = []
        self.label_merge = []
        self.batch_merge = []
        self.batch_size = None
        self.batches_onehot = None
             
 
    def prepare(self, spe_file_list, n_cell_used, hvg, percent_annotation):
        for st_id, st_file in enumerate(spe_file_list):
            
            # Load the data
            logger.info(f"Loading ST data of {st_file}...")
            adata = sc.read_h5ad(st_file)
            st_name = (Path(st_file).name).split(".h5ad")[0]
             
            # Set data layers
            if not hasattr(adata, 'layers') or self.params.data_layer not in adata.layers:
                if adata.X is not None and np.issubdtype(adata.X.dtype, np.integer):
                    logger.info(
                        f'Data layer {self.params.data_layer} not found in layers or layers missing, '
                        f'falling back to adata.X'
                    )
                    adata.X = adata.X  # Use adata.X directly
            else:
                adata.X = adata.layers[self.params.data_layer]
                
            # Filter cells based on annotation if provided
            if self.params.annotation is not None:
                adata = adata[~adata.obs[self.params.annotation].isnull()]
                label = adata.obs[self.params.annotation].values
            else:
                label = np.zeros(adata.n_obs)
            
            # Get expression array and apply GCN
            expression_array = torch.Tensor(adata[:, hvg].X.toarray())
            edge, _ = build_spatial_graph(
                adata,
                self.params.n_neighbors,
                self.params.spatial_key,
            )
            expression_array_gcn = self.gcov(expression_array, edge)
            logger.info(
                f"Graph for {st_name} has {edge.size(1)} edges, {adata.n_obs} cells."
            )

            # Downsampling
            if self.params.do_sampling:
                if self.params.annotation is None:
                    num_cell = min(adata.n_obs, n_cell_used[st_id])
                    logger.info(
                        f"Downsampling {st_name} to {num_cell} cells...")
                    random_indices = np.random.choice(
                        adata.n_obs, num_cell, replace=False
                    )
                else:
                    num_cell = (
                        (percent_annotation * n_cell_used[st_id])
                        .astype(int)
                        .loc[adata.obs[self.params.annotation].unique()]
                    )
                    logger.info(
                        f"Downsampling {st_name} to {n_cell_used[st_id]} cells..."
                    )
                    logger.info("---Including-----")
                    logger.info(
                        num_cell["cell_num"].sort_values(
                            ascending=False).to_dict()
                    )
                    sampled_cells = (
                        adata.obs.groupby(
                            self.params.annotation, group_keys=False)
                        .apply(
                            lambda x: x.sample(
                                max(min(num_cell.loc[x.name, "cell_num"], len(x)),1),
                                replace=False,
                            )
                        )
                        .index
                    )
                    random_indices = [
                        adata.obs.index.get_loc(idx) for idx in sampled_cells
                    ]

                expression_array = expression_array[random_indices]
                expression_array_gcn = expression_array_gcn[random_indices]
                label = label[random_indices]

            # Batch identifiers
            batch = [f"S{st_id}"] * expression_array.size(0)

            # Update attributes
            self.expression_merge.append(expression_array)
            self.expression_gcn_merge.append(expression_array_gcn)
            self.label_merge.append(label)
            self.batch_merge.append(batch)

        # Concatenate data
        self.expression_merge = torch.cat(self.expression_merge, dim=0)
        self.expression_gcn_merge = torch.cat(self.expression_gcn_merge, dim=0)
        self.batch_merge = torch.Tensor(
            pd.Categorical(np.concatenate(self.batch_merge)).codes
        )
        cat_labels = pd.Categorical(np.concatenate(self.label_merge))
        self.label_merge = torch.Tensor(cat_labels.codes).long()
        
        if self.params.annotation is not None:
            self.label_name = cat_labels.categories.take(np.unique(cat_labels.codes)).to_list()
        else:
            self.label_name = None
            
        # One-hot encode batches
        self.batch_size = len(torch.unique(self.batch_merge))


# Inference for each ST dataset
class InferenceData(object):
    """
    Infer cell embeddings for each spatial transcriptomics (ST) dataset.
    Attributes:
        hvg: List of highly variable genes.
        batch_size: Integer defining the batch size for inference.
        model: Model to be used for inference.
        params: Dictionary containing additional parameters for inference.
    """

    def __init__(self, hvg, batch_size, model, label_name, params):
        self.params = params
        self.gcov = GCN(self.params.K)
        self.hvg = hvg
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.label_name = label_name
        self.processed_list_path = self.params.latent_dir / 'processed.list'
    

    def infer_embedding_single(self, st_id, st_file) -> Path:
        st_name = (Path(st_file).name).split(".h5ad")[0]
        logger.info(f"Infering cell embeddings for {st_name}...")
        
        # Load data
        adata = sc.read_h5ad(st_file)
        
        # Set data layers
        if not hasattr(adata, 'layers') or self.params.data_layer not in adata.layers:
            if adata.X is not None and np.issubdtype(adata.X.dtype, np.integer):
                logger.info(
                    f'Data layer {self.params.data_layer} not found in layers or layers missing, '
                    f'falling back to adata.X'
                )
                adata.X = adata.X  # Use adata.X directly
        else:
            adata.X = adata.layers[self.params.data_layer]
        
        # Get expression array and apply GCN
        if self.params.species is not None:
            homologs = pd.read_csv(self.params.homolog_file, sep='\t')
            if homologs.shape[1] < 2:
                raise ValueError("Homologs file must have at least two columns.")
            homologs.columns = [self.params.species, 'HUMAN_GENE_SYM']
            homologs.set_index(self.params.species, inplace=True)
            hvg_filtered = [gene for gene in self.hvg if gene in adata.var_names and gene in homologs.index]
        else:
            hvg_filtered = [gene for gene in self.hvg if gene in adata.var_names]
        
        expression_array = torch.Tensor(adata[:, hvg_filtered].X.toarray())
        edge, _ = build_spatial_graph(
            adata,
            self.params.n_neighbors,
            self.params.spatial_key,
        )
        expression_array_gcn = self.gcov(expression_array, edge)
        
        # Create batch tensor
        batch_tensor = torch.full((adata.n_obs,), st_id).long()
        
        # Create dataset and dataloader
        dataset = TensorDataset(expression_array, expression_array_gcn, batch_tensor)
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Inference
        self.model.eval()
        latent_list = []
        latent_indv_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                expr, expr_gcn, batch_ids = batch
                expr = expr.to(self.device)
                expr_gcn = expr_gcn.to(self.device)
                batch_ids = batch_ids.to(self.device)
                
                # Get embeddings from model
                z, z_indv = self.model.encode([expr, expr_gcn], batch_ids)
                latent_list.append(z.cpu().numpy())
                latent_indv_list.append(z_indv.cpu().numpy())
        
        # Concatenate results
        latent = np.vstack(latent_list)
        latent_indv = np.vstack(latent_indv_list)
        
        # Save embeddings to adata
        adata.obsm['latent_GVAE'] = latent
        adata.obsm['latent_GVAE_indv'] = latent_indv
        
        # Save the updated adata
        output_path = self.params.latent_dir / f"{st_name}_with_latent.h5ad"
        adata.write(output_path)
        
        logger.info(f"Saved embeddings for {st_name} to {output_path}")
        return output_path