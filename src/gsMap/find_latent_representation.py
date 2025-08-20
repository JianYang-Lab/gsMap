import os
import torch
import numpy as np
import logging
import random
import gc
import zarr
import pandas as pd
from pathlib import Path
from torch.utils.data import (
    DataLoader,
    random_split,
    TensorDataset,
    SubsetRandomSampler,
)
from scipy.stats import rankdata
from scipy.sparse import csr_matrix
from zarr.storage import DirectoryStore, LRUStoreCache
from gsMap.GNN.TrainStep import ModelTrain
from gsMap.GNN.STmodel import StEmbeding
from gsMap.ST_process import TrainingData, find_common_hvg, InferenceData
from gsMap.config import FindLatentRepresentationsConfig
from gsMap.slice_mean import process_slice_mean

from operator import itemgetter

logger = logging.getLogger(__name__)


def csc_col_sum(mat: csr_matrix, fill_zero=False):
    if fill_zero:
        ncols = mat.shape[1]
        zero_ranks = np.log((ncols - mat.getnnz(axis=1) + 1) / 2)
        mat = mat.tocsc()
        nz_sum = mat.sum(axis=0).A1

        not_to_fill = mat != 0
        zero_sum = zero_ranks.sum() - (
            zero_ranks * not_to_fill
        )  # NOTE: `not_to_fill` is a matrix, * here is matrix multiplication

        return nz_sum + zero_sum

    return mat.tocsc().sum(0).A1


def rank_data_nz(arr, glob_positions, n_genes):
    # init res
    res = np.ones((arr.shape[0], n_genes), dtype=np.float32)
    for i in range(arr.shape[0]):
        cols = glob_positions[arr[i].indices]
        res[i, cols] = rankdata(arr[i].data) + n_genes - arr[i].nnz
    return res


class ZarrBackedCSR:
    def __init__(
        self,
        path: str,
        ncols: int,
        mode: str = "r",
        chunks_indptr: int = 1000,
        chunks_data: int = 10000,
        lru_cache_size: int = 1,
        **kwargs,
    ):
        self.path = path
        self.ncols = ncols
        store = DirectoryStore(self.path)
        lru_cache_store = LRUStoreCache(
            store, max_size=lru_cache_size * 2024**3
        )
        self.zarr = zarr.open(lru_cache_store, mode=mode)

        if "indptr" not in self.zarr.array_keys():
            self.zarr.create(
                "indptr",
                shape=(0,),
                dtype=np.int64,
                chunks=(chunks_indptr,),
                **kwargs,
            )
            self.zarr.create(
                "indices",
                shape=(0,),
                dtype=np.uint16,
                chunks=(chunks_data,),
                **kwargs,
            )
            self.zarr.create(
                "data",
                shape=(0,),
                dtype=np.float32,
                chunks=(chunks_data,),
                **kwargs,
            )
            self.zarr.attrs["ncols"] = ncols  # Store critical metadata

        self._indptr = self.zarr["indptr"]
        self._indices = self.zarr["indices"]
        self._data = self.zarr["data"]
        self._binary_instance = None

    @classmethod
    def open(cls, path: str | Path, mode="r", **kwargs):
        z = zarr.open(path, mode="r")
        attrs = z.attrs
        assert "ncols" in attrs, "ncols not found in attrs"
        return cls(path, ncols=attrs["ncols"], mode=mode, **kwargs)

    @property
    def bool(self):
        if self._binary_instance is None:
            binary_instance = self.open(self.path, mode="r")
            binary_instance._data = np.ones(self.nnz, dtype=np.uint8)
            self._binary_instance = binary_instance
        return self._binary_instance

    @property
    def shape(self):
        return (self._indptr.shape[0] - 1, self.ncols)

    @property
    def nnz(self):
        return len(self._data)

    def append(self, mat: csr_matrix):
        if mat.shape[1] != self.ncols:
            raise ValueError("Number of columns must match")
        if mat.nnz == 0:  # Skip empty matrices
            return

        if self._indptr.shape[0] == 0:
            self._indptr.append(np.array([0], dtype=np.int64))
        new_indptr = mat.indptr[1:] + self._indptr[-1]

        self._indptr.append(new_indptr)
        self._data.append(mat.data)
        self._indices.append(mat.indices)

        self._binary_instance = None  # Reset binary instance

    def tail_add_dummy(self):
        """Add a dummy row at the end of the matrix."""
        self._indptr.append(np.array([self._indptr[-1]]))


def _parse_spe_file_list(spe_file_list: str | list[str]):
    if isinstance(spe_file_list, str):
        spe_file_list = Path(spe_file_list)
        if not spe_file_list.exists():
            logger.error(f"Path not found: {spe_file_list}")
            raise FileNotFoundError(f"Path not found: {spe_file_list}")
        #-
        if spe_file_list.is_dir():
            final_list = [
                spe_file_list / file_path for file_path in os.listdir(spe_file_list)
            ]
        elif spe_file_list.is_file():
            with open(spe_file_list, "r") as f:
                final_list = f.readlines()
            final_list = [file.strip() for file in final_list]
    elif isinstance(spe_file_list, list):
        final_list = spe_file_list
    else:
        raise ValueError(
            "Invalid type for spe_file_list,"
            f"expected str or list, got {type(spe_file_list)}"
        )
    #-
    return final_list


def set_seed(seed_value):
    """
    Set seed for reproducibility in PyTorch and other libraries.
    """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        logger.info("Using GPU for computations.")
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    else:
        logger.info("Using CPU for computations.")


def index_splitter(n, splits):
    idx = torch.arange(n)
    splits_tensor = torch.as_tensor(splits)
    multiplier = n / splits_tensor.sum()
    splits_tensor = (multiplier * splits_tensor).long()
    diff = n - splits_tensor.sum()
    splits_tensor[0] += diff
    return random_split(idx, splits_tensor)


def run_find_latent_representation(args: FindLatentRepresentationsConfig):
    logger.info(f'Project dir: {args.project_dir}')
    set_seed(2024)
    
    # Find the hvg
    spe_file_list = args.spe_file_list
    spe_file_list = _parse_spe_file_list(spe_file_list)
    hvg, n_cell_used, percent_annotation, gene_name_dict = find_common_hvg(spe_file_list, args)
    common_genes = np.array(list(gene_name_dict.keys()))
    
    # Prepare the trainning data
    get_trainning_data = TrainingData(args)
    get_trainning_data.prepare(spe_file_list, n_cell_used, hvg, percent_annotation)
    
    # Configure the distribution
    if args.data_layer in ["count", "counts"]:
        distribution = args.distribution
        variational = True
        use_tf = args.use_tf
    else:
        distribution = "gaussian"
        variational = False
        use_tf = False

    # Instantiation the LGCN VAE
    input_size = [
        get_trainning_data.expression_merge.size(1),
        get_trainning_data.expression_gcn_merge.size(1),
    ]


    class_size = len(torch.unique(get_trainning_data.label_merge))
    batch_size = get_trainning_data.batch_size
    cell_size, out_size = get_trainning_data.expression_merge.shape
    label_name = get_trainning_data.label_name
    
    # Configure the batch embedding dim
    batch_embedding_size = 64

    # Configure the model
    gsmap_lgcn_model = StEmbeding(
        # parameter of VAE
        input_size=input_size,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        batch_embedding_size=batch_embedding_size,
        out_put_size=out_size,
        batch_size=batch_size,
        class_size=class_size,
        # parameter of transformer
        module_dim=args.module_dim,
        hidden_gmf=args.hidden_gmf,
        n_modules=args.n_modules,
        nhead=args.nhead,
        n_enc_layer=args.n_enc_layer,
        # parameter of model structure
        distribution=distribution,
        use_tf=use_tf,
        variational=variational,
    )

    # Configure the optimizer
    optimizer = torch.optim.Adam(gsmap_lgcn_model.parameters(), lr=1e-3)
    logger.info(
        f"gsMap-LGCN parameters: {sum(p.numel() for p in gsmap_lgcn_model.parameters())}."
    )
    logger.info(f"Number of cells used in trainning: {cell_size}.")

    # Split the data to trainning (80%) and validation (20%).
    train_idx, val_idx = index_splitter(
        get_trainning_data.expression_gcn_merge.size(0), [80, 20]
    )
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Configure the data loader
    dataset = TensorDataset(
        get_trainning_data.expression_gcn_merge,
        get_trainning_data.batch_merge,
        get_trainning_data.expression_merge,
        get_trainning_data.label_merge,
    )
    train_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, sampler=train_sampler
    )
    val_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, sampler=val_sampler
    )

    # Model trainning
    gsMap_embedding_finder = ModelTrain(
        gsmap_lgcn_model,
        optimizer,
        distribution,
        mode="reconstruction",
        lr=1e-3,
        model_path=args.model_path,
    )
    gsMap_embedding_finder.set_loaders(train_loader, val_loader)
    print(gsMap_embedding_finder.model)
    
    if not os.path.exists(args.model_path):
        # reconstruction
        gsMap_embedding_finder.train(args.itermax, patience=args.patience)

        # classification
        if args.two_stage and args.annotation is not None:
            gsMap_embedding_finder.model.load_state_dict(torch.load(args.model_path))
            gsMap_embedding_finder.mode = "classification"
            gsMap_embedding_finder.train(args.itermax, patience=args.patience)
    else:
        logger.info(f"Model found at {args.model_path}. Skipping training.")
        
    # Load the best model
    gsMap_embedding_finder.model.load_state_dict(torch.load(args.model_path))
    gsmap_embedding_model = gsMap_embedding_finder.model

    # Configure the inference
    infer = InferenceData(hvg, batch_size, gsmap_embedding_model, label_name, args)
    
    print(args.zarr_group_path)
    
    # Initialize zarr storage for ranks
    n_genes = len(common_genes)
    output_zarr_path = args.latent_dir / "ranks.zarr"
    log_ranks = ZarrBackedCSR(str(output_zarr_path), ncols=n_genes, mode="w")
    
    # Initialize arrays for mean calculation
    sum_log_ranks = np.zeros(n_genes, dtype=np.float32)
    sum_frac = np.zeros(n_genes, dtype=np.float32)
    total_cells = 0
    
    # Do inference
    for st_id, st_file in enumerate(spe_file_list):
        st_name = (Path(st_file).name).split(".h5ad")[0]
        logger.info(f"Processing {st_name} ({st_id + 1}/{len(spe_file_list)})...")
        
        output_path = args.latent_dir / f"{st_name}_add_latent.h5ad"  
        
        # Infer the embedding
        adata = infer.infer_embedding_single(st_id, st_file)
        # adata.obs_names = st_name + '_' + adata.obs_names
        
        # Transfer the gene name
        common_genes = np.array(list(gene_name_dict.keys()))
        common_genes_transfer = np.array(itemgetter(*common_genes)(gene_name_dict))
        adata = adata[:,common_genes].copy()
        adata.var_names = common_genes_transfer
        
        # Get expression data for ranking
        if args.data_layer in ["count", "counts"]:
            X = adata.layers[args.data_layer]
        else:
            X = adata.X
        
        # Convert to sparse if not already
        if not hasattr(X, 'tocsr'):
            X = csr_matrix(X)
        else:
            X = X.tocsr()
        
        # Calculate ranks using non-zero method
        global_positions = np.arange(n_genes)  # All genes are already aligned
        study_log_ranks = np.log(rank_data_nz(X, global_positions, n_genes))
        
        # Check for anomalous ranks
        max_study_log_ranks = study_log_ranks.max()
        if max_study_log_ranks > 11:
            logger.warning(
                f"Max log rank {max_study_log_ranks} is larger than 11 in {st_name}"
            )
        
        # Convert to sparse and append to zarr
        study_log_ranks_sparse = csr_matrix(study_log_ranks)
        log_ranks.append(study_log_ranks_sparse)
        
        # Update sums for mean calculation
        sum_log_ranks += csc_col_sum(study_log_ranks_sparse, fill_zero=True)
        sum_frac += (X != 0).tocsc().sum(0).A1
        total_cells += X.shape[0]
        
        # Compute the slice mean (original method for backward compatibility)
        process_slice_mean(args, st_name, adata)

        # Compute the depth
        if args.data_layer in ["count", "counts"]:
            adata.obs['depth'] = np.array(adata.layers[args.data_layer].sum(axis=1)).flatten()
            
        # Save the ST data with embeddings
        adata.write_h5ad(output_path)
        
        # Clean up memory
        del adata, X, study_log_ranks, study_log_ranks_sparse
        gc.collect()
    
    # Finalize zarr storage
    log_ranks.tail_add_dummy()
    
    # Calculate mean log ranks and mean fraction
    mean_log_ranks = sum_log_ranks / total_cells
    mean_frac = sum_frac / total_cells
    
    # Save mean and fraction to parquet file
    mean_frac_df = pd.DataFrame(
        data=dict(
            G_Mean=mean_log_ranks,
            frac=mean_frac,
            gene_name=common_genes_transfer,
        ),
        index=common_genes_transfer,
    )
    parquet_path = args.latent_dir / "mean_frac.parquet"
    mean_frac_df.to_parquet(
        parquet_path,
        index=True,
        compression="gzip",
    )
    logger.info(f"Mean fraction data saved to {parquet_path}")


    
