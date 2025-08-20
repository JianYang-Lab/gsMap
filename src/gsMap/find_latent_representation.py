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


logger = logging.getLogger(__name__)
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap
    from jax.scipy.stats import rankdata as jax_rankdata
    JAX_AVAILABLE = True
    # Set JAX to use GPU if available, otherwise CPU
    try:
        _ = jax.devices('gpu')
        logger.info("JAX GPU backend available")
    except:
        logger.info("JAX CPU backend will be used")
except ImportError:
    JAX_AVAILABLE = False
    logger.warning("JAX not available, using pure Python implementation")

try:
    import zarr.codecs
    # Sharding will be configured via array creation options
    logger.info("Using zarr with sharding support")
except ImportError:
    raise RuntimeError("Zarr codecs not available, please upgrade zarr")

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


def rank_data_nz_vectorized(arr, n_genes):
    """Vectorized version of rank_data_nz for better performance."""
    res = np.ones((arr.shape[0], n_genes), dtype=np.float32)
    
    # Process in batches for better cache efficiency
    batch_size = min(100, arr.shape[0])
    
    for batch_start in range(0, arr.shape[0], batch_size):
        batch_end = min(batch_start + batch_size, arr.shape[0])
        
        for i in range(batch_start, batch_end):
            row = arr.getrow(i)
            if row.nnz > 0:
                # Data is already aligned, so indices directly correspond to gene positions
                res[i, row.indices] = rankdata(row.data, method='average') + n_genes - row.nnz
    
    return res


if JAX_AVAILABLE:
    @jit
    def jax_rank_single_row(data, indices, n_genes):
        """JAX-optimized ranking for a single row."""
        n_nonzero = data.shape[0]
        
        # Create full row with zeros
        full_row = jnp.zeros(n_genes, dtype=jnp.float32)
        full_row = full_row.at[indices].set(data)
        
        # Rank non-zero elements
        mask = full_row != 0
        ranks = jnp.zeros_like(full_row)
        
        # Get ranks for non-zero elements
        nonzero_data = jnp.where(mask, full_row, jnp.inf)
        sorted_indices = jnp.argsort(nonzero_data)
        
        # Compute ranks
        rank_values = jnp.arange(1, n_genes + 1, dtype=jnp.float32)
        ranks = ranks.at[sorted_indices].set(rank_values)
        
        # Adjust ranks for zeros (they get max rank)
        ranks = jnp.where(mask, ranks, 1.0)
        
        # Adjust for sparsity
        ranks = jnp.where(mask, ranks + n_genes - n_nonzero, 1.0)
        
        return ranks
    
    @jit
    def jax_rank_dense_batch(dense_matrix):
        """JAX-optimized batch ranking for dense matrices."""
        n_rows, n_cols = dense_matrix.shape
        
        # Vectorized ranking
        # Add small noise to break ties consistently
        noise = jax.random.uniform(jax.random.PRNGKey(0), dense_matrix.shape) * 1e-10
        matrix_with_noise = dense_matrix + noise
        
        # Get argsort for each row
        sorted_indices = jnp.argsort(matrix_with_noise, axis=1)
        
        # Create ranks
        ranks = jnp.zeros_like(dense_matrix)
        row_indices = jnp.arange(n_rows)[:, None]
        col_ranks = jnp.arange(1, n_cols + 1)[None, :]
        
        ranks = ranks.at[row_indices, sorted_indices].set(col_ranks)
        
        # Handle zeros by setting their rank to 1
        ranks = jnp.where(dense_matrix != 0, ranks, 1.0)
        
        return ranks
    
    def rank_data_nz_jax(arr, n_genes):
        """JAX-optimized rank calculation for sparse matrices."""
        n_rows = arr.shape[0]
        
        # Convert sparse to dense for JAX processing
        # Process in chunks to manage memory
        chunk_size = min(1000, n_rows)
        all_ranks = []
        
        for start_idx in range(0, n_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_rows)
            
            # Convert chunk to dense
            chunk = arr[start_idx:end_idx].toarray().astype(np.float32)
            
            # Compute ranks using JAX
            chunk_ranks = jax_rank_dense_batch(jnp.array(chunk))
            
            # Convert back to numpy
            all_ranks.append(np.array(chunk_ranks))
        
        # Concatenate all results
        result = np.vstack(all_ranks) if all_ranks else np.ones((n_rows, n_genes), dtype=np.float32)
        
        return result


def rank_data_nz(arr, n_genes, use_jax=True):
    """Main ranking function with JAX acceleration."""
    if use_jax and JAX_AVAILABLE and arr.nnz > 0:
        try:
            return rank_data_nz_jax(arr, n_genes)
        except Exception as e:
            logger.warning(f"JAX ranking failed, falling back to vectorized: {e}")
            return rank_data_nz_vectorized(arr, n_genes)
    else:
        return rank_data_nz_vectorized(arr, n_genes)




class ZarrBackedCSR:

    def __init__(
        self,
        path: str,
        ncols: int,
        mode: str = "r",
        chunks_rows: int = 1000,
        chunks_data: int = 5_000,
        shards: int = 10_000_000,
        lru_cache_size: int = 2,
        **kwargs,
    ):
        zarr
        # Check zarr version
        zarr_version = tuple(map(int, zarr.__version__.split('.')[:2]))
        if zarr_version < (2, 18):
            raise RuntimeError(
                f"Zarr version {zarr.__version__} is not supported. "
                f"Please upgrade to zarr>=2.18 for sharding support: pip install 'zarr>=2.18'"
            )
        
        self.path = path
        self.ncols = ncols
        
        # Configure storage with sharding
        store = DirectoryStore(self.path)
        
        # Configure sharding for fewer files

        # Use LRU cache for better performance
        lru_cache_store = LRUStoreCache(
            store, max_size=lru_cache_size * 1024**3
        )
        self.zarr_array = zarr.open(lru_cache_store, mode=mode)

        if "indptr" not in self.zarr_array.array_keys():
            # Configure compression
            compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=zarr.Blosc.BITSHUFFLE)
            
            self.zarr_array.create(
                "indptr",
                shape=(0,),
                dtype=np.int32,
                chunks=(chunks_rows,),
                compressor=compressor,
                **kwargs,
            )
            
            # Only create combined array (no backward compatibility)
            dt = np.dtype([('idx', np.uint16), ('val', np.float32)])
            self.zarr_array.create(
                "data_indices",
                shape=(0,),
                dtype=dt,
                chunks=(chunks_data,),
                compressor=compressor,
                shards = (shards,),
                **kwargs,
            )
            
            self.zarr_array.attrs["ncols"] = ncols
            self.zarr_array.attrs["version"] = "3.0"

        # Load indptr entirely into memory (it's small)
        self._indptr_zarr = self.zarr_array["indptr"]
        self._indptr = np.array(self._indptr_zarr[:], dtype=np.int32) if len(self._indptr_zarr) > 0 else np.array([], dtype=np.int32)
        
        # Keep data_indices as zarr array for lazy loading
        self._data_indices = self.zarr_array["data_indices"]
        
        self._binary_instance = None

    @classmethod
    def open(cls, path: str | Path, mode="r", **kwargs):
        z = zarr.open(path, mode="r")
        attrs = z.attrs
        if "ncols" not in attrs:
            raise ValueError("Invalid ZarrBackedCSR: ncols not found in attributes")
        if attrs.get("version", "1.0") < "3.0":
            raise ValueError(f"Unsupported ZarrBackedCSR version {attrs.get('version', '1.0')}. Please recreate with version 3.0")
        return cls(path, ncols=attrs["ncols"], mode=mode, **kwargs)

    @property
    def shape(self):
        return (len(self._indptr) - 1, self.ncols) if len(self._indptr) > 0 else (0, self.ncols)

    @property
    def nnz(self):
        return self._indptr[-1] if len(self._indptr) > 0 else 0

    def append(self, mat: csr_matrix):
        if mat.shape[1] != self.ncols:
            raise ValueError(f"Number of columns must match. Expected {self.ncols}, got {mat.shape[1]}")
        if mat.nnz == 0:  # Skip empty matrices
            return

        # Update in-memory indptr
        if len(self._indptr) == 0:
            self._indptr = np.array([0], dtype=np.int32)
            self._indptr_zarr.append(np.array([0], dtype=np.int32))
        
        current_nnz = self._indptr[-1]
        new_indptr = mat.indptr[1:].astype(np.int32) + current_nnz
        
        # Update zarr indptr
        self._indptr_zarr.append(new_indptr)
        
        # Update in-memory indptr
        self._indptr = np.append(self._indptr, new_indptr)
        
        # Store in combined array only
        combined = np.empty(mat.nnz, dtype=[('idx', np.uint16), ('val', np.float32)])
        combined['idx'] = mat.indices.astype(np.uint16)
        combined['val'] = mat.data.astype(np.float32)
        self._data_indices.append(combined)
    
    def __getitem__(self, indices):
        """Optimized slicing with combined data access."""
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
        
        indices_array = np.asarray(indices, dtype=np.int32)
        M = len(indices_array)
        
        if M == 0:
            raise ValueError("Expected non-empty indices")
        
        nrows = len(self._indptr) - 1
        if nrows == 0:
            return csr_matrix((M, self.ncols), dtype=np.float32)
        
        # Handle negative indices
        negative_mask = indices_array < 0
        if negative_mask.any():
            indices_array[negative_mask] += nrows
        
        # Use in-memory indptr for fast access
        start_ptrs = self._indptr[indices_array]
        end_ptrs = self._indptr[indices_array + 1]
        
        # Calculate total size needed
        row_nnz = end_ptrs - start_ptrs
        res_indptr = np.zeros(M + 1, dtype=np.int32)
        np.cumsum(row_nnz, out=res_indptr[1:])
        
        total_nnz = res_indptr[-1]
        
        if total_nnz == 0:
            return csr_matrix((M, self.ncols), dtype=np.float32)
        
        # Optimize data fetching by merging consecutive ranges
        ranges = [(start, end) for start, end in zip(start_ptrs, end_ptrs) if end > start]
        
        if not ranges:
            return csr_matrix((M, self.ncols), dtype=np.float32)
        
        # Merge nearby ranges to reduce I/O
        merged_ranges = []
        current_start, current_end = ranges[0]
        
        for start, end in ranges[1:]:
            if start <= current_end + 5000:  # Merge if gap < 5000 elements
                current_end = max(current_end, end)
            else:
                merged_ranges.append((current_start, current_end))
                current_start, current_end = start, end
        merged_ranges.append((current_start, current_end))
        
        # Fetch data in merged chunks
        all_data = []
        for chunk_start, chunk_end in merged_ranges:
            chunk_data = self._data_indices[chunk_start:chunk_end]
            all_data.append(chunk_data)
        
        # Combine and extract needed elements
        if len(all_data) == 1:
            combined_data = all_data[0]
        else:
            combined_data = np.concatenate(all_data)
        
        # Extract only the needed elements
        needed_indices = []
        base_offset = merged_ranges[0][0]
        for start, end in zip(start_ptrs, end_ptrs):
            if end > start:
                needed_indices.extend(range(start - base_offset, end - base_offset))
        
        if needed_indices:
            sub_combined = combined_data[needed_indices]
            sub_indices = sub_combined['idx']
            sub_data = sub_combined['val']
        else:
            sub_indices = np.array([], dtype=np.uint16)
            sub_data = np.array([], dtype=np.float32)
        
        return csr_matrix(
            (sub_data, sub_indices, res_indptr),
            shape=(M, self.ncols),
            dtype=np.float32
        )

    def tail_add_dummy(self):
        """Add a dummy row at the end of the matrix."""
        if len(self._indptr) > 0:
            last_val = self._indptr[-1]
            self._indptr_zarr.append(np.array([last_val], dtype=np.int32))
            self._indptr = np.append(self._indptr, last_val)
    
    def optimize_chunks(self):
        """Dynamically optimize chunk sizes based on data characteristics."""
        if self._indptr.shape[0] > 1:
            # Calculate average row density
            total_nnz = self._indptr[-1]
            n_rows = self._indptr.shape[0] - 1
            avg_nnz_per_row = total_nnz / n_rows if n_rows > 0 else 0
            
            # Optimize row chunks: aim for ~100KB per chunk
            optimal_rows_per_chunk = max(100, min(2000, int(100000 / (avg_nnz_per_row * 6))))  # 6 bytes per entry
            
            # Optimize data chunks: aim for ~500KB per chunk  
            optimal_data_chunk = max(10000, min(200000, int(500000 / 6)))
            
            logger.info(f"Optimal chunking - rows: {optimal_rows_per_chunk}, data: {optimal_data_chunk}")
            logger.info(f"Average NNZ per row: {avg_nnz_per_row:.1f}")
            
            return optimal_rows_per_chunk, optimal_data_chunk
        return 500, 50000


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
    
    # Check for JAX availability
    if JAX_AVAILABLE:
        logger.info("JAX acceleration is available for rank calculations")
        # Check if GPU is available
        try:
            devices = jax.devices()
            logger.info(f"JAX devices available: {devices}")
        except:
            pass
    else:
        logger.info("JAX not available, using vectorized numpy implementation")
    
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
        
        # Efficient sparse matrix conversion
        if not hasattr(X, 'tocsr'):
            X = csr_matrix(X, dtype=np.float32)  # Use float32 for memory efficiency
        else:
            X = X.tocsr()
            if X.dtype != np.float32:
                X = X.astype(np.float32)  # Convert to float32 if needed
        
        # Pre-allocate output arrays for efficiency
        X.sort_indices()  # Sort indices for better cache performance
        
        # Calculate ranks using optimized method
        n_cells = X.shape[0]
        
        # Choose ranking method based on data size and available resources
        if JAX_AVAILABLE and n_cells > 50:
            # Use JAX for medium to large datasets
            logger.debug(f"Using JAX acceleration for {n_cells} cells")
            study_ranks = rank_data_nz(X, n_genes, use_jax=True)
        else:
            # Use vectorized numpy for small datasets
            logger.debug(f"Using vectorized numpy for {n_cells} cells")
            study_ranks = rank_data_nz(X, n_genes, use_jax=False)
        
        study_log_ranks = np.log(study_ranks,)
        
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


    
