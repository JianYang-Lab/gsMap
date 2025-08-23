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
from scipy.sparse import csr_matrix
from zarr.storage import DirectoryStore
from tqdm import tqdm
import threading
import queue
import anndata as ad

logger = logging.getLogger(__name__)
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

try:
    import zarr.codecs
    # Sharding will be configured via array creation options
    logger.info("Using zarr with sharding support")
except ImportError:
    raise RuntimeError("Zarr codecs not available, please upgrade zarr>=2.18.0 to use sharding features. Install with: pip install 'zarr>=2.18'")

from gsMap.GNN.TrainStep import ModelTrain
from gsMap.GNN.STmodel import StEmbeding
from gsMap.ST_process import TrainingData, find_common_hvg, InferenceData
from gsMap.config import FindLatentRepresentationsConfig
from gsMap.slice_mean import process_slice_mean

from operator import itemgetter

logger = logging.getLogger(__name__)



@jit
def jax_process_chunk(dense_matrix, n_genes):
    """JAX-optimized processing: ranking + accumulator calculations."""
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

    # Compute log ranks
    log_ranks = jnp.log(ranks)

    # Compute accumulators for fill_zero logic
    nonzero_mask = dense_matrix != 0
    n_nonzero_per_row = nonzero_mask.sum(axis=1, keepdims=True)
    zero_log_ranks = jnp.log((n_genes - n_nonzero_per_row + 1) / 2)

    # Sum log ranks (with fill_zero)
    sum_log_ranks = log_ranks.sum(axis=0)
    sum_log_ranks += zero_log_ranks.sum() - (zero_log_ranks * nonzero_mask).sum(axis=0)

    # Sum fraction (count of non-zeros)
    sum_frac = nonzero_mask.sum(axis=0)

    return log_ranks, sum_log_ranks, sum_frac

def rank_data_jax(X: csr_matrix, n_genes,
                  zarr_csr = None,
                  metadata=None,
                  chunk_size=1000,
                  write_interval=10):
    """JAX-optimized rank calculation with batched writing."""
    assert X.nnz != 0, "Input matrix must not be empty"

    n_rows, n_cols = X.shape

    # Initialize accumulators
    sum_log_ranks = jnp.zeros(n_genes, dtype=jnp.float32)
    sum_frac = jnp.zeros(n_genes, dtype=jnp.float32)

    # Process in chunks to manage memory
    chunk_size = min(chunk_size, n_rows)
    pending_chunks = []  # Buffer for batching writes
    chunks_processed = 0

    # Setup progress bar
    study_name = metadata.get('name', 'unknown') if metadata else 'unknown'

    with tqdm(total=n_rows, desc=f"Ranking {study_name}", unit="cells") as pbar:
        for start_idx in range(0, n_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, n_rows)

            # Convert chunk to dense
            chunk_X = X[start_idx:end_idx]
            chunk_dense = chunk_X.toarray().astype(np.float32)
            chunk_jax = jnp.array(chunk_dense)

            # Process chunk with JIT-compiled function (ranking + accumulators)
            chunk_log_ranks, chunk_sum_log_ranks, chunk_sum_frac = jax_process_chunk(chunk_jax, n_genes)

            # Update global accumulators
            sum_log_ranks += chunk_sum_log_ranks
            sum_frac += chunk_sum_frac

            # Convert JAX array to numpy
            chunk_log_ranks_np = np.array(chunk_log_ranks)
            pending_chunks.append(chunk_log_ranks_np)
            chunks_processed += 1

            # Write to zarr periodically
            if zarr_csr and chunks_processed % write_interval == 0:
                # Combine pending chunks and convert to CSR
                zarr_csr.append_batch(np.vstack(pending_chunks))
                pending_chunks = []

            # Update progress bar
            pbar.update(end_idx - start_idx)

    # Write any remaining chunks
    if zarr_csr and pending_chunks:
        zarr_csr.append_batch(np.vstack(pending_chunks))

    return np.array(sum_log_ranks), np.array(sum_frac)


class ZarrBackedCSR:
    """CSR matrix storage with single background writer thread and batched appends."""
    
    def __init__(
        self,
        path: str,
        ncols: int,
        mode: str = "r",
        chunks_rows: int = 1000,
        chunks_data: int = 100_000,
        **kwargs,
    ):
        # Check zarr version
        zarr_version = tuple(map(int, zarr.__version__.split('.')[:2]))
        if zarr_version < (2, 18):
            raise RuntimeError(
                f"Zarr version {zarr.__version__} is not supported. "
                f"Please upgrade to zarr>=2.18: pip install 'zarr>=2.18'"
            )
        
        self.path = Path(path)
        self.ncols = ncols
        self.current_row = 0
        self.current_nnz = 0
        self.mode = mode
        
        # Check if file exists and is complete before opening
        if mode == 'w' and self.path.exists():
            try:
                # Open in read mode first to check integrity
                store = DirectoryStore(self.path)
                existing = zarr.open(store, mode='r')
                if 'integrity_mark' in existing.attrs and existing.attrs['integrity_mark'] == 'complete':
                    raise ValueError(
                        f"ZarrBackedCSR at {path} already exists and is marked as complete. "
                        f"Please delete it manually if you want to overwrite: rm -rf {path}"
                    )
                else:
                    logger.warning(f"ZarrBackedCSR at {path} exists but is incomplete. Deleting and recreating.")
                    import shutil
                    shutil.rmtree(self.path)
            except ValueError:
                # Re-raise the complete file error
                raise
            except Exception as e:
                logger.warning(f"Could not read existing Zarr at {path}: {e}. Deleting and recreating.")
                import shutil
                if self.path.exists():
                    shutil.rmtree(self.path)
        
        # Open zarr array
        store = DirectoryStore(self.path)
        self.zarr_array = zarr.open(store, mode=mode)
        
        # Check integrity for read mode
        if mode == 'r':
            if 'integrity_mark' not in self.zarr_array.attrs:
                raise ValueError(f"ZarrBackedCSR at {path} is incomplete or corrupted. Missing integrity mark.")
            if self.zarr_array.attrs['integrity_mark'] != 'complete':
                raise ValueError(f"ZarrBackedCSR at {path} is incomplete. Integrity mark: {self.zarr_array.attrs.get('integrity_mark')}")
            logger.debug(f"ZarrBackedCSR loaded successfully with integrity check passed")
        
        # Only initialize arrays if in write mode
        if mode == 'w' and "indptr" not in self.zarr_array.array_keys():
            # Initialize with 0 for CSR format
            self.zarr_array.create(
                "indptr",
                shape=(1,),
                dtype=np.int64,
                chunks=(chunks_rows,),
                **kwargs,
            )
            self.zarr_array["indptr"][0] = 0
            
            # Combined array for indices and values
            dt = np.dtype([('idx', np.uint16), ('val', np.float32)])
            self.zarr_array.create(
                "data_indices",
                shape=(0,),
                dtype=dt,
                chunks=(chunks_data,),
                order='C',
                **kwargs,
            )
            
            self.zarr_array.attrs["ncols"] = ncols
            self.zarr_array.attrs["version"] = "0.1"
            self.zarr_array.attrs["integrity_mark"] = "incomplete"
        
        # Keep references to zarr arrays
        self._indptr_zarr = self.zarr_array["indptr"]
        self._data_indices = self.zarr_array["data_indices"]
        
        # Load current indptr for tracking
        if len(self._indptr_zarr) > 0:
            self._indptr = np.array(self._indptr_zarr[:], dtype=np.int64)
            self.current_row = len(self._indptr) - 1
            self.current_nnz = self._indptr[-1]
        else:
            self._indptr = np.array([0], dtype=np.int64)
            self.current_row = 0
            self.current_nnz = 0
        
        # Single background writer thread
        self.write_queue = queue.Queue()
        self.writer_thread = None
        self.stop_writer = threading.Event()
        
        if mode == 'w':
            self._start_writer_thread()

    def _start_writer_thread(self):
        """Start the background writer thread."""
        def writer_worker():
            while not self.stop_writer.is_set():
                try:
                    # Get batch from queue with timeout
                    mat = self.write_queue.get(timeout=0.1)
                    if mat is None:  # Sentinel value
                        break
                    self._append_to_zarr(mat)
                    self.write_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Writer thread error: {e}")
                    raise
        
        self.writer_thread = threading.Thread(target=writer_worker, daemon=True)
        self.writer_thread.start()
    
    @classmethod
    def open(cls, path: str | Path, mode="r", **kwargs):
        z = zarr.open(path, mode="r")
        attrs = z.attrs
        if "ncols" not in attrs:
            raise ValueError("Invalid ZarrBackedCSR: ncols not found in attributes")
        return cls(path, ncols=attrs["ncols"], mode=mode, **kwargs)

    @property
    def shape(self):
        return (self.current_row, self.ncols)

    @property
    def nnz(self):
        return self.current_nnz

    def append_batch(self, mat):

        # Queue for background writing
        self.write_queue.put(mat)
    
    def _append_to_zarr(self, mat):

        if not isinstance(mat, csr_matrix):
            if isinstance(mat, np.ndarray):
                mat = csr_matrix(mat)
            else:
                raise TypeError(f"Expected csr_matrix or numpy array, got {type(mat)}")

        if mat.shape[1] != self.ncols:
            raise ValueError(f"Number of columns must match. Expected {self.ncols}, got {mat.shape[1]}")

        n_rows = mat.shape[0]
        nnz = mat.nnz
        
        # Append to indptr (offset by current nnz)
        new_indptr = mat.indptr[1:].astype(np.int64) + self.current_nnz
        self._indptr_zarr.append(new_indptr)
        
        # Append to data_indices as structured array
        if nnz > 0:
            combined = np.empty(nnz, dtype=[('idx', np.uint16), ('val', np.float32)])
            combined['idx'] = mat.indices.astype(np.uint16)
            combined['val'] = mat.data.astype(np.float32)
            self._data_indices.append(combined)
        
        # Update counters
        self.current_row += n_rows
        self.current_nnz += nnz
        
        # Update in-memory indptr
        self._indptr = np.append(self._indptr, new_indptr)
    
    def mark_complete(self):
        """Mark the zarr array as complete by setting integrity mark."""
        if self.mode == 'w':
            self.zarr_array.attrs['integrity_mark'] = 'complete'
            logger.info(f"Marked ZarrBackedCSR at {self.path} as complete")
    
    def close(self):
        """Clean up resources and wait for pending writes."""
        if hasattr(self, 'writer_thread') and self.writer_thread and self.writer_thread.is_alive():
            logger.info("Closing ZarrBackedCSR, waiting for writer thread to finish...")
            # Wait for queue to empty
            self.write_queue.join()
            # Send sentinel to stop thread
            self.write_queue.put(None)
            # Wait for thread to finish
            self.writer_thread.join(timeout=5)
            if self.writer_thread.is_alive():
                logger.warning("Writer thread did not finish in time")
        
        # Mark as complete when closing in write mode
        if self.mode == 'w':
            self.mark_complete()

    def __getitem__(self, indices):
        nrows = len(self._indptr) - 1
        
        # Handle different index types
        if isinstance(indices, slice):
            # Convert slice to array of indices
            start, stop, step = indices.indices(nrows)
            indices = list(range(start, stop, step))
            if len(indices) == 0:
                return csr_matrix((0, self.ncols), dtype=np.float32)
        elif isinstance(indices, (int, np.integer)):
            indices = [indices]
        elif isinstance(indices, (list, tuple, np.ndarray)):
            indices = list(indices)
        else:
            raise TypeError(f"Indices must be an integer, slice, or array-like, got {type(indices)}")
        
        indices_array = np.asarray(indices, dtype=np.int64)
        M = len(indices_array)
        
        if M == 0:
            return csr_matrix((0, self.ncols), dtype=np.float32)
        
        if nrows == 0:
            return csr_matrix((M, self.ncols), dtype=np.float32)
        
        # Handle negative indices
        negative_mask = indices_array < 0
        if negative_mask.any():
            indices_array[negative_mask] += nrows
        
        # Use in-memory indptr for fast access
        start_ptrs = self._indptr[indices_array]
        end_ptrs = self._indptr[indices_array + 1]
        
        # Calculate total size needed and build result indptr
        row_nnz = end_ptrs - start_ptrs
        res_indptr = np.zeros(M + 1, dtype=np.int64)
        np.cumsum(row_nnz, out=res_indptr[1:])

        total_nnz = res_indptr[-1]
        
        if total_nnz == 0:
            return csr_matrix((M, self.ncols), dtype=np.float32)
        
        # Build array of all indices to fetch
        all_indices = np.empty(total_nnz, dtype=np.int64)
        
        # Create ranges for each row's data
        ranges = [
            np.arange(start, end)
            for start, end in zip(start_ptrs, end_ptrs)
            if end > start
        ]
        
        # Concatenate all ranges efficiently
        if ranges:
            if len(ranges) == 1:
                all_indices[:] = ranges[0]
            else:
                np.concatenate(ranges, out=all_indices)
        
        # Fetch all data at once using fancy indexing (Zarr handles this efficiently)
        data_elements = self._data_indices[all_indices]
        
        # Extract indices and values from structured array
        sub_indices = np.ascontiguousarray(data_elements['idx'])
        sub_data = np.ascontiguousarray(data_elements['val'])
        
        return csr_matrix(
            (sub_data, sub_indices, res_indptr),
            shape=(M, self.ncols),
            dtype=np.float32
        )



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


    # Initialize zarr storage for ranks with simple append-based writing
    n_genes = len(common_genes)
    output_zarr_path = args.latent_dir / "ranks.zarr"
    log_ranks_zarr = ZarrBackedCSR(str(output_zarr_path), ncols=n_genes, mode="w")
    
    # Initialize arrays for mean calculation
    sum_log_ranks = np.zeros(n_genes, dtype=np.float64)  # Use float64 for accumulation precision
    sum_frac = np.zeros(n_genes, dtype=np.float64)
    total_cells = 0
    
    # Initialize list to collect AnnData objects for concatenation
    adata_list = []
    
    # Do inference
    logger.info(f"Processing {len(spe_file_list)} spatial transcriptomics files")
    
    for st_id, st_file in tqdm(enumerate(spe_file_list), 
                                total=len(spe_file_list), 
                                desc="Processing studies", 
                                unit="study"):
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

        # Compute the depth
        if args.data_layer in ["count", "counts"]:
            adata.obs['depth'] = np.array(adata.layers[args.data_layer].sum(axis=1)).flatten()

        # Add slice_id to obs
        adata.obs['slice_id'] = st_name
        adata.obs['slice_numeric_id'] = st_id

        # Filter cells based on annotation group size if annotation is provided
        # This must be done BEFORE adding to rank zarr to maintain index consistency
        if args.annotation is not None and args.annotation in adata.obs.columns:
            min_cells_per_type = 21  # Minimum number of homogeneous neighbors
            annotation_counts = adata.obs[args.annotation].value_counts()
            valid_annotations = annotation_counts[annotation_counts >= min_cells_per_type].index
            
            # Check if any filtering is needed
            if len(valid_annotations) < len(annotation_counts):
                n_cells_before = adata.n_obs
                mask = adata.obs[args.annotation].isin(valid_annotations)
                adata = adata[mask, :].copy()
                n_cells_after = adata.n_obs
                
                logger.info(f"  Filtered {st_name} based on annotation group size (min={min_cells_per_type})")
                logger.info(f"    - Cells before: {n_cells_before}, after: {n_cells_after}, removed: {n_cells_before - n_cells_after}")
                
                # Log which groups were removed
                removed_groups = annotation_counts[~annotation_counts.index.isin(valid_annotations)]
                if len(removed_groups) > 0:
                    logger.debug(f"    - Removed groups: {removed_groups.to_dict()}")
        
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
        
        # Use JAX with incremental writing
        logger.debug(f"Processing {n_cells} cells with JAX")
        
        # Metadata for this study
        study_metadata = {'name': st_name, 'cells': n_cells, 'study_id': st_id}
        
        batch_sum_log_ranks, batch_frac = rank_data_jax(
            X,
            n_genes,
            zarr_csr=log_ranks_zarr,
            metadata=study_metadata,
            chunk_size=1_000,
            write_interval=5  # Batch 10 chunks before writing
        )

        # Update global sums
        sum_log_ranks += batch_sum_log_ranks
        sum_frac += batch_frac
        total_cells += n_cells
        
        # Create a minimal AnnData with empty X matrix but keep obs and obsm
        # This saves memory while preserving the needed metadata
        minimal_adata = ad.AnnData(
            X=csr_matrix((adata.n_obs, n_genes), dtype=np.float32),  # Empty sparse matrix
            obs=adata.obs.copy(),
            var=pd.DataFrame(index=common_genes_transfer),
            obsm=adata.obsm.copy()  # Keep all the latent representations
        )
        
        # # Add annotation if exists
        # if args.annotation is not None and args.annotation in adata.obs.columns:
        #     minimal_adata.obs[args.annotation] = adata.obs[args.annotation]
        
        # Append to list for later concatenation
        adata_list.append(minimal_adata)
        
        # Save the ST data with embeddings (optional, keeping original behavior)
        # adata.write_h5ad(output_path)
        
        # Clean up memory
        del adata, X, minimal_adata
        gc.collect()

    log_ranks_zarr.close()
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
    
    # Concatenate all AnnData objects
    logger.info("Concatenating all AnnData objects...")
    if adata_list:
        # Concatenate along the observation axis
        concatenated_adata = ad.concat(adata_list, axis=0, join='outer', merge='same')
        
        # Ensure the var_names are the common genes
        concatenated_adata.var_names = common_genes_transfer

        # Save the concatenated AnnData
        concat_output_path = args.latent_dir / "concatenated_latent_adata.h5ad"
        concatenated_adata.write_h5ad(concat_output_path)
        logger.info(f"Concatenated AnnData saved to {concat_output_path}")
        logger.info(f"  - Total cells: {concatenated_adata.n_obs}")
        logger.info(f"  - Total genes: {concatenated_adata.n_vars}")
        logger.info(f"  - Latent representations in obsm: {list(concatenated_adata.obsm.keys())}")
        if 'slice_id' in concatenated_adata.obs.columns:
            logger.info(f"  - Number of slices: {concatenated_adata.obs['slice_id'].nunique()}")
        
        # Clean up
        del adata_list, concatenated_adata
        gc.collect()


    
