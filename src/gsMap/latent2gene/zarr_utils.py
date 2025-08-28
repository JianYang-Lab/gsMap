"""
Zarr-backed storage utilities for efficient large-scale data handling
Including ZarrBackedCSR for sparse matrices and ZarrBackedDense for dense matrices
"""

import logging
import queue
import shutil
import threading
from pathlib import Path
from typing import Optional, Tuple, Union
import time

import numpy as np
import zarr
from zarr import DirectoryStore
from scipy.sparse import csr_matrix
from zarr.sync import ThreadSynchronizer

logger = logging.getLogger(__name__)


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
                    shutil.rmtree(self.path)
            except ValueError:
                # Re-raise the complete file error
                raise
            except Exception as e:
                logger.warning(f"Could not read existing Zarr at {path}: {e}. Deleting and recreating.")
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
    def open(cls, path: Union[str, Path], mode="r", **kwargs):
        """Open an existing ZarrBackedCSR file"""
        z = zarr.open(path, mode="r")
        attrs = z.attrs
        if "ncols" not in attrs:
            raise ValueError("Invalid ZarrBackedCSR: ncols not found in attributes")
        return cls(path, ncols=attrs["ncols"], mode=mode, **kwargs)

    @classmethod
    def create(cls, path: Union[str, Path], ncols: int, mode: str = 'w', **kwargs):
        """Create a new ZarrBackedCSR file"""
        return cls(path, ncols=ncols, mode=mode, **kwargs)

    @property
    def shape(self):
        return (self.current_row, self.ncols)

    @property
    def nnz(self):
        return self.current_nnz
    
    def add_row(self, indices: np.ndarray, values: np.ndarray):
        """Add a single row to the CSR matrix"""
        if len(indices) != len(values):
            raise ValueError("Indices and values must have the same length")
        
        # Create a CSR matrix for this single row
        if len(indices) > 0:
            row_mat = csr_matrix(
                (values, indices, [0, len(indices)]),
                shape=(1, self.ncols),
                dtype=np.float32
            )
        else:
            row_mat = csr_matrix((1, self.ncols), dtype=np.float32)
        
        self.append_batch(row_mat)

    def append_batch(self, mat):
        """Queue a batch for background writing"""
        # Queue for background writing
        self.write_queue.put(mat)
    
    def _append_to_zarr(self, mat):
        """Internal method to append data to zarr arrays"""
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
        """Get rows from the CSR matrix"""
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


class ZarrBackedDense:
    """Dense matrix storage with async multi-threaded writing"""
    
    def __init__(
        self,
        path: Union[str, Path],
        shape: Tuple[int, int],
        dtype=np.float32,
        chunks: Optional[Tuple[int, int]] = None,
        mode: str = 'w',
        num_write_workers: int = 4
    ):
        self.path = Path(path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.num_write_workers = num_write_workers
        
        # Default chunks: (1, n_genes) for optimal row-wise writing
        if chunks is None:
            chunks = (1, shape[1])
        self.chunks = chunks
        
        # Initialize Zarr array with integrity checking
        if mode == 'w':
            if self.path.exists():
                # Check if it's already complete
                try:
                    existing = zarr.open(str(self.path), mode='r')
                    if 'integrity_mark' in existing.attrs and existing.attrs['integrity_mark'] == 'complete':
                        raise ValueError(
                            f"ZarrBackedDense at {self.path} already exists and is marked as complete. "
                            f"Please delete it manually if you want to overwrite: rm -rf {self.path}"
                        )
                    else:
                        logger.warning(f"ZarrBackedDense at {self.path} exists but is incomplete. Deleting and recreating.")
                        time.sleep(0.1)  # Brief pause to ensure files are released
                        shutil.rmtree(self.path, ignore_errors=True)
                        # Double-check deletion
                        if self.path.exists():
                            import os
                            os.system(f"rm -rf {self.path}")
                except ValueError:
                    # Re-raise the complete file error
                    raise
                except Exception as e:
                    logger.warning(f"Could not read existing Zarr at {self.path}: {e}. Deleting and recreating.")
                    time.sleep(0.1)
                    shutil.rmtree(self.path, ignore_errors=True)
                    if self.path.exists():
                        import os
                        os.system(f"rm -rf {self.path}")

            store = zarr.DirectoryStore(str(self.path))
            self.zarr_array = zarr.open(
                store,
                mode='w',
                shape=shape,
                dtype=dtype,
                chunks=chunks,
                compressor=zarr.Blosc(cname='lz4', clevel=3)
            )
            # Mark as incomplete initially
            self.zarr_array.attrs['integrity_mark'] = 'incomplete'
        else:
            # Read mode - validate integrity
            self.zarr_array = zarr.open(str(self.path), mode='r')
            if 'integrity_mark' not in self.zarr_array.attrs:
                raise ValueError(f"ZarrBackedDense at {self.path} is incomplete or corrupted (no integrity mark).")
            if self.zarr_array.attrs['integrity_mark'] != 'complete':
                raise ValueError(f"ZarrBackedDense at {self.path} is incomplete (marked as {self.zarr_array.attrs['integrity_mark']}).")
        
        # Async writing setup (only if still in write mode)
        self.write_queue = queue.Queue(maxsize=100)
        self.writer_threads = []
        self.stop_writer = threading.Event()
        
        if mode == 'w' and self.mode == 'w':  # Only start writer if we're actually writing
            self._start_writer_threads()
    
    def _start_writer_threads(self):
        """Start multiple background writer threads"""
        def writer_worker(worker_id):
            while not self.stop_writer.is_set():
                try:
                    item = self.write_queue.get(timeout=0.1)
                    if item is None:
                        break
                    data, row_slice, col_slice = item
                    self.zarr_array[row_slice, col_slice] = data
                    self.write_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Writer thread {worker_id} error: {e}")
                    raise
        
        # Start multiple writer threads
        for i in range(self.num_write_workers):
            thread = threading.Thread(target=writer_worker, args=(i,), daemon=True)
            thread.start()
            self.writer_threads.append(thread)
        logger.info(f"Started {self.num_write_workers} writer threads for ZarrBackedDense")
    
    def write_batch(self, data: np.ndarray, row_indices: Union[int, np.ndarray], col_slice=slice(None)):
        """Queue batch for async writing
        
        Args:
            data: Data to write
            row_indices: Either a single row index or array of row indices
            col_slice: Column slice (default: all columns)
        """
        if self.mode != 'w':
            logger.warning("Cannot write to read-only ZarrBackedDense")
            return
        
        # Handle both single index and array of indices
        if isinstance(row_indices, (int, np.integer)):
            row_slice = slice(row_indices, row_indices + data.shape[0])
        else:
            row_slice = row_indices
        
        self.write_queue.put((data, row_slice, col_slice))
    
    def mark_complete(self):
        """Mark the zarr array as complete by setting integrity mark."""
        if self.mode == 'w':
            self.zarr_array.attrs['integrity_mark'] = 'complete'
            logger.info(f"Marked ZarrBackedDense at {self.path} as complete")
    
    def close(self):
        """Clean up resources"""
        if self.writer_threads:
            logger.info("Closing ZarrBackedDense, waiting for writes...")
            self.write_queue.join()
            self.stop_writer.set()
            # Send stop signal to all threads
            for _ in self.writer_threads:
                self.write_queue.put(None)
            # Wait for all threads to finish
            for thread in self.writer_threads:
                thread.join(timeout=5.0)
        
        # Mark as complete when closing in write mode
        if self.mode == 'w':
            self.mark_complete()