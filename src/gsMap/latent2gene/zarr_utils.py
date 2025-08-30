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
                compressor=zarr.Blosc(cname='lz4', clevel=3),
                synchronizer=ThreadSynchronizer(),
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