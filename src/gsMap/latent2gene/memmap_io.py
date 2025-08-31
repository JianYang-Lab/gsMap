"""
Memory-mapped I/O utilities for efficient large-scale data handling
Replaces Zarr-backed storage with NumPy memory maps for better performance
"""

import logging
import json
import queue
import shutil
import threading
from pathlib import Path
from typing import Optional, Tuple, Union
import time

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


class MemMapDense:
    """Dense matrix storage using NumPy memory maps with async multi-threaded writing"""
    
    def __init__(
        self,
        path: Union[str, Path],
        shape: Tuple[int, int],
        dtype=np.float32,
        mode: str = 'w',
        num_write_workers: int = 4,
        flush_interval: float = 30,
    ):
        """
        Initialize a memory-mapped dense matrix.
        
        Args:
            path: Path to the memory-mapped file (without extension)
            shape: Shape of the matrix (n_rows, n_cols)
            dtype: Data type of the matrix
            mode: 'w' for write (create/overwrite), 'r' for read, 'r+' for read/write
            num_write_workers: Number of worker threads for async writing
        """
        self.path = Path(path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.num_write_workers = num_write_workers
        self.flush_interval = flush_interval
        # File paths
        self.data_path = self.path.with_suffix('.dat')
        self.meta_path = self.path.with_suffix('.meta.json')
        
        # Initialize memory map
        if mode == 'w':
            self._create_memmap()
        elif mode == 'r':
            self._open_memmap_readonly()
        elif mode == 'r+':
            self._open_memmap_readwrite()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'w', 'r', or 'r+'")
        
        # Async writing setup (only for write modes)
        self.write_queue = queue.Queue(maxsize=100)
        self.writer_threads = []
        self.stop_writer = threading.Event()

        if mode in ('w', 'r+'):
            self._start_writer_threads()
    
    def _create_memmap(self):
        """Create a new memory-mapped file"""
        # Check if already exists and is complete
        if self.meta_path.exists():
            try:
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get('complete', False):
                    raise ValueError(
                        f"MemMapDense at {self.path} already exists and is marked as complete. "
                        f"Please delete it manually if you want to overwrite: rm {self.data_path} {self.meta_path}"
                    )
                else:
                    logger.warning(f"MemMapDense at {self.path} exists but is incomplete. Recreating.")
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Invalid metadata at {self.meta_path}. Recreating.")
        
        # Create new memory map
        self.memmap = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode='w+',
            shape=self.shape
        )
        
        # # Initialize to zeros
        # self.memmap[:] = 0
        # self.memmap.flush()
        
        # Write metadata
        meta = {
            'shape': self.shape,
            'dtype': np.dtype(self.dtype).name,  # Use dtype.name for proper serialization
            'complete': False,
            'created_at': time.time()
        }
        with open(self.meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        logger.info(f"Created MemMapDense at {self.data_path} with shape {self.shape}")
    
    def _open_memmap_readonly(self):
        """Open an existing memory-mapped file in read-only mode"""
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")
        
        # Read metadata
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        
        if not meta.get('complete', False):
            raise ValueError(f"MemMapDense at {self.path} is incomplete")
        
        # Validate shape and dtype
        if tuple(meta['shape']) != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {tuple(meta['shape'])}"
            )
        
        # Open memory map
        self.memmap = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode='r',
            shape=self.shape
        )
        
        logger.info(f"Opened MemMapDense at {self.data_path} in read-only mode")
    
    def _open_memmap_readwrite(self):
        """Open an existing memory-mapped file in read-write mode"""
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")
        
        # Read metadata
        with open(self.meta_path, 'r') as f:
            meta = json.load(f)
        
        # Open memory map
        self.memmap = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode='r+',
            shape=tuple(meta['shape'])
        )
        
        logger.info(f"Opened MemMapDense at {self.data_path} in read-write mode")
    
    def _start_writer_threads(self):
        """Start multiple background writer threads"""
        def writer_worker(worker_id):
            last_flush_time = time.time()  # Track last flush time for worker 0
            
            while not self.stop_writer.is_set():
                try:
                    item = self.write_queue.get(timeout=1)
                    if item is None:
                        break
                    data, row_indices, col_slice = item
                    
                    # Write data with thread safety
                    if isinstance(row_indices, slice):
                        self.memmap[row_indices, col_slice] = data
                    elif isinstance(row_indices, (int, np.integer)):
                        start_row = row_indices
                        end_row = start_row + data.shape[0]
                        self.memmap[start_row:end_row, col_slice] = data
                    else:
                        # Handle array of indices
                        self.memmap[row_indices, col_slice] = data

                    # Periodic flush every 1 second for worker 0
                    if worker_id == 0:
                        current_time = time.time()
                        if current_time - last_flush_time >= self.flush_interval:
                            self.memmap.flush()
                            last_flush_time = time.time()
                            logger.debug(f"Worker 0 flushed memmap at {last_flush_time:.2f}")

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
        logger.info(f"Started {self.num_write_workers} writer threads for MemMapDense")
    
    def write_batch(self, data: np.ndarray, row_indices: Union[int, slice, np.ndarray], col_slice=slice(None)):
        """Queue batch for async writing
        
        Args:
            data: Data to write
            row_indices: Either a single row index, slice, or array of row indices
            col_slice: Column slice (default: all columns)
        """
        if self.mode not in ('w', 'r+'):
            logger.warning("Cannot write to read-only MemMapDense")
            return
        
        self.write_queue.put((data, row_indices, col_slice))
    
    def read_batch(self, row_indices: Union[int, slice, np.ndarray], col_slice=slice(None)) -> np.ndarray:
        """Read batch of data
        
        Args:
            row_indices: Row indices to read
            col_slice: Column slice (default: all columns)
            
        Returns:
            NumPy array with the requested data
        """
        if isinstance(row_indices, (int, np.integer)):
            return self.memmap[row_indices:row_indices+1, col_slice].copy()
        else:
            return self.memmap[row_indices, col_slice].copy()
    
    def __getitem__(self, key):
        """Direct array access for compatibility"""
        return self.memmap[key]
    
    def __setitem__(self, key, value):
        """Direct array access for compatibility"""
        if self.mode not in ('w', 'r+'):
            raise ValueError("Cannot write to read-only MemMapDense")
        self.memmap[key] = value
    
    def mark_complete(self):
        """Mark the memory map as complete"""
        if self.mode in ('w', 'r+'):
            logger.info("Marking memmap as complete")
            # Ensure all writes are flushed
            if self.writer_threads and not self.write_queue.empty():
                logger.info("Waiting for remaining writes before marking complete...")
                self.write_queue.join()
            
            # Flush memory map to disk
            logger.info("Flushing memmap to disk...")
            self.memmap.flush()
            logger.info("Memmap flush complete")
            
            # Update metadata
            with open(self.meta_path, 'r') as f:
                meta = json.load(f)
            meta['complete'] = True
            meta['completed_at'] = time.time()
            # Ensure dtype is properly serialized
            if 'dtype' in meta and not isinstance(meta['dtype'], str):
                meta['dtype'] = np.dtype(self.dtype).name
            with open(self.meta_path, 'w') as f:
                json.dump(meta, f, indent=2)
            
            logger.info(f"Marked MemMapDense at {self.path} as complete")

    def close(self):
        """Clean up resources"""
        logger.info("MemMapDense.close() called")
        if self.writer_threads:
            logger.info("Closing MemMapDense: waiting for queued writes...")
            self.write_queue.join()
            logger.info("All queued writes have been processed")
            self.stop_writer.set()
            logger.info("Stop signal set for writer threads")

            # Send stop signal to all threads
            for _ in self.writer_threads:
                self.write_queue.put(None)
            logger.info("Stop sentinels queued for writer threads")

            # Wait for all threads to finish
            for thread in self.writer_threads:
                thread.join(timeout=5.0)

        # Final flush
        if self.mode in ('w', 'r+'):
            self.mark_complete()

    def __enter__(self):
        return self

    @property
    def attrs(self):
        """Compatibility property for accessing metadata"""
        if hasattr(self, '_attrs'):
            return self._attrs
        
        if self.meta_path.exists():
            with open(self.meta_path, 'r') as f:
                self._attrs = json.load(f)
        else:
            self._attrs = {}
        return self._attrs
    
    def delete(self):
        """Delete the memory-mapped files"""
        self.close()
        if self.data_path.exists():
            self.data_path.unlink()
        if self.meta_path.exists():
            self.meta_path.unlink()
        logger.info(f"Deleted MemMapDense files at {self.path}")


class ParallelMemMapReader:
    """Multi-threaded reader for memory-mapped data"""
    
    def __init__(
        self,
        memmap: Union[MemMapDense, str, Path],
        num_workers: int = 4
    ):
        """
        Initialize parallel reader for memory-mapped data.
        
        Args:
            memmap: MemMapDense instance or path to memory map
            num_workers: Number of worker threads
        """
        if isinstance(memmap, (str, Path)):
            # Open as MemMapDense in read mode
            path = Path(memmap)
            # Get shape from metadata
            meta_path = path.with_suffix('.meta.json')
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            self.memmap = MemMapDense(
                path=path,
                shape=tuple(meta['shape']),
                dtype=np.dtype(meta['dtype']),
                mode='r'
            )
            self.data = self.memmap.memmap
        else:
            self.memmap = memmap
            self.data = memmap.memmap
        
        self.shape = self.data.shape
        self.num_workers = num_workers
        
        # Queues for communication
        self.read_queue = queue.Queue()
        self.result_queue = queue.Queue(maxsize=self.num_workers * 4)
        
        # Start worker threads
        self.workers = []
        self.stop_workers = threading.Event()
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker(self, worker_id: int):
        """Worker thread for reading batches"""
        logger.info(f"Reader worker {worker_id} started")
        
        while not self.stop_workers.is_set():
            try:
                # Get batch request
                item = self.read_queue.get(timeout=0.1)
                if item is None:
                    break
                
                batch_id, indices = item
                
                # Read data
                if len(indices.shape) == 2:
                    # Handle 2D neighbor indices
                    flat_indices = np.unique(indices.flatten())
                    data = self.data[flat_indices].copy()
                else:
                    # Handle 1D indices
                    data = self.data[indices].copy()
                
                # Queue result
                self.result_queue.put((batch_id, indices, data))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Reader worker {worker_id} error: {e}")
                self.result_queue.put((None, None, e))
    
    def queue_read(self, batch_id: int, indices: np.ndarray):
        """Queue a read request"""
        self.read_queue.put((batch_id, indices))
    
    def get_result(self, timeout: Optional[float] = None):
        """Get a read result"""
        return self.result_queue.get(timeout=timeout)
    
    def close(self):
        """Clean up resources"""
        self.stop_workers.set()
        for _ in self.workers:
            self.read_queue.put(None)
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        if hasattr(self, 'memmap'):
            self.memmap.close()