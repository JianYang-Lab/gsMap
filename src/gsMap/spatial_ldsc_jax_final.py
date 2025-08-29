"""
JAX-optimized implementation of spatial LDSC.
"""

import gc
import json
import logging
import os
import psutil
import queue
import threading
import multiprocessing as mp
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Tuple, Dict, List, Callable
import time
from datetime import datetime

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from functools import partial
from jax import jit, vmap
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import zarr

from .config import SpatialLDSCConfig
from .utils.regression_read import _read_ref_ld_v2, _read_sumstats, _read_w_ld

logger = logging.getLogger("gsMap.spatial_ldsc_jax")

# Configure JAX for optimal performance and memory efficiency
jax.config.update('jax_enable_x64', False)  # Use float32 for speed and memory efficiency

# Platform selection - comment/uncomment as needed
# jax.config.update('jax_platform_name', 'cpu')  # Force CPU usage
# jax.config.update('jax_platform_name', 'gpu')  # Force GPU usage

# Memory configuration for environments with limited resources
# os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
# os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')

# ============================================================================
# Chunk metadata
# ============================================================================

class ChunkMetadata:
    """Metadata for chunk processing results using absolute positions."""
    
    def __init__(self, n_spots: int, n_snps: int, trait_name: str, project_name: str,
                 start_spot: int, end_spot: int, total_spots: int):
        # Position-based metadata (required)
        self.start_spot = start_spot  # Absolute start position in global adata
        self.end_spot = end_spot      # Absolute end position in global adata (exclusive)
        self.total_spots = total_spots  # Total spots in the global adata
        self.n_spots = n_spots  # Number of spots in this chunk
        
        # Processing metadata
        self.n_snps = n_snps
        self.trait_name = trait_name
        self.project_name = project_name
        
        # System metadata
        self.timestamp = datetime.now().isoformat()
        self.hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
        self.pid = os.getpid()
        
    def to_dict(self) -> dict:
        return vars(self)
    
    def get_filename(self, extension: str = 'csv.gz') -> str:
        """Generate filename using absolute position ranges."""
        return (f"{self.project_name}_{self.trait_name}_"
                f"start{self.start_spot}_end{self.end_spot}_total{self.total_spots}.{extension}")


# ============================================================================
# Memory monitoring
# ============================================================================

def log_memory_usage(message=""):
    """Log current memory usage."""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        rss_gb = mem_info.rss / 1024**3
        logger.debug(f"Memory usage {message}: {rss_gb:.2f} GB")
        return rss_gb
    except:
        return 0.0


# ============================================================================
# Core computational functions
# ============================================================================

def prepare_snp_data_for_blocks(data: dict, n_blocks: int) -> dict:
    """Prepare SNP-related data arrays for equal-sized blocks."""
    if 'chisq' in data:
        n_snps = len(data['chisq'])
    elif 'N' in data:
        n_snps = len(data['N'])
    else:
        raise ValueError("Cannot determine number of SNPs from data")
    
    block_size = n_snps // n_blocks
    n_snps_used = block_size * n_blocks
    n_dropped = n_snps - n_snps_used
    
    if n_dropped > 0:
        logger.info(f"Truncating SNP data: dropping {n_dropped} SNPs "
                   f"({n_dropped/n_snps*100:.3f}%) for {n_blocks} blocks of size {block_size}")
    
    truncated = {}
    snp_keys = ['baseline_ld_sum', 'w_ld', 'chisq', 'N']
    
    for key, value in data.items():
        if key in snp_keys and isinstance(value, (np.ndarray, jnp.ndarray)):
            truncated[key] = value[:n_snps_used]
        elif key == 'baseline_ld':
            truncated[key] = value.iloc[:n_snps_used]
        else:
            truncated[key] = value
    
    truncated['block_size'] = block_size
    truncated['n_blocks'] = n_blocks
    truncated['n_snps_used'] = n_snps_used
    truncated['n_snps_original'] = n_snps
    
    return truncated


@partial(jit, static_argnums=(0, 1))
def process_chunk_jit(n_blocks: int,
                      batch_size: int,
                      spatial_ld: jnp.ndarray,
                      baseline_ld_sum: jnp.ndarray, 
                      chisq: jnp.ndarray,
                      N: jnp.ndarray,
                      baseline_ann: jnp.ndarray,
                      w_ld: jnp.ndarray,
                      Nbar: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Process an entire chunk of spots with JIT compilation and batch processing.
    Processes spots in batches to reduce memory usage.
    """
    def process_single_spot(spot_ld):
        """Process a single spot."""
        # Compute initial weights
        x_tot = spot_ld + baseline_ld_sum
        
        # Aggregate for weight calculation
        hsq = 10000.0 * (jnp.mean(chisq) - 1.0) / jnp.mean(x_tot * N)
        hsq = jnp.clip(hsq, 0.0, 1.0)
        
        # Compute weights efficiently
        ld_clip = jnp.maximum(x_tot, 1.0)
        w_ld_clip = jnp.maximum(w_ld, 1.0)
        c = hsq * N / 10000.0
        weights = jnp.sqrt(1.0 / (2 * jnp.square(1.0 + c * ld_clip) * w_ld_clip))
        
        # Scale weights
        weights = weights.reshape(-1, 1)
        weights_scaled = weights / jnp.sum(weights)
        
        # Apply weights and combine features
        x_focal = jnp.concatenate([
            (spot_ld.reshape(-1, 1) * weights_scaled),
            (baseline_ann * weights_scaled)
        ], axis=1)
        y_weighted = chisq.reshape(-1, 1) * weights_scaled
        
        # Reshape for block computation
        n_snps_used = x_focal.shape[0]
        block_size = n_snps_used // n_blocks
        
        x_blocks = x_focal.reshape(n_blocks, block_size, -1)
        y_blocks = y_weighted.reshape(n_blocks, block_size, -1)
        
        # Compute block values
        xty_blocks = jnp.einsum('nbp,nb->np', x_blocks, y_blocks.squeeze())
        xtx_blocks = jnp.einsum('nbp,nbq->npq', x_blocks, x_blocks)
        
        # Jackknife regression
        xty_total = jnp.sum(xty_blocks, axis=0)
        xtx_total = jnp.sum(xtx_blocks, axis=0)
        est = jnp.linalg.solve(xtx_total, xty_total)
        
        # Delete-one estimates using vectorized solve
        xty_del = xty_total - xty_blocks
        xtx_del = xtx_total - xtx_blocks
        delete_ests = jnp.linalg.solve(xtx_del, xty_del[..., None]).squeeze(-1)
        
        # Pseudovalues and standard error
        pseudovalues = n_blocks * est - (n_blocks - 1) * delete_ests
        jknife_est = jnp.mean(pseudovalues, axis=0)
        jknife_cov = jnp.cov(pseudovalues.T, ddof=1) / n_blocks
        jknife_se = jnp.sqrt(jnp.diag(jknife_cov))
        
        # Return spatial coefficient (first element)
        return jknife_est[0] / Nbar, jknife_se[0] / Nbar
    
    # Process in batches to reduce memory usage
    n_spots = spatial_ld.shape[1]
    
    if batch_size == 0 or batch_size >= n_spots:
        # Process all spots at once (batch_size=0 means no batching)
        betas, ses = vmap(process_single_spot, in_axes=1, out_axes=0)(spatial_ld)
    else:
        # Process in smaller batches
        betas_list = []
        ses_list = []
        
        for start_idx in range(0, n_spots, batch_size):
            end_idx = min(start_idx + batch_size, n_spots)
            batch_ld = spatial_ld[:, start_idx:end_idx]
            
            batch_betas, batch_ses = vmap(process_single_spot, in_axes=1, out_axes=0)(batch_ld)
            betas_list.append(batch_betas)
            ses_list.append(batch_ses)
        
        betas = jnp.concatenate(betas_list)
        ses = jnp.concatenate(ses_list)
    
    return betas, ses


# ============================================================================
# Data loading and preparation
# ============================================================================

def load_and_prepare_data(config: SpatialLDSCConfig, 
                         trait_name: str,
                         sumstats_file: str) -> Tuple[dict, pd.Index]:
    """Load and prepare all data for a single trait."""
    logger.info(f"Loading data for {trait_name}...")
    
    log_memory_usage("before loading data")
    
    # Load weights and baseline LD scores
    w_ld = _read_w_ld(config.w_file)
    w_ld.set_index("SNP", inplace=True)
    
    # Use ldscore_save_dir which is set to quick_mode_resource_dir when in quick mode
    baseline_ld_path = f"{config.ldscore_save_dir}/baseline/baseline."
    baseline_ld = _read_ref_ld_v2(baseline_ld_path)
    
    log_memory_usage("after loading baseline")
    
    # Find common SNPs
    common_snps = baseline_ld.index.intersection(w_ld.index)
    
    # Load and process summary statistics
    sumstats = _read_sumstats(fh=sumstats_file, alleles=False, dropna=False)
    sumstats.set_index("SNP", inplace=True)
    sumstats = sumstats.astype(np.float32)
    
    # Filter by chi-squared
    chisq_max = config.chisq_max
    if chisq_max is None:
        chisq_max = max(0.001 * sumstats.N.max(), 80)
    sumstats["chisq"] = sumstats.Z ** 2
    sumstats = sumstats[sumstats.chisq < chisq_max]
    logger.info(f"Filtered to {len(sumstats)} SNPs with chi^2 < {chisq_max}")
    
    # Find common SNPs with sumstats
    common_snps = common_snps.intersection(sumstats.index)
    logger.info(f"Common SNPs: {len(common_snps)}")
    
    if len(common_snps) < 200000:
        logger.warning(f"WARNING: Only {len(common_snps)} common SNPs")
    
    # Get SNP positions
    snp_positions = baseline_ld.index.get_indexer(common_snps)
    
    # Subset all data to common SNPs
    baseline_ld = baseline_ld.loc[common_snps]
    w_ld = w_ld.loc[common_snps]
    sumstats = sumstats.loc[common_snps]
    
    # Load additional baseline if needed
    if config.use_additional_baseline_annotation:
        # Use ldscore_save_dir which points to the correct directory
        additional_path = f"{config.ldscore_save_dir}/additional_baseline/baseline."
        additional_ld = _read_ref_ld_v2(additional_path)
        additional_ld = additional_ld.loc[common_snps]
        baseline_ld = pd.concat([baseline_ld, additional_ld], axis=1)
    
    # Prepare data dictionary
    data = {
        'baseline_ld': baseline_ld,
        'baseline_ld_sum': baseline_ld.sum(axis=1).values.astype(np.float32),
        'w_ld': w_ld.LD_weights.values.astype(np.float32),
        'sumstats': sumstats,
        'chisq': sumstats.chisq.values.astype(np.float32),
        'N': sumstats.N.values.astype(np.float32),
        'Nbar': np.float32(sumstats.N.mean()),
        'snp_positions': snp_positions
    }
    
    return data, common_snps


# ============================================================================
# Result Accumulator
# ============================================================================

class ResultAccumulator:
    """Accumulates chunk results for spatial LDSC without checkpointing."""
    
    def __init__(self, output_dir: Path, config: SpatialLDSCConfig, 
                 trait_name: str, n_snps_used: int, quick_handler=None):
        self.output_dir = output_dir
        self.config = config
        self.trait_name = trait_name
        self.n_snps_used = n_snps_used
        self.quick_handler = quick_handler
        
        # Storage for accumulated results
        self.results = []  # List of (chunk_idx, betas, ses, spot_names, metadata) tuples
        self.processed_chunks = set()  # Track which chunks have been processed
        
        # Track the actual range of spots covered
        self.min_spot_start = float('inf')
        self.max_spot_end = 0
        self.total_spots = None
        
    def add_chunk_result(self, chunk_idx: int, betas: np.ndarray, ses: np.ndarray,
                        spot_names: pd.Index) -> None:
        """Add chunk result to accumulated results."""
        # Get absolute spot positions
        if self.quick_handler and hasattr(self.quick_handler, 'get_chunk_spot_range'):
            start_spot, end_spot, total_spots = self.quick_handler.get_chunk_spot_range(chunk_idx)
        else:
            # Fallback for non-quick mode
            start_spot = chunk_idx * len(spot_names)
            end_spot = start_spot + len(spot_names)
            total_spots = -1  # Unknown
        
        # Update the overall range covered
        self.min_spot_start = min(self.min_spot_start, start_spot)
        self.max_spot_end = max(self.max_spot_end, end_spot)
        if self.total_spots is None:
            self.total_spots = total_spots
        
        # Create metadata
        metadata = ChunkMetadata(
            n_spots=len(spot_names),
            n_snps=self.n_snps_used,
            trait_name=self.trait_name,
            project_name=self.config.project_name,
            start_spot=start_spot,
            end_spot=end_spot,
            total_spots=total_spots
        )
        
        # Store result
        self.results.append((chunk_idx, betas, ses, spot_names, metadata))
        self.processed_chunks.add(chunk_idx)
        
    def get_merged_results(self) -> pd.DataFrame:
        """Merge all accumulated results into a single DataFrame."""
        if not self.results:
            raise ValueError("No results to merge")
        
        # Sort results by chunk index to ensure correct order
        sorted_results = sorted(self.results, key=lambda x: x[0])
        
        dfs = []
        for chunk_idx, betas, ses, spot_names, metadata in sorted_results:
            # Convert to float64 for accurate p-value calculation
            betas_f64 = betas.astype(np.float64)
            ses_f64 = ses.astype(np.float64)
            
            # Calculate statistics
            z_scores = betas_f64 / ses_f64
            p_values = norm.sf(z_scores)
            log10_p = -np.log10(np.maximum(p_values, 1e-300))
            
            results_df = pd.DataFrame({
                'spot': spot_names,
                'beta': betas,
                'se': ses,
                'z': z_scores.astype(np.float32),
                'p': p_values,
                'neg_log10_p': log10_p
            })
            dfs.append(results_df)
        
        merged_df = pd.concat(dfs, ignore_index=True)
        return merged_df
    
    def get_coverage_filename(self, base_name: str) -> str:
        """Generate filename with coverage range if incomplete."""
        if self.total_spots is not None and self.total_spots > 0:
            # Check if we have complete coverage
            if self.min_spot_start == 0 and self.max_spot_end == self.total_spots:
                # Complete coverage - use simple name
                return f"{base_name}.csv.gz"
            else:
                # Partial coverage - include range in filename
                return f"{base_name}_start{self.min_spot_start}_end{self.max_spot_end}_total{self.total_spots}.csv.gz"
        else:
            # No total spots info - just use the range we have
            return f"{base_name}_start{self.min_spot_start}_end{self.max_spot_end}.csv.gz"
    
    def is_complete(self, total_chunks: int) -> bool:
        """Check if all chunks have been processed."""
        return len(self.processed_chunks) == total_chunks


# ============================================================================
# Producer-Consumer Pattern with Writer
# ============================================================================

class ChunkProducer(threading.Thread):
    """Thread that loads spatial LD chunks and puts them in a queue."""
    
    def __init__(self, chunk_queue: queue.Queue, config: SpatialLDSCConfig,
                 data: dict, chunk_indices: List[int], quick_handler=None,
                 worker_id: int = 0):
        super().__init__()
        self.chunk_queue = chunk_queue
        self.config = config
        self.data = data
        self.chunk_indices = chunk_indices
        self.quick_handler = quick_handler
        self.worker_id = worker_id
        self.daemon = True
    
    def run(self):
        """Load chunks and put them in the queue."""
        for chunk_idx in self.chunk_indices:
            try:
                chunk_data = self.load_chunk(chunk_idx)
                self.chunk_queue.put((chunk_idx, chunk_data))
            except Exception as e:
                logger.error(f"Worker {self.worker_id}: Error loading chunk {chunk_idx}: {e}")
                self.chunk_queue.put((chunk_idx, None))
        
        # Signal completion from this worker
        self.chunk_queue.put((f"DONE_{self.worker_id}", None))
    
    def load_chunk(self, chunk_index: int) -> Tuple[jnp.ndarray, pd.Index]:
        """Load a single chunk of spatial LD scores."""
        n_snps_used = self.data['n_snps_used']
        
        if self.config.ldscore_save_format == "feather":
            # Use ldscore_save_dir from config
            ld_file = f"{self.config.ldscore_save_dir}/{self.config.project_name}_chunk{chunk_index}/{self.config.project_name}."
            ref_ld = _read_ref_ld_v2(ld_file)
            ref_ld = ref_ld.iloc[self.data['snp_positions']]
            spatial_ld = ref_ld.values.astype(np.float32)
            spot_names = ref_ld.columns
            del ref_ld
            
        elif self.config.ldscore_save_format == "quick_mode":
            if self.quick_handler is None:
                raise ValueError("quick_handler required for quick mode")
            spatial_ld, spot_names = self.quick_handler.fetch_ldscore_by_chunk(chunk_index)  # chunk_index is already 0-based
        else:
            raise ValueError(f"Unsupported format: {self.config.ldscore_save_format}")
        
        # Truncate spatial LD
        spatial_ld = spatial_ld[:n_snps_used]
        
        # Convert to JAX array - this is fast, actual conversion happens on GPU
        return spatial_ld, spot_names


class ParallelChunkLoader:
    """Manages multiple producer threads for parallel chunk loading."""
    
    def __init__(self, config: SpatialLDSCConfig, data: dict, 
                 chunk_indices: List[int], quick_handler=None,
                 n_workers: int = 2):
        self.config = config
        self.data = data
        self.chunk_indices = chunk_indices
        self.quick_handler = quick_handler
        self.n_workers = min(n_workers, len(chunk_indices))
        self.workers = []
        self.chunk_queue = queue.Queue(maxsize=self.n_workers * 10)  # Increased queue size for buffering
        
    def start(self):
        """Start all producer threads."""
        # Divide chunks among workers
        chunks_per_worker = len(self.chunk_indices) // self.n_workers
        remainder = len(self.chunk_indices) % self.n_workers
        
        start_idx = 0
        for i in range(self.n_workers):
            # Calculate chunk range for this worker
            n_chunks = chunks_per_worker + (1 if i < remainder else 0)
            end_idx = start_idx + n_chunks
            worker_chunks = self.chunk_indices[start_idx:end_idx]
            
            # Create and start worker
            worker = ChunkProducer(
                self.chunk_queue, self.config, self.data,
                worker_chunks, self.quick_handler, worker_id=i
            )
            worker.start()
            self.workers.append(worker)
            
            start_idx = end_idx
            
        logger.debug(f"Started {self.n_workers} chunk loader threads")
        
    def get_next_chunk(self):
        """Get the next available chunk from any worker."""
        return self.chunk_queue.get()

    def wait_for_completion(self):
        """Wait for all workers to complete."""
        for worker in self.workers:
            worker.join()


def process_chunks_with_queue(config: SpatialLDSCConfig,
                             data_truncated: dict,
                             chunk_indices: List[int],
                             trait_name: str,
                             output_dir: Path,
                             quick_handler=None) -> ResultAccumulator:
    """
    Process chunks using parallel producer-consumer pattern.
    
    Returns:
        ResultAccumulator with accumulated results
    """
    # Prepare static JAX arrays (same for all chunks)
    n_snps_used = data_truncated['n_snps_used']
    
    # Prepare baseline annotation
    baseline_ann = (data_truncated['baseline_ld'].values.astype(np.float32) * 
                   data_truncated['N'].reshape(-1, 1).astype(np.float32) / 
                   data_truncated['Nbar'])
    baseline_ann = np.concatenate([baseline_ann, 
                                  np.ones((n_snps_used, 1), dtype=np.float32)], axis=1)
    
    # Convert to JAX arrays (move to GPU once)
    baseline_ld_sum_jax = jnp.asarray(data_truncated['baseline_ld_sum'], dtype=jnp.float32)
    chisq_jax = jnp.asarray(data_truncated['chisq'], dtype=jnp.float32)
    N_jax = jnp.asarray(data_truncated['N'], dtype=jnp.float32)
    baseline_ann_jax = jnp.asarray(baseline_ann, dtype=jnp.float32)
    w_ld_jax = jnp.asarray(data_truncated['w_ld'], dtype=jnp.float32)
    
    del baseline_ann
    gc.collect()
    
    # Create result accumulator
    result_accumulator = ResultAccumulator(output_dir, config, trait_name, n_snps_used, quick_handler)
    
    # Determine number of loader threads based on platform
    if jax.default_backend() == 'gpu':
        n_loader_threads = 10  # More threads for GPU to keep it fed
    else:
        n_loader_threads = 2  # Fewer threads for CPU
    
    # Create parallel chunk loader
    loader = ParallelChunkLoader(
        config, data_truncated, chunk_indices, 
        quick_handler, n_workers=n_loader_threads
    )
    loader.start()
    
    # Track completion
    n_workers_done = 0
    n_chunks_processed = 0
    
    # Process chunks as they become available
    with tqdm(total=len(chunk_indices), desc="Processing chunks") as pbar:
        while n_chunks_processed < len(chunk_indices):
            # Get chunk from queue
            chunk_idx, chunk_data = loader.get_next_chunk()

            # Check for worker completion signal
            if isinstance(chunk_idx, str) and chunk_idx.startswith("DONE_"):
                n_workers_done += 1
                if n_workers_done >= loader.n_workers:
                    break
                continue
            
            if chunk_data is None:
                logger.error(f"Skipping chunk {chunk_idx} due to loading error")
                pbar.update(1)
                n_chunks_processed += 1
                continue

            spatial_ld, spot_names = chunk_data
            spatial_ld_jax = jnp.asarray(spatial_ld, dtype=jnp.float32)

            # Process chunk with JIT-compiled function
            batch_size = min(50, spot_names.shape[0])  # Adaptive batch size
            betas, ses = process_chunk_jit(
                config.n_blocks,
                batch_size,
                spatial_ld_jax,
                baseline_ld_sum_jax,
                chisq_jax,
                N_jax,
                baseline_ann_jax,
                w_ld_jax,
                data_truncated['Nbar']
            )
            # Ensure computation completes before proceeding
            betas.block_until_ready()
            ses.block_until_ready()

            # Convert to numpy (transfer from GPU to CPU)
            betas_np = np.array(betas)
            ses_np = np.array(ses)
            
            # Add chunk result to accumulator
            result_accumulator.add_chunk_result(chunk_idx, betas_np, ses_np, spot_names)
            
            # Clean up GPU memory
            del spatial_ld_jax, betas, ses
            
            pbar.update(1)
            n_chunks_processed += 1
            
            # Periodic memory monitoring
            if n_chunks_processed % 10 == 0:
                log_memory_usage(f"after {n_chunks_processed} chunks")
    
    # Wait for all loaders to finish
    loader.wait_for_completion()
    
    return result_accumulator




# ============================================================================
# Quick mode support
# ============================================================================

class QuickModeLDScore:
    """Handler for quick mode LD score loading with optimized matrix operations."""
    
    def __init__(self, config: SpatialLDSCConfig, snp_positions: np.ndarray):
        """Initialize quick mode with SNP-gene weights."""
        logger.info("Loading quick mode data...")

        if config.marker_score_format == "zarr":
            # Use the auto-generated path from ConfigWithAutoPaths
            mk_score_zarr_array_path = Path(config.marker_scores_zarr_path)
            
            if not mk_score_zarr_array_path.exists():
                raise FileNotFoundError(f"Marker scores zarr not found at {mk_score_zarr_array_path}")

            # Open zarr array in read mode - keeps data on disk
            logger.info(f"Opening marker scores zarr array from {mk_score_zarr_array_path}")
            self.mkscore_zarr = zarr.open(str(mk_score_zarr_array_path), mode='r')

            # The zarr array shape should be (n_spots, n_genes)
            n_spots_zarr, n_genes_zarr = self.mkscore_zarr.shape
            logger.info(f"Marker scores zarr shape: (n_spots={n_spots_zarr}, n_genes={n_genes_zarr})")


            # Load concatenated latent adata to get gene names and spot names
            # Use the auto-generated path from ConfigWithAutoPaths
            concat_adata_path = Path(config.concatenated_latent_adata_path)

            logger.info(f"Loading gene names and spot names from {concat_adata_path}")
            concat_adata = ad.read_h5ad(concat_adata_path, backed='r')
            gene_names_from_adata = concat_adata.var_names.to_numpy()
            self.spot_names_all = concat_adata.obs_names.to_numpy()
            
            # Filter by sample_name if provided
            if config.sample_name:
                logger.info(f"Filtering spots by sample_name: {config.sample_name}")
                # Get sample information from obs
                sample_info = concat_adata.obs['sample'].to_numpy() if 'sample' in concat_adata.obs.columns else None
                if sample_info is None:
                    logger.warning("No 'sample' column found in obs. Checking for sample_name column...")
                    sample_info = concat_adata.obs['sample_name'].to_numpy() if 'sample_name' in concat_adata.obs.columns else None
                
                if sample_info is None:
                    concat_adata.file.close()
                    raise ValueError("No 'sample' or 'sample_name' column found in concatenated_latent_adata.obs. Cannot filter by sample.")
                
                # Get indices of spots for the specified sample
                self.spot_indices = np.where(sample_info == config.sample_name)[0]
                # Verify continuous monotony (increment 1) for efficient zarr slicing
                assert self.spot_indices.tolist() == list(range(self.spot_indices[0], self.spot_indices[-1] + 1)), "Spot indices for sample_name must be continuous"

                if len(self.spot_indices) == 0:
                    concat_adata.file.close()
                    raise ValueError(f"No spots found for sample_name '{config.sample_name}'. Available samples: {np.unique(sample_info)}")
                
                # Store the absolute start offset for this sample (for contiguous slicing)
                self.sample_start_offset = self.spot_indices[0]
                
                logger.info(f"Found {len(self.spot_indices)} spots for sample '{config.sample_name}' (absolute range: [{self.sample_start_offset}, {self.sample_start_offset + len(self.spot_indices)})")
                self.spot_names_filtered = self.spot_names_all[self.spot_indices]
            else:
                # Use all spots if no sample_name specified
                self.spot_indices = np.arange(n_spots_zarr)
                self.spot_names_filtered = self.spot_names_all
                self.sample_start_offset = 0
            
            concat_adata.file.close()
            assert len(self.spot_names_all) == n_spots_zarr
            assert len(gene_names_from_adata) == n_genes_zarr
            del concat_adata
            

            # Load SNP-gene weight data
            # The path is set in config.__post_init__ when quick_mode_resource_dir is provided
            snp_gene_weight_path = Path(config.snp_gene_weight_adata_path)
            if not snp_gene_weight_path.exists():
                raise FileNotFoundError(f"SNP-gene weight matrix not found at {snp_gene_weight_path}")
            snp_gene_weight_adata = ad.read_h5ad(snp_gene_weight_path)
            
            # Find common genes between zarr and SNP-gene weights
            zarr_genes_series = pd.Series(gene_names_from_adata)
            common_genes_mask = zarr_genes_series.isin(snp_gene_weight_adata.var.index)
            common_genes = gene_names_from_adata[common_genes_mask]
            
            # Get indices for gene alignment
            self.zarr_gene_indices = np.where(common_genes_mask)[0]
            snp_gene_indices = [snp_gene_weight_adata.var.index.get_loc(g) for g in common_genes]
            
            logger.info(f"Found {len(common_genes)} common genes between marker scores and SNP-gene weights")
            
            # Extract SNP-gene weight matrix for common genes
            self.snp_gene_weight_sparse = snp_gene_weight_adata[snp_positions, snp_gene_indices].X
            
            if hasattr(self.snp_gene_weight_sparse, 'nnz'):
                logger.info(f"Using sparse SNP-gene matrix "
                           f"(shape: {self.snp_gene_weight_sparse.shape}, "
                           f"density: {self.snp_gene_weight_sparse.nnz / np.prod(self.snp_gene_weight_sparse.shape):.2%})")
            
            # Convert sparse matrix to CSR format for faster multiplication
            if hasattr(self.snp_gene_weight_sparse, 'tocsr'):
                self.snp_gene_weight_sparse = self.snp_gene_weight_sparse.tocsr()
            
            # Set up chunking for zarr reading - based on filtered spots
            self.chunk_size = config.spots_per_chunk_quick_mode
            self.n_spots_filtered = len(self.spot_indices)
            self.chunk_starts = list(range(0, self.n_spots_filtered, self.chunk_size))
            self.n_spots = n_spots_zarr  # Keep original for zarr indexing
            
            # Store flag for zarr mode
            self.use_zarr = True
            
        else:
            # Original feather-based implementation
            if config.mkscore_feather_path is None:
                raise ValueError("mkscore_feather_path must be provided when marker_score_format='feather'")
            
            mkscore_path = Path(config.mkscore_feather_path)
            if not mkscore_path.exists():
                raise FileNotFoundError(f"Marker score feather file not found at {mkscore_path}")
            
            mk_score = pd.read_feather(mkscore_path)
            mk_score.set_index("HUMAN_GENE_SYM", inplace=True)

            snp_gene_weight_path = Path(config.snp_gene_weight_adata_path)
            if not snp_gene_weight_path.exists():
                raise FileNotFoundError(f"SNP-gene weight matrix not found at {snp_gene_weight_path}")
            snp_gene_weight_adata = ad.read_h5ad(snp_gene_weight_path)
            
            common_genes = mk_score.index.intersection(snp_gene_weight_adata.var.index)
            
            self.snp_gene_weight_sparse = snp_gene_weight_adata[snp_positions, common_genes.to_list()].X
            
            if hasattr(self.snp_gene_weight_sparse, 'nnz'):
                logger.info(f"Using sparse SNP-gene matrix "
                           f"(shape: {self.snp_gene_weight_sparse.shape}, "
                           f"density: {self.snp_gene_weight_sparse.nnz / np.prod(self.snp_gene_weight_sparse.shape):.2%})")
            
            # Convert sparse matrix to CSR format for faster multiplication
            if hasattr(self.snp_gene_weight_sparse, 'tocsr'):
                self.snp_gene_weight_sparse = self.snp_gene_weight_sparse.tocsr()
            
            self.mk_score = mk_score.loc[common_genes]
            self.chunk_size = config.spots_per_chunk_quick_mode
            self.chunk_starts = list(range(0, self.mk_score.shape[1], self.chunk_size))
            
            # Pre-convert mk_score to float32 for efficiency
            self.mk_score_values = self.mk_score.values.astype(np.float32)
            self.use_zarr = False
    
    def fetch_ldscore_by_chunk(self, chunk_index: int) -> Tuple[np.ndarray, pd.Index]:
        """Fetch LD score chunk using optimized sparse matrix multiplication."""
        start = self.chunk_starts[chunk_index]
        
        if self.use_zarr:
            # Zarr-based implementation - read only the needed chunk from disk
            end = min(start + self.chunk_size, self.n_spots_filtered)
            
            # Since we verified spots are continuous, could use direct slicing
            zarr_start = self.sample_start_offset + start
            zarr_end = self.sample_start_offset + end
            
            # Select contiguous spots and aligned genes
            mk_score_chunk = self.mkscore_zarr[zarr_start:zarr_end, self.zarr_gene_indices]
            # Transpose to (n_genes, n_spots) for matrix multiplication
            mk_score_chunk = mk_score_chunk.T
            
            # Ensure it's float32 for consistency
            mk_score_chunk = mk_score_chunk.astype(np.float32)
            
            # Efficient sparse matrix multiplication
            ldscore_chunk = self.snp_gene_weight_sparse @ mk_score_chunk
            
            # Convert sparse result to dense if needed
            if hasattr(ldscore_chunk, 'toarray'):
                ldscore_chunk = ldscore_chunk.toarray()
            
            # Use actual spot names from the filtered spots
            spot_names = pd.Index(self.spot_names_filtered[start:end])
            
            return ldscore_chunk.astype(np.float32), spot_names
            
        else:
            # Original feather-based implementation
            end = min(start + self.chunk_size, self.mk_score.shape[1])
            
            # Use pre-converted float32 values
            mk_score_chunk = self.mk_score_values[:, start:end]
            
            # Efficient sparse matrix multiplication
            ldscore_chunk = self.snp_gene_weight_sparse @ mk_score_chunk
            
            # Convert sparse result to dense if needed
            if hasattr(ldscore_chunk, 'toarray'):
                ldscore_chunk = ldscore_chunk.toarray()
            
            return ldscore_chunk.astype(np.float32), self.mk_score.columns[start:end]
    
    def get_total_chunks(self) -> int:
        """Return total number of chunks."""
        return len(self.chunk_starts)
    
    def get_chunk_spot_range(self, chunk_index: int) -> Tuple[int, int, int]:
        """Get the absolute start, end positions in global adata, and total spots."""
        if self.use_zarr:
            # Get the chunk boundaries in filtered space
            start_in_filtered = self.chunk_starts[chunk_index]
            end_in_filtered = min(start_in_filtered + self.chunk_size, self.n_spots_filtered)
            
            # Convert to absolute positions in global adata
            # spot_indices contains the absolute positions of filtered spots
            absolute_start = self.spot_indices[start_in_filtered]
            absolute_end = self.spot_indices[end_in_filtered - 1] + 1 if end_in_filtered > start_in_filtered else absolute_start
            
            # Return absolute positions and total spots in original data
            return int(absolute_start), int(absolute_end), self.n_spots
        else:
            # For feather mode, positions are already absolute
            start = self.chunk_starts[chunk_index]
            end = min(start + self.chunk_size, self.mk_score.shape[1])
            return start, end, self.mk_score.shape[1]


# ============================================================================
# Main entry point
# ============================================================================

def run_spatial_ldsc_single_trait(config: SpatialLDSCConfig,
                                 trait_name: str,
                                 sumstats_file: str) -> pd.DataFrame:
    """
    Run spatial LDSC for a single trait.
    
    This function:
    1. Uses queue-based processing for efficiency
    2. Accumulates results in memory
    3. Saves a single output file with position info in filename if incomplete
    4. Returns merged DataFrame when processing completes
    
    Args:
        config: Configuration object
        trait_name: Name of the trait
        sumstats_file: Path to summary statistics file
    
    Returns:
        Merged DataFrame with results
    """
    logger.info("=" * 70)
    logger.info(f"Running Spatial LDSC (JAX-optimized Final Version)")
    logger.info(f"Project: {config.project_name}, Trait: {trait_name}")
    if config.sample_name:
        logger.info(f"Filtering by sample: {config.sample_name}")
    if config.cell_indices_range:
        logger.info(f"Cell indices range: {config.cell_indices_range}")
    logger.info("=" * 70)
    
    # Create output directory using the auto-generated path
    output_dir = config.ldsc_save_dir  # This is auto-created by @ensure_path_exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and prepare data
    data, common_snps = load_and_prepare_data(config, trait_name, sumstats_file)
    data_truncated = prepare_snp_data_for_blocks(data, config.n_blocks)
    
    # Initialize handlers
    quick_handler = None
    
    if config.ldscore_save_format == "quick_mode":
        logger.info("Initializing quick mode...")
        quick_handler = QuickModeLDScore(config, data_truncated['snp_positions'])
        total_chunks = len(quick_handler.chunk_starts)
    elif config.ldscore_save_format == "feather":
        # Check for chunk directories in ldscore_save_dir
        ldscore_dir = Path(config.ldscore_save_dir)
        if not ldscore_dir.exists():
            raise ValueError(f"ldscore_save_dir not found: {ldscore_dir}")
        chunk_dirs = [d for d in os.listdir(ldscore_dir) if "chunk" in d]
        total_chunks = len(chunk_dirs)
    else:
        raise ValueError(f"Unsupported format: {config.ldscore_save_format}")
    
    logger.info(f"Total chunks: {total_chunks}")
    
    # Determine chunk indices to process
    if config.cell_indices_range:
        # Convert cell indices to chunk indices
        start_cell, end_cell = config.cell_indices_range  # 0-based [start, end)
        chunk_size = config.spots_per_chunk_quick_mode
        
        # Calculate which chunks contain these cells
        start_chunk = start_cell // chunk_size  # 0-based chunk index
        end_chunk = (end_cell - 1) // chunk_size if end_cell > 0 else 0  # 0-based, inclusive
        
        # Validate chunk indices
        if start_chunk >= total_chunks:
            raise ValueError(f"cell_indices_range start ({start_cell}) maps to chunk {start_chunk} which is >= total chunks ({total_chunks})")
        if end_chunk >= total_chunks:
            logger.warning(f"cell_indices_range end ({end_cell}) maps to chunk {end_chunk} which is >= total chunks ({total_chunks}). Capping to last chunk.")
            end_chunk = total_chunks - 1
        
        # Create list of 0-based chunk indices
        chunk_indices = list(range(start_chunk, end_chunk + 1))
        logger.info(f"Cell range [{start_cell}, {end_cell}) maps to chunks {start_chunk}-{end_chunk} (0-based)")
    else:
        # Process all chunks (0-based indexing)
        start_chunk, end_chunk = 0, total_chunks - 1
        chunk_indices = list(range(0, total_chunks))
    
    # Process chunks with result accumulator
    start_time = time.time()
    result_accumulator = process_chunks_with_queue(
        config, data_truncated, chunk_indices, 
        trait_name, output_dir, quick_handler
    )
    elapsed_time = time.time() - start_time
    
    h, rem = divmod(elapsed_time, 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Processed {len(result_accumulator.processed_chunks)} chunks in {int(h)}h {int(m)}m {s:.2f}s")
    
    # Get merged results
    merged_df = result_accumulator.get_merged_results()
    
    # Generate appropriate filename based on coverage
    base_name = f"{config.project_name}_{trait_name}"
    output_filename = result_accumulator.get_coverage_filename(base_name)
    final_file = Path(config.ldsc_save_dir) / output_filename
    
    # Save and log statistics
    save_results_and_log_statistics(merged_df, final_file)
    
    # Log coverage information if incomplete
    if not result_accumulator.is_complete(total_chunks):
        logger.info(f"Partial results saved: spots [{result_accumulator.min_spot_start}, {result_accumulator.max_spot_end}) "
                   f"out of {result_accumulator.total_spots} total")
        logger.info("To process remaining chunks, adjust cell_indices_range accordingly")
    
    return merged_df


def save_results_and_log_statistics(merged_df: pd.DataFrame, 
                                    output_file: Path) -> None:
    """
    Save merged results and log statistical summary.
    
    Args:
        merged_df: Merged DataFrame with p-values and betas
        output_file: Path to save the compressed CSV file
    """
    # Ensure output directory exists
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save final results
    merged_df.to_csv(output_file, index=False, compression='gzip')

    # Calculate statistics
    n_spots = len(merged_df)
    bonferroni_threshold = 0.05 / n_spots
    n_bonferroni_sig = (merged_df['p'] < bonferroni_threshold).sum()
    
    # Calculate FDR for informational purposes only
    from statsmodels.stats.multitest import multipletests
    _, fdr_corrected_pvals, _, _ = multipletests(merged_df['p'], alpha=0.001, method='fdr_bh')
    n_fdr_sig = fdr_corrected_pvals.sum()
    
    # Log summary
    logger.info("=" * 70)
    logger.info(f"STATISTICAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total spots: {n_spots:,}")
    logger.info(f"Max -log10(p): {merged_df['neg_log10_p'].max():.2f}")
    logger.info("-" * 70)
    logger.info(f"Nominally significant (p < 0.05): {(merged_df['p'] < 0.05).sum():,}")
    logger.info(f"Bonferroni threshold: 0.05/{n_spots:,} = {bonferroni_threshold:.2e}")
    logger.info(f"Bonferroni-corrected significant spots (p < {bonferroni_threshold:.2e}): {n_bonferroni_sig:,}")
    logger.info(f"FDR-corrected significant spots (alpha=0.001): {n_fdr_sig:,} (informational only)")
    logger.info("=" * 70)
    logger.info(f"Final results saved to {output_file}")


def run_spatial_ldsc_jax(config: SpatialLDSCConfig):
    """
    Wrapper for compatibility with existing code.
    Processes all traits from config.sumstats_config_dict.
    """
    for trait_name, sumstats_file in config.sumstats_config_dict.items():
        run_spatial_ldsc_single_trait(config, trait_name, sumstats_file)

