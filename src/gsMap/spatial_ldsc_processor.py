"""
Unified spatial LDSC processor combining chunk production, parallel loading, and result accumulation.
"""

import gc
import json
import logging
import os
import queue
import threading
import multiprocessing as mp
from multiprocessing import Queue, Process
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import anndata as ad
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .config import SpatialLDSCConfig

logger = logging.getLogger("gsMap.spatial_ldsc_processor")


class SpatialLDSCProcessor:
    """
    Unified processor for spatial LDSC that combines:
    - ChunkProducer: Loading spatial LD chunks
    - ParallelChunkLoader: Managing parallel chunk loading with adjacent fetching
    - QuickModeLDScore: Handling memory-mapped marker scores
    - ResultAccumulator: Validating, merging and saving results
    """
    
    def __init__(self, 
                 config: SpatialLDSCConfig,
                 trait_name: str,
                 data_truncated: dict,
                 output_dir: Path,
                 n_loader_threads: int = 10):
        """
        Initialize the unified processor.
        
        Args:
            config: Configuration object
            trait_name: Name of the trait being processed
            data_truncated: Truncated SNP data dictionary
            output_dir: Output directory for results
            n_loader_threads: Number of parallel loader threads
        """
        self.config = config
        self.trait_name = trait_name
        self.data_truncated = data_truncated
        self.output_dir = output_dir
        self.n_loader_threads = n_loader_threads
        
        # Check marker score format
        if config.marker_score_format != "memmap":
            raise NotImplementedError(
                f"Only 'memmap' marker score format is supported. Got: {config.marker_score_format}"
            )
        
        # Initialize QuickModeLDScore components
        self._initialize_quick_mode()
        
        # Result accumulation
        self.results = []
        self.processed_chunks = set()
        self.min_spot_start = float('inf')
        self.max_spot_end = 0
        
        # Threading components
        self.chunk_queue = queue.Queue(maxsize=n_loader_threads * 2)
        self.result_queue = queue.Queue()
        self.workers = []
        
    def _initialize_quick_mode(self):
        """Initialize quick mode components for memory-mapped marker scores."""
        logger.info("Initializing memory-mapped marker scores...")
        
        # Load memory-mapped marker scores
        mk_score_data_path = Path(self.config.marker_scores_memmap_path)
        mk_score_meta_path = mk_score_data_path.with_suffix('.meta.json')
        
        if not mk_score_meta_path.exists():
            raise FileNotFoundError(f"Marker scores metadata not found at {mk_score_meta_path}")
        
        # Load metadata
        with open(mk_score_meta_path, 'r') as f:
            meta = json.load(f)
        
        n_spots_memmap = meta['shape'][0]
        n_genes_memmap = meta['shape'][1]
        dtype = np.dtype(meta['dtype'])
        
        # Open memory-mapped array
        self.mkscore_memmap = np.memmap(
            mk_score_data_path,
            dtype=dtype,
            mode='r',
            shape=(n_spots_memmap, n_genes_memmap)
        )
        
        logger.info(f"Marker scores shape: (n_spots={n_spots_memmap}, n_genes={n_genes_memmap})")
        
        # Load concatenated latent adata for metadata
        concat_adata_path = Path(self.config.concatenated_latent_adata_path)
        concat_adata = ad.read_h5ad(concat_adata_path, backed='r')
        gene_names_from_adata = concat_adata.var_names.to_numpy()
        self.spot_names_all = concat_adata.obs_names.to_numpy()
        
        # Filter by sample if specified
        if self.config.sample_name:
            logger.info(f"Filtering spots by sample_name: {self.config.sample_name}")
            sample_info = concat_adata.obs.get('sample', concat_adata.obs.get('sample_name', None))
            
            if sample_info is None:
                concat_adata.file.close()
                raise ValueError("No 'sample' or 'sample_name' column found in obs")
            
            sample_info = sample_info.to_numpy()
            self.spot_indices = np.where(sample_info == self.config.sample_name)[0]
            
            # Verify spots are contiguous for efficient slicing
            expected_range = list(range(self.spot_indices[0], self.spot_indices[-1] + 1))
            if self.spot_indices.tolist() != expected_range:
                concat_adata.file.close()
                raise ValueError("Spot indices for sample must be contiguous")
            
            self.sample_start_offset = self.spot_indices[0]
            self.spot_names_filtered = self.spot_names_all[self.spot_indices]
            logger.info(f"Found {len(self.spot_indices)} spots for sample '{self.config.sample_name}'")
        else:
            self.spot_indices = np.arange(n_spots_memmap)
            self.spot_names_filtered = self.spot_names_all
            self.sample_start_offset = 0
        
        concat_adata.file.close()
        self.n_spots = n_spots_memmap
        self.n_spots_filtered = len(self.spot_indices)
        
        # Load SNP-gene weights
        snp_gene_weight_path = Path(self.config.snp_gene_weight_adata_path)
        if not snp_gene_weight_path.exists():
            raise FileNotFoundError(f"SNP-gene weight matrix not found at {snp_gene_weight_path}")
        
        snp_gene_weight_adata = ad.read_h5ad(snp_gene_weight_path)
        
        # Find common genes
        memmap_genes_series = pd.Series(gene_names_from_adata)
        common_genes_mask = memmap_genes_series.isin(snp_gene_weight_adata.var.index)
        common_genes = gene_names_from_adata[common_genes_mask]
        
        self.memmap_gene_indices = np.where(common_genes_mask)[0]
        snp_gene_indices = [snp_gene_weight_adata.var.index.get_loc(g) for g in common_genes]
        
        logger.info(f"Found {len(common_genes)} common genes")
        
        # Get SNP positions from data_truncated
        snp_positions = self.data_truncated.get('snp_positions', None)
        if snp_positions is None:
            raise ValueError("snp_positions not found in data_truncated")
        
        # Extract SNP-gene weight matrix
        self.snp_gene_weight_sparse = snp_gene_weight_adata[snp_positions, snp_gene_indices].X
        
        if hasattr(self.snp_gene_weight_sparse, 'tocsr'):
            self.snp_gene_weight_sparse = self.snp_gene_weight_sparse.tocsr()
        
        # Set up chunking
        self.chunk_size = self.config.spots_per_chunk_quick_mode
        
        # Handle cell indices range if specified
        if self.config.cell_indices_range:
            start_cell, end_cell = self.config.cell_indices_range
            # Adjust for filtered spots
            start_cell = max(0, start_cell)
            end_cell = min(end_cell, self.n_spots_filtered)
            self.chunk_starts = list(range(start_cell, end_cell, self.chunk_size))
            logger.info(f"Processing cell range [{start_cell}, {end_cell})")
        else:
            self.chunk_starts = list(range(0, self.n_spots_filtered, self.chunk_size))
        
        self.total_chunks = len(self.chunk_starts)
        logger.info(f"Total chunks to process: {self.total_chunks}")
        
    def _fetch_ldscore_chunk(self, chunk_index: int) -> Tuple[np.ndarray, pd.Index, int, int]:
        """
        Fetch LD score chunk for given index.
        
        Returns:
            Tuple of (ldscore_array, spot_names, absolute_start, absolute_end)
        """
        if chunk_index >= len(self.chunk_starts):
            raise ValueError(f"Invalid chunk index {chunk_index}")
        
        start = self.chunk_starts[chunk_index]
        end = min(start + self.chunk_size, self.n_spots_filtered)
        
        # Calculate absolute positions in memmap
        memmap_start = self.sample_start_offset + start
        memmap_end = self.sample_start_offset + end
        
        # Load chunk from memmap
        mk_score_chunk = self.mkscore_memmap[memmap_start:memmap_end, self.memmap_gene_indices]
        mk_score_chunk = mk_score_chunk.T.astype(np.float32)
        
        # Compute LD scores via sparse matrix multiplication
        ldscore_chunk = self.snp_gene_weight_sparse @ mk_score_chunk
        
        if hasattr(ldscore_chunk, 'toarray'):
            ldscore_chunk = ldscore_chunk.toarray()
        
        # Get spot names
        spot_names = pd.Index(self.spot_names_filtered[start:end])
        
        # Calculate absolute positions in original data
        absolute_start = self.spot_indices[start] if start < len(self.spot_indices) else start
        absolute_end = self.spot_indices[end - 1] + 1 if end > 0 else absolute_start
        
        return ldscore_chunk.astype(np.float32), spot_names, absolute_start, absolute_end
    
    def _worker_fetch_chunks(self, worker_id: int, chunk_indices: List[int]):
        """
        Worker function to fetch chunks in order (adjacent chunks for each worker).
        
        Args:
            worker_id: ID of this worker
            chunk_indices: List of chunk indices for this worker to process
        """
        logger.debug(f"Worker {worker_id} starting with {len(chunk_indices)} chunks")
        
        for chunk_idx in chunk_indices:
            try:
                # Fetch the chunk
                ldscore, spot_names, abs_start, abs_end = self._fetch_ldscore_chunk(chunk_idx)
                
                # Truncate to match SNP data
                n_snps_used = self.data_truncated.get('n_snps_used', ldscore.shape[0])
                ldscore = ldscore[:n_snps_used]
                
                # Put result in queue
                self.result_queue.put({
                    'chunk_idx': chunk_idx,
                    'ldscore': ldscore,
                    'spot_names': spot_names,
                    'abs_start': abs_start,
                    'abs_end': abs_end,
                    'worker_id': worker_id,
                    'success': True
                })
                
            except Exception as e:
                logger.error(f"Worker {worker_id} error on chunk {chunk_idx}: {e}")
                self.result_queue.put({
                    'chunk_idx': chunk_idx,
                    'worker_id': worker_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Signal completion
        self.result_queue.put({
            'worker_id': worker_id,
            'completed': True
        })
    
    def _distribute_chunks_to_workers(self) -> List[List[int]]:
        """
        Distribute chunks to workers ensuring adjacent chunks go to same worker.
        
        Returns:
            List of chunk index lists, one per worker
        """
        total_chunks = self.total_chunks
        n_workers = self.n_loader_threads
        
        # Calculate base chunks per worker and remainder
        chunks_per_worker = total_chunks // n_workers
        remainder = total_chunks % n_workers
        
        worker_chunks = []
        start_idx = 0
        
        for i in range(n_workers):
            # This worker gets one extra chunk if we have remainder
            n_chunks = chunks_per_worker + (1 if i < remainder else 0)
            if n_chunks == 0:
                break
                
            # Assign contiguous chunk indices to this worker
            end_idx = start_idx + n_chunks
            worker_chunks.append(list(range(start_idx, end_idx)))
            start_idx = end_idx
        
        return worker_chunks
    
    def process_all_chunks(self, process_chunk_jit_fn) -> pd.DataFrame:
        """
        Process all chunks using parallel loading and computation.
        
        Args:
            process_chunk_jit_fn: JIT-compiled function for processing chunks
            
        Returns:
            Merged DataFrame with all results
        """
        # Prepare static JAX arrays
        n_snps_used = self.data_truncated['n_snps_used']
        
        baseline_ann = (self.data_truncated['baseline_ld'].values.astype(np.float32) * 
                       self.data_truncated['N'].reshape(-1, 1).astype(np.float32) / 
                       self.data_truncated['Nbar'])
        baseline_ann = np.concatenate([baseline_ann, 
                                      np.ones((n_snps_used, 1), dtype=np.float32)], axis=1)
        
        # Convert to JAX arrays
        baseline_ld_sum_jax = jnp.asarray(self.data_truncated['baseline_ld_sum'], dtype=jnp.float32)
        chisq_jax = jnp.asarray(self.data_truncated['chisq'], dtype=jnp.float32)
        N_jax = jnp.asarray(self.data_truncated['N'], dtype=jnp.float32)
        baseline_ann_jax = jnp.asarray(baseline_ann, dtype=jnp.float32)
        w_ld_jax = jnp.asarray(self.data_truncated['w_ld'], dtype=jnp.float32)
        
        del baseline_ann
        gc.collect()
        
        # Distribute chunks to workers
        worker_chunk_assignments = self._distribute_chunks_to_workers()
        
        # Start worker threads
        with ThreadPoolExecutor(max_workers=len(worker_chunk_assignments)) as executor:
            futures = []
            for worker_id, chunk_indices in enumerate(worker_chunk_assignments):
                if chunk_indices:  # Only start worker if it has chunks
                    future = executor.submit(self._worker_fetch_chunks, worker_id, chunk_indices)
                    futures.append(future)
            
            # Process results as they come in
            n_workers_completed = 0
            n_chunks_processed = 0
            
            with tqdm(total=self.total_chunks, desc="Processing chunks") as pbar:
                while n_chunks_processed < self.total_chunks:
                    try:
                        result = self.result_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue
                    
                    # Check for worker completion
                    if result.get('completed', False):
                        n_workers_completed += 1
                        logger.debug(f"Worker {result['worker_id']} completed")
                        if n_workers_completed >= len(futures):
                            break
                        continue
                    
                    # Skip failed chunks
                    if not result.get('success', False):
                        logger.error(f"Skipping chunk {result.get('chunk_idx')} due to error")
                        n_chunks_processed += 1
                        pbar.update(1)
                        continue
                    
                    # Process successful chunk
                    chunk_idx = result['chunk_idx']
                    ldscore = result['ldscore']
                    spot_names = result['spot_names']
                    abs_start = result['abs_start']
                    abs_end = result['abs_end']
                    
                    # Convert to JAX and process
                    spatial_ld_jax = jnp.asarray(ldscore, dtype=jnp.float32)
                    
                    # Process with JIT function
                    batch_size = min(50, spot_names.shape[0])
                    betas, ses = process_chunk_jit_fn(
                        self.config.n_blocks,
                        batch_size,
                        spatial_ld_jax,
                        baseline_ld_sum_jax,
                        chisq_jax,
                        N_jax,
                        baseline_ann_jax,
                        w_ld_jax,
                        self.data_truncated['Nbar']
                    )
                    
                    # Ensure computation completes
                    betas.block_until_ready()
                    ses.block_until_ready()
                    
                    # Convert to numpy
                    betas_np = np.array(betas)
                    ses_np = np.array(ses)
                    
                    # Add to results
                    self._add_chunk_result(
                        chunk_idx, betas_np, ses_np, spot_names,
                        abs_start, abs_end
                    )
                    
                    # Clean up
                    del spatial_ld_jax, betas, ses
                    
                    n_chunks_processed += 1
                    pbar.update(1)
                    
                    # Periodic memory check
                    if n_chunks_processed % 10 == 0:
                        gc.collect()
            
            # Wait for all workers to complete
            for future in futures:
                future.result()
        
        # Validate and merge results
        return self._validate_merge_and_save()
    
    def _add_chunk_result(self, chunk_idx: int, betas: np.ndarray, ses: np.ndarray,
                         spot_names: pd.Index, abs_start: int, abs_end: int):
        """Add processed chunk result to accumulator."""
        # Update coverage tracking
        self.min_spot_start = min(self.min_spot_start, abs_start)
        self.max_spot_end = max(self.max_spot_end, abs_end)
        
        # Store result
        self.results.append({
            'chunk_idx': chunk_idx,
            'betas': betas,
            'ses': ses,
            'spot_names': spot_names,
            'abs_start': abs_start,
            'abs_end': abs_end
        })
        self.processed_chunks.add(chunk_idx)
    
    def _validate_merge_and_save(self) -> pd.DataFrame:
        """
        Validate completeness, merge results, and save with appropriate filename.
        
        Returns:
            Merged DataFrame with all results
        """
        if not self.results:
            raise ValueError("No results to merge")
        
        # Check completeness
        expected_chunks = set(range(self.total_chunks))
        missing_chunks = expected_chunks - self.processed_chunks
        
        if missing_chunks:
            logger.warning(f"Missing chunks: {sorted(missing_chunks)}")
            logger.warning(f"Processed {len(self.processed_chunks)}/{self.total_chunks} chunks")
        
        # Sort results by chunk index
        sorted_results = sorted(self.results, key=lambda x: x['chunk_idx'])
        
        # Merge all results
        dfs = []
        for result in sorted_results:
            betas = result['betas'].astype(np.float64)
            ses = result['ses'].astype(np.float64)
            
            # Calculate statistics
            z_scores = betas / ses
            p_values = norm.sf(z_scores)
            log10_p = -np.log10(np.maximum(p_values, 1e-300))
            
            chunk_df = pd.DataFrame({
                'spot': result['spot_names'],
                'beta': result['betas'],
                'se': result['ses'],
                'z': z_scores.astype(np.float32),
                'p': p_values,
                'neg_log10_p': log10_p
            })
            dfs.append(chunk_df)
        
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # Generate filename with cell range information
        filename = self._generate_output_filename()
        output_path = self.output_dir / filename
        
        # Save results
        logger.info(f"Saving results to {output_path}")
        merged_df.to_csv(output_path, index=False, compression='gzip')
        
        # Log statistics
        self._log_statistics(merged_df, output_path)
        
        return merged_df
    
    def _generate_output_filename(self) -> str:
        """Generate output filename including cell range information."""
        base_name = f"{self.config.project_name}_{self.trait_name}"
        
        # If we have cell indices range, include it in filename
        if self.config.cell_indices_range:
            start_cell, end_cell = self.config.cell_indices_range
            # Adjust for actual processed range
            actual_start = max(self.min_spot_start, start_cell)
            actual_end = min(self.max_spot_end, end_cell)
            return f"{base_name}_cells_{actual_start}_{actual_end}.csv.gz"
        
        # Check if we have complete coverage
        if self.min_spot_start == 0 and self.max_spot_end == self.n_spots:
            return f"{base_name}.csv.gz"
        
        # Partial coverage without explicit range
        return f"{base_name}_start{self.min_spot_start}_end{self.max_spot_end}_total{self.n_spots}.csv.gz"
    
    def _log_statistics(self, df: pd.DataFrame, output_path: Path):
        """Log statistical summary of results."""
        n_spots = len(df)
        bonferroni_threshold = 0.05 / n_spots
        n_bonferroni_sig = (df['p'] < bonferroni_threshold).sum()
        
        # FDR correction
        _, fdr_corrected_pvals, _, _ = multipletests(
            df['p'], alpha=0.001, method='fdr_bh'
        )
        n_fdr_sig = fdr_corrected_pvals.sum()
        
        logger.info("=" * 70)
        logger.info("STATISTICAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total spots: {n_spots:,}")
        logger.info(f"Cell range processed: [{self.min_spot_start}, {self.max_spot_end})")
        logger.info(f"Max -log10(p): {df['neg_log10_p'].max():.2f}")
        logger.info("-" * 70)
        logger.info(f"Nominally significant (p < 0.05): {(df['p'] < 0.05).sum():,}")
        logger.info(f"Bonferroni threshold: {bonferroni_threshold:.2e}")
        logger.info(f"Bonferroni significant: {n_bonferroni_sig:,}")
        logger.info(f"FDR significant (alpha=0.001): {n_fdr_sig:,}")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {output_path}")
        
        # Warn if incomplete
        if len(self.processed_chunks) < self.total_chunks:
            logger.warning(f"WARNING: Only processed {len(self.processed_chunks)}/{self.total_chunks} chunks")
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'mkscore_memmap'):
            del self.mkscore_memmap
            logger.debug("Cleaned up memory-mapped arrays")
        
        # Clean up other resources
        gc.collect()
    
    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass