"""Bounded, ordered result persistence for spatial LDSC."""

from __future__ import annotations

import gzip
import logging
import os
import queue
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.stats import norm

if TYPE_CHECKING:
    from typing import TextIO

    from gsMap.config import SpatialLDSCConfig

logger = logging.getLogger("gsMap.spatial_ldsc.result_writer")


def result_filename(config, trait_name, n_spots, start, end) -> str:
    """Canonical result name for a validated selection of source rows."""
    base = f"{config.project_name}_{trait_name}"
    if config.cell_indices_range:
        left, right = config.cell_indices_range
        return f"{base}_cells_{max(start, left)}_{min(end, right)}.csv.gz"
    if start == 0 and end == n_spots:
        return f"{base}.csv.gz"
    if config.sample_filter:
        base += f"_{config.sample_filter}"
    return f"{base}_start{start}_end{end}_total{n_spots}.csv.gz"


@dataclass(slots=True)
class ChunkResult:
    """Regression output for one cell chunk."""

    chunk_idx: int
    betas: np.ndarray
    ses: np.ndarray
    spot_names: pd.Index
    abs_start: int
    abs_end: int


class ResultWriter:
    """Write out-of-order compute results as one ordered atomic CSV stream.

    Compute workers may finish out of order. Results are held only until the
    next consecutive chunk is available, then coalesced into a moderate output
    batch and sent to a bounded writer thread, so CSV formatting/compression
    does not block GPU dispatch. The completed file is atomically published only
    after every expected chunk and row is validated.
    """

    compression_level = 1

    def __init__(
        self,
        config: SpatialLDSCConfig,
        trait_name: str,
        output_dir: Path,
        n_spots: int,
        total_chunks: int,
        expected_cells: int,
        output_batch_rows: int = 50_000,
        max_pending_chunks: int = 64,
        emit_header: bool = True,
    ):
        self.config = config
        self.trait_name = trait_name
        self.output_dir = output_dir
        self.n_spots = n_spots
        self.total_chunks = total_chunks
        self.expected_cells = expected_cells
        self.output_batch_rows = output_batch_rows
        self.max_pending_chunks = max(1, max_pending_chunks)
        self.condition = threading.Condition()

        self.pending: dict[int, ChunkResult] = {}
        self.ready: list[ChunkResult] = []
        self.ready_rows = 0
        self.next_chunk = 0
        self.seen_chunks = np.zeros(total_chunks, dtype=np.bool_)
        self.min_spot_start = float("inf")
        self.max_spot_end = 0

        self.p_values = np.empty(expected_cells, dtype=np.float64)
        self.rows_written = 0
        self.nominal_significant = 0
        self.failed_cells = 0
        self.max_neg_log10_p = float("-inf")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".spatial_ldsc_",
            suffix=".partial.csv.gz",
            dir=self.output_dir,
        )
        os.close(descriptor)
        self.temporary_path = Path(temporary_name)
        self.handle: TextIO | None = None
        # Private multi-GPU shards after rank zero are headerless gzip members.
        # Completeness is independently checked using chunk and row counts.
        self.header_written = not emit_header
        self.final_path: Path | None = None
        self.closed = False
        self._closing = False
        self._writer_error: BaseException | None = None
        self._stop_writer = threading.Event()
        self._output_queue: queue.Queue = queue.Queue(maxsize=2)
        self._finalize_lock = threading.Lock()
        self._writer_thread = threading.Thread(
            target=self._write_worker, name="ldsc-result-writer", daemon=True
        )
        self._writer_thread.start()

    def add_chunk_result(
        self,
        chunk_idx: int,
        betas: np.ndarray,
        ses: np.ndarray,
        spot_names: pd.Index,
        abs_start: int,
        abs_end: int,
    ) -> None:
        """Accept one compute result and drain newly consecutive chunks."""
        betas = np.asarray(betas).reshape(-1)
        ses = np.asarray(ses).reshape(-1)
        if not (betas.size == ses.size == len(spot_names)):
            raise ValueError(
                f"Chunk {chunk_idx} output length mismatch: beta={betas.size}, "
                f"se={ses.size}, spots={len(spot_names)}"
            )

        with self.condition:
            self.check_errors()
            if self.closed or self._closing:
                raise RuntimeError("Cannot add a result after the LDSC writer was closed")
            if not 0 <= chunk_idx < self.total_chunks:
                raise ValueError(
                    f"Invalid result chunk {chunk_idx}; expected 0..{self.total_chunks - 1}"
                )
            if self.seen_chunks[chunk_idx]:
                raise ValueError(f"Duplicate result chunk {chunk_idx}")

            while (
                len(self.pending) >= self.max_pending_chunks
                and chunk_idx != self.next_chunk
                and not self.closed
                and not self._closing
            ):
                self.condition.wait(timeout=0.1)
                self.check_errors()
            if self.closed or self._closing:
                raise RuntimeError("Cannot add a result after the LDSC writer was closed")
            # Another producer may have submitted the same chunk while this
            # producer released the condition lock to wait for buffer space.
            if self.seen_chunks[chunk_idx]:
                raise ValueError(f"Duplicate result chunk {chunk_idx}")

            self.seen_chunks[chunk_idx] = True
            self.min_spot_start = min(self.min_spot_start, abs_start)
            self.max_spot_end = max(self.max_spot_end, abs_end)
            self.pending[chunk_idx] = ChunkResult(
                chunk_idx=chunk_idx,
                betas=betas.copy(),
                ses=ses.copy(),
                spot_names=spot_names.copy(deep=True),
                abs_start=abs_start,
                abs_end=abs_end,
            )
            self._drain_consecutive()
            self.condition.notify_all()

    def _drain_consecutive(self) -> None:
        while self.next_chunk in self.pending:
            result = self.pending.pop(self.next_chunk)
            self.ready.append(result)
            self.ready_rows += len(result.spot_names)
            self.next_chunk += 1
            if self.ready_rows >= self.output_batch_rows:
                self._flush_ready()

    def _flush_ready(self) -> None:
        if not self.ready:
            return
        self._enqueue_output(self.ready)
        self.ready = []
        self.ready_rows = 0

    def check_errors(self) -> None:
        """Surface asynchronous output failures to producers and the monitor."""
        if self._writer_error is not None:
            raise RuntimeError(
                f"LDSC result writer failed: {self._writer_error}"
            ) from self._writer_error

    def _enqueue_output(self, batch: list[ChunkResult] | None) -> None:
        """Keep publication ordered while waiting for bounded output space."""
        while True:
            self.check_errors()
            if self.closed or self._stop_writer.is_set():
                raise RuntimeError("LDSC result writer was aborted")
            try:
                self._output_queue.put(batch, timeout=0.05)
                return
            except queue.Full:
                # The consumer never takes condition. Retain it here so another
                # producer cannot overtake or mutate the batch being published.
                # Abort sets its event before acquiring condition to unblock us.
                continue

    def _write_worker(self) -> None:
        try:
            while not self._stop_writer.is_set():
                try:
                    batch = self._output_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if batch is None:
                        break
                    if not self._stop_writer.is_set():
                        self._write_batch(batch)
                finally:
                    self._output_queue.task_done()
        except BaseException as error:  # noqa: BLE001 - relay worker failures to the caller
            self._writer_error = error
        finally:
            # Only this thread owns the compression stream. Abort/finalize wait
            # for it before removing or publishing the temporary file.
            if self.handle is not None:
                try:
                    self.handle.close()
                except BaseException as error:  # noqa: BLE001 - close failures also invalidate output
                    if self._writer_error is None:
                        self._writer_error = error
                finally:
                    self.handle = None

    def _write_batch(self, results: list[ChunkResult]) -> None:
        """Format/write a consecutive batch; exclusively owned by the writer."""

        betas = np.concatenate([result.betas for result in results]).astype(np.float64, copy=False)
        ses = np.concatenate([result.ses for result in results]).astype(np.float64, copy=False)
        spot_names = np.concatenate(
            [np.asarray(result.spot_names, dtype=object) for result in results]
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            z_scores = betas / ses
        p_values = norm.sf(z_scores)
        neg_log10_p = -np.log10(np.maximum(p_values, 1e-300))
        # A failed or degenerate regression (non-finite beta/SE, or SE = 0) has no
        # p-value. Record it as missing rather than as p = 1 (no evidence) or p = 0.
        failed = ~np.isfinite(z_scores)
        if failed.any():
            p_values[failed] = np.nan
            neg_log10_p[failed] = np.nan
            self.failed_cells += int(failed.sum())

        end_row = self.rows_written + p_values.size
        if end_row > self.expected_cells:
            raise ValueError(
                f"LDSC writer received {end_row:,} rows, exceeding expected "
                f"{self.expected_cells:,}"
            )
        self.p_values[self.rows_written : end_row] = p_values
        self.rows_written = end_row
        self.nominal_significant += int(np.count_nonzero(p_values < 0.05))
        if np.isfinite(neg_log10_p).any():
            self.max_neg_log10_p = max(
                self.max_neg_log10_p,
                float(np.nanmax(neg_log10_p)),
            )

        frame = pd.DataFrame(
            {
                "spot": spot_names,
                "beta": betas.astype(np.float32),
                "se": ses.astype(np.float32),
                "z": z_scores.astype(np.float32),
                "p": p_values,
                "neg_log10_p": neg_log10_p,
            }
        )
        if self.handle is None:
            self.handle = gzip.open(
                self.temporary_path, mode="wt", newline="", compresslevel=self.compression_level
            )
        frame.to_csv(self.handle, index=False, header=not self.header_written)
        self.header_written = True

    def finalize(self) -> Path:
        """Validate completeness and atomically publish the final result."""
        with self._finalize_lock:
            try:
                with self.condition:
                    self.check_errors()
                    if self.closed:
                        if self.final_path is None:
                            raise RuntimeError("LDSC result writer was aborted")
                        return self.final_path
                    self._closing = True
                    self._drain_consecutive()
                    if self.next_chunk != self.total_chunks:
                        missing = np.flatnonzero(~self.seen_chunks)
                        preview = ", ".join(map(str, missing[:10]))
                        raise RuntimeError(
                            f"Missing {missing.size:,}/{self.total_chunks:,} LDSC chunks"
                            f" (first: {preview})"
                        )
                    self._flush_ready()
                    self._enqueue_output(None)
                self._writer_thread.join()
                with self.condition:
                    self.check_errors()
                    if self.closed:
                        raise RuntimeError("LDSC result writer was aborted")
                    if self.rows_written != self.expected_cells:
                        raise RuntimeError(
                            f"LDSC result row mismatch: wrote {self.rows_written:,}, "
                            f"expected {self.expected_cells:,}"
                        )
                    if not self.header_written:
                        raise RuntimeError("No LDSC results were written")
                    output_path = self.output_dir / self._generate_output_filename()
                    os.replace(self.temporary_path, output_path)
                    self.final_path = output_path
                    self.closed = True
                    self.condition.notify_all()
                self._log_statistics(output_path)
                return output_path
            except BaseException:
                self.abort()
                raise

    def abort(self) -> None:
        """Discard an incomplete stream; no final-looking output is retained."""
        self._stop_writer.set()
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self._writer_thread.join()
        self.temporary_path.unlink(missing_ok=True)
        with self.condition:
            self.pending.clear()
            self.ready.clear()
            self.ready_rows = 0
            while True:
                try:
                    self._output_queue.get_nowait()
                    self._output_queue.task_done()
                except queue.Empty:
                    break

    def _generate_output_filename(self) -> str:
        return result_filename(
            self.config,
            self.trait_name,
            self.n_spots,
            int(self.min_spot_start),
            self.max_spot_end,
        )

    def _fdr_significant_count(self, alpha: float = 0.001) -> int:
        sorted_p = np.sort(self.p_values[: self.rows_written])
        n_values = sorted_p.size
        maximum_rank = 0
        block_size = 1_000_000
        for start in range(0, n_values, block_size):
            end = min(start + block_size, n_values)
            ranks = np.arange(start + 1, end + 1, dtype=np.float64)
            passing = np.flatnonzero(sorted_p[start:end] <= alpha * ranks / n_values)
            if passing.size:
                maximum_rank = start + int(passing[-1]) + 1
        return maximum_rank

    def _log_statistics(self, output_path: Path) -> None:
        bonferroni_threshold = 0.05 / self.rows_written
        p_values = self.p_values[: self.rows_written]
        n_bonferroni = int(np.count_nonzero(p_values < bonferroni_threshold))
        n_fdr = self._fdr_significant_count(alpha=0.001)

        logger.info("=" * 70)
        logger.info("STATISTICAL SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total spots: {self.rows_written:,}")
        logger.info(f"Cell range processed: [{self.min_spot_start}, {self.max_spot_end})")
        logger.info(f"Max -log10(p): {self.max_neg_log10_p:.2f}")
        logger.info("-" * 70)
        if self.failed_cells:
            logger.warning(
                f"Regression failed for {self.failed_cells:,} spots (non-finite beta or SE); "
                "their p-values are left missing and are excluded downstream."
            )
        logger.info(f"Nominally significant (p < 0.05): {self.nominal_significant:,}")
        logger.info(f"Bonferroni threshold: {bonferroni_threshold:.2e}")
        logger.info(f"Bonferroni significant: {n_bonferroni:,}")
        logger.info(f"FDR significant (alpha=0.001): {n_fdr:,}")
        logger.info("=" * 70)
        logger.info(f"Results saved to: {output_path}")
