"""
Memory-mapped I/O utilities for efficient large-scale data handling
Replaces Zarr-backed storage with NumPy memory maps for better performance
"""

import json
import logging
import mmap
import os
import queue
import shutil
import socket
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gsMap.utils.matrix_io import pwrite_rows

logger = logging.getLogger(__name__)

_LUSTRE_STRIPE_SIZE_BYTES = 4 * 1024**2
_TMP_MINIMUM_HEADROOM_BYTES = 5 * 1024**3
_COPY_BUFFER_BYTES = 16 * 1024**2


_SCRATCH_OWNER_FILE = "gsmap_owner.json"


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True


def remove_orphaned_scratch(tmp_dir: Path) -> None:
    """Remove scratch copies left by killed gsMap processes on this host.

    Only ``memmap_*`` directories with an owner record from this host whose process
    no longer exists are removed; anything else (other hosts, live or unknown owners)
    is left alone.
    """
    host = socket.gethostname()
    try:
        candidates = list(Path(tmp_dir).glob("memmap_*"))
    except OSError:
        return
    for directory in candidates:
        owner_path = directory / _SCRATCH_OWNER_FILE
        try:
            owner = json.loads(owner_path.read_text(encoding="utf-8"))
            pid = int(owner["pid"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if owner.get("host") != host or pid == os.getpid() or _process_alive(pid):
            continue
        size = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        logger.warning(
            f"Removing scratch copy left by a stopped gsMap process (pid {pid}): "
            f"{directory} ({size / 1024**3:.1f} GiB)"
        )
        shutil.rmtree(directory, ignore_errors=True)


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Publish JSON metadata atomically (and durably) in the destination directory."""
    from gsMap.utils.atomic_io import atomic_write_text

    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _filesystem_type(path: Path) -> str | None:
    """Resolve the Linux mount type without requiring an external command."""
    target = path if path.exists() else path.parent
    try:
        target = target.resolve()
        best_match = ""
        best_type = None
        with Path("/proc/self/mountinfo").open(encoding="utf-8") as handle:
            for line in handle:
                left, separator, right = line.rstrip().partition(" - ")
                if not separator:
                    continue
                fields = left.split()
                right_fields = right.split()
                if len(fields) < 5 or not right_fields:
                    continue
                mountpoint = fields[4].replace("\\040", " ").replace("\\011", "\t")
                target_text = str(target)
                prefix = mountpoint.rstrip("/") + "/"
                if target_text == mountpoint or target_text.startswith(prefix):
                    if len(mountpoint) > len(best_match):
                        best_match = mountpoint
                        best_type = right_fields[0]
        return best_type
    except OSError:
        return None


def _automatic_lustre_stripe_count(file_bytes: int) -> int:
    """Choose bounded parallelism for large files without a user-facing knob."""
    if file_bytes < 8 * 1024**3:
        return 1
    if file_bytes < 128 * 1024**3:
        return 4
    if file_bytes < 1024**4:
        return 8
    return 16


def _same_filesystem(first: Path, second: Path) -> bool:
    """Return whether two paths reside on the same mounted filesystem."""

    def existing_parent(path: Path) -> Path:
        candidate = path
        while not candidate.exists() and candidate != candidate.parent:
            candidate = candidate.parent
        return candidate

    try:
        return existing_parent(first).stat().st_dev == existing_parent(second).stat().st_dev
    except OSError:
        return False


def _should_use_tmp_storage(
    original_path: Path,
    tmp_dir: Path,
    file_bytes: int,
) -> bool:
    """Use optional scratch only when it is distinct and safely large enough."""
    if _same_filesystem(original_path, tmp_dir):
        logger.info(
            f"Temporary directory {tmp_dir} shares the output filesystem; "
            "using the original path directly"
        )
        return False

    try:
        free_bytes = shutil.disk_usage(tmp_dir).free
    except OSError as error:
        logger.warning(
            f"Could not inspect temporary storage {tmp_dir}; using the original path: {error}"
        )
        return False

    headroom = max(_TMP_MINIMUM_HEADROOM_BYTES, file_bytes // 10)
    required_bytes = file_bytes + headroom
    if free_bytes < required_bytes:
        logger.warning(
            f"Temporary storage {tmp_dir} has {free_bytes / 1024**3:.1f} GiB free, "
            f"but this memmap requires {file_bytes / 1024**3:.1f} GiB plus "
            f"{headroom / 1024**3:.1f} GiB headroom; using the original path directly"
        )
        return False
    return True


def _prepare_memmap_data_file(data_path: Path, file_bytes: int) -> dict[str, object]:
    """Create a large Lustre file with automatic striping, falling back portably."""
    data_path.unlink(missing_ok=True)
    filesystem_type = _filesystem_type(data_path)
    storage_layout: dict[str, object] = {
        "filesystem_type": filesystem_type or "unknown",
        "automatic_striping": False,
    }
    if filesystem_type != "lustre":
        return storage_layout

    lfs = shutil.which("lfs")
    if lfs is None:
        logger.warning("Lustre filesystem detected but lfs is unavailable; using default striping")
        return storage_layout

    stripe_count = _automatic_lustre_stripe_count(file_bytes)
    command = [
        lfs,
        "setstripe",
        "-c",
        str(stripe_count),
        "-S",
        str(_LUSTRE_STRIPE_SIZE_BYTES),
        str(data_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        data_path.unlink(missing_ok=True)
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        logger.warning(f"Could not configure Lustre striping; using defaults: {detail}")
        return storage_layout

    with data_path.open("r+b") as handle:
        handle.truncate(file_bytes)
    storage_layout.update(
        {
            "automatic_striping": True,
            "stripe_count": stripe_count,
            "stripe_size_bytes": _LUSTRE_STRIPE_SIZE_BYTES,
        }
    )
    logger.info(
        f"Created Lustre memmap with {stripe_count} stripes of "
        f"{_LUSTRE_STRIPE_SIZE_BYTES // 1024**2} MiB: {data_path}"
    )
    return storage_layout


def set_memmap_advice(rank_memmap: np.memmap, advice: str = "random") -> bool:
    """Apply a portable best-effort access hint to a rank memory map.

    Returns ``True`` when the operating system accepted the hint. Unsupported
    platforms retain their default behavior and return ``False``.
    """

    advice_value = getattr(advice, "value", advice)
    normalized = str(advice_value).lower()
    constant_names = {
        "normal": "MADV_NORMAL",
        "random": "MADV_RANDOM",
        "sequential": "MADV_SEQUENTIAL",
    }
    if normalized not in constant_names:
        choices = ", ".join(constant_names)
        raise ValueError(f"Invalid rank mmap advice {advice!r}; expected one of: {choices}")

    underlying_mmap = getattr(rank_memmap, "_mmap", None)
    madvise = getattr(underlying_mmap, "madvise", None)
    advice_constant = getattr(mmap, constant_names[normalized], None)
    if madvise is None or advice_constant is None:
        logger.warning(
            "Rank mmap advice '%s' is unsupported on this platform; using OS defaults",
            normalized,
        )
        return False

    try:
        madvise(advice_constant)
    except (OSError, ValueError) as error:
        logger.warning(
            "Could not apply rank mmap advice '%s'; using OS defaults: %s",
            normalized,
            error,
        )
        return False

    logger.info("Applied rank mmap advice: %s", normalized)
    return True


class MemMapDense:
    """Dense matrix storage using NumPy memory maps with async multi-threaded writing"""

    def __init__(
        self,
        path: str | Path,
        shape: tuple[int, int],
        dtype=np.float16,
        mode: str = "w",
        num_write_workers: int = 4,
        flush_interval: float = 30,
        tmp_dir: str | Path | None = None,
    ):
        """
        Initialize a memory-mapped dense matrix.

        Args:
            path: Path to the memory-mapped file (without extension)
            shape: Shape of the matrix (n_rows, n_cols)
            dtype: Data type of the matrix
            mode: 'w' for write (create/overwrite), 'r' for read, 'r+' for read/write
            num_write_workers: Number of worker threads for async writing
            tmp_dir: Optional temporary directory for faster I/O on slow filesystems.
                    If provided, files will be created/copied to tmp_dir for operations
                    and synced back to the original path on close.
        """
        self.original_path = Path(path)
        self.shape = shape
        self.dtype = dtype
        self.mode = mode
        self.num_write_workers = num_write_workers
        if self.num_write_workers < 0:
            raise ValueError("num_write_workers cannot be negative")
        self.flush_interval = flush_interval
        self.tmp_dir = Path(tmp_dir) if tmp_dir else None
        self.using_tmp = False
        self.tmp_path = None
        self._published = False
        self._closed = False
        self._writers_stopped = False
        self._write_fd: int | None = None

        # A scratch directory must not bypass the normal protection against
        # overwriting a completed final output.
        if mode == "w":
            original_meta_path = self.original_path.with_suffix(".meta.json")
            if original_meta_path.is_file():
                try:
                    with original_meta_path.open(encoding="utf-8") as handle:
                        original_meta = json.load(handle)
                    if original_meta.get("complete", False):
                        raise ValueError(
                            f"MemMapDense at {self.original_path} already exists and is "
                            "marked as complete. Please delete it manually before overwriting."
                        )
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid metadata at {original_meta_path}. Recreating the memmap."
                    )

        file_bytes = int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize

        # Scratch is an optional optimization, never a capacity requirement.
        # This check is repeated for each concurrently live memmap, so free
        # space consumed by an already staged rank file is accounted for when
        # deciding whether the marker output can also be staged.
        if self.tmp_dir and _should_use_tmp_storage(
            self.original_path,
            self.tmp_dir,
            file_bytes,
        ):
            try:
                self._setup_tmp_paths()
            except BaseException:
                self._cleanup_tmp()
                raise
        else:
            self.tmp_dir = None
            self.path = self.original_path

        # File paths
        self.data_path = self.path.with_suffix(".dat")
        self.meta_path = self.path.with_suffix(".meta.json")

        # Initialize the mapping. Constructor failures occur before a caller can
        # hold and close this object, so clean any staged scratch copy here.
        try:
            if mode == "w":
                self._create_memmap()
            elif mode == "r":
                self._open_memmap_readonly()
            elif mode == "r+":
                self._open_memmap_readwrite()
            else:
                raise ValueError(f"Invalid mode: {mode}. Must be 'w', 'r', or 'r+'")
        except BaseException:
            self._close_memmap_handle()
            self._cleanup_tmp()
            raise

        # Async writing setup (only for write modes)
        self.write_queue = queue.Queue(maxsize=100)
        self.writer_threads = []
        self.stop_writer = threading.Event()
        # First exception raised by any writer thread, surfaced on close/mark_complete
        # so a failed write cannot silently wedge write_queue.join() forever.
        self.writer_error: BaseException | None = None

        if mode in ("w", "r+"):
            try:
                self._start_writer_threads()
            except BaseException:
                self.stop_writer.set()
                for thread in self.writer_threads:
                    thread.join()
                self._close_memmap_handle()
                self._cleanup_tmp()
                raise

    def _setup_tmp_paths(self):
        """Set up temporary paths for memory-mapped files"""
        remove_orphaned_scratch(self.tmp_dir)
        # Create a unique subdirectory in tmp_dir to avoid conflicts
        unique_id = str(uuid.uuid4())[:8]
        self.tmp_subdir = self.tmp_dir / f"memmap_{unique_id}"
        self.tmp_subdir.mkdir(parents=True, exist_ok=True)
        # Record the owner so a later run can remove this copy if this process is killed.
        _atomic_write_json(
            self.tmp_subdir / _SCRATCH_OWNER_FILE,
            {"host": socket.gethostname(), "pid": os.getpid(), "created": time.time()},
        )

        # Create tmp path with same structure as original
        self.tmp_path = self.tmp_subdir / self.original_path.name
        self.path = self.tmp_path
        self.using_tmp = True

        logger.info(f"Using temporary directory for memmap: {self.tmp_subdir}")

        # If reading, copy existing files to tmp directory
        if self.mode in ("r", "r+"):
            original_data_path = self.original_path.with_suffix(".dat")
            original_meta_path = self.original_path.with_suffix(".meta.json")

            if original_data_path.exists() and original_meta_path.exists():
                tmp_data_path = self.tmp_path.with_suffix(".dat")
                tmp_meta_path = self.tmp_path.with_suffix(".meta.json")

                logger.info("Copying memmap files to temporary directory for faster access...")
                shutil.copy2(original_data_path, tmp_data_path)
                shutil.copy2(original_meta_path, tmp_meta_path)
                logger.info(f"Memmap files copied to {self.tmp_subdir}")

    def _sync_tmp_to_original(self):
        """Sequentially publish scratch data to its final storage layout."""
        if not self.using_tmp:
            return

        tmp_data_path = self.tmp_path.with_suffix(".dat")
        tmp_meta_path = self.tmp_path.with_suffix(".meta.json")
        original_data_path = self.original_path.with_suffix(".dat")
        original_meta_path = self.original_path.with_suffix(".meta.json")

        if not tmp_data_path.is_file() or not tmp_meta_path.is_file():
            raise FileNotFoundError("Temporary memmap data or metadata disappeared before sync")

        self.original_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp_meta_path.open(encoding="utf-8") as handle:
            final_meta = json.load(handle)

        # Publish incomplete metadata before replacing data. Readers therefore
        # cannot accept a partial cross-filesystem copy as a completed memmap.
        incomplete_meta = {**final_meta, "complete": False}
        _atomic_write_json(original_meta_path, incomplete_meta)

        file_bytes = tmp_data_path.stat().st_size
        storage_layout = _prepare_memmap_data_file(original_data_path, file_bytes)
        destination_mode = "r+b" if original_data_path.exists() else "wb"
        logger.info("Syncing memmap data from temporary to original storage sequentially...")
        with (
            tmp_data_path.open("rb") as source,
            original_data_path.open(destination_mode) as destination,
        ):
            shutil.copyfileobj(source, destination, length=_COPY_BUFFER_BYTES)
            destination.truncate(file_bytes)
            destination.flush()
            os.fsync(destination.fileno())

        final_meta["storage_layout"] = storage_layout
        _atomic_write_json(original_meta_path, final_meta)

        logger.info(f"Memmap files synced to {self.original_path}")

    def _cleanup_tmp(self):
        """Clean up temporary directory"""
        if self.using_tmp and self.tmp_subdir and self.tmp_subdir.exists():
            try:
                shutil.rmtree(self.tmp_subdir)
                logger.debug(f"Cleaned up temporary directory: {self.tmp_subdir}")
            except OSError as e:
                logger.warning(f"Could not clean up temporary directory {self.tmp_subdir}: {e}")

    def _create_memmap(self):
        """Create a new memory-mapped file"""
        # Check if already exists and is complete
        if self.meta_path.exists():
            try:
                with open(self.meta_path) as f:
                    meta = json.load(f)
                if meta.get("complete", False):
                    raise ValueError(
                        f"MemMapDense at {self.path} already exists and is marked as complete. "
                        f"Please delete it manually if you want to overwrite: rm {self.data_path} {self.meta_path}"
                    )
                else:
                    logger.warning(
                        f"MemMapDense at {self.path} exists but is incomplete. Recreating."
                    )
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Invalid metadata at {self.meta_path}. Recreating.")

        file_bytes = int(np.prod(self.shape, dtype=np.int64)) * np.dtype(self.dtype).itemsize
        storage_layout = _prepare_memmap_data_file(self.data_path, file_bytes)
        memmap_mode = "r+" if storage_layout["automatic_striping"] else "w+"
        self.memmap = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode=memmap_mode,
            shape=self.shape,
        )

        # Write metadata
        meta = {
            "shape": self.shape,
            "dtype": np.dtype(self.dtype).name,  # Use dtype.name for proper serialization
            "complete": False,
            "created_at": time.time(),
            "storage_layout": storage_layout,
        }
        _atomic_write_json(self.meta_path, meta)

        logger.info(f"Created MemMapDense at {self.data_path} with shape {self.shape}")

    def _open_memmap_readonly(self):
        """Open an existing memory-mapped file in read-only mode"""
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        # Read metadata
        with open(self.meta_path) as f:
            meta = json.load(f)

        if not meta.get("complete", False):
            raise ValueError(f"MemMapDense at {self.path} is incomplete")

        # Validate shape and dtype
        if tuple(meta["shape"]) != self.shape:
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {tuple(meta['shape'])}")
        metadata_dtype = np.dtype(meta["dtype"])
        requested_dtype = np.dtype(self.dtype)
        if metadata_dtype != requested_dtype:
            raise ValueError(
                f"Dtype mismatch: expected {requested_dtype.name}, got {metadata_dtype.name}"
            )

        # Open memory map
        self.memmap = np.memmap(self.data_path, dtype=self.dtype, mode="r", shape=self.shape)

        logger.info(f"Opened MemMapDense at {self.data_path} in read-only mode")

    def _open_memmap_readwrite(self):
        """Open an existing memory-mapped file in read-write mode"""
        if not self.meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.meta_path}")

        # Read metadata
        with open(self.meta_path) as f:
            meta = json.load(f)

        if tuple(meta["shape"]) != self.shape:
            raise ValueError(f"Shape mismatch: expected {self.shape}, got {tuple(meta['shape'])}")
        metadata_dtype = np.dtype(meta["dtype"])
        requested_dtype = np.dtype(self.dtype)
        if metadata_dtype != requested_dtype:
            raise ValueError(
                f"Dtype mismatch: expected {requested_dtype.name}, got {metadata_dtype.name}"
            )

        # Open memory map
        self.memmap = np.memmap(self.data_path, dtype=self.dtype, mode="r+", shape=self.shape)

        logger.info(f"Opened MemMapDense at {self.data_path} in read-write mode")

    def _start_writer_threads(self):
        """Start background writers sharing one positional descriptor when supported."""

        def writer_worker(worker_id):
            last_flush_time = time.time()  # Track last flush time for worker 0

            while not self.stop_writer.is_set():
                try:
                    item = self.write_queue.get(timeout=1)
                except queue.Empty:
                    continue

                # Once an item is dequeued we MUST call task_done() for it on
                # every path, or write_queue.join() blocks forever. A write error
                # is recorded and the queue drained (without writing) so join()
                # can return and the error is surfaced on close/mark_complete.
                try:
                    if item is None:
                        break
                    if self.writer_error is not None:
                        # A sibling worker already failed; stop writing but keep
                        # draining so join() completes.
                        continue
                    data, row_indices, col_slice = item

                    self._write_batch_data(data, row_indices, col_slice)

                    # Only the mmap fallback needs periodic flushes. Positional
                    # output is made durable once, before completion is published.
                    if worker_id == 0 and self._write_fd is None:
                        current_time = time.time()
                        if current_time - last_flush_time >= self.flush_interval:
                            self.memmap.flush()
                            last_flush_time = time.time()
                            logger.debug(f"Worker 0 flushed memmap at {last_flush_time:.2f}")
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Writer thread {worker_id} error: {e}")
                    if self.writer_error is None:
                        self.writer_error = e
                finally:
                    self.write_queue.task_done()

        # Open once before threads start; all positional writers share this fd.
        if self.num_write_workers and callable(getattr(os, "pwrite", None)):
            self._write_fd = os.open(self.data_path, os.O_WRONLY)
        logger.info(
            f"Matrix async write backend={'pwrite' if self._write_fd is not None else 'mmap'}"
        )
        # Start multiple writer threads
        for i in range(self.num_write_workers):
            thread = threading.Thread(target=writer_worker, args=(i,), daemon=True)
            thread.start()
            self.writer_threads.append(thread)
        logger.info(f"Started {self.num_write_workers} writer threads for MemMapDense")

    def _write_batch_data(
        self,
        data: np.ndarray,
        row_indices: int | slice | np.ndarray,
        col_slice: slice | np.ndarray,
    ) -> None:
        if isinstance(row_indices, int | np.integer):
            # In this API an integer means the first row of a whole batch.
            start = int(row_indices)
            if not -self.shape[0] <= start < self.shape[0]:
                raise IndexError("Matrix batch start row out of bounds")
            if start < 0:
                start += self.shape[0]
            row_indices = slice(start, start + data.shape[0])
        all_columns = isinstance(col_slice, slice) and col_slice.indices(self.shape[1]) == (
            0,
            self.shape[1],
            1,
        )
        if self._write_fd is not None and all_columns:
            pwrite_rows(self._write_fd, self.shape, self.dtype, data, row_indices)
        else:
            self.memmap[row_indices, col_slice] = data

    def write_batch(
        self, data: np.ndarray, row_indices: int | slice | np.ndarray, col_slice=slice(None)
    ):
        """Queue batch for async writing

        Args:
            data: Data to write
            row_indices: Either a single row index, slice, or array of row indices
            col_slice: Column slice (default: all columns)
        """
        if self.mode not in ("w", "r+"):
            logger.warning("Cannot write to read-only MemMapDense")
            return
        if self._closed or self._published or self._writers_stopped:
            raise RuntimeError("Cannot write to a closed or published matrix")
        if self.writer_error is not None:
            raise RuntimeError(
                f"Matrix write already failed: {self.writer_error}"
            ) from self.writer_error

        # A zero-worker mapping is useful when a caller already serializes its
        # output. Keep write_batch functional instead of enqueuing work that no
        # thread can ever consume.
        if not self.writer_threads:
            if self._write_fd is None and callable(getattr(os, "pwrite", None)):
                self._write_fd = os.open(self.data_path, os.O_WRONLY)
            try:
                self._write_batch_data(data, row_indices, col_slice)
            except Exception as error:
                self.writer_error = error
                raise
            return

        self.write_queue.put((data, row_indices, col_slice))

    def read_batch(
        self, row_indices: int | slice | np.ndarray, col_slice=slice(None)
    ) -> np.ndarray:
        """Read batch of data

        Args:
            row_indices: Row indices to read
            col_slice: Column slice (default: all columns)

        Returns:
            NumPy array with the requested data
        """
        if isinstance(row_indices, int | np.integer):
            return self.memmap[row_indices : row_indices + 1, col_slice].copy()
        else:
            return self.memmap[row_indices, col_slice].copy()

    def __getitem__(self, key):
        """Direct array access for compatibility"""
        return self.memmap[key]

    def __setitem__(self, key, value):
        """Direct array access for compatibility"""
        if self.mode not in ("w", "r+"):
            raise ValueError("Cannot write to read-only MemMapDense")
        self.memmap[key] = value

    def mark_complete(self, metadata_updates: dict | None = None):
        """Mark the memory map as complete"""
        if self.mode in ("w", "r+"):
            logger.info("Marking memmap as complete")
            # Ensure all writes are flushed
            if self.writer_threads:
                logger.info("Waiting for remaining writes before marking complete...")
                self.write_queue.join()

            # A failed background write must not be silently marked complete.
            if self.writer_error is not None:
                raise RuntimeError(
                    "Background memmap write failed; refusing to mark incomplete data "
                    f"as complete: {self.writer_error}"
                ) from self.writer_error

            # Flush memory map to disk
            logger.info("Flushing memmap to disk...")
            self.memmap.flush()
            if self._write_fd is not None:
                try:
                    os.fsync(self._write_fd)
                except OSError as error:
                    self.writer_error = error
                    raise
            logger.info("Memmap flush complete")

            # Update metadata
            with open(self.meta_path) as f:
                meta = json.load(f)
            meta["complete"] = True
            meta["completed_at"] = time.time()
            if metadata_updates:
                meta.update(metadata_updates)
            # Ensure dtype is properly serialized
            if "dtype" in meta and not isinstance(meta["dtype"], str):
                meta["dtype"] = np.dtype(self.dtype).name
            _atomic_write_json(self.meta_path, meta)

            logger.info(f"Marked MemMapDense at {self.path} as complete")

    @classmethod
    def check_complete(
        cls, memmap_path: str | Path, meta_path: str | Path | None = None
    ) -> tuple[bool, dict | None]:
        """
        Check if a memory map file is complete without opening it.

        Args:
            memmap_path: Path to the memory-mapped file (without extension)
            meta_path: Optional path to metadata file. If not provided, will be derived from memmap_path

        Returns:
            Tuple of (is_complete, metadata_dict). metadata_dict is None if file doesn't exist or can't be read.
        """
        memmap_path = Path(memmap_path)

        if meta_path is None:
            # Derive metadata path from memmap path
            if memmap_path.suffix == ".dat":
                meta_path = memmap_path.with_suffix(".meta.json")
            elif memmap_path.name.endswith(".meta.json"):
                meta_path = memmap_path
            else:
                # Assume no extension, add .meta.json
                meta_path = memmap_path.with_suffix(".meta.json")
        else:
            meta_path = Path(meta_path)

        if not meta_path.exists():
            return False, None

        try:
            with open(meta_path) as f:
                meta = json.load(f)
            if not meta.get("complete", False):
                return False, meta

            if memmap_path.name.endswith(".meta.json"):
                data_path = memmap_path.with_name(
                    f"{memmap_path.name.removesuffix('.meta.json')}.dat"
                )
            else:
                data_path = (
                    memmap_path
                    if memmap_path.suffix == ".dat"
                    else memmap_path.with_suffix(".dat")
                )
            expected_bytes = (
                int(np.prod(meta["shape"], dtype=np.int64)) * np.dtype(meta["dtype"]).itemsize
            )
            if not data_path.is_file() or data_path.stat().st_size != expected_bytes:
                logger.warning(f"Memmap data file is missing or has the wrong size: {data_path}")
                return False, meta

            if "row_layout" in meta:
                from .marker_layout import open_marker_obs_indices

                mapping = open_marker_obs_indices(
                    data_path,
                    meta,
                    expected_obs=int(meta["shape"][0]),
                )
                if mapping is not None:
                    mapping._mmap.close()
            return True, meta
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            logger.warning(f"Could not read metadata from {meta_path}: {e}")
            return False, None

    def _stop_writer_threads(self) -> None:
        """Drain queued writes and stop the asynchronous writers exactly once."""
        if self._writers_stopped:
            return
        if self.writer_threads:
            logger.info("Waiting for queued memmap writes...")
            self.write_queue.join()
            logger.info("All queued writes have been processed")

            for _ in self.writer_threads:
                self.write_queue.put(None)

            for thread in self.writer_threads:
                thread.join(timeout=5.0)
                if thread.is_alive():
                    raise RuntimeError("Memmap writer thread did not stop within 5 seconds")
            self.stop_writer.set()
            logger.info("Memmap writer threads stopped")
        self._writers_stopped = True

    def publish(self, metadata_updates: dict | None = None) -> None:
        """Finalize output and publish one sequential copy while retaining scratch.

        A newly generated rank matrix uses this method before marker scoring so
        its project-side checkpoint is durable while the existing scratch
        mapping remains available for immediate reuse.
        """
        if self._closed:
            raise RuntimeError("Cannot publish a closed MemMapDense")
        if self.mode not in ("w", "r+"):
            return
        if self._published:
            if metadata_updates:
                raise RuntimeError("Cannot update metadata after a memmap was published")
            return

        self._stop_writer_threads()
        self.mark_complete(metadata_updates=metadata_updates)
        if self.using_tmp:
            self._sync_tmp_to_original()
        self._published = True

    def _close_memmap_handle(self) -> None:
        try:
            if self._write_fd is not None:
                fd, self._write_fd = self._write_fd, None
                os.close(fd)
        finally:
            underlying_mmap = getattr(getattr(self, "memmap", None), "_mmap", None)
            if underlying_mmap is not None and not getattr(underlying_mmap, "closed", False):
                underlying_mmap.close()

    def close(self, metadata_updates: dict | None = None):
        """Publish pending output, close the mapping, and clean temporary storage."""
        if self._closed:
            return
        logger.info("MemMapDense.close() called")

        if self.mode in ("w", "r+"):
            self.publish(metadata_updates=metadata_updates)

        self._close_memmap_handle()
        self._cleanup_tmp()
        self._closed = True

    def abort(self) -> None:
        """Close an unpublished mapping without marking or syncing it complete."""
        if self._closed:
            return
        logger.info("Aborting incomplete MemMapDense output")
        if self.mode in ("w", "r+"):
            self._stop_writer_threads()
        self._close_memmap_handle()
        self._cleanup_tmp()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        del exc_value, traceback_obj
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False

    @property
    def attrs(self):
        """Compatibility property for accessing metadata"""
        if hasattr(self, "_attrs"):
            return self._attrs

        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self._attrs = json.load(f)
        else:
            self._attrs = {}
        return self._attrs


@dataclass
class ComponentThroughput:
    """Track throughput for individual pipeline components"""

    total_batches: int = 0
    total_time: float = 0.0
    last_batch_time: float = 0.0

    def record_batch(self, elapsed_time: float):
        """Record a batch completion"""
        self.total_batches += 1
        self.total_time += elapsed_time
        self.last_batch_time = elapsed_time

    @property
    def average_time(self) -> float:
        """Average time per batch"""
        if self.total_batches > 0:
            return self.total_time / self.total_batches
        return 0.0

    @property
    def throughput(self) -> float:
        if self.average_time > 0:
            return 1.0 / self.average_time
        return 0.0


class ParallelMarkerScoreWriter:
    """Bounded marker writer using positional I/O for file-backed output.

    Writing through a Lustre mmap can synchronously fetch a page before
    modifying it, even for sequential, newly generated output. Positional
    writes send whole row blocks without faulting output pages into the
    process. The binary layout and observation mapping are unchanged.
    """

    def __init__(
        self,
        output_memmap: MemMapDense,
        num_workers: int = 4,
        input_queue: queue.Queue | None = None,
        queue_size: int = 100,
    ):
        """
        Initialize writer pool

        Args:
            output_memmap: Output memory map wrapper object
            num_workers: Number of writer threads
            input_queue: Optional input queue (from computer)
            queue_size: Queue capacity when no input queue is supplied
        """
        # Store shared memmap object
        self.shared_memmap = output_memmap.memmap
        self.memmap_path = output_memmap.path
        self.shape = output_memmap.shape
        self.dtype = output_memmap.dtype
        self.num_workers = num_workers
        if self.num_workers < 1:
            raise ValueError("Marker-score writer requires at least one worker")

        # Queue for write requests
        if queue_size < 1:
            raise ValueError("Writer queue size must be positive")
        self.write_queue = (
            input_queue if input_queue is not None else queue.Queue(maxsize=queue_size)
        )
        self.completed_count = 0
        self.completed_lock = threading.Lock()
        self.active_cell_type = None  # Track current cell type being processed

        # Throughput tracking
        self.throughput = ComponentThroughput()
        self.throughput_lock = threading.Lock()

        # Exception handling
        self.exception_queue = queue.Queue()
        self.has_error = threading.Event()
        self.error_lock = threading.Lock()
        self.writer_error: tuple[int, BaseException, str] | None = None
        self._closed = False
        self._output_memmap = output_memmap
        self._output_fd: int | None = None
        self.io_backend = "mmap"
        if output_memmap.mode not in ("w", "r+"):
            raise ValueError("Marker-score output must be writable")
        if callable(getattr(os, "pwrite", None)) and isinstance(self.shared_memmap, np.memmap):
            self._output_fd = os.open(output_memmap.data_path, os.O_WRONLY)
            self.io_backend = "pwrite"
        logger.info(
            f"Marker writer backend={self.io_backend}; "
            f"output={output_memmap.data_path}; workers={self.num_workers}"
        )

        # Start worker threads
        self.workers = []
        self.stop_workers = threading.Event()
        try:
            self._start_workers()
        except BaseException as error:
            self._record_error(-1, error, traceback.format_exc())
            self.stop_workers.set()
            for worker in self.workers:
                worker.join()
            self._close_output_fd()
            raise

    def _close_output_fd(self) -> None:
        if self._output_fd is not None:
            fd, self._output_fd = self._output_fd, None
            os.close(fd)

    def _write_rows(self, marker_scores: np.ndarray, cell_indices) -> None:
        if self._output_fd is None:
            self.shared_memmap[cell_indices] = marker_scores
        else:
            pwrite_rows(self._output_fd, self.shape, self.dtype, marker_scores, cell_indices)

    def _record_error(self, worker_id: int, error: BaseException, error_trace: str) -> None:
        with self.error_lock:
            if self.writer_error is None:
                self.writer_error = (worker_id, error, error_trace)
                self.exception_queue.put(self.writer_error)
                # Refuse publication even if a caller overlooks the writer error.
                self._output_memmap.writer_error = error
        self.has_error.set()

    def _start_workers(self):
        """Start writer worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(target=self._writer_worker, args=(i,), daemon=True)
            worker.start()
            self.workers.append(worker)
        logger.debug(f"Started {self.num_workers} marker writer threads ({self.io_backend})")

    def _writer_worker(self, worker_id: int):
        """Consume disjoint row batches using the selected output backend."""
        logger.debug(f"Writer worker {worker_id} started")

        while not self.stop_workers.is_set():
            try:
                item = self.write_queue.get(timeout=1)
            except queue.Empty:
                continue
            try:
                if item is None:
                    break

                # Once one write fails, drain subsequent requests without
                # touching the output. Every dequeued request still receives a
                # task_done() call so callers can never wedge on Queue.join().
                if self.has_error.is_set():
                    continue

                batch_idx, marker_scores, cell_indices = item

                # Track timing
                start_time = time.perf_counter()

                # Workers write disjoint absolute row ranges; positional writes
                # do not share a mutable seek offset.
                self._write_rows(marker_scores, cell_indices)

                # Track throughput
                elapsed = time.perf_counter() - start_time
                with self.throughput_lock:
                    self.throughput.record_batch(elapsed)

                # Update completed count
                with self.completed_lock:
                    self.completed_count += 1

            except Exception as e:  # noqa: BLE001
                error_trace = traceback.format_exc()
                logger.error(f"Writer worker {worker_id} error: {e}\nTraceback:\n{error_trace}")
                self._record_error(worker_id, e, error_trace)
            finally:
                self.write_queue.task_done()

        # Only close() owns the shared descriptor and final durability check.
        logger.debug(f"Writer worker {worker_id} stopping")

    def reset_for_cell_type(self, cell_type: str):
        """Reset for processing a new cell type"""
        self.active_cell_type = cell_type
        with self.completed_lock:
            self.completed_count = 0
        with self.throughput_lock:
            self.throughput = ComponentThroughput()
        logger.debug(f"Reset writer throughput for {cell_type}")

    def get_completed_count(self):
        """Get number of completed writes"""
        with self.completed_lock:
            return self.completed_count

    def get_queue_size(self):
        """Get write queue size"""
        return self.write_queue.qsize()

    def check_errors(self):
        """Check if any worker encountered an error"""
        if self.has_error.is_set():
            with self.error_lock:
                error = self.writer_error
            if error is None:
                raise RuntimeError("Writer worker failed with unknown error") from None
            worker_id, exception, error_trace = error
            raise RuntimeError(
                f"Writer worker {worker_id} failed: {exception}\nOriginal traceback:\n{error_trace}"
            ) from exception

    def close(self, *, wait: bool = True, raise_errors: bool = True):
        """Close the writer pool, optionally discarding pending writes."""
        if self._closed:
            if raise_errors:
                self.check_errors()
            return
        logger.info("Closing writer pool...")

        if wait:
            logger.info("Waiting for remaining writes...")
            self.write_queue.join()
            # Wake idle consumers immediately instead of paying one queue
            # timeout at the end of every marker calculation.
            for _ in self.workers:
                self.write_queue.put(None)
        else:
            self._record_error(
                -1,
                RuntimeError("Marker-score writing was aborted before completion"),
                "",
            )
            while True:
                try:
                    self.write_queue.get_nowait()
                except queue.Empty:
                    break
                else:
                    self.write_queue.task_done()
            self.stop_workers.set()

        for worker in self.workers:
            worker.join(timeout=5)
            if worker.is_alive():
                self._record_error(
                    -1,
                    RuntimeError("Marker-score writer thread did not stop within 5 seconds"),
                    "",
                )
                raise RuntimeError("Marker-score writer thread did not stop within 5 seconds")
        self.stop_workers.set()

        # mmap.flush() is not the durability boundary for positional writes.
        # Surface delayed filesystem failures before metadata can become complete.
        try:
            if wait and self._output_fd is not None and not self.has_error.is_set():
                started = time.perf_counter()
                os.fsync(self._output_fd)
                logger.info(
                    f"Marker output fsync completed in {time.perf_counter() - started:.2f}s"
                )
        except OSError as error:
            self._record_error(-1, error, traceback.format_exc())
        finally:
            self._close_output_fd()
        self._closed = True
        logger.info("Writer pool closed")
        if raise_errors:
            self.check_errors()

    def abort(self) -> None:
        """Stop workers and discard queued output after a failed calculation."""
        self.close(wait=False, raise_errors=False)
