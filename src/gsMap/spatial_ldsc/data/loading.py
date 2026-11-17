import fcntl
import glob
import hashlib
import json
import logging
import os
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import psutil
import pyarrow.feather as feather
import scipy.sparse as sp

from gsMap.config import SpatialLDSCConfig
from gsMap.latent2gene.marker_layout import open_marker_obs_indices
from gsMap.latent2gene.memmap_io import MemMapDense
from gsMap.spatial_ldsc.data.sumstats import read_reference_sumstats
from gsMap.utils.h5ad import read_h5ad_metadata

logger = logging.getLogger("gsMap.spatial_ldsc.io")

_MARKER_CACHE_COPY_BUFFER_BYTES = 16 * 1024**2
_MARKER_CACHE_MINIMUM_HEADROOM_BYTES = 5 * 1024**3
_UNMAPPED_WEIGHT_COLUMN = "Unmapped"


class MarkerScoreMemmapLease:
    """Keep a shared scratch copy alive until its final reader closes."""

    def __init__(
        self,
        path: Path,
        *,
        lock_handle=None,
        cache_base: Path | None = None,
    ) -> None:
        self.path = path
        self._lock_handle = lock_handle
        self._cache_base = cache_base
        self._closed = False

    @property
    def uses_scratch(self) -> bool:
        return self._cache_base is not None

    def close(self) -> None:
        """Release this reader and delete scratch after the last reader."""
        if self._closed:
            return
        self._closed = True

        lock_handle = self._lock_handle
        if lock_handle is None:
            return

        cache_base = self._cache_base
        try:
            # Drop our shared-reader lock before probing for the exclusive lock.
            # A successful probe means no other process is still using the cache.
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                logger.info(
                    f"Released shared marker-score cache reader; other readers still use "
                    f"{cache_base.with_suffix('.dat')}"
                )
                return

            cache_data = cache_base.with_suffix(".dat")
            cache_metadata = cache_base.with_suffix(".meta.json")
            removed_bytes = cache_data.stat().st_size if cache_data.is_file() else 0
            cache_data.unlink(missing_ok=True)
            cache_metadata.unlink(missing_ok=True)
            logger.info(
                f"Removed final-reader marker-score scratch cache: {cache_data} "
                f"({removed_bytes / 1024**3:.2f} GiB freed)"
            )
        except OSError as error:
            logger.warning(f"Could not remove marker-score scratch cache: {error}")
        finally:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_handle.close()
            self._lock_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback_obj):
        del exc_type, exc_value, traceback_obj
        self.close()
        return False


class _LeasedMarkerMemmapManager:
    """Close a marker mapping before releasing its scratch-reader lease."""

    def __init__(self, memmap: MemMapDense, lease: MarkerScoreMemmapLease) -> None:
        self.memmap = memmap
        self.lease = lease
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.memmap.close()
        finally:
            self.lease.close()


def _cached_marker_memmap_is_valid(cache_base: Path, expected_bytes: int) -> bool:
    data_path = cache_base.with_suffix(".dat")
    metadata_path = cache_base.with_suffix(".meta.json")
    if not data_path.is_file() or not metadata_path.is_file():
        return False
    if data_path.stat().st_size != expected_bytes:
        return False
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return bool(metadata.get("complete", False))


def _paths_share_storage_device(source: Path, destination_dir: Path) -> bool:
    return source.stat().st_dev == destination_dir.stat().st_dev


def _remove_inactive_marker_caches(
    cache_dir: Path,
    *,
    exclude_lock_path: Path,
) -> None:
    """Best-effort cleanup for caches orphaned by interrupted processes."""
    for lock_path in cache_dir.glob("marker_scores_*.lock"):
        if lock_path == exclude_lock_path:
            continue
        try:
            with lock_path.open("a+b") as lock_handle:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue

                cache_base = lock_path.with_suffix("")
                cache_data = cache_base.with_suffix(".dat")
                cache_metadata = cache_base.with_suffix(".meta.json")
                removed_bytes = cache_data.stat().st_size if cache_data.is_file() else 0
                cache_data.unlink(missing_ok=True)
                cache_metadata.unlink(missing_ok=True)
                for partial_path in cache_dir.glob(f".{cache_base.name}.*.partial"):
                    partial_path.unlink(missing_ok=True)
                if removed_bytes:
                    logger.info(
                        f"Removed inactive marker-score cache {cache_data} "
                        f"({removed_bytes / 1024**3:.2f} GiB freed)"
                    )
        except OSError as error:
            logger.warning(f"Could not inspect inactive marker-score cache {lock_path}: {error}")


def _remove_invalid_marker_cache(cache_base: Path) -> int:
    """Remove all same-key artifacts while its exclusive lock is held."""
    artifact_paths = [
        cache_base.with_suffix(".dat"),
        cache_base.with_suffix(".meta.json"),
        *cache_base.parent.glob(f".{cache_base.name}.*.partial"),
    ]
    removed_bytes = 0
    for artifact_path in artifact_paths:
        try:
            removed_bytes += artifact_path.stat().st_size
        except FileNotFoundError:
            continue
        artifact_path.unlink()

    if removed_bytes:
        logger.info(
            f"Removed invalid marker-score cache artifacts for {cache_base.name} "
            f"before capacity checking ({removed_bytes / 1024**3:.2f} GiB freed)"
        )
    return removed_bytes


def _acquire_shared_marker_cache(
    memmap_path: Path,
    tmp_dir: Path,
    marker_metadata: dict,
    *,
    populate: bool = True,
) -> MarkerScoreMemmapLease:
    """Acquire one immutable marker file per node/cache directory.

    All concurrent LDSC processes using the same ``tmp_dir`` share the cache.
    Shared advisory locks retain it while readers are active. The last reader
    removes the large data and metadata files, while the tiny lock file remains
    available for race-free coordination.
    """
    source_data = memmap_path.with_suffix(".dat")
    source_stat = source_data.stat()
    tmp_dir = tmp_dir.resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Staging onto the same filesystem cannot improve I/O and doubles capacity.
    if _paths_share_storage_device(source_data, tmp_dir):
        logger.info(
            f"Marker scratch {tmp_dir} is on the source filesystem; reading {source_data} directly"
        )
        return MarkerScoreMemmapLease(memmap_path)

    identity = "\0".join(
        [
            str(source_data.resolve()),
            str(source_stat.st_size),
            str(source_stat.st_mtime_ns),
            str(marker_metadata.get("dtype")),
            repr(marker_metadata.get("shape")),
        ]
    )
    cache_key = hashlib.blake2b(identity.encode(), digest_size=12).hexdigest()
    cache_dir = tmp_dir / "gsmap_read_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_base = cache_dir / f"marker_scores_{cache_key}"
    cache_data = cache_base.with_suffix(".dat")
    cache_metadata = cache_base.with_suffix(".meta.json")
    lock_path = cache_dir / f"marker_scores_{cache_key}.lock"

    expected_bytes = source_stat.st_size
    lock_handle = lock_path.open("a+b")
    try:
        # Every cache reader retains a shared lock for the entire mmap lifetime.
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
        _remove_inactive_marker_caches(
            cache_dir,
            exclude_lock_path=lock_path,
        )
        if _cached_marker_memmap_is_valid(cache_base, expected_bytes):
            logger.info(f"Reusing leased marker-score scratch cache: {cache_data}")
            return MarkerScoreMemmapLease(
                cache_base,
                lock_handle=lock_handle,
                cache_base=cache_base,
            )

        # Upgrade through an explicit unlock. Recovery/population is serialized
        # and validity is repeated because another process may win the race.
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        logger.info(f"Waiting for marker-score scratch recovery lock: {lock_path}")
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if _cached_marker_memmap_is_valid(cache_base, expected_bytes):
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            logger.info(
                f"Reusing marker-score scratch cache prepared by another process: {cache_data}"
            )
            return MarkerScoreMemmapLease(
                cache_base,
                lock_handle=lock_handle,
                cache_base=cache_base,
            )

        # We now hold this key's exclusive lock, so no valid reader or writer
        # can be using any same-key artifact. Reclaim incomplete data before
        # measuring capacity; otherwise a crashed 700 GiB copy could make its
        # own replacement appear not to fit.
        try:
            _remove_invalid_marker_cache(cache_base)
        except OSError as error:
            logger.warning(
                f"Could not remove invalid marker-score cache {cache_base}: {error}. "
                f"Reading source directly."
            )
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            return MarkerScoreMemmapLease(memmap_path)

        if not populate:
            logger.info(
                f"No complete marker-score scratch cache is available in {cache_dir}; "
                f"removed invalid artifacts and will read selected rows directly from "
                f"{source_data}"
            )
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            return MarkerScoreMemmapLease(memmap_path)

        free_bytes = shutil.disk_usage(cache_dir).free
        headroom = max(_MARKER_CACHE_MINIMUM_HEADROOM_BYTES, expected_bytes // 20)
        if free_bytes < expected_bytes + headroom:
            logger.warning(
                f"Shared marker cache {cache_dir} has {free_bytes / 1024**3:.1f} GiB free; "
                f"requires {expected_bytes / 1024**3:.1f} GiB plus "
                f"{headroom / 1024**3:.1f} GiB headroom. Reading source directly."
            )
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            return MarkerScoreMemmapLease(memmap_path)

        partial_data = cache_dir / f".{cache_data.name}.{os.getpid()}.partial"
        partial_metadata = cache_dir / f".{cache_metadata.name}.{os.getpid()}.partial"
        try:
            logger.info(
                f"Populating leased marker-score scratch cache: {source_data} -> {cache_data}"
            )
            with (
                source_data.open("rb") as source,
                partial_data.open("wb") as destination,
            ):
                shutil.copyfileobj(
                    source,
                    destination,
                    length=_MARKER_CACHE_COPY_BUFFER_BYTES,
                )
            if partial_data.stat().st_size != expected_bytes:
                raise OSError(
                    f"Shared marker cache copy has {partial_data.stat().st_size:,} bytes; "
                    f"expected {expected_bytes:,}"
                )

            cache_payload = {
                **marker_metadata,
                "complete": True,
                "shared_read_cache": {
                    "source": str(source_data.resolve()),
                    "source_size": expected_bytes,
                    "source_mtime_ns": source_stat.st_mtime_ns,
                },
            }
            with partial_metadata.open("w", encoding="utf-8") as handle:
                json.dump(cache_payload, handle, indent=2)
                handle.write("\n")
            os.replace(partial_data, cache_data)
            os.replace(partial_metadata, cache_metadata)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_SH)
            logger.info(f"Published leased marker-score scratch cache: {cache_data}")
            return MarkerScoreMemmapLease(
                cache_base,
                lock_handle=lock_handle,
                cache_base=cache_base,
            )
        except OSError as error:
            logger.warning(f"Could not populate shared marker-score cache: {error}")
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()
            return MarkerScoreMemmapLease(memmap_path)
        finally:
            partial_data.unlink(missing_ok=True)
            partial_metadata.unlink(missing_ok=True)
    except BaseException:
        if not lock_handle.closed:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            lock_handle.close()
        raise


def acquire_marker_scores_memmap(
    config: SpatialLDSCConfig,
    marker_metadata: dict,
    *,
    populate_cache: bool = True,
) -> MarkerScoreMemmapLease:
    """Acquire the marker memmap path for one complete read workload.

    Full spatial LDSC scans benefit from staging the complete marker matrix on
    configured scratch storage. Report PCC preparation reads only a sampled set
    of rows, so it can set ``populate_cache=False`` to reuse an existing cache
    without paying to copy the whole matrix when no cache exists yet.
    """

    memmap_path = Path(config.marker_scores_memmap_path)
    if config.memmap_tmp_dir is None:
        return MarkerScoreMemmapLease(memmap_path)
    return _acquire_shared_marker_cache(
        memmap_path,
        Path(config.memmap_tmp_dir),
        marker_metadata,
        populate=populate_cache,
    )


def _read_sumstats(fh, alleles=False, dropna=False, *, float_dtype=float):
    """Parse GWAS summary statistics."""
    fh = str(fh)
    logger.info(f"Reading summary statistics from {fh} ...")

    # Determine compression type
    compression = None
    if fh.endswith("gz"):
        compression = "gzip"
    elif fh.endswith("bz2"):
        compression = "bz2"

    # Define columns and dtypes
    dtype_dict = {"SNP": str, "Z": float_dtype, "N": float_dtype, "A1": str, "A2": str}
    usecols = ["SNP", "Z", "N"]
    if alleles:
        usecols += ["A1", "A2"]

    # Read the file
    try:
        sumstats = pd.read_csv(
            fh,
            sep=r"\s+",
            na_values=".",
            usecols=usecols,
            dtype=dtype_dict,
            compression=compression,
        )
    except (AttributeError, ValueError) as e:
        logger.error(f"Failed to parse sumstats file: {str(e.args)}")
        raise ValueError("Improperly formatted sumstats file: " + str(e.args)) from e

    # Drop NA values if specified
    if dropna:
        sumstats = sumstats.dropna(how="any")

    logger.info(f"Read summary statistics for {len(sumstats)} SNPs.")

    # Drop duplicates
    m = len(sumstats)
    sumstats = sumstats.drop_duplicates(subset="SNP")
    if m > len(sumstats):
        logger.info(f"Dropped {m - len(sumstats)} SNPs with duplicated rs numbers.")

    return sumstats


def _read_chr_files(base_path, suffix, expected_count=22):
    """Read chromosome files using glob pattern matching."""
    # Create the pattern to search for files
    file_pattern = f"{base_path}[1-9]*{suffix}*"

    # Find all matching files
    all_files = glob.glob(file_pattern)

    # Extract chromosome numbers
    chr_files = []
    for file in all_files:
        try:
            # Extract the chromosome number from filename
            file_name = os.path.basename(file)
            base_name = os.path.basename(base_path)
            chr_part = file_name.replace(base_name, "").split(suffix)[0]
            chr_num = int(chr_part)
            if 1 <= chr_num <= expected_count:
                chr_files.append((chr_num, file))
        except (ValueError, IndexError):
            continue

    # Check if we have the expected number of chromosome files
    if len(chr_files) != expected_count:
        logger.warning(
            f"❗ SEVERE WARNING ❗ Expected {expected_count} chromosome files, but found {len(chr_files)}! "
            f"⚠️ For human GWAS data, all 22 autosomes must be present. Please verify your input files."
        )

    # Sort by chromosome number and return file paths
    chr_files.sort()
    return [file for _, file in chr_files]


def _read_file(file_path):
    """Read a file based on its format/extension."""
    file_path = str(file_path)
    try:
        if file_path.endswith(".feather"):
            return pd.read_feather(file_path)
        elif file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif file_path.endswith(".gz"):
            return pd.read_csv(file_path, compression="gzip", sep="\t")
        elif file_path.endswith(".bz2"):
            return pd.read_csv(file_path, compression="bz2", sep="\t")
        else:
            return pd.read_csv(file_path, sep="\t")
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {str(e)}")
        raise


def _read_w_ld(w_ld_dir):
    """Read LD weights for all chromosomes."""
    suffix = ".l2.ldscore"
    w_file_pattern = str(Path(w_ld_dir) / "weights.")
    logger.info(f"Reading LD score annotations from {w_file_pattern}[1-22]{suffix}...")

    chr_files = _read_chr_files(w_file_pattern, suffix)

    if not chr_files:
        logger.error(
            f"No LD score files found matching pattern: {w_file_pattern}*{suffix}* inside {w_ld_dir}"
        )
        raise FileNotFoundError(
            f"No LD score files found matching pattern: {w_file_pattern}*{suffix}* inside {w_ld_dir}"
        )

    # Read and process each file
    w_array = []
    for file in chr_files:
        x = _read_file(file)

        # Sort if possible
        if "CHR" in x.columns and "BP" in x.columns:
            x = x.sort_values(by=["CHR", "BP"])

        # Drop unnecessary columns
        columns_to_drop = ["MAF", "CM", "Gene", "TSS", "CHR", "BP"]
        columns_to_drop = [col for col in columns_to_drop if col in x.columns]
        if columns_to_drop:
            x = x.drop(columns=columns_to_drop)

        w_array.append(x)

    # Concatenate and set column names
    w_ld = pd.concat(w_array, axis=0)
    logger.info(f"Loaded {len(w_ld)} SNPs from LD weight files")

    # Set column names
    w_ld.columns = (
        ["SNP", "LD_weights"] + list(w_ld.columns[2:])
        if len(w_ld.columns) > 2
        else ["SNP", "LD_weights"]
    )

    return w_ld


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
    except Exception:  # noqa: BLE001
        return 0.0


# ============================================================================
# Data loading and preparation
# ============================================================================


def construct_baseline_ld_from_snp_gene_weights(
    snp_gene_weight_adata: ad.AnnData,
) -> pd.DataFrame:
    """Construct the ``base`` and ``all_gene`` baseline annotations.

    The SNP-by-gene weight-matrix format stores mapped genes first and one
    terminal ``Unmapped`` column last. ``base`` is the sum over every column,
    while ``all_gene`` excludes the terminal column. Validate this layout
    explicitly so that a malformed or reordered resource cannot silently make
    the final mapped gene behave as the unmapped annotation.
    """

    if snp_gene_weight_adata.n_vars < 2:
        raise ValueError(
            "SNP-gene weight matrix must contain at least one mapped-gene column "
            f"followed by {_UNMAPPED_WEIGHT_COLUMN!r}"
        )

    column_names = snp_gene_weight_adata.var_names.to_numpy(dtype=str)
    unmapped_positions = np.flatnonzero(column_names == _UNMAPPED_WEIGHT_COLUMN)
    if unmapped_positions.size != 1:
        raise ValueError(
            "SNP-gene weight matrix must contain exactly one column named "
            f"{_UNMAPPED_WEIGHT_COLUMN!r}; found {unmapped_positions.size}"
        )

    unmapped_position = int(unmapped_positions[0])
    expected_position = snp_gene_weight_adata.n_vars - 1
    if unmapped_position != expected_position:
        raise ValueError(
            f"SNP-gene weight matrix column {_UNMAPPED_WEIGHT_COLUMN!r} must be last "
            f"(position {expected_position}), but it is at position {unmapped_position}"
        )

    weights = snp_gene_weight_adata.X
    base = np.asarray(weights.sum(axis=1)).reshape(-1)
    unmapped = weights[:, unmapped_position]
    if sp.issparse(unmapped):
        unmapped = unmapped.toarray()
    unmapped = np.asarray(unmapped).reshape(-1)
    all_gene = base - unmapped

    baseline_ld = pd.DataFrame(
        np.column_stack((base, all_gene)),
        columns=["base", "all_gene"],
        index=snp_gene_weight_adata.obs_names,
    )
    baseline_ld.index.name = "SNP"
    return baseline_ld


def load_common_resources(
    config: SpatialLDSCConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, ad.AnnData]:
    """
    Load resources common to all traits (weights, baseline, SNP-gene matrix).
    Returns (baseline_ld, w_ld, snp_gene_weight_adata)
    baseline_ld and w_ld are guaranteed to have the same index (intersection of available SNPs).
    """
    logger.info("Loading common resources...")
    log_memory_usage("before loading common resources")

    # 1. Load weights
    w_ld = _read_w_ld(config.w_ld_dir)
    w_ld.set_index("SNP", inplace=True)

    # 2. Load SNP-gene weight matrix
    logger.info(f"Loading SNP-gene weight matrix from {config.snp_gene_weight_adata_path}...")
    snp_gene_weight_adata = ad.read_h5ad(config.snp_gene_weight_adata_path)

    # 3. Construct baseline LD from snp_gene_weight_adata
    baseline_ld = construct_baseline_ld_from_snp_gene_weights(snp_gene_weight_adata)
    logger.info(f"Constructed baseline LD from SNP-gene weights. Shape: {baseline_ld.shape}")

    # 4. Find common SNPs between baseline and weights
    common_snps = baseline_ld.index.intersection(w_ld.index)

    # 5. Load additional baselines and update common SNPs
    if config.additional_baseline_h5ad_path_list:
        logger.info(
            f"Loading {len(config.additional_baseline_h5ad_path_list)} additional baseline annotations..."
        )

        # We need to process additional baselines carefully to maintain the dataframe structure
        # First, ensure we only work with currently common SNPs
        baseline_ld = baseline_ld.loc[common_snps]

        for i, h5ad_path in enumerate(config.additional_baseline_h5ad_path_list):
            logger.info(f"Loading additional baseline {i + 1}: {h5ad_path}")
            add_adata = ad.read_h5ad(h5ad_path)

            # Intersect with current common SNPs
            common_in_add = common_snps.intersection(add_adata.obs_names)

            if len(common_in_add) < len(common_snps):
                logger.warning(
                    f"Additional baseline {h5ad_path} only has {len(common_in_add)}/{len(common_snps)} common SNPs. Intersecting..."
                )
                common_snps = common_in_add
                baseline_ld = baseline_ld.loc[common_snps]

            # Extract data
            add_X = add_adata[common_snps].X
            if hasattr(add_X, "toarray"):
                add_X = add_X.toarray()

            add_df = pd.DataFrame(add_X, index=common_snps, columns=add_adata.var_names)

            # Concatenate
            baseline_ld = pd.concat([baseline_ld, add_df], axis=1)

    # Final subsetting
    baseline_ld = baseline_ld.loc[common_snps]
    w_ld = w_ld.loc[common_snps]

    log_memory_usage("after loading common resources")
    return baseline_ld, w_ld, snp_gene_weight_adata


def load_common_resources_direct_ldscore(
    config: SpatialLDSCConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, ad.AnnData]:
    """
    Load resources for direct ldscore input mode (pre-computed SNP × Cell h5ad).

    Returns (baseline_ld, w_ld, ldscore_adata).
    - baseline_ld is built entirely from additional_baseline_h5ad_path_list.
    - ldscore_adata has obs=SNPs, var=cells, X=pre-computed LD scores.
    """
    logger.info("Loading common resources (direct ldscore mode)...")
    log_memory_usage("before loading direct ldscore resources")

    # 1. Load LD weights
    w_ld = _read_w_ld(config.w_ld_dir)
    w_ld.set_index("SNP", inplace=True)

    # 2. Load the pre-computed ldscore h5ad into memory. Chunks are cell-column
    # slices, which are fast in memory but strided (slow) reads from a backed file.
    logger.info(f"Loading pre-computed ldscore h5ad into memory: {config.ldscore_h5ad_path}")
    ldscore_adata = ad.read_h5ad(
        config.ldscore_h5ad_path,
    )
    matrix = ldscore_adata.X
    matrix_bytes = (
        matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        if hasattr(matrix, "indptr")
        else getattr(matrix, "nbytes", 0)
    )
    logger.info(
        f"ldscore_adata: {ldscore_adata.n_obs} SNPs × {ldscore_adata.n_vars} cells "
        f"({matrix_bytes / 2**30:.2f} GiB in memory)"
    )

    # 3. Start with the intersection of ldscore SNPs and w_ld SNPs
    common_snps = ldscore_adata.obs_names.intersection(w_ld.index)
    logger.info(f"Common SNPs (ldscore_adata ∩ w_ld): {len(common_snps)}")

    # 4. Build baseline_ld from additional_baseline_h5ad_path_list
    baseline_ld: pd.DataFrame | None = None
    for i, h5ad_path in enumerate(config.additional_baseline_h5ad_path_list):
        logger.info(f"Loading baseline annotation {i + 1}: {h5ad_path}")
        add_adata = ad.read_h5ad(h5ad_path)

        common_in_add = common_snps.intersection(add_adata.obs_names)
        if len(common_in_add) < len(common_snps):
            logger.warning(
                f"Baseline {h5ad_path} only has {len(common_in_add)}/{len(common_snps)} "
                f"common SNPs. Intersecting..."
            )
            common_snps = common_in_add

        add_X = add_adata[common_snps].X
        if hasattr(add_X, "toarray"):
            add_X = add_X.toarray()

        add_df = pd.DataFrame(np.asarray(add_X), index=common_snps, columns=add_adata.var_names)
        baseline_ld = add_df if baseline_ld is None else pd.concat([baseline_ld, add_df], axis=1)

    if baseline_ld is None:
        raise ValueError(
            "No baseline annotations loaded. additional_baseline_h5ad_path_list is empty."
        )

    # 5. Final subset to common_snps
    baseline_ld = baseline_ld.loc[common_snps]
    w_ld = w_ld.loc[common_snps]

    logger.info(f"Final common SNPs: {len(common_snps)}")
    logger.info(f"Baseline LD shape: {baseline_ld.shape}")
    log_memory_usage("after loading direct ldscore resources")
    return baseline_ld, w_ld, ldscore_adata


def prepare_trait_data(
    config: SpatialLDSCConfig,
    trait_name: str,
    sumstats_file: str,
    baseline_ld: pd.DataFrame,
    w_ld: pd.DataFrame,
    snp_gene_weight_adata: ad.AnnData,
) -> tuple[dict, pd.Index]:
    """
    Prepare data for a specific trait using pre-loaded common resources.
    """
    logger.info(f"Preparing data for {trait_name}...")

    # Keep full-file QC statistics even when the fast reader materializes only
    # reference SNPs. Other whitespace formats use the established reader.
    fast = read_reference_sumstats(sumstats_file, baseline_ld.index)
    if fast is None:
        sumstats = _read_sumstats(
            fh=sumstats_file, alleles=False, dropna=False, float_dtype=np.float32
        )
        n_max = sumstats.N.max()
        lambda_gc = np.median(sumstats.Z**2) / 0.4559364
    else:
        sumstats, n_max, lambda_gc = fast
    sumstats.set_index("SNP", inplace=True)

    # Filter by chi-squared
    chisq_max = config.chisq_max
    if chisq_max is None:
        chisq_max = max(0.001 * n_max, 80)
    sumstats["chisq"] = sumstats.Z**2

    # Calculate genomic control lambda (λGC) before filtering
    logger.info(f"Lambda GC (genomic control λ): {lambda_gc:.4f}")

    # A missing or non-positive sample size would make every weight (and so every
    # spot's regression) NaN; drop such SNPs like missing Z values.
    valid_n = np.isfinite(sumstats.N.to_numpy()) & (sumstats.N.to_numpy() > 0)
    n_invalid_n = int((~valid_n).sum())
    if n_invalid_n:
        logger.warning(f"Dropping {n_invalid_n:,} SNPs with a missing or non-positive N")
        sumstats = sumstats[valid_n]
    sumstats = sumstats[sumstats.chisq < chisq_max]
    logger.info(
        "Filtered to %d %sSNPs with chi^2 < %s",
        len(sumstats),
        "reference " if fast is not None else "",
        chisq_max,
    )

    # Intersect trait sumstats with common resources
    common_snps = baseline_ld.index.intersection(sumstats.index)
    logger.info(f"Common SNPs: {len(common_snps)}")

    if len(common_snps) < 200000:
        logger.warning(f"WARNING: Only {len(common_snps)} common SNPs")

    # Get SNP positions relative to the original snp_gene_weight_adata
    # This is crucial for QuickMode to know which rows of the weight matrix to pick
    snp_positions = snp_gene_weight_adata.obs_names.get_indexer(common_snps)

    # Subset data
    trait_baseline_ld = baseline_ld.loc[common_snps]
    trait_w_ld = w_ld.loc[common_snps]
    trait_sumstats = sumstats.loc[common_snps]

    # Prepare data dictionary
    data = {
        "baseline_ld": trait_baseline_ld,
        "baseline_ld_sum": trait_baseline_ld.sum(axis=1).values.astype(np.float32),
        "w_ld": trait_w_ld.LD_weights.values.astype(np.float32),
        "sumstats": trait_sumstats,
        "chisq": trait_sumstats.chisq.values.astype(np.float32),
        "N": trait_sumstats.N.values.astype(np.float32),
        "Nbar": np.float32(trait_sumstats.N.mean()),
        "snp_positions": snp_positions,
    }

    return data, common_snps


def load_marker_scores_memmap_format(
    config: SpatialLDSCConfig,
    *,
    populate_cache: bool = True,
) -> ad.AnnData:
    """
    Load marker scores memmap and wrap it in an AnnData object with metadata
    from a reference h5ad file using configuration.

    Args:
        config: SpatialLDSCConfig containing paths and settings.
        populate_cache: Copy the full matrix to configured scratch when no
            shared copy exists. Full LDSC scans should retain the default;
            report workloads should disable it because they read selected data.

    Returns:
        AnnData object with X backed by the memory map
    """
    memmap_path = Path(config.marker_scores_memmap_path)
    metadata_path = Path(config.concatenated_latent_adata_path)
    mode = "r"  # Read-only mode for loading

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    # check complete
    is_complete, marker_metadata = MemMapDense.check_complete(memmap_path)
    if not is_complete:
        raise ValueError(
            f"Marker score at {memmap_path} is incomplete or corrupted. Please recompute."
        )

    if marker_metadata is None:
        raise ValueError(f"Marker score metadata could not be loaded: {memmap_path}")

    logger.info(f"Loading metadata from {metadata_path}")
    # Backed AnnData still eagerly loads obsm, including enormous neighbor arrays.
    # The marker wrapper needs only metadata and its optional spatial coordinates.
    src_adata = read_h5ad_metadata(
        metadata_path,
        obsm_keys=[getattr(config, "spatial_key", None)],
        include_uns=True,
        include_varm=True,
    )
    row_order = None
    mm = None
    marker_lease = None
    try:
        shape = (src_adata.n_obs, src_adata.n_vars)
        if tuple(marker_metadata["shape"]) != shape:
            raise ValueError(
                f"Marker score shape {tuple(marker_metadata['shape'])} does not match "
                f"AnnData shape {shape}"
            )

        marker_lease = acquire_marker_scores_memmap(
            config,
            marker_metadata,
            populate_cache=populate_cache,
        )

        mm = MemMapDense(
            marker_lease.path,
            shape=shape,
            dtype=np.dtype(marker_metadata["dtype"]),
            mode=mode,
            tmp_dir=None,
        )

        row_order = open_marker_obs_indices(
            memmap_path,
            marker_metadata,
            expected_obs=src_adata.n_obs,
        )
        spatial_key = getattr(config, "spatial_key", None)
        obsm_keys = (
            [spatial_key] if spatial_key is not None and spatial_key in src_adata.obsm else []
        )
        if row_order is None:
            obs = src_adata.obs.copy()
            obsm = {}
            for key in obsm_keys:
                value = src_adata.obsm[key]
                if hasattr(value, "to_memory"):
                    value = value.to_memory()
                obsm[key] = value.copy()
            logger.info("Using legacy identity marker-row layout")
        else:
            logger.info("Aligning AnnData observations to sequential marker-row layout")
            row_indices = np.asarray(row_order)
            obs = src_adata.obs.iloc[row_indices].copy()
            obsm = {}
            for key in obsm_keys:
                value = src_adata.obsm[key]
                if hasattr(value, "to_memory"):
                    value = value.to_memory()
                if isinstance(value, pd.DataFrame):
                    obsm[key] = value.iloc[row_indices].copy()
                elif sp.issparse(value):
                    obsm[key] = value[row_indices].copy()
                else:
                    # Materialize backed arrays before arbitrary permutation;
                    # h5py fancy indexing requires monotonically ordered rows.
                    obsm[key] = np.asarray(value)[row_indices].copy()

        if obsm_keys:
            logger.info(f"Loaded aligned spatial coordinates from obsm[{spatial_key!r}]")
        logger.info("Constructing AnnData wrapper...")
        adata = ad.AnnData(
            X=mm.memmap,
            obs=obs,
            var=src_adata.var.copy(),
            uns=src_adata.uns.copy(),
            obsm=obsm,
            varm=src_adata.varm.copy(),
        )
    except Exception:
        if mm is not None:
            mm.close()
        if marker_lease is not None:
            marker_lease.close()
        raise
    finally:
        if row_order is not None:
            row_order._mmap.close()
        if src_adata.isbacked:
            src_adata.file.close()

    # Closing this manager closes the mmap first, then removes the node-local
    # scratch copy if this process was its final reader.
    adata.uns["memmap_manager"] = _LeasedMarkerMemmapManager(mm, marker_lease)
    if row_order is not None:
        adata.uns["marker_score_row_layout"] = marker_metadata["row_layout"]

    return adata


def generate_expected_output_filename(
    config: SpatialLDSCConfig, trait_name: str, n_spots: int | None = None
) -> str | None:
    """Name of a complete result for ``trait_name``, or None when it cannot be predicted.

    Matches ``writer.result_filename``: a ``cell_indices_range`` is clamped to the
    ``n_spots`` input spots when that number is known.
    """
    base_name = f"{config.project_name}_{trait_name}"

    if config.cell_indices_range:
        start_cell, end_cell = config.cell_indices_range
        if n_spots is not None:
            start_cell, end_cell = max(start_cell, 0), min(end_cell, n_spots)
        return f"{base_name}_cells_{start_cell}_{end_cell}.csv.gz"

    # With a sample filter the processed range is known only after filtering.
    if config.sample_filter:
        return None

    return f"{base_name}.csv.gz"


def log_existing_result_statistics(result_path: Path, trait_name: str):
    """Log a reused result without re-reading it (statistics were logged when it was written)."""
    try:
        size_mib = result_path.stat().st_size / 2**20
    except OSError as error:
        logger.warning(f"Could not stat existing result for {trait_name}: {error}")
        return
    logger.info(f"Reusing existing result for {trait_name}: {result_path} ({size_mib:.1f} MiB)")


class LazyFeatherX:
    """
    A proxy for the 'X' matrix that slices directly from the Feather file
    via memory mapping without loading the full data.
    """

    def __init__(self, arrow_table, feature_names, transpose=False):
        self.table = arrow_table
        self.feature_names = feature_names
        self.transpose = transpose
        if not transpose:
            # Standard AnnData shape: (n_obs, n_vars)
            self.shape = (self.table.num_rows, len(self.feature_names))
        else:
            # Transposed shape: (n_obs, n_vars) where n_obs = num_features, n_vars = num_rows
            self.shape = (len(self.feature_names), self.table.num_rows)

    def __getitem__(self, key):
        """
        Handles slicing: adata.X[start:end], adata.X[start:end, :], etc.
        """
        # Normalize the key to always be a tuple of (row_slice, col_slice)
        if not isinstance(key, tuple):
            row_key = key
            col_key = slice(None)  # Select all columns
        else:
            row_key, col_key = key

        if not self.transpose:
            # --- 1. Handle Row Slicing ---
            if isinstance(row_key, slice):
                start = row_key.start or 0
                stop = row_key.stop or self.shape[0]
                step = row_key.step or 1

                # Calculate length based on step (simplified for step=1)
                # For complex steps, we might need more logic, but basic slicing:
                if step == 1:
                    length = stop - start
                    sliced_table = self.table.slice(offset=start, length=length)
                else:
                    # Fallback for stepped slicing: Read range, then step in pandas
                    # (Slightly less efficient but works)
                    length = stop - start
                    sliced_table = self.table.slice(offset=start, length=length)
                    # We will handle the stepping after conversion

            elif isinstance(row_key, int):
                # Single row request
                sliced_table = self.table.slice(offset=row_key, length=1)
            else:
                raise NotImplementedError(
                    "Only slice objects (start:stop) or integers are supported for rows."
                )

            # --- 2. Handle Column Slicing ---
            final_cols = self.feature_names

            # Helper to map integer indices to column names
            def get_col_names_by_indices(indices):
                return [self.feature_names[i] for i in indices]

            if isinstance(col_key, slice):
                final_cols = self.feature_names[col_key]
                sliced_table = sliced_table.select(final_cols)

            elif isinstance(col_key, list | np.ndarray):
                # Check if it's integers or strings
                if len(col_key) > 0:
                    if isinstance(col_key[0], int | np.integer):
                        final_cols = get_col_names_by_indices(col_key)
                    else:
                        # Assume strings
                        final_cols = col_key
                sliced_table = sliced_table.select(final_cols)

            elif isinstance(col_key, int):
                final_cols = [self.feature_names[col_key]]
                sliced_table = sliced_table.select(final_cols)

            # --- 3. Materialize to NumPy ---
            # FIX: We convert to Pandas first, because pyarrow.Table
            # doesn't have a direct to_numpy() for 2D structures.
            df = sliced_table.to_pandas()

            # If we had a row step > 1, apply it now on the small DataFrame
            if isinstance(row_key, slice) and row_key.step and row_key.step != 1:
                df = df.iloc[:: row_key.step]

            return df.to_numpy()

        else:
            # Transposed mode:
            # row_key selects from feature_names (which were columns in feather)
            # col_key selects from table rows

            # --- 1. Determine which columns to read (Observations) ---
            if isinstance(row_key, slice):
                selected_cols = self.feature_names[row_key]
            elif isinstance(row_key, list | np.ndarray):
                if len(row_key) > 0 and isinstance(row_key[0], int | np.integer):
                    selected_cols = [self.feature_names[i] for i in row_key]
                else:
                    selected_cols = row_key
            elif isinstance(row_key, int):
                selected_cols = [self.feature_names[row_key]]
            else:
                raise NotImplementedError(
                    "Row key type not supported for transposed FeatherAnnData."
                )

            # --- 2. Determine which rows to read (Variables) ---
            if isinstance(col_key, slice):
                start = col_key.start or 0
                stop = col_key.stop or self.table.num_rows
                length = stop - start
                sliced_table = self.table.slice(offset=start, length=length)
                # Select columns and convert
                df = sliced_table.select(selected_cols).to_pandas()
                # Apply step for rows if needed
                if col_key.step and col_key.step != 1:
                    df = df.iloc[:: col_key.step]
            elif isinstance(col_key, int):
                sliced_table = self.table.slice(offset=col_key, length=1)
                df = sliced_table.select(selected_cols).to_pandas()
            elif isinstance(col_key, list | np.ndarray):
                # Use take for arbitrary row selection
                # Note: col_key should be integer indices
                sliced_table = self.table.take(col_key)
                df = sliced_table.select(selected_cols).to_pandas()
            else:
                raise NotImplementedError(
                    "Column key type not supported for transposed FeatherAnnData."
                )

            # The dataframe 'df' has table rows as index and selected_cols as columns.
            # In transposed mode, we want selected_cols as rows and table rows as columns.
            return df.to_numpy().T


class FeatherAnnData:
    """
    A minimal AnnData-like class backed by a Feather file.
    Mimics the behavior of anndata.AnnData without loading X into memory.
    """

    def __init__(self, file_path, index_col=None, transpose=False):
        # 1. Open with memory mapping (Zero RAM usage for data)
        self._table = feather.read_table(file_path, memory_map=True)

        # 2. Setup Index (Obs Names) and Columns (Var Names)
        all_cols = self._table.column_names

        if transpose:
            if index_col:
                # Genes are rows in file, but we want them as vars
                self.var_names = self._table.column(index_col).to_pylist()
                # Cells are columns in file, we want them as obs
                self.obs_names = [c for c in all_cols if c != index_col]
            else:
                self.obs_names = all_cols
                self.var_names = [str(i) for i in range(self._table.num_rows)]
        else:
            if index_col:
                # Load the index column specifically
                self.obs_names = self._table.column(index_col).to_pylist()
                # The variables (genes) are all columns MINUS the index column
                self.var_names = [c for c in all_cols if c != index_col]
            else:
                # Fallback: Assume all columns are genes
                self.obs_names = [str(i) for i in range(self._table.num_rows)]
                self.var_names = all_cols

        # 3. Setup Metadata DataFrames
        self.obs = pd.DataFrame(index=self.obs_names)
        self.var = pd.DataFrame(index=self.var_names)

        # 4. Setup Attributes (n_obs, n_vars)
        self.n_obs = len(self.obs_names)
        self.n_vars = len(self.var_names)

        # 5. Setup the Lazy X
        if transpose:
            # When transposed, 'feature_names' in LazyFeatherX refers to the obs columns
            self.X = LazyFeatherX(self._table, self.obs_names, transpose=True)
        else:
            # Standard: 'feature_names' refers to the var columns
            self.X = LazyFeatherX(self._table, self.var_names, transpose=False)

        self.uns = {}

        # 6. Setup Shape
        self.shape = (self.n_obs, self.n_vars)

    def __repr__(self):
        return (
            f"FeatherAnnData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}\n"
            f"    obs: {list(self.obs.columns)}\n"
            f"    var: {list(self.var.columns)}\n"
            f"    uns: (Empty)\n"
            f"    obsm: (Empty)\n"
            f"    varm: (Empty)\n"
            f"    layers: (Empty)\n"
            f"    Backing: PyArrow Memory Mapping (Read-Only)"
        )
