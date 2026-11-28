from __future__ import annotations

import gc
import hashlib
import logging
import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pa_compute
import pyarrow.csv as pa_csv
import pyarrow.parquet as pa_parquet
import scipy as sp
from anndata.io import read_elem

from gsMap.config.fingerprint import record_fingerprint
from gsMap.logging_config import logged_stage
from gsMap.utils.atomic_io import atomic_output

if TYPE_CHECKING:
    from gsMap.config.cauchy_config import CauchyCombinationConfig

logger = logging.getLogger(__name__)

_CSV_BLOCK_SIZE = 32 << 20
_PARQUET_ROW_GROUP_SIZE = 250_000
_MAX_READER_WORKERS = 4
# Smallest p-value used downstream; matches the LDSC writer's -log10(p) ceiling of 300.
_MIN_P_VALUE = 1e-300
_MAX_CALCULATION_WORKERS = 4


@dataclass(frozen=True)
class _MetadataBundle:
    """Compact metadata loaded without materializing the AnnData expression matrix."""

    h5ad_path: Path
    index_key: str
    n_spots: int
    spot_fingerprint: str
    annotations: dict[str, pd.Categorical]
    sample_name: pd.Categorical | None
    in_memory_spots: np.ndarray | None = None


@dataclass(frozen=True)
class _TraitMatrix:
    """Dense trait-major LDSC matrix with one canonical row order."""

    traits: tuple[str, ...]
    values: np.ndarray
    valid_spots: np.ndarray


@dataclass(frozen=True)
class _GroupIndex:
    """CSR-like mapping from a group to canonical matrix row positions."""

    indices: np.ndarray
    offsets: np.ndarray
    annotations: tuple[str, ...]
    background_sizes: np.ndarray
    sample_names: tuple[str, ...] | None = None


def _usable_cpu_count() -> int:
    if hasattr(os, "process_cpu_count"):
        count = os.process_cpu_count()
        if count:
            return count
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _filled_categorical(values) -> pd.Categorical:
    categorical = values if isinstance(values, pd.Categorical) else pd.Categorical(values)
    if np.any(categorical.codes < 0):
        if "NaN" not in categorical.categories:
            categorical = categorical.add_categories(["NaN"])
        categorical = categorical.fillna("NaN")
    return categorical


def _update_hash_from_python_strings(hasher, values: np.ndarray) -> None:
    """Hash concatenated UTF-8 values without constructing a persistent string index."""

    materialized = values.tolist()
    if not materialized:
        return
    if isinstance(materialized[0], bytes):
        if any(b"\0" in value for value in materialized):
            raise ValueError("AnnData spot identifiers cannot contain NUL characters.")
        hasher.update(b"\0".join(materialized) + b"\0")
    else:
        if any("\0" in str(value) for value in materialized):
            raise ValueError("AnnData spot identifiers cannot contain NUL characters.")
        hasher.update(("\0".join(str(value) for value in materialized) + "\0").encode())


def _update_hash_from_arrow_strings(hasher, values: pa.Array) -> None:
    """Hash Arrow string data without converting every value to a Python object."""

    if not len(values):
        return
    if values.null_count:
        raise ValueError("LDSC spot identifiers cannot contain missing values.")
    if bool(pa_compute.any(pa_compute.match_substring(values, "\0")).as_py()):
        raise ValueError("LDSC spot identifiers cannot contain NUL characters.")
    # Include string boundaries: ["ab", "c"] must not hash as ["a", "bc"].
    # Per-string delimiters also make this independent of CSV/H5AD chunk sizes.
    values = pa_compute.binary_join_element_wise(values, "\0", "")
    if pa.types.is_string(values.type):
        offset_dtype = np.int32
    elif pa.types.is_large_string(values.type):
        offset_dtype = np.int64
    else:
        raise TypeError(f"Expected an Arrow string column, found {values.type}.")

    _, offset_buffer, data_buffer = values.buffers()
    offsets = np.frombuffer(offset_buffer, dtype=offset_dtype)
    first = int(offsets[values.offset])
    last = int(offsets[values.offset + len(values)])
    hasher.update(memoryview(data_buffer)[first:last])


def _finalize_spot_hash(hasher, n_spots: int) -> str:
    hasher.update(struct.pack("<Q", n_spots))
    return hasher.hexdigest()


def _read_metadata(
    h5ad_path: Path,
    annotation_columns: list[str],
) -> _MetadataBundle:
    """Read only obs names and requested categorical columns from an H5AD file."""

    started = time.perf_counter()
    with h5py.File(h5ad_path, "r") as handle:
        if "obs" not in handle:
            raise ValueError(f"AnnData file has no obs table: {h5ad_path}")
        obs = handle["obs"]
        index_key = obs.attrs.get("_index", "_index")
        if isinstance(index_key, bytes):
            index_key = index_key.decode()
        if index_key not in obs:
            raise ValueError(f"AnnData obs index '{index_key}' is missing from {h5ad_path}.")

        index_node = obs[index_key]
        in_memory_spots = None
        spot_hasher = hashlib.blake2b(digest_size=16)
        if isinstance(index_node, h5py.Dataset):
            n_spots = len(index_node)
            for start in range(0, n_spots, _PARQUET_ROW_GROUP_SIZE):
                values = index_node[start : start + _PARQUET_ROW_GROUP_SIZE]
                _update_hash_from_python_strings(spot_hasher, values)
        else:
            # Categorical obs indices are unusual, but retain a correct fallback.
            in_memory_spots = np.asarray(read_elem(index_node).astype(str))
            n_spots = len(in_memory_spots)
            for start in range(0, n_spots, _PARQUET_ROW_GROUP_SIZE):
                _update_hash_from_python_strings(
                    spot_hasher,
                    in_memory_spots[start : start + _PARQUET_ROW_GROUP_SIZE],
                )
        spot_fingerprint = _finalize_spot_hash(spot_hasher, n_spots)

        requested = list(dict.fromkeys([*annotation_columns, "sample_name"]))
        missing_annotations = [column for column in annotation_columns if column not in obs]
        if missing_annotations:
            raise ValueError(
                f"Annotation columns are missing from {h5ad_path}: {missing_annotations}"
            )

        columns: dict[str, pd.Categorical] = {}
        for column in requested:
            if column in obs:
                columns[column] = _filled_categorical(read_elem(obs[column]))
                if len(columns[column]) != n_spots:
                    raise ValueError(
                        f"Metadata column '{column}' has {len(columns[column]):,} rows; "
                        f"expected {n_spots:,}."
                    )

    elapsed = time.perf_counter() - started
    logger.info(
        f"Loaded compact metadata for {n_spots:,} spots and {len(columns)} columns "
        f"in {elapsed:.2f}s"
    )
    return _MetadataBundle(
        h5ad_path=h5ad_path,
        index_key=str(index_key),
        n_spots=n_spots,
        spot_fingerprint=spot_fingerprint,
        annotations={column: columns[column] for column in annotation_columns},
        sample_name=columns.get("sample_name"),
        in_memory_spots=in_memory_spots,
    )


def _read_trait_into_row(
    trait: str,
    path: Path,
    destination: np.ndarray,
    expected_rows: int,
) -> tuple[str, str, float]:
    """Stream one compressed LDSC file into its exclusive shared matrix row."""

    started = time.perf_counter()
    reader = pa_csv.open_csv(
        path,
        read_options=pa_csv.ReadOptions(block_size=_CSV_BLOCK_SIZE, use_threads=False),
        convert_options=pa_csv.ConvertOptions(
            include_columns=["spot", "p"],
            column_types={"spot": pa.string(), "p": pa.float64()},
        ),
    )

    spot_hasher = hashlib.blake2b(digest_size=16)
    offset = 0
    for batch in reader:
        n_rows = batch.num_rows
        end = offset + n_rows
        if end > expected_rows:
            raise ValueError(
                f"LDSC result for {trait} contains more than {expected_rows:,} rows: {path}"
            )

        spots = batch.column("spot")
        _update_hash_from_arrow_strings(spot_hasher, spots)
        p_values = batch.column("p").to_numpy(zero_copy_only=False)
        if np.any((p_values < 0) | (p_values > 1)):
            raise ValueError(f"LDSC result for {trait} contains p-values outside [0, 1]: {path}")
        if np.any(p_values == 0):
            # p underflows to 0 for z > ~38. Use the writer's -log10(p) ceiling (300)
            # so the outlier filter and ACAT stay finite instead of failing on +inf.
            p_values = np.where(p_values == 0, _MIN_P_VALUE, p_values)
        with np.errstate(divide="ignore", invalid="ignore"):
            np.log10(p_values, out=destination[offset:end])
            np.negative(destination[offset:end], out=destination[offset:end])
        offset = end

    if offset != expected_rows:
        raise ValueError(
            f"LDSC result for {trait} contains {offset:,} rows; expected {expected_rows:,}: {path}"
        )

    elapsed = time.perf_counter() - started
    return trait, _finalize_spot_hash(spot_hasher, offset), elapsed


def _load_trait_matrix(
    paths: dict[str, Path],
    metadata: _MetadataBundle,
    max_workers: int | None = None,
) -> _TraitMatrix:
    """Load all traits into one compact in-process matrix using bounded reader threads."""

    traits = tuple(paths)
    if not traits:
        raise ValueError("At least one LDSC result is required.")
    if len(set(traits)) != len(traits):
        raise ValueError("Trait names must be unique.")
    reserved_columns = {"spot", *metadata.annotations}
    if metadata.sample_name is not None:
        reserved_columns.add("sample_name")
    collisions = sorted(reserved_columns.intersection(traits))
    if collisions:
        raise ValueError(f"Trait names collide with metadata columns: {collisions}")

    matrix_gib = len(traits) * metadata.n_spots * np.dtype(np.float64).itemsize / 2**30
    logger.info(
        f"Allocating compact LDSC matrix: {len(traits)} traits x {metadata.n_spots:,} "
        f"spots, float64, {matrix_gib:.2f} GiB"
    )
    values = np.empty((len(traits), metadata.n_spots), dtype=np.float64, order="C")

    worker_count = min(
        max_workers or _MAX_READER_WORKERS,
        _usable_cpu_count(),
        len(traits),
    )
    logger.info(f"Loading LDSC traits with {worker_count} shared-memory reader threads")
    started = time.perf_counter()
    canonical_spots = None
    cached_fingerprint = None
    cached_positions = None
    trait_positions = {trait: index for index, trait in enumerate(traits)}
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="cauchy-reader") as pool:
        futures = {
            pool.submit(
                _read_trait_into_row,
                trait,
                Path(paths[trait]),
                values[trait_index],
                metadata.n_spots,
            ): trait
            for trait_index, trait in enumerate(traits)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            futures.pop(future)
            trait, fingerprint, elapsed = future.result()
            if fingerprint != metadata.spot_fingerprint:
                # Sequential marker layouts legitimately write LDSC in marker-row
                # order. Restore canonical metadata order by ID, never by position.
                # Materialize names only on this fallback and cache one permutation
                # (normally shared by every trait), not one index per trait.
                if canonical_spots is None:
                    if metadata.in_memory_spots is not None:
                        canonical_spots = pd.Index(metadata.in_memory_spots.astype(str))
                    else:
                        with h5py.File(metadata.h5ad_path, "r") as handle:
                            canonical_spots = pd.Index(
                                read_elem(handle["obs"][metadata.index_key]).astype(str)
                            )
                    if not canonical_spots.is_unique:
                        raise ValueError("Canonical AnnData spot identifiers must be unique.")
                if fingerprint != cached_fingerprint:
                    spot_table = pa_csv.read_csv(
                        paths[trait],
                        read_options=pa_csv.ReadOptions(use_threads=False),
                        convert_options=pa_csv.ConvertOptions(
                            include_columns=["spot"], column_types={"spot": pa.string()}
                        ),
                    )
                    result_spots = pd.Index(spot_table.column("spot").to_pandas())
                    if not result_spots.is_unique:
                        raise ValueError(f"Duplicate spot identifiers in LDSC result for {trait}.")
                    cached_positions = result_spots.get_indexer(canonical_spots)
                    if np.any(cached_positions < 0):
                        raise ValueError(
                            f"Spot universe in LDSC result for {trait} does not match the canonical concatenated AnnData."
                        )
                    cached_fingerprint = fingerprint
                row = values[trait_positions[trait]]
                row[:] = row[cached_positions]
                logger.info(
                    f"Aligned LDSC trait {trait} from marker-row order to canonical AnnData order by spot ID"
                )
            logger.info(
                f"Loaded LDSC trait {completed}/{len(traits)}: {trait}; "
                f"{metadata.n_spots / max(elapsed, 1e-12):,.0f} spots/s"
            )

    # A spot is kept when any trait has a p-value; each trait skips only its own
    # missing values, so one trait's failed cells never change another trait's result.
    # Avoid a traits x spots temporary boolean matrix (1.28 GiB for the macaque run).
    valid_spots = np.zeros(metadata.n_spots, dtype=bool)
    for trait, trait_values in zip(traits, values, strict=True):
        missing = np.isnan(trait_values)
        n_missing = int(missing.sum())
        if n_missing:
            logger.warning(
                f"{trait}: {n_missing:,} spots have no p-value (failed regression) "
                "and are excluded from this trait's Cauchy statistics."
            )
        np.logical_or(valid_spots, ~missing, out=valid_spots)
    n_invalid = int((~valid_spots).sum())
    if n_invalid:
        logger.warning(f"Excluding {n_invalid:,} spots with no p-value in any trait.")
    elapsed = time.perf_counter() - started
    logger.info(
        f"Loaded {len(traits)} LDSC traits ({metadata.n_spots * len(traits):,} values) "
        f"in {elapsed:.2f}s"
    )
    return _TraitMatrix(traits=traits, values=values, valid_spots=valid_spots)


def _categorical_arrow_array(
    categorical: pd.Categorical,
    start: int,
    end: int,
    mask: np.ndarray | None,
) -> pa.DictionaryArray:
    codes = np.asarray(categorical.codes[start:end], dtype=np.int32)
    if mask is not None:
        codes = codes[mask]
    if np.any(codes < 0):
        raise ValueError("Categorical metadata must not contain unresolved missing values.")
    return pa.DictionaryArray.from_arrays(
        pa.array(codes, type=pa.int32()),
        pa.array([str(value) for value in categorical.categories], type=pa.string()),
    )


def _parquet_schema(
    traits: tuple[str, ...],
    metadata_columns: list[str],
) -> pa.Schema:
    fields = [pa.field("spot", pa.string())]
    fields.extend(pa.field(trait, pa.float64()) for trait in traits)
    fields.extend(
        pa.field(column, pa.dictionary(pa.int32(), pa.string())) for column in metadata_columns
    )
    return pa.schema(fields)


def _write_combined_parquet(
    path: Path,
    matrix: _TraitMatrix,
    metadata: _MetadataBundle,
) -> None:
    """Write the legacy wide Parquet artifact without constructing a wide DataFrame."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    metadata_columns = [*metadata.annotations]
    categoricals = dict(metadata.annotations)
    if metadata.sample_name is not None and "sample_name" not in metadata_columns:
        metadata_columns.append("sample_name")
        categoricals["sample_name"] = metadata.sample_name
    schema = _parquet_schema(matrix.traits, metadata_columns)

    logger.info(f"Writing combined LDSC Parquet in {_PARQUET_ROW_GROUP_SIZE:,}-row groups: {path}")
    started = time.perf_counter()
    try:
        with (
            h5py.File(metadata.h5ad_path, "r") as h5ad,
            pa_parquet.ParquetWriter(
                temporary,
                schema,
                compression="snappy",
                use_dictionary=metadata_columns,
                write_statistics=True,
            ) as writer,
        ):
            index_node = h5ad["obs"][metadata.index_key]
            for start in range(0, metadata.n_spots, _PARQUET_ROW_GROUP_SIZE):
                end = min(start + _PARQUET_ROW_GROUP_SIZE, metadata.n_spots)
                local_valid = matrix.valid_spots[start:end]
                local_mask = None if bool(local_valid.all()) else local_valid

                if metadata.in_memory_spots is None:
                    raw_spots = index_node[start:end]
                else:
                    raw_spots = metadata.in_memory_spots[start:end]
                if local_mask is not None:
                    raw_spots = raw_spots[local_mask]

                arrays: list[pa.Array] = [pa.array(raw_spots, type=pa.string())]
                for trait_index in range(len(matrix.traits)):
                    values = matrix.values[trait_index, start:end]
                    if local_mask is not None:
                        values = values[local_mask]
                    arrays.append(pa.array(values, type=pa.float64()))
                arrays.extend(
                    _categorical_arrow_array(categoricals[column], start, end, local_mask)
                    for column in metadata_columns
                )
                batch = pa.RecordBatch.from_arrays(arrays, schema=schema)
                writer.write_batch(batch, row_group_size=batch.num_rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

    elapsed = time.perf_counter() - started
    logger.info(
        f"Combined LDSC Parquet complete: {int(matrix.valid_spots.sum()):,} rows in "
        f"{elapsed:.2f}s ({path.stat().st_size / 2**30:.2f} GiB)"
    )


def _acat_test(pvalues: np.ndarray, weights=None):
    """Aggregated Cauchy Association Test (ACAT)."""

    if np.any(np.isnan(pvalues)):
        raise ValueError("Cannot have NAs in the p-values.")
    if np.any((pvalues > 1) | (pvalues < 0)):
        raise ValueError("P-values must be between 0 and 1.")

    if np.any(pvalues == 0):
        return 0.0
    if np.any(pvalues == 1) and np.all(pvalues == 1):
        return 1.0

    equal_weights = weights is None
    if not equal_weights:
        if len(weights) != len(pvalues):
            raise ValueError("Length of weights and p-values differs.")
        if np.any(np.asarray(weights) < 0):
            raise ValueError("All weights must be positive.")
        weights = np.asarray(weights) / np.sum(weights)

    is_small = pvalues < 1e-16
    cct_stat = 0.0
    if equal_weights:
        inverse_count = 1 / len(pvalues)
        if np.any(is_small):
            cct_stat += np.sum((inverse_count / pvalues[is_small]) / np.pi)
        if np.any(~is_small):
            cct_stat += inverse_count * np.sum(np.tan((0.5 - pvalues[~is_small]) * np.pi))
    else:
        if np.any(is_small):
            cct_stat += np.sum((weights[is_small] / pvalues[is_small]) / np.pi)
        if np.any(~is_small):
            cct_stat += np.sum(weights[~is_small] * np.tan((0.5 - pvalues[~is_small]) * np.pi))

    if cct_stat > 1e15:
        return (1 / cct_stat) / np.pi
    return 1 - sp.stats.cauchy.cdf(cct_stat)


def remove_outliers_IQR(data, threshold_factor=3.0):
    """Remove values above median + ``threshold_factor`` times the IQR."""

    if len(data) == 0:
        return data, np.array([], dtype=bool)

    median = np.median(data)
    q75, q25 = np.percentile(data, [75, 25])
    cutoff = median + threshold_factor * (q75 - q25)
    mask = data < cutoff
    return data[mask], mask


def _summarize_group(
    trait: str,
    annotation_col: str,
    annotation: str,
    log10p: np.ndarray,
    background_size: int,
    sample_name: str | None,
) -> dict[str, object]:
    filtered_log10p, _ = remove_outliers_IQR(log10p)
    p_values = np.power(10.0, -filtered_log10p)

    if len(p_values) == 0:
        p_cauchy = 1.0
        p_median = 1.0
        top_95_quantile = 0.0
    else:
        p_cauchy = _acat_test(p_values)
        p_median = np.median(p_values)
        top_95_quantile = np.quantile(filtered_log10p, 0.95)

    significance_threshold = 0.05 / background_size
    result: dict[str, object] = {
        "trait": trait,
        "annotation_name": annotation_col,
        "annotation": annotation,
        "p_cauchy": p_cauchy,
        "p_median": p_median,
        "top_95_quantile": top_95_quantile,
        "sig_spots": int(np.sum(p_values < significance_threshold)),
        "total_spots": len(p_values),
    }
    if sample_name is not None:
        result["sample_name"] = sample_name
    return result


def _matrix_positions(valid_spots: np.ndarray) -> np.ndarray:
    n_spots = len(valid_spots)
    index_dtype = np.uint32 if n_spots <= np.iinfo(np.uint32).max else np.uint64
    if bool(valid_spots.all()):
        return np.arange(n_spots, dtype=index_dtype)
    return np.flatnonzero(valid_spots).astype(index_dtype, copy=False)


def _build_group_index(
    annotation: pd.Categorical,
    valid_positions: np.ndarray,
    sample_name: pd.Categorical | None = None,
) -> _GroupIndex:
    annotation_codes = np.asarray(annotation.codes[valid_positions], dtype=np.int32)
    if np.any(annotation_codes < 0):
        raise ValueError("Annotation codes contain unresolved missing values.")

    if sample_name is None:
        group_codes = annotation_codes
        n_samples = 0
        sample_codes = None
        sample_counts = None
    else:
        sample_codes = np.asarray(sample_name.codes[valid_positions], dtype=np.int32)
        if np.any(sample_codes < 0):
            raise ValueError("Sample codes contain unresolved missing values.")
        n_samples = len(sample_name.categories)
        group_codes = annotation_codes * n_samples + sample_codes
        sample_counts = np.bincount(sample_codes, minlength=n_samples)

    order = np.argsort(group_codes, kind="stable")
    sorted_codes = group_codes[order]
    if len(sorted_codes):
        boundaries = np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1
        offsets = np.concatenate(
            (np.array([0], dtype=np.int64), boundaries, np.array([len(order)], dtype=np.int64))
        )
        unique_codes = sorted_codes[offsets[:-1]]
    else:
        offsets = np.array([0], dtype=np.int64)
        unique_codes = np.array([], dtype=np.int32)

    indices = valid_positions[order]
    annotation_labels: list[str] = []
    sample_labels: list[str] = []
    background_sizes: list[int] = []
    for group_code in unique_codes:
        if sample_name is None:
            annotation_code = int(group_code)
            annotation_labels.append(str(annotation.categories[annotation_code]))
            background_sizes.append(len(valid_positions))
        else:
            annotation_code, sample_code = divmod(int(group_code), n_samples)
            annotation_labels.append(str(annotation.categories[annotation_code]))
            sample_labels.append(str(sample_name.categories[sample_code]))
            background_sizes.append(int(sample_counts[sample_code]))

    return _GroupIndex(
        indices=indices,
        offsets=offsets,
        annotations=tuple(annotation_labels),
        background_sizes=np.asarray(background_sizes, dtype=np.int64),
        sample_names=tuple(sample_labels) if sample_name is not None else None,
    )


def _summarize_trait_groups(
    trait: str,
    values: np.ndarray,
    annotation_col: str,
    groups: _GroupIndex,
    sample_name: pd.Categorical | None = None,
) -> list[dict[str, object]]:
    background_sizes = groups.background_sizes
    missing = None
    if np.isnan(values[groups.indices]).any():
        # This trait lacks some spots that other traits have: count only its own
        # spots in each Bonferroni background (all spots, or the spot's sample).
        missing = np.isnan(values)
        present = groups.indices[~missing[groups.indices]]
        if groups.sample_names is None:
            background_sizes = np.full(len(groups.annotations), len(present), dtype=np.int64)
        else:
            per_sample = np.bincount(
                np.asarray(sample_name.codes[present], dtype=np.int64),
                minlength=len(sample_name.categories),
            )
            label_to_code = {str(label): code for code, label in enumerate(sample_name.categories)}
            background_sizes = np.asarray(
                [per_sample[label_to_code[label]] for label in groups.sample_names],
                dtype=np.int64,
            )
    rows = []
    for group_index, annotation in enumerate(groups.annotations):
        start = groups.offsets[group_index]
        end = groups.offsets[group_index + 1]
        positions = groups.indices[start:end]
        if missing is not None:
            positions = positions[~missing[positions]]
            if len(positions) == 0:
                continue
        group_sample = (
            groups.sample_names[group_index] if groups.sample_names is not None else None
        )
        rows.append(
            _summarize_group(
                trait=trait,
                annotation_col=annotation_col,
                annotation=annotation,
                log10p=values[positions],
                background_size=int(background_sizes[group_index]),
                sample_name=group_sample,
            )
        )
    return rows


def _result_frame(rows: list[dict[str, object]], sample_level: bool) -> pd.DataFrame:
    columns = [
        "trait",
        "annotation_name",
        "annotation",
        "p_cauchy",
        "p_median",
        "top_95_quantile",
        "sig_spots",
        "total_spots",
    ]
    if sample_level:
        columns.append("sample_name")
    frame = pd.DataFrame(rows, columns=columns)
    if not frame.empty:
        sort_columns = ["trait", "p_median"]
        if sample_level:
            sort_columns.insert(0, "sample_name")
        frame.sort_values(sort_columns, inplace=True)
    return frame


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    with atomic_output(path) as temporary:
        frame.to_csv(temporary, index=False)


def _run_cauchy_for_annotation(
    config: CauchyCombinationConfig,
    matrix: _TraitMatrix,
    metadata: _MetadataBundle,
    annotation_col: str,
    valid_positions: np.ndarray,
) -> None:
    logger.info(f"Building group indices for annotation: {annotation_col}")
    annotation_groups = _build_group_index(metadata.annotations[annotation_col], valid_positions)
    sample_groups = None
    if metadata.sample_name is not None:
        sample_groups = _build_group_index(
            metadata.annotations[annotation_col],
            valid_positions,
            sample_name=metadata.sample_name,
        )

    worker_count = min(_MAX_CALCULATION_WORKERS, _usable_cpu_count(), len(matrix.traits))
    logger.info(
        f"Calculating exact Cauchy statistics for {annotation_col} with "
        f"{worker_count} shared-memory trait workers"
    )

    def calculate_trait(trait_index: int):
        trait = matrix.traits[trait_index]
        values = matrix.values[trait_index]
        annotation_rows = _summarize_trait_groups(trait, values, annotation_col, annotation_groups)
        sample_rows = (
            _summarize_trait_groups(
                trait, values, annotation_col, sample_groups, sample_name=metadata.sample_name
            )
            if sample_groups is not None
            else []
        )
        return trait, annotation_rows, sample_rows

    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="cauchy-calc",
    ) as pool:
        futures = {
            pool.submit(calculate_trait, trait_index): trait
            for trait_index, trait in enumerate(matrix.traits)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            futures.pop(future)
            trait, annotation_rows, sample_rows = future.result()
            _atomic_to_csv(
                _result_frame(annotation_rows, sample_level=False),
                config.get_cauchy_result_file(trait, annotation=annotation_col, all_samples=True),
            )
            if sample_groups is not None:
                _atomic_to_csv(
                    _result_frame(sample_rows, sample_level=True),
                    config.get_cauchy_result_file(
                        trait, annotation=annotation_col, all_samples=False
                    ),
                )
            logger.info(
                f"Completed Cauchy trait {completed}/{len(matrix.traits)} for "
                f"{annotation_col}: {trait}"
            )

    elapsed = time.perf_counter() - started
    logger.info(f"Cauchy annotation complete: {annotation_col} in {elapsed:.2f}s")


# The DataFrame implementation remains as a small-data reference used for numerical
# regression tests. Production runs use the compact matrix path above.
def process_trait(trait, anno_data, all_data, annotation, annotation_col):
    sig_threshold = 0.05 / len(all_data)
    if trait not in anno_data.columns:
        logger.warning(f"Trait {trait} not found in data columns.")
        return None
    log10p, _ = remove_outliers_IQR(anno_data[trait].values)
    p_values = np.power(10.0, -log10p)
    if len(p_values) == 0:
        p_cauchy_val = 1.0
        p_median_val = 1.0
        top_95_quantile = 0.0
    else:
        p_cauchy_val = _acat_test(p_values)
        p_median_val = np.median(p_values)
        top_95_quantile = np.quantile(log10p, 0.95)
    return {
        "trait": trait,
        "annotation_name": annotation_col,
        "annotation": annotation,
        "p_cauchy": p_cauchy_val,
        "p_median": p_median_val,
        "top_95_quantile": top_95_quantile,
        "sig_spots": np.sum(p_values < sig_threshold),
        "total_spots": len(p_values),
    }


def run_cauchy_on_dataframe(df, annotation_col, trait_cols=None, extra_group_col=None):
    if trait_cols is None:
        trait_cols = [
            column
            for column in df.columns
            if column not in [annotation_col, extra_group_col, "spot"]
        ]

    grouped = (
        df.groupby([annotation_col, extra_group_col], observed=True)
        if extra_group_col
        else df.groupby(annotation_col, observed=True)
    )
    rows = []
    for trait in trait_cols:
        for group_key, group in grouped:
            if extra_group_col:
                annotation, extra = group_key
                background = df[df[extra_group_col] == extra]
            else:
                annotation = group_key
                extra = None
                background = df
            result = process_trait(trait, group, background, annotation, annotation_col)
            if result is not None:
                if extra_group_col:
                    result[extra_group_col] = extra
                rows.append(result)
    return _result_frame(rows, sample_level=extra_group_col is not None)


@logged_stage("cauchy_combination")
def run_Cauchy_combination(config: CauchyCombinationConfig):
    traits_dict = config.ldsc_traits_result_path_dict
    latent_adata_path = config.concatenated_latent_adata_path
    if not latent_adata_path.exists():
        raise FileNotFoundError(
            f"Latent AnnData with annotations not found at: {latent_adata_path}"
        )

    logger.info(
        f"Loading compact Cauchy inputs for {len(traits_dict)} traits from {latent_adata_path}"
    )
    metadata = _read_metadata(latent_adata_path, config.annotation_list)
    if metadata.sample_name is None:
        logger.warning(
            "'sample_name' is absent from AnnData obs; sample-level Cauchy will be skipped."
        )

    matrix = _load_trait_matrix(traits_dict, metadata)
    logger.info(f"Saving combined data to {config.ldsc_combined_parquet_path}")
    _write_combined_parquet(config.ldsc_combined_parquet_path, matrix, metadata)

    valid_positions = _matrix_positions(matrix.valid_spots)
    logger.info(f"Calculating Cauchy metrics with all {len(valid_positions):,} valid spots")
    for annotation_col in config.annotation_list:
        _run_cauchy_for_annotation(
            config,
            matrix,
            metadata,
            annotation_col,
            valid_positions,
        )
        gc.collect()

    record_fingerprint("cauchy", config)
    logger.info("Cauchy combination processing completed.")
