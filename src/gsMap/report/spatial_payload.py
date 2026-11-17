"""Compact, browser-oriented payloads for spatial report visualizations."""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from gsMap.config.gwas import trait_file_token
from gsMap.utils.atomic_io import atomic_output, atomic_write_text

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SPATIAL_PAYLOAD_VERSION = 11
P_VALUE_COLORS = [
    "#313695",
    "#4575b4",
    "#74add1",
    "#fee090",
    "#fdae61",
    "#f46d43",
    "#d73027",
    "#a50026",
]


def safe_name(value: str) -> str:
    """Return the filename-safe convention used throughout the report."""

    return "".join(character if character.isalnum() else "_" for character in str(value))


def encode_typed_array(values, dtype: str = "float32") -> dict:
    """Encode a one-dimensional NumPy-compatible array for compact local JS loading."""

    dtype_map = {
        "float32": np.dtype("<f4"),
        "uint16": np.dtype("<u2"),
        "uint32": np.dtype("<u4"),
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported browser payload dtype: {dtype}")

    array = np.ascontiguousarray(values, dtype=dtype_map[dtype]).reshape(-1)
    return {
        "encoding": "base64",
        "dtype": dtype,
        "length": int(array.size),
        "data": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
    }


def encode_nonnegative_quantized_array(values, scale: float = 0.1) -> dict:
    """Quantize nonnegative values to uint16 while preserving NaN explicitly."""

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Quantization scale must be finite and positive")
    values_array = np.asarray(values, dtype=np.float32).reshape(-1)
    nan_value = np.iinfo(np.uint16).max
    maximum_code = nan_value - 1
    codes = np.full(values_array.size, nan_value, dtype=np.uint16)
    present = ~np.isnan(values_array)
    if present.any():
        maximum_value = maximum_code * scale
        bounded = np.clip(values_array[present], 0, maximum_value)
        quantized = np.rint(bounded / scale)
        codes[present] = np.clip(quantized, 0, maximum_code).astype(np.uint16)
    payload = encode_typed_array(codes, dtype="uint16")
    payload.update({"scale": float(scale), "offset": 0.0, "nan_value": int(nan_value)})
    return payload


def decode_typed_array(payload: dict) -> np.ndarray:
    """Decode a payload in Python; primarily useful for validation and tests."""

    dtype_map = {
        "float32": np.dtype("<f4"),
        "uint16": np.dtype("<u2"),
        "uint32": np.dtype("<u4"),
    }
    raw = base64.b64decode(payload["data"])
    array = np.frombuffer(raw, dtype=dtype_map[payload["dtype"]], count=payload["length"])
    if "scale" not in payload:
        return array
    decoded = array.astype(np.float32)
    nan_value = payload.get("nan_value")
    missing = array == nan_value if nan_value is not None else None
    decoded *= float(payload["scale"])
    decoded += float(payload.get("offset", 0.0))
    if missing is not None:
        decoded[missing] = np.nan
    return decoded


def _atomic_write_text(path: Path, content: str) -> None:
    atomic_write_text(path, content)


def _atomic_to_parquet(frame: pd.DataFrame, path: Path) -> None:
    with atomic_output(path) as temporary:
        frame.to_parquet(temporary, index=False)


def _write_keyed_payload(path: Path, root_name: str, key: str, payload: dict) -> None:
    serialized_key = json.dumps(str(key), ensure_ascii=False)
    serialized_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    _atomic_write_text(
        path,
        f"window.{root_name}=window.{root_name}||{{}};"
        f"window.{root_name}[{serialized_key}]={serialized_payload};",
    )


def _categorical_schema(values: pd.Series | np.ndarray) -> tuple[list[str], list[str]]:
    series = pd.Series(values, copy=False)
    if isinstance(series.dtype, pd.CategoricalDtype):
        categories = [str(category) for category in series.cat.categories]
        if (series.cat.codes < 0).any():
            categories.append("NaN")
        categories = sorted(set(categories), key=str)
    else:
        string_series = series.astype("string").fillna("NaN").astype(str)
        categories = sorted(string_series.unique().tolist(), key=str)

    from gsMap.report.visualize import _create_color_map

    color_map = _create_color_map(categories, hex=True, rng=42)
    colors = [color_map[category] for category in categories]
    return categories, colors


def _categorical_payload(
    values: pd.Series | np.ndarray,
    categories: list[str] | None = None,
    colors: list[str] | None = None,
) -> dict:
    series = pd.Series(values, copy=False).astype("string").fillna("NaN").astype(str)
    if categories is None or colors is None:
        categories, colors = _categorical_schema(series)
    categorical = pd.Categorical(series, categories=categories)
    if (categorical.codes < 0).any():
        raise ValueError("Categorical schema does not include every observed value")
    dtype = "uint16" if len(categories) <= np.iinfo(np.uint16).max else "uint32"
    return {
        "codes": encode_typed_array(categorical.codes, dtype=dtype),
        "categories": categories,
        "colors": colors,
    }


def _categorical_code_arrays(
    metadata: pd.DataFrame,
    schemas: dict[str, tuple[list[str], list[str]]],
) -> dict[str, tuple[np.ndarray, str]]:
    """Encode full-row annotation codes against canonical category schemas."""

    encoded = {}
    for annotation, (categories, _) in schemas.items():
        values = metadata[annotation].astype("string").fillna("NaN").astype(str)
        categorical = pd.Categorical(values, categories=categories)
        if (categorical.codes < 0).any():
            raise ValueError(f"Categorical schema for {annotation} is incomplete")
        dtype = "uint16" if len(categories) <= np.iinfo(np.uint16).max else "uint32"
        numpy_dtype = np.dtype("<u2" if dtype == "uint16" else "<u4")
        encoded[annotation] = (
            np.asarray(categorical.codes, dtype=numpy_dtype),
            dtype,
        )
    return encoded


def _stratified_positions(
    metadata: pd.DataFrame,
    target_size: int,
    annotation_columns: list[str],
    random_state: int = 42,
) -> np.ndarray:
    """Select an exact-size, annotation-aware display sample.

    Very large inputs first draw a deterministic candidate reservoir. This bounds
    the expensive grouping work by the browser display size while preserving the
    observed sample/annotation composition.
    """

    n_rows = len(metadata)
    if target_size >= n_rows:
        return np.arange(n_rows, dtype=np.int64)

    candidate_size = min(n_rows, max(250_000, target_size * 12))
    if candidate_size < n_rows:
        rng = np.random.default_rng(random_state)
        relevant_columns = [
            column for column in ["sample_name", *annotation_columns[:1]] if column in metadata
        ]
        required_positions = []
        for column in relevant_columns:
            first_occurrence = ~metadata[column].duplicated().to_numpy()
            required_positions.append(np.flatnonzero(first_occurrence))
        required_positions = (
            np.unique(np.concatenate(required_positions))
            if required_positions
            else np.empty(0, dtype=np.int64)
        )
        if required_positions.size > candidate_size:
            required_positions = required_positions[:candidate_size]

        random_positions = rng.choice(n_rows, size=candidate_size, replace=False, shuffle=False)
        candidate_positions = np.union1d(random_positions, required_positions)
        if candidate_positions.size > candidate_size:
            optional_positions = np.setdiff1d(
                candidate_positions, required_positions, assume_unique=True
            )
            optional_positions = rng.choice(
                optional_positions,
                size=candidate_size - required_positions.size,
                replace=False,
            )
            candidate_positions = np.sort(np.concatenate((required_positions, optional_positions)))
        candidate_metadata = metadata.iloc[candidate_positions][relevant_columns]
        selected_candidates = _stratified_positions(
            candidate_metadata,
            target_size,
            annotation_columns,
            random_state=random_state + 1,
        )
        return np.sort(candidate_positions[selected_candidates])

    sample_strata = (
        metadata["sample_name"].astype("string").fillna("NaN")
        if "sample_name" in metadata
        else pd.Series("all", index=metadata.index, dtype="string")
    )
    sample_codes, sample_categories = pd.factorize(sample_strata, sort=False)
    n_sample_categories = max(1, len(sample_categories))
    codes = sample_codes.astype(np.int64, copy=False)
    if annotation_columns:
        primary = annotation_columns[0]
        if primary in metadata:
            annotation_strata = metadata[primary].astype("string").fillna("NaN")
            annotation_codes, annotation_categories = pd.factorize(annotation_strata, sort=False)
            n_annotation_categories = max(1, len(annotation_categories))
            codes = codes * n_annotation_categories + annotation_codes

    n_groups = max(
        n_sample_categories,
        int(codes.max()) + 1,
    )
    counts = np.bincount(codes, minlength=n_groups)
    exact = target_size * counts / n_rows
    allocations = np.floor(exact).astype(np.int64)
    preserve_each_stratum = np.count_nonzero(counts) <= target_size
    if preserve_each_stratum:
        allocations[(counts > 0) & (allocations == 0)] = 1

    while allocations.sum() > target_size:
        candidates = np.flatnonzero(allocations > int(preserve_each_stratum))
        if candidates.size == 0:
            break
        candidate = candidates[np.argmax(allocations[candidates] - exact[candidates])]
        allocations[candidate] -= 1

    remainder_order = np.argsort(-(exact - np.floor(exact)))
    remaining = target_size - int(allocations.sum())
    while remaining > 0:
        progressed = False
        for group in remainder_order:
            if allocations[group] < counts[group]:
                allocations[group] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            break

    rng = np.random.default_rng(random_state)
    order = np.argsort(codes, kind="stable")
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    selected = []
    for group, allocation in enumerate(allocations):
        if allocation <= 0:
            continue
        members = order[offsets[group] : offsets[group + 1]]
        selected.append(rng.choice(members, size=int(allocation), replace=False))

    positions = np.sort(np.concatenate(selected).astype(np.int64, copy=False))
    if positions.size != target_size:
        raise RuntimeError(
            f"Spatial display sampling selected {positions.size:,} rows; expected {target_size:,}."
        )
    return positions


def _robust_trait_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    upper = float(np.quantile(finite, 0.999))
    return upper if upper > 0 else max(1.0, float(np.max(finite)))


def _trait_q95(values: np.ndarray) -> float:
    """Return the shared per-trait upper anchor used by signal opacity."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return max(0.0, float(np.quantile(finite, 0.95)))


def _finite_range(values: np.ndarray | None) -> float:
    if values is None:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.max(finite) - np.min(finite))


def _point_size_3d(n_points: int) -> float:
    """Choose a conservative screen-space size from the displayed spot count.

    Anchor 100,000 displayed spots at size 1.0 (about 3 CSS pixels in the
    renderer). Inverse-square-root scaling roughly preserves total glyph area,
    with bounds matching the viewer slider. This is not a physical cell size.
    """
    return round(float(np.clip(np.sqrt(100_000.0 / max(1, n_points)), 0.5, 4.0)), 1)


def _point_geometry_2d(coordinates: np.ndarray) -> dict:
    """Measure robust display spacing, without making the browser build a KNN tree.

    Circular glyphs need a larger diameter on a square lattice than on a hexagonal
    lattice to cover the space between centers. These are display glyph sizes, not
    estimates of biological cell/spot diameters. Query at most 8192 positions against
    the full unique coordinate set; duplicates and invalid rows do not bias spacing.
    """
    from scipy.spatial import cKDTree

    xy = np.asarray(coordinates, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("2D point sizing requires an (n, 2) coordinate array")
    xy = np.unique(xy[np.isfinite(xy).all(axis=1)], axis=0)
    if len(xy) < 2:
        return {"spacing": 0.0, "diameter": 0.0, "lattice": "degenerate", "point_size": 6.0}
    rows = np.linspace(0, len(xy) - 1, min(len(xy), 8192), dtype=np.int64)
    distances, _ = cKDTree(xy).query(xy[rows], k=min(7, len(xy)), workers=1)
    q10, spacing, q90 = np.quantile(distances[:, 1], [0.1, 0.5, 0.9])
    lattice, coverage = "irregular", 1.15
    if len(xy) >= 32 and q10 > 0 and q90 / q10 < 1.2:
        # Six equidistant neighbors identify a hexagonal lattice; four identify
        # a square lattice. Sparse borders do not dominate these median ratios.
        neighbor_ratios = np.median(distances[:, 1:], axis=0) / spacing
        if neighbor_ratios[5] < 1.2:
            lattice, coverage = "hexagonal", 1.15
        elif neighbor_ratios[3] < 1.2:
            lattice, coverage = "square", float(np.sqrt(2))
    diameter = float(spacing * coverage)
    extent = float(np.ptp(xy, axis=0).max())
    # Fallback for consumers without a measured viewport. The browser uses the
    # data-unit diameter with its actual equal-aspect axes, including after zoom.
    pixels = diameter * 560 / extent if extent > 0 else 6.0
    return {
        "spacing": float(spacing),
        "diameter": diameter,
        "lattice": lattice,
        "point_size": round(float(np.clip(pixels, 1.25, 32.0)), 2),
    }


def _point_size_2d(coordinates: np.ndarray) -> float:
    return _point_geometry_2d(coordinates)["point_size"]


def _outputs_exist(root: Path, outputs: list[str]) -> bool:
    return bool(outputs) and all(
        (root / relative_path).is_file() and (root / relative_path).stat().st_size > 0
        for relative_path in outputs
    )


def export_spatial_3d_payloads(
    metadata: pd.DataFrame,
    traits: list[str],
    annotations: list[str],
    report_data_dir: Path,
    web_report_dir: Path,
    max_points: int,
    force: bool = False,
    coordinate_info: dict | None = None,
) -> dict | None:
    """Create shared 3D coordinates and lazy feature payloads for Plotly rendering."""

    coordinate_columns = ["3d_x", "3d_y", "3d_z"]
    if any(column not in metadata for column in coordinate_columns):
        logger.warning("3D coordinates are incomplete; skipping the 3D report viewer.")
        return None

    all_coordinates = np.column_stack(
        [
            pd.to_numeric(metadata[column], errors="coerce").to_numpy(dtype=np.float32)
            for column in coordinate_columns
        ]
    )
    if not np.isfinite(all_coordinates).any(axis=0).all():
        logger.warning("At least one 3D coordinate axis has no finite values; skipping 3D.")
        return None

    coordinate_bounds = {
        column: [
            float(np.nanmin(all_coordinates[:, index])),
            float(np.nanmax(all_coordinates[:, index])),
        ]
        for index, column in enumerate(coordinate_columns)
    }
    coordinate_ranges = [bounds[1] - bounds[0] for bounds in coordinate_bounds.values()]

    available_traits = [trait for trait in traits if trait in metadata]
    available_annotations = [annotation for annotation in annotations if annotation in metadata]
    target_size = min(len(metadata), max(1, int(max_points)))
    metadata_file = report_data_dir / "spot_metadata.csv"
    source_signature = None
    if metadata_file.is_file():
        source_stat = metadata_file.stat()
        source_signature = {
            "inode": source_stat.st_ino,
            "size": source_stat.st_size,
            "mtime_ns": source_stat.st_mtime_ns,
        }
    signature = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "n_total": len(metadata),
        "n_display": target_size,
        "point_size": _point_size_3d(target_size),
        "traits": available_traits,
        "annotations": available_annotations,
        "coordinate_info": coordinate_info,
        "source": source_signature,
        "coordinate_bounds": coordinate_bounds,
    }

    data_dir = report_data_dir / "spatial_3d"
    provenance_path = data_dir / "spatial_3d_display_spots.parquet"
    manifest_path = data_dir / "spatial_3d_manifest.json"
    if not force and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        if (
            manifest.get("signature") == signature
            and _outputs_exist(web_report_dir, manifest.get("outputs", []))
            and provenance_path.is_file()
        ):
            logger.info("Compact 3D viewer payloads are current. Skipping export.")
            return manifest.get("viewer_info")

    logger.info(
        f"Exporting compact 3D viewer payloads for {len(metadata):,} spots; "
        f"browser display sample={target_size:,}"
    )
    started = time.perf_counter()
    positions = _stratified_positions(metadata, target_size, available_annotations)
    selected = metadata.iloc[positions]
    coordinates = all_coordinates[positions]

    sample_payload = None
    if "sample_name" in selected:
        sample_payload = _categorical_payload(selected["sample_name"])
    annotation_schemas = {
        annotation: _categorical_schema(metadata[annotation])
        for annotation in available_annotations
    }
    base_payload = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "n_points": target_size,
        "n_points_original": len(metadata),
        "x": encode_typed_array(coordinates[:, 0]),
        "y": encode_typed_array(coordinates[:, 1]),
        "z": encode_typed_array(coordinates[:, 2]),
        "coordinate_ranges": coordinate_ranges,
        "sample": sample_payload,
        "point_size": signature["point_size"],
    }
    base_relative = "js_data/spatial_3d/spatial_3d_base.js"
    _atomic_write_text(
        web_report_dir / base_relative,
        "window.GSMAP_SPATIAL_3D_BASE="
        + json.dumps(base_payload, ensure_ascii=False, separators=(",", ":"))
        + ";",
    )

    trait_index = {}
    outputs = [base_relative]
    for trait in available_traits:
        values = pd.to_numeric(metadata[trait], errors="coerce").to_numpy(dtype=np.float32)
        payload = {
            "values": encode_typed_array(values[positions]),
            "cmin": 0.0,
            "cmax": _robust_trait_max(values),
            "q95": _trait_q95(values),
            "scale_scope": "all_spots",
        }
        filename = f"trait_{trait_file_token(trait)}.js"
        relative = f"js_data/spatial_3d/{filename}"
        _write_keyed_payload(web_report_dir / relative, "GSMAP_SPATIAL_3D_TRAITS", trait, payload)
        trait_index[trait] = {"file": relative}
        outputs.append(relative)

    annotation_index = {}
    for annotation in available_annotations:
        categories, colors = annotation_schemas[annotation]
        payload = _categorical_payload(selected[annotation], categories, colors)
        filename = f"annotation_{safe_name(annotation)}.js"
        relative = f"js_data/spatial_3d/{filename}"
        _write_keyed_payload(
            web_report_dir / relative,
            "GSMAP_SPATIAL_3D_ANNOTATIONS",
            annotation,
            payload,
        )
        annotation_index[annotation] = {"file": relative}
        outputs.append(relative)

    index_payload = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "base_file": base_relative,
        "n_points": target_size,
        "n_points_original": len(metadata),
        "downsampled": target_size < len(metadata),
        "traits": trait_index,
        "annotations": annotation_index,
        "colorscale": P_VALUE_COLORS,
        "coordinate_info": coordinate_info,
    }
    index_relative = "js_data/spatial_3d/spatial_3d_index.js"
    _atomic_write_text(
        web_report_dir / index_relative,
        "window.GSMAP_SPATIAL_3D_INDEX="
        + json.dumps(index_payload, ensure_ascii=False, separators=(",", ":"))
        + ";",
    )
    outputs.append(index_relative)

    provenance = pd.DataFrame(
        {
            "display_index": np.arange(target_size, dtype=np.int64),
            "source_position": positions,
            "spot": selected["spot"].astype(str).to_numpy()
            if "spot" in selected
            else selected.index.astype(str).to_numpy(),
            "sample_name": selected["sample_name"].astype(str).to_numpy()
            if "sample_name" in selected
            else "all",
        }
    )
    _atomic_to_parquet(provenance, provenance_path)

    viewer_info = {
        "schema_version": SPATIAL_PAYLOAD_VERSION,
        "index_file": index_relative,
        "n_points": target_size,
        "n_points_original": len(metadata),
        "downsampled": target_size < len(metadata),
        "point_size": base_payload["point_size"],
        "coordinate_info": coordinate_info,
    }
    manifest = {
        "version": SPATIAL_PAYLOAD_VERSION,
        "signature": signature,
        "outputs": sorted(outputs),
        "viewer_info": viewer_info,
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )
    elapsed = time.perf_counter() - started
    logger.info(
        f"Compact 3D viewer export complete in {elapsed:.2f}s: "
        f"{len(outputs)} payload files, {target_size:,}/{len(metadata):,} display spots"
    )
    return viewer_info


def export_sample_spatial_payloads(
    metadata: pd.DataFrame,
    samples: list[str],
    traits: list[str],
    annotations: list[str],
    js_data_dir: Path,
    max_points: int,
    coordinate_info: dict | None = None,
) -> tuple[dict, list[str]]:
    """Write compact per-sample bases and independently loadable trait arrays.

    Trait colors share one robust scale across every sample. Display spots are
    sampled once per tissue and reused by every trait.
    """

    payload_dir = js_data_dir / "spatial_2d"
    payload_dir.mkdir(parents=True, exist_ok=True)
    sample_index = {}
    outputs = []

    sample_values = metadata["sample_name"].astype(str).to_numpy()
    has_spatial_coordinates = "sx" in metadata and "sy" in metadata
    sx_all = (
        pd.to_numeric(metadata["sx"], errors="coerce").to_numpy(dtype=np.float32)
        if has_spatial_coordinates
        else None
    )
    sy_all = (
        pd.to_numeric(metadata["sy"], errors="coerce").to_numpy(dtype=np.float32)
        if has_spatial_coordinates
        else None
    )
    sample_positions = {}
    sample_all_positions = {}
    annotation_schemas = {
        annotation: _categorical_schema(metadata[annotation])
        for annotation in annotations
        if annotation in metadata
    }
    annotation_codes = _categorical_code_arrays(metadata, annotation_schemas)
    section_planes = (coordinate_info or {}).get("section_planes", {})

    for sample_name in samples:
        all_positions = np.flatnonzero(sample_values == str(sample_name))
        if all_positions.size == 0:
            raise ValueError(f"No spot metadata found for configured sample: {sample_name}")
        n_original = int(all_positions.size)
        if has_spatial_coordinates and n_original > max_points:
            local_positions = _stratified_positions(
                metadata.iloc[all_positions],
                max_points,
                annotations,
                random_state=42,
            )
            positions = all_positions[local_positions]
        else:
            local_positions = np.arange(n_original, dtype=np.int64)
            positions = all_positions
        sample_positions[sample_name] = positions
        sample_all_positions[sample_name] = all_positions

        if has_spatial_coordinates:
            sx = sx_all[positions]
            sy = sy_all[positions]
            point_geometry = _point_geometry_2d(np.column_stack((sx, sy)))
            point_size = point_geometry["point_size"]
        else:
            sx = sy = None
            point_size = 5.0
            point_geometry = None

        annotation_payloads = {}
        for annotation, (categories, colors) in annotation_schemas.items():
            codes, dtype = annotation_codes[annotation]
            annotation_payloads[annotation] = {
                "codes": encode_typed_array(codes[positions], dtype=dtype),
                "categories": categories,
                "colors": colors,
            }
        include_spot_labels = len(positions) <= 50_000
        section_plane = section_planes.get(str(sample_name), {})
        base_payload = {
            "version": SPATIAL_PAYLOAD_VERSION,
            "n_points": int(len(positions)),
            "n_points_original": n_original,
            "point_size": round(float(point_size), 2),
            "point_geometry": point_geometry,
            "coordinate_ranges": [
                _finite_range(sx_all[all_positions]) if sx_all is not None else 0.0,
                _finite_range(sy_all[all_positions]) if sy_all is not None else 0.0,
            ],
            "axis_titles": section_plane.get("axis_titles", ["X", "Y"]),
            "cut_axis_title": section_plane.get("cut_axis_title"),
            "sx": encode_typed_array(sx) if sx is not None else None,
            "sy": encode_typed_array(sy) if sy is not None else None,
            "spot": metadata["spot"].iloc[positions].astype(str).tolist()
            if include_spot_labels and "spot" in metadata
            else None,
            "annotations": annotation_payloads,
        }

        safe_sample = safe_name(sample_name)
        base_filename = f"sample_{safe_sample}_base.js"
        base_relative = f"spatial_2d/{base_filename}"
        _write_keyed_payload(
            js_data_dir / base_relative, "GSMAP_SAMPLE_BASES", sample_name, base_payload
        )
        outputs.append(base_relative)

        sample_index[sample_name] = {
            "version": SPATIAL_PAYLOAD_VERSION,
            "base_file": base_relative,
            "trait_files": {},
            "n_spots": int(len(positions)),
            "n_spots_original": n_original,
            "point_size": round(float(point_size), 2),
            "has_spot_labels": include_spot_labels,
        }

    # Convert each full trait column once, then export the selected rows for every sample.
    for trait in traits:
        if trait not in metadata:
            continue
        all_values = pd.to_numeric(metadata[trait], errors="coerce").to_numpy(dtype=np.float32)
        quantization_scale = 0.1
        maximum_encoded_value = (np.iinfo(np.uint16).max - 1) * quantization_scale
        shared_cmax = min(_robust_trait_max(all_values), maximum_encoded_value)
        shared_q95 = min(_trait_q95(all_values), maximum_encoded_value)
        for sample_name in samples:
            positions = sample_positions[sample_name]
            values = all_values[positions]
            all_positions = sample_all_positions[sample_name]
            sample_values = all_values[all_positions]
            payload = {
                "values": encode_nonnegative_quantized_array(values, quantization_scale),
                "cmin": 0.0,
                "cmax": shared_cmax,
                "q95": shared_q95,
                "n_clipped": int(np.count_nonzero(sample_values > shared_cmax)),
                "scale_scope": "all_samples",
            }
            filename = f"sample_{safe_name(sample_name)}_trait_{trait_file_token(trait)}.js"
            relative = f"spatial_2d/{filename}"
            compound_key = f"{sample_name}\x1f{trait}"
            _write_keyed_payload(
                js_data_dir / relative, "GSMAP_SAMPLE_TRAITS", compound_key, payload
            )
            sample_index[sample_name]["trait_files"][trait] = {"file": relative}
            outputs.append(relative)

    for sample_name in samples:
        info = sample_index[sample_name]
        logger.info(
            f"Exported compact sample payload: {sample_name}, "
            f"{info['n_spots']:,}/{info['n_spots_original']:,} spots, "
            f"{len(info['trait_files'])} lazy traits"
        )

    return sample_index, outputs
