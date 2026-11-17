"""Record what produced each step's outputs; warn when a rerun reuses outputs made differently.

Every step writes a small fingerprint next to its outputs when it completes: the settings
that change its results and the identity (path, size, modification time) of its inputs.
When a later run skips the step because outputs exist, the stored fingerprint is compared
with the current settings and inputs. Differences are logged as a warning that names what
changed and how to recompute. Outputs are still reused: this never stops or reruns a step.
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)

FINGERPRINT_VERSION = 1

# Settings that change each step's results (not threads, batch sizes or scratch paths).
STEP_SETTINGS: dict[str, tuple[str, ...]] = {
    "latent2gene": (
        "dataset_type",
        "annotation",
        "data_layer",
        "expression_data_type",
        "latent_representation_niche",
        "latent_representation_cell",
        "spatial_key",
        "spatial_neighbors",
        "homogeneous_neighbors",
        "cell_embedding_similarity_threshold",
        "spatial_domain_similarity_threshold",
        "homogeneous_selection_mode",
        "no_expression_fraction",
        "adjacent_slice_spatial_neighbors",
        "n_adjacent_slices",
        "cross_slice_selection_mode",
        "cross_slice_max_distance",
        "cross_slice_z_scale",
        "cross_slice_marker_score_strategy",
        "high_quality_neighbor_filter",
        "fix_cross_slice_homogenous_neighbors",
        "min_cells_per_type",
    ),
    "spatial_ldsc": (
        "chisq_max",
        "n_blocks",
        "cell_indices_range",
        "sample_filter",
        "marker_score_format",
    ),
    "cauchy": ("annotation_list",),
    "report": (
        "annotation_list",
        "top_corr_genes",
        "downsampling_n_spots_pcc",
        "downsampling_n_spots_umap",
        "downsampling_n_spots_3d",
        "downsampling_n_spots_2d",
        "plot_origin",
        "export_spatialvista",
        "spatialvista_max_spots",
    ),
}

RECOMPUTE_HINT = {
    "latent2gene": "delete the latent_to_gene directory (or use a new --workdir) to recompute",
    "spatial_ldsc": "delete this trait's result in spatial_ldsc to recompute it",
    "cauchy": "delete the cauchy_combination directory to recompute",
    "report": "use force_report_re_run or delete the report directories to regenerate",
}


def _plain(value: Any) -> Any:
    """JSON-ready value (paths as strings, enums as values, tuples as lists)."""
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(value, list | tuple | set | frozenset):
        items = [_plain(item) for item in value]
        return sorted(items, key=str) if isinstance(value, set | frozenset) else items
    if value is None or isinstance(value, bool | int | float | str):
        return value
    return str(value)


def file_identity(path: str | Path | None) -> dict[str, Any] | None:
    """Path, size and modification time of a file, or a digest of a directory listing."""
    if path is None:
        return None
    path = Path(path)
    try:
        resolved = str(path.resolve())
        if path.is_dir():
            entries = sorted(
                (child.name, child.stat().st_size, child.stat().st_mtime_ns)
                for child in path.iterdir()
                if child.is_file()
            )
            digest = hashlib.sha256(json.dumps(entries).encode()).hexdigest()[:16]
            return {"path": resolved, "files": len(entries), "listing": digest}
        stat = path.stat()
        return {"path": resolved, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    except OSError:
        return {"path": str(path), "missing": True}


def _step_inputs(step: str, config, trait: str | None) -> dict[str, Any]:
    """Identities of the inputs each step reads."""
    get = lambda name: getattr(config, name, None)  # noqa: E731
    if step == "latent2gene":
        upstream = get("find_latent_metadata_path")
        if upstream is not None and Path(upstream).is_file():
            return {"find_latent": file_identity(upstream)}
        return {
            f"h5ad:{name}": file_identity(path)
            for name, path in sorted((get("sample_h5ad_dict") or {}).items())
        }
    if step == "spatial_ldsc":
        sumstats = (get("sumstats_config_dict") or {}).get(trait)
        inputs = {
            "sumstats": file_identity(sumstats),
            "w_ld_dir": file_identity(get("w_ld_dir")),
            "snp_gene_weights": file_identity(get("snp_gene_weight_adata_path")),
            "ldscore_h5ad": file_identity(get("ldscore_h5ad_path")),
            "marker_scores": file_identity(get("marker_scores_memmap_path")),
        }
        for index, path in enumerate(get("additional_baseline_h5ad_path_list") or []):
            inputs[f"baseline_{index}"] = file_identity(path)
        return inputs
    if step == "cauchy":
        inputs = {"annotations": file_identity(get("concatenated_latent_adata_path"))}
        for name in sorted(get("sumstats_config_dict") or {}):
            inputs[f"ldsc:{name}"] = file_identity(config.get_ldsc_result_file(name))
        return inputs
    if step == "report":
        return {
            "combined_ldsc": file_identity(get("ldsc_combined_parquet_path")),
            "cauchy": file_identity(get("cauchy_save_dir")),
            "marker_scores": file_identity(get("marker_scores_memmap_path")),
        }
    raise ValueError(f"Unknown step: {step}")


def build_fingerprint(step: str, config, trait: str | None = None) -> dict[str, Any]:
    settings = {name: _plain(getattr(config, name, None)) for name in STEP_SETTINGS[step]}
    if step in ("cauchy", "report"):
        settings["traits"] = sorted(getattr(config, "sumstats_config_dict", None) or {})
    return {
        "version": FINGERPRINT_VERSION,
        "step": step,
        "trait": trait,
        "settings": settings,
        "inputs": _plain(_step_inputs(step, config, trait)),
    }


def fingerprint_path(step: str, config, trait: str | None = None) -> Path:
    if step == "latent2gene":
        return Path(config.latent2gene_metadata_path).with_name("fingerprint.json")
    if step == "spatial_ldsc":
        return Path(config.ldsc_save_dir) / "fingerprints" / f"{trait}.json"
    if step == "cauchy":
        return Path(config.cauchy_save_dir) / "fingerprint.json"
    if step == "report":
        return Path(config.report_data_dir) / "fingerprint.json"
    raise ValueError(f"Unknown step: {step}")


def record_fingerprint(step: str, config, trait: str | None = None) -> None:
    """Store the fingerprint of outputs that were just completed (never raises)."""
    from gsMap.utils.atomic_io import atomic_write_text

    try:
        payload = build_fingerprint(step, config, trait)
        atomic_write_text(
            fingerprint_path(step, config, trait), json.dumps(payload, indent=2) + "\n"
        )
    except Exception as error:  # noqa: BLE001 - bookkeeping must not fail a finished step
        logger.debug(f"Could not record the {step} fingerprint: {error}")


def _differences(stored: dict, current: dict) -> list[str]:
    changes = []
    for section in ("settings", "inputs"):
        old, new = stored.get(section) or {}, current.get(section) or {}
        for key in sorted(set(old) | set(new)):
            if old.get(key) != new.get(key):
                if section == "inputs":
                    path = (new.get(key) or old.get(key) or {}).get("path", key)
                    changes.append(f"input {key} ({path}) changed")
                else:
                    changes.append(f"{key}: {old.get(key)!r} -> {new.get(key)!r}")
    return changes


def warn_if_outputs_differ(step: str, config, trait: str | None = None) -> bool:
    """Warn when reused outputs were made with other settings or inputs.

    Returns False when differences were found. Missing fingerprints (outputs from an
    older gsMap) are only noted at debug level. Never raises.
    """
    try:
        path = fingerprint_path(step, config, trait)
        if not path.is_file():
            logger.debug(f"No fingerprint recorded for the {step} outputs at {path}")
            return True
        stored = json.loads(path.read_text())
        changes = _differences(stored, build_fingerprint(step, config, trait))
    except Exception as error:  # noqa: BLE001 - checking must never block a run
        logger.debug(f"Could not compare the {step} fingerprint: {error}")
        return True
    if changes:
        label = f"{step} ({trait})" if trait else step
        shown = "; ".join(changes[:8]) + (
            f"; and {len(changes) - 8} more" if len(changes) > 8 else ""
        )
        logger.warning(
            f"Reusing existing {label} outputs, but they were made with different settings "
            f"or inputs: {shown}. To recompute, {RECOMPUTE_HINT[step]}."
        )
        return False
    return True
