"""Print the public gsMap Python API (documented entry points and config fields) as JSON.

tests/config/test_public_api.py compares the current API with
tests/config/public_api_snapshot.json. After an intended, documented API change,
regenerate the snapshot:

    python tests/tools/public_api_snapshot.py > tests/config/public_api_snapshot.json
"""

import dataclasses
import importlib
import inspect
import json
import sys

ENTRY_POINTS = [
    ("gsMap.pipeline", "run_quick_mode"),
    ("gsMap.find_latent", "run_find_latent_representation"),
    ("gsMap.latent2gene", "run_latent_to_gene"),
    ("gsMap.spatial_ldsc.spatial_ldsc_jax", "run_spatial_ldsc_jax"),
    ("gsMap.cauchy_combination_test", "run_Cauchy_combination"),
    ("gsMap.format_sumstats", "gwas_format"),
    ("gsMap.spatial_ldsc.io", "load_marker_scores_memmap_format"),
    ("gsMap.ldscore.pipeline", "LDScorePipeline"),
    ("gsMap.report", "run_report"),
]
CONFIGS = [
    "QuickModeConfig",
    "FindLatentRepresentationsConfig",
    "LatentToGeneConfig",
    "SpatialLDSCConfig",
    "CauchyCombinationConfig",
    "LDScoreConfig",
    "FormatSumstatsConfig",
    "ReportConfig",
]


def default(field):
    if field.default is not dataclasses.MISSING:
        return repr(field.default)
    if field.default_factory is not dataclasses.MISSING:
        return f"factory:{field.default_factory()!r}"
    return "required"


def build_snapshot() -> dict:
    snapshot = {"entry_points": {}, "configs": {}}
    for module, name in ENTRY_POINTS:
        obj = getattr(importlib.import_module(module), name)
        target = inspect.unwrap(obj.__init__ if inspect.isclass(obj) else obj)
        # Names, kinds and defaults are the contract; annotation text depends on the
        # installed library versions (e.g. anndata.AnnData vs anndata._core...).
        snapshot["entry_points"][f"{module}.{name}"] = [
            [
                param.name,
                param.kind.name,
                "required" if param.default is param.empty else repr(param.default),
            ]
            for param in inspect.signature(target).parameters.values()
        ]
    config = importlib.import_module("gsMap.config")
    snapshot["exports"] = sorted(name for name in dir(config) if not name.startswith("_"))
    from gsMap.config import DatasetType

    snapshot["DatasetType"] = sorted(member.value for member in DatasetType)
    for name in CONFIGS:
        cls = getattr(config, name)
        snapshot["configs"][name] = {f.name: default(f) for f in dataclasses.fields(cls) if f.init}
    return snapshot


if __name__ == "__main__":
    json.dump(build_snapshot(), sys.stdout, indent=1, sort_keys=True)
