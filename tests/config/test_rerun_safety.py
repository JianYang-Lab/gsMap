"""Reruns: warnings for outputs made differently, atomic outputs and resume defaults."""

import json
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest

from gsMap.config.fingerprint import record_fingerprint, warn_if_outputs_differ
from gsMap.utils.atomic_io import atomic_output, atomic_write_text

# ---------------------------------------------------------------- fingerprints


def _cauchy_config(tmp_path, **changes):
    results = tmp_path / "spatial_ldsc"
    results.mkdir(exist_ok=True)
    values = dict(
        annotation_list=["celltype"],
        sumstats_config_dict={"IQ": tmp_path / "IQ.sumstats.gz"},
        concatenated_latent_adata_path=tmp_path / "latent.h5ad",
        cauchy_save_dir=tmp_path / "cauchy_combination",
        get_ldsc_result_file=lambda trait: results / f"p_{trait}.csv.gz",
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def cauchy_outputs(tmp_path):
    (tmp_path / "latent.h5ad").write_text("x")
    (tmp_path / "spatial_ldsc").mkdir()
    (tmp_path / "spatial_ldsc" / "p_IQ.csv.gz").write_text("first run")
    record_fingerprint("cauchy", _cauchy_config(tmp_path))
    return tmp_path


def test_unchanged_rerun_reuses_outputs_silently(cauchy_outputs, caplog):
    with caplog.at_level(logging.WARNING):
        assert warn_if_outputs_differ("cauchy", _cauchy_config(cauchy_outputs))
    assert not caplog.records


def test_changed_setting_is_named_in_a_warning(cauchy_outputs, caplog):
    config = _cauchy_config(cauchy_outputs, annotation_list=["celltype", "region"])
    assert not warn_if_outputs_differ("cauchy", config)
    assert "annotation_list" in caplog.text
    assert "delete the cauchy_combination directory" in caplog.text


def test_rerun_upstream_result_is_named_in_a_warning(cauchy_outputs, caplog):
    (cauchy_outputs / "spatial_ldsc" / "p_IQ.csv.gz").write_text("rerun with a corrected GWAS")
    assert not warn_if_outputs_differ("cauchy", _cauchy_config(cauchy_outputs))
    assert "input ldsc:IQ" in caplog.text


def test_missing_or_unreadable_fingerprints_never_block(tmp_path, caplog):
    config = _cauchy_config(tmp_path)
    assert warn_if_outputs_differ("cauchy", config)  # outputs from an older gsMap
    config.cauchy_save_dir.mkdir(parents=True)
    (config.cauchy_save_dir / "fingerprint.json").write_text("{not json")
    with caplog.at_level(logging.WARNING):
        assert warn_if_outputs_differ("cauchy", config)
    assert not caplog.records


# ---------------------------------------------------------------- atomic outputs


def test_interrupted_write_keeps_the_previous_file(tmp_path):
    target = tmp_path / "out.h5ad"
    atomic_write_text(target, "complete previous output")

    def interrupted_write():
        with atomic_output(target) as temporary:
            temporary.write_text("half")
            raise RuntimeError("killed mid-write")

    with pytest.raises(RuntimeError):
        interrupted_write()
    assert target.read_text() == "complete previous output"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out.h5ad"]


def test_anndata_sidecar_is_replaced_atomically(tmp_path):
    target = tmp_path / "concatenated_latent_adata.h5ad"
    ad.AnnData(np.zeros((2, 2))).write_h5ad(target)
    with atomic_output(target) as temporary:
        ad.AnnData(np.ones((3, 2))).write_h5ad(temporary)
    assert ad.read_h5ad(target).shape == (3, 2)
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]


# ---------------------------------------------------------------- quick mode


def test_force_report_re_run_regenerates_an_existing_report(monkeypatch):
    from gsMap.pipeline import quick_mode

    calls = []
    monkeypatch.setattr(quick_mode, "check_report_done", lambda *args, **kwargs: True)
    monkeypatch.setattr(quick_mode, "run_report", lambda config: calls.append("report"))
    config = SimpleNamespace(
        start_step="report",
        stop_step=None,
        requires_spatial_inputs=False,
        sumstats_config_dict={},
        force_report_re_run=True,
    )
    quick_mode.run_quick_mode.__wrapped__(config)
    assert calls == ["report"]


def _write_input(path, n=12):
    data = ad.AnnData(np.ones((n, 4), dtype=np.float32))
    data.obs["celltype"] = "typeA"
    data.obsm["spatial"] = np.random.default_rng(1).random((n, 2))
    data.obsm["emb_cell"] = np.ones((n, 3), dtype=np.float32)
    data.write_h5ad(path)
    return path


@pytest.mark.parametrize("find_latent_done", [True, False])
def test_resume_at_latent2gene_uses_the_full_run_defaults(tmp_path, monkeypatch, find_latent_done):
    from gsMap.config import QuickModeConfig

    monkeypatch.setattr("gsMap.config.latent2gene_config.configure_jax_platform", lambda _: None)
    (tmp_path / "IQ.sumstats.gz").touch()
    source = _write_input(tmp_path / "s0.h5ad")
    if find_latent_done:
        metadata = tmp_path / "proj" / "find_latent_representations" / "find_latent_metadata.yaml"
        metadata.parent.mkdir(parents=True)
        metadata.write_text("outputs: {}\n")
    config = QuickModeConfig(
        workdir=tmp_path,
        project_name="proj",
        h5ad_path=[source],
        annotation="celltype",
        sumstats_config_dict={"IQ": tmp_path / "IQ.sumstats.gz"},
        start_step="latent2gene",
        use_gpu=False,
    )
    # A full run sets the niche key; a resumed run from its outputs now does too.
    assert config.latent_representation_niche == ("emb_niche" if find_latent_done else None)


def test_removed_rank_checkpoint_does_not_invalidate_final_outputs(tmp_path, monkeypatch):
    from gsMap.config import latent2gene_config
    from gsMap.latent2gene.memmap_io import MemMapDense

    paths = {
        name: tmp_path / file
        for name, file in [
            ("concatenated_latent_adata_path", "concatenated_latent_adata.h5ad"),
            ("rank_memmap_path", "ranks"),
            ("mean_frac_path", "mean_frac.parquet"),
            ("marker_scores_memmap_path", "marker_scores"),
            ("latent2gene_metadata_path", "metadata.yaml"),
        ]
    }
    for name, path in paths.items():
        if name != "rank_memmap_path":
            path.write_text("outputs: {}\n" if name == "latent2gene_metadata_path" else "x")
    config = SimpleNamespace(_resolve_h5ad_inputs=lambda: None, **paths)
    monkeypatch.setattr(MemMapDense, "check_complete", staticmethod(lambda path: (True, {})))
    assert latent2gene_config.check_latent2gene_done(config)

    # A rank checkpoint that exists but is incomplete still means "not done".
    Path(f"{paths['rank_memmap_path']}.meta.json").write_text("{}")
    monkeypatch.setattr(
        MemMapDense,
        "check_complete",
        staticmethod(lambda path: (Path(path).name != "ranks", {})),
    )
    assert not latent2gene_config.check_latent2gene_done(config)


# ---------------------------------------------------------------- scratch and names


def _dead_pid() -> int:
    process = subprocess.Popen([sys.executable, "-c", "pass"])
    process.wait()
    return process.pid


def test_only_orphaned_scratch_copies_from_this_host_are_removed(tmp_path):
    from gsMap.latent2gene.memmap_io import remove_orphaned_scratch

    def scratch(name, owner):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "data.dat").write_bytes(b"\0" * 16)
        if owner is not None:
            (directory / "gsmap_owner.json").write_text(json.dumps(owner))
        return directory

    host = socket.gethostname()
    orphan = scratch("memmap_dead", {"host": host, "pid": _dead_pid()})
    live = scratch("memmap_live", {"host": host, "pid": os.getppid()})
    remote = scratch("memmap_remote", {"host": host + "-other", "pid": _dead_pid()})
    legacy = scratch("memmap_legacy", None)
    other = scratch("unrelated", {"host": host, "pid": _dead_pid()})

    remove_orphaned_scratch(tmp_path)
    assert not orphan.exists()
    for kept in (live, remote, legacy, other):
        assert kept.exists()


def test_samples_sharing_a_report_file_name_fail_before_any_work():
    from gsMap.report.report import _check_report_sample_file_names

    _check_report_sample_file_names(["E1S1", "E1S2", "Mouse_E11.5"])
    with pytest.raises(ValueError, match="'E1-1' and 'E1_1'"):
        _check_report_sample_file_names(["E1-1", "E1_1", "E2"])
