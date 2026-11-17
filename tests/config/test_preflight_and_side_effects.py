"""Quick-mode preflight checks and side effects that gsMap must not leave behind."""

import gzip
import logging
import warnings
from types import SimpleNamespace

import pandas as pd
import pytest

from tests.logging_guard import preserved_gsmap_logging


def _ldsc_config(tmp_path, **changes):
    values = dict(
        start_step="find_latent",
        stop_step=None,
        requires_spatial_inputs=False,
        sumstats_config_dict={"IQ": tmp_path / "IQ.sumstats.gz"},
        get_ldsc_result_file=lambda trait: tmp_path / f"p_{trait}.csv.gz",
        get_cauchy_result_file=lambda trait, annotation, all_samples: tmp_path / "missing.csv",
        annotation_list=["celltype"],
        ldscore_save_dir=tmp_path / "generate_ldscore",
        w_ld_dir=None,
        snp_gene_weight_adata_path=tmp_path / "weights.h5ad",
        sample_h5ad_dict={},
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.fixture
def no_steps(monkeypatch):
    from gsMap.pipeline import quick_mode

    calls = []
    for name in ("run_find_latent_representation", "run_latent_to_gene", "run_report"):
        monkeypatch.setattr(quick_mode, name, lambda *args, _n=name: calls.append(_n))
    monkeypatch.setattr(quick_mode, "check_find_latent_done", lambda config: False)
    return quick_mode, calls


def test_missing_weights_fail_before_the_first_step(tmp_path, no_steps):
    quick_mode, calls = no_steps
    with pytest.raises(ValueError, match="w-ld-dir"):
        quick_mode.run_quick_mode.__wrapped__(_ldsc_config(tmp_path))
    assert calls == []  # previously found only when spatial LDSC started


def test_trait_named_like_a_metadata_column_fails_before_the_first_step(tmp_path, no_steps):
    quick_mode, calls = no_steps
    (tmp_path / "generate_ldscore" / "w_ld").mkdir(parents=True)
    (tmp_path / "weights.h5ad").touch()
    config = _ldsc_config(tmp_path, sumstats_config_dict={"celltype": tmp_path / "x.gz"})
    with pytest.raises(ValueError, match="rename the traits"):
        quick_mode.run_quick_mode.__wrapped__(config)
    assert calls == []


def test_finished_steps_are_not_checked_again(tmp_path):
    from gsMap.pipeline.quick_mode import _preflight

    (tmp_path / "p_IQ.csv.gz").touch()  # LDSC already done; its weights may have moved
    _preflight(_ldsc_config(tmp_path), ["spatial_ldsc"])


def test_a_logging_session_restores_the_logger_it_found(tmp_path):
    from gsMap.logging_config import stage_logging

    with preserved_gsmap_logging():
        logger = logging.getLogger("gsMap")
        logger.setLevel(logging.WARNING)
        logger.propagate = True
        with stage_logging(tmp_path, "latent_to_gene"):
            assert logger.propagate is False
        assert logger.level == logging.WARNING
        assert logger.propagate is True


def test_importing_the_report_keeps_other_warnings_visible():
    import gsMap.report.visualize  # noqa: F401

    blanket = [
        entry
        for entry in warnings.filters
        if entry[0] == "ignore" and entry[1] is None and entry[2] is Warning and entry[3] is None
    ]
    assert blanket == []


@pytest.mark.parametrize("compression", [None, "gzip", "bz2"])
def test_report_reads_gwas_files_that_spatial_ldsc_accepts(tmp_path, compression):
    from gsMap.report.diagnosis import load_gwas_data

    suffix = {None: ".txt", "gzip": ".txt.gz", "bz2": ".txt.bz2"}[compression]
    path = tmp_path / f"trait{suffix}"
    frame = pd.DataFrame(
        {"SNP": ["rs1", "rs2", "rs3"], "A1": ["A", "C", "G"], "Z": [1.0, -2.0, 3.5]}
    )
    text = frame.to_string(index=False)  # space-separated, like many GWAS files
    if compression == "gzip":
        path.write_bytes(gzip.compress(text.encode()))
    elif compression == "bz2":
        import bz2

        path.write_bytes(bz2.compress(text.encode()))
    else:
        path.write_text(text)
    data = load_gwas_data(path, usecols=["SNP", "Z"])
    assert data["SNP"].tolist() == ["rs1", "rs2", "rs3"]
    assert "P" in data.columns
