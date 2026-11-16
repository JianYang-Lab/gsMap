"""Annotation module-score filtering is opt-in across entry points."""

import anndata as ad
import numpy as np
import pytest
from typer.main import get_command

from gsMap.cli import app
from gsMap.config import FindLatentRepresentationsConfig, LatentToGeneConfig, QuickModeConfig


@pytest.fixture(autouse=True)
def spatial_input(tmp_path):
    data = ad.AnnData(np.ones((5, 2)))
    data.obsm["spatial"] = np.zeros((5, 2))
    data.write_h5ad(tmp_path / "input.h5ad")


@pytest.mark.parametrize("config", [FindLatentRepresentationsConfig, QuickModeConfig])
def test_module_score_qc_defaults_off(config):
    assert config.__dataclass_fields__["high_quality_cell_qc"].default is False


@pytest.mark.parametrize("command", ["find-latent", "quick-mode"])
def test_cli_module_score_qc_defaults_off_with_explicit_opt_in(command):
    params = get_command(app).commands[command].params
    option = next(p for p in params if p.name == "high_quality_cell_qc")
    assert option.default is False
    assert "--high-quality-cell-qc" in option.opts
    assert "--no-high-quality-cell-qc" in option.secondary_opts


@pytest.mark.parametrize("enabled", [False, True])
def test_quick_mode_propagates_explicit_qc_choice(tmp_path, enabled):
    sumstats = tmp_path / "trait.sumstats.gz"
    sumstats.touch()
    config = QuickModeConfig(
        workdir=tmp_path,
        h5ad_path=[tmp_path / "input.h5ad"],
        project_name="qc",
        annotation="annotation",
        sumstats_config_dict={"trait": sumstats},
        start_step="find_latent",
        stop_step="latent2gene",
        high_quality_cell_qc=enabled,
    )
    assert config.high_quality_cell_qc is enabled
    assert config.high_quality_neighbor_filter is enabled


def test_marker_only_resume_does_not_enable_quality_filter(tmp_path):
    sumstats = tmp_path / "trait.sumstats.gz"
    sumstats.touch()
    config = QuickModeConfig(
        workdir=tmp_path,
        h5ad_path=[tmp_path / "input.h5ad"],
        project_name="qc",
        annotation="annotation",
        sumstats_config_dict={"trait": sumstats},
        start_step="latent2gene",
        stop_step="latent2gene",
    )
    assert config.high_quality_cell_qc is False
    assert config.high_quality_neighbor_filter is False
    assert LatentToGeneConfig.__dataclass_fields__["high_quality_neighbor_filter"].default is False
