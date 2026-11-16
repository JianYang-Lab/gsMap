"""Noncounts are an explicit external-embedding path, separate from count QC."""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pytest
from typer.testing import CliRunner

from gsMap.config import LatentToGeneConfig, QuickModeConfig
from gsMap.config.latent2gene_config import ExpressionDataType, validate_expression_contract


def test_expression_contract_defaults_to_strict_counts():
    assert validate_expression_contract(SimpleNamespace()) == ExpressionDataType.COUNTS
    assert LatentToGeneConfig.__dataclass_fields__["expression_data_type"].default == "counts"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"no_expression_fraction": False}, "requires no_expression_fraction"),
        ({"high_quality_neighbor_filter": True}, "high-quality cell filtering"),
        ({"high_quality_cell_qc": True}, "high-quality cell filtering"),
        ({"cell_qc_min_counts": 1}, "cannot use cell_qc_min_counts"),
    ],
)
def test_noncounts_rejects_count_only_filters(changes, message):
    params = dict(expression_data_type="noncounts", no_expression_fraction=True)
    params.update(changes)
    with pytest.raises(ValueError, match=message):
        validate_expression_contract(SimpleNamespace(**params))


def test_noncounts_quick_mode_rejects_find_latent_before_loading_inputs(tmp_path):
    with pytest.raises(ValueError, match="skip find_latent"):
        QuickModeConfig(
            workdir=tmp_path,
            project_name="noncounts",
            expression_data_type="noncounts",
            no_expression_fraction=True,
            start_step="find_latent",
        )


@pytest.mark.parametrize("entrypoint", ["quick", "standalone"])
def test_noncounts_config_supports_spatial_external_pca(tmp_path, monkeypatch, entrypoint):
    monkeypatch.setattr("gsMap.config.latent2gene_config.configure_jax_platform", lambda _: None)
    data = ad.AnnData(np.tile(np.arange(-12, 13, dtype=np.float32), (4, 1)))
    data.obs["celltype"] = "example"
    data.obsm["spatial"] = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    data.obsm["cell_emb_pca"] = np.ones((4, 2), dtype=np.float32)
    path = tmp_path / "pca.h5ad"
    data.write_h5ad(path)
    params = dict(
        workdir=tmp_path,
        project_name="noncounts",
        h5ad_path=[path],
        dataset_type="spatial2D",
        annotation="celltype",
        expression_data_type="noncounts",
        no_expression_fraction=True,
        latent_representation_cell="cell_emb_pca",
        use_gpu=False,
    )
    if entrypoint == "quick":
        sumstats = tmp_path / "trait.sumstats.gz"
        sumstats.touch()
        params["sumstats_config_dict"] = {"trait": sumstats}
        config = QuickModeConfig(**params, start_step="latent2gene", stop_step="latent2gene")
        config = config.latent2gene_config
    else:
        config = LatentToGeneConfig(**params)
    assert config.expression_data_type == "noncounts"
    assert config.latent_representation_cell == "cell_emb_pca"
    assert config.no_expression_fraction is True
    assert config.high_quality_neighbor_filter is False
    assert config.to_dict_with_paths_as_strings()["expression_data_type"] == "noncounts"


def test_standalone_cli_accepts_explicit_noncounts_and_cell_embedding(tmp_path, monkeypatch):
    from gsMap.cli import app

    monkeypatch.setattr("gsMap.config.latent2gene_config.configure_jax_platform", lambda _: None)
    data = ad.AnnData(np.tile(np.arange(-12, 13, dtype=np.float32), (4, 1)))
    data.obsm["spatial"] = np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    data.obsm["cell_emb_pca"] = np.ones((4, 2), dtype=np.float32)
    path = tmp_path / "pca.h5ad"
    data.write_h5ad(path)
    seen = []
    monkeypatch.setattr("gsMap.latent2gene.run_latent_to_gene", lambda config: seen.append(config))
    result = CliRunner().invoke(
        app,
        [
            "latent-to-gene",
            "--workdir",
            str(tmp_path),
            "--project-name",
            "cli",
            "--h5ad-path",
            str(path),
            "--dataset-type",
            "spatial2D",
            "--expression-data-type",
            "noncounts",
            "--latent-representation-cell",
            "cell_emb_pca",
            "--no-expression-fraction",
            "--no-gpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(seen) == 1
    assert seen[0].expression_data_type == "noncounts"
    assert seen[0].latent_representation_cell == "cell_emb_pca"
