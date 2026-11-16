import json
import logging
import shlex
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from gsMap.cli import app
from gsMap.config import QuickModeConfig
from gsMap.config.report_config import (
    ReportConfig,
    should_export_spatialvista,
    spatialvista_output_is_current,
)
from gsMap.config.spatial import report_geometry
from gsMap.report.spatialvista import export_spatialvista


@pytest.fixture
def project(tmp_path):
    project = tmp_path / "embryo"
    (project / "report_data").mkdir(parents=True)
    (project / "gsmap_web_report").mkdir()
    metadata = pd.DataFrame(
        {
            "spot": ["d|S2", "a|S1", "c|S2", "b|S1"],
            "sample_name": ["S2", "S1", "S2", "S1"],
            "annotation": ["01", "NA", "Heart", ""],
            "3d_x": [40.0, 10.0, 30.0, 20.0],
            "3d_y": [1.0, 2.0, 3.0, 4.0],
            "3d_z": [20.0, 10.0, 20.0, 10.0],
            "IQ": [4.0, 1.0, np.nan, 2.0],
            "AF": [1.0, 2.0, 3.0, 4.0],
            "a/b": [1.0, 2.0, 3.0, 4.0],
            "a_b": [4.0, 3.0, 2.0, 1.0],
        }
    )
    metadata.to_csv(project / "report_data/spot_metadata.csv", index=False)
    (project / "gsmap_web_report/report_meta.json").write_text(
        json.dumps(
            {
                "traits": ["IQ", "AF", "a/b", "a_b"],
                "annotations": ["annotation"],
                "samples": ["S1", "S2"],
                "is_3d": True,
            }
        )
    )
    (project / "report_data/spatial_3d_coordinate_info.json").write_text(
        json.dumps({"z_source": "ordinal_section_order", "z_is_physical": False})
    )
    return project


def test_export_preserves_rows_labels_values_and_geometry(project):
    output = export_spatialvista(project)
    result = ad.read_h5ad(output)
    assert result.X is None
    assert result.shape == (4, 0)
    assert result.obs_names.tolist() == ["d|S2", "a|S1", "c|S2", "b|S1"]
    assert result.obs.annotation.tolist() == ["01", "NA", "Heart", "Unknown"]
    assert result.obs.sample_name.cat.categories.tolist() == ["S1", "S2"]
    np.testing.assert_equal(result.obsm["spatial"][:, 0], [40, 10, 30, 20])
    schema = result.uns["gsmap"]
    assert schema["schema_version"] == 1
    assert schema["value_scale"] == "neg_log10_p"
    assert schema["sampling"] == "all"
    assert len(schema["traits"]) == 4
    key = next(key for key, label in schema["traits"].items() if label == "IQ")
    np.testing.assert_allclose(result.obs[key], [4, 1, np.nan, 2], equal_nan=True)
    assert all("/" not in key for key in schema["traits"])
    assert not json.loads(schema["coordinate_provenance_json"])["z_is_physical"]


def test_export_overwrite_guard_and_cleanup(project):
    output = export_spatialvista(project)
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        export_spatialvista(project)
    assert output.read_bytes() == before
    export_spatialvista(project, overwrite=True)
    assert not list(project.glob(".embryo.*.h5ad"))


def test_trait_selection_and_deterministic_sampling(project, tmp_path):
    a = ad.read_h5ad(export_spatialvista(project, tmp_path / "a.h5ad", traits=["AF"], max_spots=2))
    b = ad.read_h5ad(export_spatialvista(project, tmp_path / "b.h5ad", traits=["AF"], max_spots=2))
    assert a.obs_names.tolist() == b.obs_names.tolist()
    assert list(a.uns["gsmap"]["traits"].values()) == ["AF"]
    assert a.uns["gsmap"]["source_n_spots"] == 4
    assert a.n_obs == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"traits": []},
        {"traits": ["missing"]},
        {"traits": ["IQ", "IQ"]},
        {"annotations": []},
        {"annotations": ["missing"]},
        {"max_spots": 0},
        {"max_spots": True},
        {"output": "bad.txt"},
    ],
)
def test_bad_options(project, kwargs):
    with pytest.raises(ValueError):
        export_spatialvista(project, **kwargs)


@pytest.mark.parametrize(
    ("column", "value"), [("3d_x", np.inf), ("IQ", -1), ("IQ", np.inf), ("sample_name", "unknown")]
)
def test_bad_metadata(project, column, value):
    path = project / "report_data/spot_metadata.csv"
    data = pd.read_csv(path)
    data.loc[0, column] = value
    data.to_csv(path, index=False)
    with pytest.raises(ValueError):
        export_spatialvista(project)
    assert not list(project.glob("*.h5ad"))


def test_duplicate_rows_rejected(project):
    path = project / "report_data/spot_metadata.csv"
    data = pd.read_csv(path)
    data.loc[0, "spot"] = data.loc[1, "spot"]
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        export_spatialvista(project)


@pytest.mark.parametrize("values", [[True] * 4, [1 + 2j] * 4, ["1"] * 4])
def test_export_rejects_non_real_scores(project, values):
    metadata = pd.read_csv(project / "report_data/spot_metadata.csv")
    metadata["IQ"] = values
    with pytest.raises(ValueError, match="real numeric"):
        export_spatialvista(project, metadata=metadata)


@pytest.mark.parametrize("maximum", [None, 2])
def test_report_hook_exports_in_memory_metadata(project, monkeypatch, caplog, maximum):
    from gsMap.report import report

    config = SimpleNamespace(
        dataset_type="spatial3D",
        project_dir=project,
        project_name=project.name,
        report_data_dir=project / "report_data",
        spatialvista_max_spots=maximum,
    )
    manager = Mock()
    manager.metadata = pd.read_csv(project / "report_data/spot_metadata.csv")
    # The manager starts with is_3d=False, before run() loads coordinates.
    manager.is_3d = False
    manager.run.return_value = project / "gsmap_web_report"
    monkeypatch.setattr(report, "ReportDataManager", Mock(return_value=manager))
    with caplog.at_level(logging.INFO):
        report.run_report.__wrapped__(config)
    manager.run.assert_called_once()
    exported = ad.read_h5ad(project / f"{project.name}.spatialvista.h5ad")
    assert exported.n_obs == (maximum or 4)
    assert exported.uns["gsmap"]["requested_max_spots"] == (maximum or 0)
    assert spatialvista_output_is_current(config)
    assert "Open gsMap report: gsmap report-view" in caplog.text
    assert "spatialvista --input" in caplog.text
    assert "Report generated successfully!" in caplog.text
    assert caplog.text.index("SpatialVista export completed") < caplog.text.index(
        "Report generated successfully!"
    )


@pytest.mark.parametrize(
    ("mode", "enabled"),
    [("spatial2D", True), ("scRNA", True), ("spatial3D", False)],
)
def test_non_3d_or_opt_out_skips_export(project, monkeypatch, caplog, mode, enabled):
    from gsMap.report import report

    config = SimpleNamespace(
        dataset_type=mode,
        export_spatialvista=enabled,
        project_dir=project,
        project_name=project.name,
        report_data_dir=project / "report_data",
    )
    manager = Mock()
    manager.run.return_value = project / "gsmap_web_report"
    monkeypatch.setattr(report, "ReportDataManager", Mock(return_value=manager))
    with caplog.at_level(logging.INFO):
        report.run_report.__wrapped__(config)
    manager.run.assert_called_once()
    assert not list(project.glob("*.h5ad"))
    assert "gsmap report-view" in caplog.text
    assert "spatialvista --input" not in caplog.text


@pytest.mark.parametrize("mode", ["spatial3D", "spatial2D", "scRNA"])
def test_default_export_is_specific_to_3d(mode):
    config = SimpleNamespace(dataset_type=mode)
    assert should_export_spatialvista(config) == (mode == "spatial3D")
    config.export_spatialvista = False
    assert not should_export_spatialvista(config)


def test_report_failure_does_not_announce_success(project, monkeypatch, caplog):
    from gsMap.report import report

    manager = Mock()
    manager.run.return_value = project / "gsmap_web_report"
    manager.metadata = pd.read_csv(project / "report_data/spot_metadata.csv")
    manager.metadata["IQ"] = -1
    monkeypatch.setattr(report, "ReportDataManager", Mock(return_value=manager))
    with caplog.at_level(logging.INFO), pytest.raises(ValueError):
        report.run_report.__wrapped__(
            SimpleNamespace(
                dataset_type="spatial3D", project_dir=project, project_name=project.name
            )
        )
    assert "Report generated successfully!" not in caplog.text
    assert "spatialvista --input" not in caplog.text
    assert not (project / "gsmap_web_report/index.html").exists()


def test_export_flag_requires_export_for_report_completion(project):
    from gsMap.config.quick_mode_config import get_report_missing_files

    config = SimpleNamespace(
        dataset_type="spatial3D",
        project_dir=project,
        report_data_dir=project / "report_data",
        web_report_dir=project / "gsmap_web_report",
        trait_name_list=[],
        annotation_list=[],
        generate_multi_sample_plots=False,
        concatenated_latent_adata_path=None,
        sample_h5ad_dict={},
    )
    output = project / f"{project.name}.spatialvista.h5ad"
    assert output in get_report_missing_files(config)[0]
    export_spatialvista(project)
    assert output not in get_report_missing_files(config)[0]
    config.spatialvista_max_spots = 2
    assert output in get_report_missing_files(config)[0]
    export_spatialvista(project, max_spots=2, overwrite=True)
    assert output not in get_report_missing_files(config)[0]
    output.write_bytes(b"")
    assert output in get_report_missing_files(config)[0]
    output.write_bytes(b"interrupted")
    assert output in get_report_missing_files(config)[0]
    config.export_spatialvista = False
    assert output not in get_report_missing_files(config)[0]


def test_report_config_and_cli_defaults():
    from click import Context
    from typer.main import get_command

    runner = CliRunner()
    result = runner.invoke(app, ["export-spatialvista", "--help"])
    assert result.exit_code != 0
    result = runner.invoke(app, ["quick-mode", "--help"], env={"COLUMNS": "240"})
    assert result.exit_code == 0
    assert "--export-spatialvista" in result.output
    assert "--no-export-spatialvista" in result.output
    assert "--spatialvista-max-spots" in result.output
    for config_type in (QuickModeConfig, ReportConfig):
        field_map = {field.name: field for field in fields(config_type)}
        assert field_map["export_spatialvista"].default is True
        assert field_map["spatialvista_max_spots"].default is None
    command = get_command(app).commands["quick-mode"]
    flag = next(param for param in command.params if param.name == "export_spatialvista")
    assert flag.default is True
    limit = next(param for param in command.params if param.name == "spatialvista_max_spots")
    assert limit.type.convert("20000", limit, Context(command)) == 20000


def test_report_completion_detects_changed_export_settings(project):
    from gsMap.config.quick_mode_config import _report_visualization_config_matches

    config = SimpleNamespace(
        dataset_type="spatial3D",
        web_report_dir=project / "gsmap_web_report",
        downsampling_n_spots_umap=50000,
        downsampling_n_spots_2d=300000,
        downsampling_n_spots_3d=200000,
        top_corr_genes=10,
        trait_name_list=["IQ"],
    )
    path = config.web_report_dir / "report_meta.json"
    metadata = dict(
        trait_labels={"IQ": "IQ"},
        trait_payload_version=2,
        spatial_layout="sections",
        spatial_geometry=report_geometry(config),
        downsampling_n_spots_umap=50000,
        downsampling_n_spots_2d=300000,
        downsampling_n_spots_3d=200000,
        top_corr_genes=10,
    )
    path.write_text(json.dumps(metadata))
    assert not _report_visualization_config_matches(config)
    metadata.update(export_spatialvista=True, spatialvista_max_spots=None)
    path.write_text(json.dumps(metadata))
    assert _report_visualization_config_matches(config)
    config.spatialvista_max_spots = 2
    assert not _report_visualization_config_matches(config)
    metadata["spatialvista_max_spots"] = 2
    path.write_text(json.dumps(metadata))
    assert _report_visualization_config_matches(config)


def test_failed_replacement_cannot_make_old_export_current(project):
    output = export_spatialvista(project)
    before = output.read_bytes()
    metadata = pd.read_csv(project / "report_data/spot_metadata.csv")
    metadata["IQ"] = -1
    with pytest.raises(ValueError):
        export_spatialvista(project, max_spots=2, metadata=metadata, overwrite=True)
    assert output.read_bytes() == before
    assert not spatialvista_output_is_current(
        SimpleNamespace(project_dir=project, spatialvista_max_spots=2)
    )


@pytest.mark.parametrize("maximum", [0, -1, True, 1.5])
def test_invalid_python_limit_fails_before_report_preparation(monkeypatch, maximum):
    from gsMap.report import report

    manager = Mock()
    monkeypatch.setattr(report, "ReportDataManager", manager)
    with pytest.raises(ValueError, match="spatialvista_max_spots"):
        report.run_report.__wrapped__(
            SimpleNamespace(dataset_type="spatial3D", spatialvista_max_spots=maximum)
        )
    manager.assert_not_called()


def test_completion_messages_quote_paths_and_are_saved(project, monkeypatch, caplog):
    from gsMap.report import report

    spaced = project.with_name("embryo [demo] space")
    project.rename(spaced)
    config = SimpleNamespace(
        dataset_type="spatial3D",
        project_dir=spaced,
        project_name=spaced.name,
        report_data_dir=spaced / "report_data",
    )
    manager = Mock()
    manager.metadata = pd.read_csv(spaced / "report_data/spot_metadata.csv")
    manager.run.return_value = spaced / "gsmap_web_report"
    monkeypatch.setattr(report, "ReportDataManager", Mock(return_value=manager))
    logfile = spaced / "report.log"
    handler = logging.FileHandler(logfile)
    report.logger.addHandler(handler)
    try:
        with caplog.at_level(logging.INFO):
            report.run_report.__wrapped__(config)
    finally:
        report.logger.removeHandler(handler)
        handler.close()
    log = logfile.read_text()
    output = spaced / f"{spaced.name}.spatialvista.h5ad"
    assert f"spatialvista --input {shlex.quote(str(output))}" in log
    assert f"gsmap report-view {shlex.quote(str(manager.run.return_value))}" in log
