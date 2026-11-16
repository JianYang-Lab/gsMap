from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from PIL import Image
from scipy import sparse
from scipy.spatial import KDTree

from gsMap.report import report_data as rd
from gsMap.report.diagnosis import load_gwas_data
from gsMap.report.diagnostic_io import read_diagnostic_expression
from gsMap.report.report_data import _read_report_csv, _render_single_sample_gene_plot_task
from gsMap.report.table_io import read_metadata_companion, write_metadata


@pytest.mark.parametrize("recorded_limit", [10, 50, None, "corrupt", []])
def test_manhattan_cache_tracks_top_gene_limit(tmp_path, monkeypatch, recorded_limit):
    import json
    from types import SimpleNamespace

    from gsMap.report import report_data

    manager = report_data.ReportDataManager.__new__(report_data.ReportDataManager)
    manager.report_data_dir = tmp_path / "data"
    manager.web_report_dir = tmp_path / "web"
    manager.js_data_dir = manager.web_report_dir / "js_data"
    manager.report_data_dir.mkdir()
    manager.js_data_dir.mkdir(parents=True)
    manager.report_config = SimpleNamespace(
        top_corr_genes=10, snp_gene_weight_adata_path=tmp_path / "weights.h5ad"
    )
    manager.force_re_run = False
    manager.traits = ["IQ"]
    data = manager.report_data_dir / "manhattan_data"
    data.mkdir()
    (data / "IQ_manhattan.csv").write_text("cached")
    (manager.js_data_dir / "manhattan_IQ.js").write_text("cached")
    if recorded_limit is not None:
        contents = (
            "{"
            if recorded_limit == "corrupt"
            else json.dumps(
                recorded_limit
                if isinstance(recorded_limit, list)
                else {"top_corr_genes": recorded_limit}
            )
        )
        (manager.web_report_dir / "report_meta.json").write_text(contents)
    monkeypatch.setattr(report_data, "_load_chrom_tick_positions", lambda _: {1: 1.0})
    monkeypatch.setattr(report_data, "_manhattan_js_has_embedded_ticks", lambda _: True)
    monkeypatch.setattr(report_data, "trait_payload_current", lambda *_: True)

    class RebuildRequested(Exception):
        pass

    def load_weights(_):
        raise RebuildRequested

    monkeypatch.setattr(report_data.ad, "read_h5ad", load_weights)
    if recorded_limit == 10:
        manager.prepare_manhattan_data()
        assert manager.chrom_tick_positions == {"IQ": {1: 1.0}}
    else:
        with pytest.raises(RebuildRequested):
            manager.prepare_manhattan_data()


def test_metadata_csv_parquet_roundtrip_and_stale_cache(tmp_path):
    frame = pd.DataFrame(
        {
            "spot": ["01", "NA", "03"],
            "annotation": ["01", "NA", ""],
            "value": np.array([0, 1.2345678, np.nan], dtype=np.float32),
        }
    )
    path = tmp_path / "spot_metadata.csv"
    write_metadata(frame, path)
    pd.testing.assert_frame_equal(pd.read_parquet(path.with_suffix(".parquet")), frame)
    via_csv = pd.read_csv(path, keep_default_na=False)
    assert via_csv["spot"].tolist() == ["01", "NA", "03"]
    got = _read_report_csv(path, labels=["spot", "annotation"])
    assert got["spot"].tolist() == ["01", "NA", "03"]
    assert pd.isna(got["annotation"].iloc[-1])
    path.write_text("spot,annotation,value\nnew,new,10\n")
    assert read_metadata_companion(path) is None
    assert _read_report_csv(path)["spot"].tolist() == ["new"]


def test_metadata_companion_is_optional_and_corrupt_cache_falls_back(tmp_path):
    path = tmp_path / "spot_metadata.csv"
    write_metadata(pd.DataFrame({"spot": ["001"], "p": [0.1]}), path)
    path.with_suffix(".parquet").write_bytes(b"broken")
    assert _read_report_csv(path, labels=["spot"])["spot"].tolist() == ["001"]


@pytest.mark.parametrize("encoding", ["dense", "csr", "csc"])
@pytest.mark.parametrize("logged", [False, True])
def test_selective_expression_matches_original_all_gene_normalization(tmp_path, encoding, logged):
    matrix = np.array([[0, 0, 0, 0], [1, 4, 2, 10], [8, 0, 3, 11]], dtype=np.float32)
    if encoding != "dense":
        matrix = getattr(sparse, f"{encoding}_matrix")(matrix)
    a = ad.AnnData(
        X=matrix,
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame(index=["g0", "g1", "g2", "g3"]),
    )
    a.layers["counts"] = matrix.copy()
    if logged:
        sc.pp.normalize_total(a, target_sum=1e4)
        sc.pp.log1p(a)
    a.obsm["irrelevant"] = np.ones((3, 100))
    path = tmp_path / "sample.h5ad"
    a.write_h5ad(path)
    reference = a.copy()
    if not logged:
        sc.pp.normalize_total(reference, target_sum=1e4)
        sc.pp.log1p(reference)
    genes, got = read_diagnostic_expression(
        path, ["c|s", "a|s", "b|s", "c|s"], ["g2", "missing", "g0"], "s", "counts"
    )
    assert genes == ["g2", "g0"]
    expected = reference[["c", "a", "b", "c"], genes].X
    if sparse.issparse(expected):
        expected = expected.toarray()
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    with pytest.raises(KeyError):
        read_diagnostic_expression(path, ["absent|s"], ["g0"], "s", "counts")


def test_cached_geometry_preserves_png_pixels(tmp_path):
    coords = np.random.default_rng(4).normal(size=(500, 2))
    task = dict(
        gene="GENE",
        sample_name="sample",
        plot_type="gss",
        coords=coords,
        values=np.arange(500),
        output_path=tmp_path / "before.png",
    )
    _render_single_sample_gene_plot_task(task)
    distances, _ = KDTree(coords).query(coords, k=2)
    geometry = dict(
        nearest_neighbor_radius=float(distances[:, 1].mean() / 2),
        x_limits=(coords[:, 0].min(), coords[:, 0].max()),
        y_limits=(coords[:, 1].min(), coords[:, 1].max()),
    )
    _render_single_sample_gene_plot_task(
        {**task, "geometry": geometry, "output_path": tmp_path / "after.png"}
    )
    np.testing.assert_array_equal(
        np.asarray(Image.open(tmp_path / "before.png")),
        np.asarray(Image.open(tmp_path / "after.png")),
    )


def test_dense_count_probe_checks_past_first_row(tmp_path):
    a = ad.AnnData(
        X=np.array([[0, 0], [1.2, 2.3]], dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g0", "g1"]),
    )
    path = tmp_path / "sample.h5ad"
    a.write_h5ad(path)
    _, got = read_diagnostic_expression(path, ["b|s"], ["g0"], "s", "X")
    np.testing.assert_array_equal(got, [[np.float32(1.2)]])


def test_null_x_falls_back_to_counts_and_empty_selections(tmp_path):
    a = ad.AnnData(obs=pd.DataFrame(index=["a", "b"]), var=pd.DataFrame(index=["g0", "g1"]))
    a.layers["counts"] = sparse.csr_matrix([[1, 2], [2, 1]], dtype=np.float32)
    a.uns["log1p"] = {"base": None}
    path = tmp_path / "sample.h5ad"
    a.write_h5ad(path)
    _, got = read_diagnostic_expression(path, ["b|s"], ["g0"], "s", "counts")
    np.testing.assert_allclose(got, np.log1p([[1e4 * 2 / 3]]), rtol=1e-6)
    _, empty_rows = read_diagnostic_expression(path, [], ["g0"], "s", "counts")
    assert empty_rows.shape == (0, 1)
    _, empty_genes = read_diagnostic_expression(path, ["a|s"], [], "s", "counts")
    assert empty_genes.shape == (1, 0)


def test_manhattan_duplicate_snps_keep_first_in_weight_order(tmp_path):
    source = tmp_path / "trait.gz"
    pd.DataFrame({"SNP": ["rs2", "rs1", "rs2"], "Z": [2.0, 3.0, 20.0]}).to_csv(
        source, sep="\t", index=False, compression="gzip"
    )
    context = rd._ManhattanWeightContext(
        snp_index=pd.Index(["rs1", "rs2"]),
        chromosome=np.array([1, 1]),
        base_pair=np.array([100, 200]),
        matrix=sparse.csr_matrix([[2.0], [3.0]]),
        target_gene_indices=np.array([0]),
        target_genes=np.array(["g"]),
    )
    config = SimpleNamespace(sumstats_config_dict={"trait": source})
    rd._prepare_manhattan_for_trait(config, "trait", context, None, tmp_path)
    result = pd.read_csv(tmp_path / "trait_manhattan.csv")
    assert result["SNP"].tolist() == ["rs1", "rs2"]
    assert result["Z"].tolist() == [3.0, 2.0]


def test_gene_batch_preserves_order_and_render_settings(monkeypatch):
    received = []
    monkeypatch.setattr(rd, "_render_single_sample_gene_pair_task", received.append)
    task = dict(
        requests=[{"gene": "a"}, {"gene": "b"}],
        coords=np.ones((3, 2)),
        geometry={"nearest_neighbor_radius": 1.0},
        expression=np.arange(6).reshape(3, 2),
        markers=np.arange(6, 12).reshape(3, 2),
        plot_origin="lower",
        dpi=200,
        fig_width=7,
    )
    rd._render_sample_gene_batch_task(task)
    assert [item["gene"] for item in received] == ["a", "b"]
    for i, item in enumerate(received):
        np.testing.assert_array_equal(item["exp_values"], task["expression"][:, i])
        np.testing.assert_array_equal(item["gss_values"], task["markers"][:, i])
        assert item["coords"] is task["coords"]
        assert item["geometry"] is task["geometry"]
        assert item["dpi"] == 200
        assert item["fig_width"] == 7
        assert item["plot_origin"] == "lower"


def test_arrow_gwas_filter_preserves_columns_null_dtypes_and_duplicate_order(tmp_path):
    source = tmp_path / "gwas.gz"
    frame = pd.DataFrame(
        {
            "SNP": ["rs2", "absent", "rs1", "rs2", "NA", "absent2", "absent3"],
            "A1": ["A", "T", "C", "G", "G", "C", "A"],
            "Z": [2.0, 0.0, 3.0, 7.0, 0.0, np.nan, np.inf],
            "N": pd.array([100, None, 100, 200, 100, 100, 100], dtype="Int64"),
            "FLAG": [True, None, False, True, True, True, True],
        }
    )
    frame.to_csv(source, sep="\t", index=False, compression="gzip")
    reference = pd.Index(["rs1", "rs2"])
    baseline = load_gwas_data(source, calculate_p=False)
    expected = baseline.loc[baseline.SNP.isin(reference)].reset_index(drop=True)
    actual = load_gwas_data(source, calculate_p=False, snp_filter=reference)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual["SNP"].tolist() == ["rs2", "rs1", "rs2"]
    assert actual.attrs["source_rows"] == 7


def test_arrow_gwas_filter_preserves_empty_reference_error(tmp_path):
    source = tmp_path / "gwas.gz"
    pd.DataFrame({"SNP": ["rs1", "rs2"], "Z": [1.0, 2.0]}).to_csv(
        source, sep="\t", index=False, compression="gzip"
    )
    actual = load_gwas_data(source, calculate_p=False, snp_filter=pd.Index([], dtype=str))
    assert actual.empty
    assert actual.attrs["source_rows"] == 2
