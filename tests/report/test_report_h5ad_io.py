"""Report input reads must not materialize expression or homogeneous-neighbor arrays."""

import json
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from gsMap.report.report_data import _prepare_umap_data, _read_diagnostic_marker_values
from gsMap.utils import h5ad


@pytest.fixture
def source(tmp_path):
    data = ad.AnnData(
        X=np.ones((8, 3)),
        obs=pd.DataFrame(
            {
                "sample_name": pd.Categorical(["001"] * 4 + ["002"] * 4),
                "domain": pd.Categorical(["01", "02", "NA", None] * 2),
                "irrelevant": np.arange(8),
            },
            index=pd.Index([f"spot-{i}" for i in range(8)], name="cell_id"),
        ),
        var=pd.DataFrame(index=["a", "b", "c"]),
    )
    data.obsm["cell"] = np.arange(32, dtype=np.float32).reshape(8, 4)
    data.obsm["niche"] = data.obsm["cell"] + 100
    data.obsm["X_umap"] = data.obsm["cell"][:, :2].copy()
    data.obsm["X_umap_niche"] = data.obsm["niche"][:, :2].copy()
    data.obsm["unused_neighbors"] = np.ones((8, 200))
    data.layers["counts"] = data.X.copy()
    data.uns["kept"] = "metadata"
    data.varm["kept"] = np.ones((3, 2))
    path = tmp_path / "input.h5ad"
    data.write_h5ad(path)
    return path, data


def test_selected_metadata_never_reads_unused_arrays(source, monkeypatch):
    path, original = source
    read_elem = h5ad.read_elem
    reads = []

    def guard(node):
        reads.append(node.name)
        assert node.name not in {"/X", "/layers", "/obsm", "/obsm/unused_neighbors", "/obs"}
        return read_elem(node)

    monkeypatch.setattr(h5ad, "read_elem", guard)
    result = h5ad.read_h5ad_metadata(
        path, obs_columns=["sample_name", "domain"], obsm_keys=["cell", None, "missing"]
    )
    assert result.X is None
    assert not result.isbacked
    assert list(result.obsm) == ["cell"]
    assert not result.layers
    assert not result.uns
    assert not result.varm
    pd.testing.assert_frame_equal(result.obs, original.obs[["sample_name", "domain"]])
    pd.testing.assert_frame_equal(result.var, original.var)
    np.testing.assert_array_equal(result.obsm["cell"], original.obsm["cell"])
    assert "/obsm/cell" in reads


def test_metadata_wrapper_can_keep_uns_and_varm_without_obsm(source):
    path, original = source
    result = h5ad.read_h5ad_metadata(path, include_uns=True, include_varm=True)
    pd.testing.assert_frame_equal(result.obs, original.obs)
    assert result.uns["kept"] == "metadata"
    np.testing.assert_array_equal(result.varm["kept"], original.varm["kept"])
    assert not result.obsm


@pytest.mark.parametrize("encoding", ["dense", "csr", "csc", "dataframe"])
def test_embedding_subset_preserves_unsorted_and_duplicate_rows(tmp_path, encoding):
    matrix = np.arange(100001 * 3, dtype=np.float32).reshape(100001, 3)
    data = ad.AnnData(obs=pd.DataFrame(index=[f"s{i}" for i in range(len(matrix))]))
    value = {
        "dense": lambda: matrix,
        "csr": lambda: sparse.csr_matrix(matrix),
        "csc": lambda: sparse.csc_matrix(matrix),
        "dataframe": lambda: pd.DataFrame(matrix, index=data.obs_names, columns=["a", "b", "c"]),
    }[encoding]()
    data.obsm["embedding"] = value
    path = tmp_path / f"{encoding}.h5ad"
    data.write_h5ad(path)
    rows = np.array([100000, 4, 65536, 1, 65535, 4, 0])
    result = h5ad.read_obsm_rows(path, ["embedding", None, "missing"], rows)["embedding"]
    actual = result.toarray() if sparse.issparse(result) else np.asarray(result)
    np.testing.assert_array_equal(actual, matrix[rows])
    empty = h5ad.read_obsm_rows(path, ["embedding"], np.array([], dtype=int))["embedding"]
    assert empty.shape == (0, 3)
    with pytest.raises(IndexError):
        h5ad.read_obsm_rows(path, ["embedding"], np.array([-1]))
    with pytest.raises(IndexError):
        h5ad.read_obsm_rows(path, ["embedding"], np.array([len(matrix)]))


@pytest.mark.parametrize("limit", [4, 20])
def test_umap_preparation_uses_only_selected_embeddings_and_preserves_traits(
    source, monkeypatch, tmp_path, limit
):
    path, original = source
    config = SimpleNamespace(
        concatenated_latent_adata_path=path,
        latent_representation_cell="cell",
        latent_representation_niche="niche",
        annotation_list=["domain"],
        trait_name_list=["IQ", "Height"],
        downsampling_n_spots_umap=limit,
    )
    metadata = pd.DataFrame(
        {"spot": original.obs_names, "IQ": np.arange(8) + 0.2, "Height": np.arange(8) + 20.2}
    )
    seen = []

    def calculate(data, key, umap_key):
        assert data.X is None
        assert data.n_vars == 0
        assert set(data.obsm) == {"cell", "niche", "X_umap", "X_umap_niche"}
        rows = original.obs_names.get_indexer(data.obs_names)
        np.testing.assert_array_equal(data.obsm[key], original.obsm[key][rows])
        seen.append(data.obs_names.tolist())
        return data.obsm[umap_key].copy()

    monkeypatch.setattr("gsMap.report.report_data._calculate_umap_from_embeddings", calculate)
    monkeypatch.setattr(ad, "read_h5ad", lambda *args, **kwargs: pytest.fail("Full H5AD read"))
    info = _prepare_umap_data(config, metadata, tmp_path)
    result = pd.read_csv(
        tmp_path / "umap_data.csv", converters={"domain": str, "sample_name": str}
    )
    expected = metadata.set_index("spot").loc[result.spot]
    np.testing.assert_allclose(result.IQ, expected.IQ)
    np.testing.assert_allclose(result.Height, expected.Height)
    assert seen[0] == seen[1] == result.spot.tolist()
    assert info["n_points"] == min(limit, 8)
    assert set(result.sample_name) == {"001", "002"}


@pytest.mark.parametrize("n_workers", [1, 8])
@pytest.mark.parametrize("layout", ["source", "permuted", "offset", "view", "dense", "sparse"])
def test_diagnostic_marker_read_preserves_wrapper_order(tmp_path, layout, monkeypatch, n_workers):
    from gsMap.report import report_data

    read_rows = report_data._read_marker_score_rows
    calls = []

    def tracked(*args, **kwargs):
        calls.append(kwargs["n_workers"])
        return read_rows(*args, **kwargs)

    monkeypatch.setattr(report_data, "_read_marker_score_rows", tracked)
    values = np.arange(60, dtype=np.float32).reshape(12, 5)
    obs = pd.DataFrame(index=[f"spot-{i}" for i in range(12)])
    var = pd.DataFrame(index=[f"gene-{i}" for i in range(5)])
    if layout == "permuted":
        order = np.array([8, 3, 11, 2, 0, 6, 7, 9, 1, 10, 5, 4])
        values = values[order]
        obs = obs.iloc[order]
    path = tmp_path / "markers.dat"
    offset = 64 if layout == "offset" else 0
    matrix = np.memmap(path, mode="w+", dtype=np.float32, shape=values.shape, offset=offset)
    matrix[:] = values
    matrix.flush()
    x = matrix
    if layout == "view":
        x, obs = matrix[2:], obs.iloc[2:]
    elif layout == "dense":
        x = values.copy()
    elif layout == "sparse":
        x = sparse.csr_matrix(values)
    data = ad.AnnData(X=x, obs=obs, var=var)
    spots = [data.obs_names[7], data.obs_names[2], data.obs_names[7], data.obs_names[0]]
    genes = ["gene-4", "gene-0", "gene-2"]
    expected = np.asarray(x.toarray() if sparse.issparse(x) else x)[
        np.ix_(data.obs_names.get_indexer(spots), data.var_names.get_indexer(genes))
    ]
    result = _read_diagnostic_marker_values(data, spots, genes, n_workers=n_workers)
    np.testing.assert_array_equal(
        result.toarray() if sparse.issparse(result) else result, expected
    )
    assert calls == ([n_workers] if layout in {"source", "permuted"} else [])
    matrix._mmap.close()


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "embedding_source",
        "embedding_key",
        "annotations",
        "traits",
        "trait_values",
        "limit",
        "missing_js",
        "edited_csv",
        "corrupt_manifest",
        "non_dict_manifest",
        "force",
        "removed_embedding",
        "removed_source",
    ],
)
def test_umap_coordinate_cache_tracks_inputs_and_removes_stale_payloads(
    source, tmp_path, monkeypatch, change
):
    import h5py

    from gsMap.report import report_data

    path, original = source
    manager = report_data.ReportDataManager.__new__(report_data.ReportDataManager)
    manager.report_config = SimpleNamespace(
        concatenated_latent_adata_path=path,
        latent_representation_cell="cell",
        latent_representation_niche="niche",
        annotation_list=["domain"],
        trait_name_list=["IQ"],
        downsampling_n_spots_umap=8,
    )
    manager.report_data_dir = tmp_path / "data"
    manager.js_data_dir = tmp_path / "js"
    manager.report_data_dir.mkdir()
    manager.js_data_dir.mkdir()
    manager.force_re_run = False
    manager.metadata = pd.DataFrame(
        {
            "spot": original.obs_names,
            "IQ": np.arange(8) + 0.2,
            "Height": np.arange(8) + 1.2,
        }
    )
    prepare = report_data._prepare_umap_data
    calls = []

    def tracked(*args):
        calls.append(True)
        return prepare(*args)

    monkeypatch.setattr(report_data, "_prepare_umap_data", tracked)
    manager.prepare_umap_data()
    csv = manager.report_data_dir / "umap_data.csv"
    js = manager.js_data_dir / "umap_data.js"
    manifest = manager.report_data_dir / "umap_coordinates_manifest.json"
    if change == "embedding_source":
        with h5py.File(path, "r+") as handle:
            handle["obsm/X_umap"][0, 0] += 100
    elif change == "embedding_key":
        manager.report_config.latent_representation_cell = "niche"
    elif change == "annotations":
        manager.report_config.annotation_list = []
    elif change == "traits":
        manager.report_config.trait_name_list = ["Height"]
    elif change == "trait_values":
        manager.metadata.loc[0, "IQ"] = 42
    elif change == "limit":
        manager.report_config.downsampling_n_spots_umap = 4
    elif change == "missing_js":
        js.unlink()
    elif change == "edited_csv":
        csv.write_text("incomplete")
    elif change == "corrupt_manifest":
        manifest.write_text("{")
    elif change == "non_dict_manifest":
        manifest.write_text("[]")
    elif change == "force":
        manager.force_re_run = True
    elif change == "removed_embedding":
        manager.report_config.latent_representation_cell = "absent"
    elif change == "removed_source":
        path.unlink()
    manager.prepare_umap_data()
    assert len(calls) == (1 if change in {"none", "missing_js"} else 2)
    if change in {"removed_embedding", "removed_source"}:
        assert manager.umap_info is None
        assert not csv.exists()
        assert not js.exists()
        assert not manifest.exists()
        assert not (manager.js_data_dir / "umap_export_manifest.json").exists()
    else:
        payload = json.loads(js.read_text().split("=", 1)[1].rstrip(";"))
        assert len(payload["spot"]) == (4 if change == "limit" else 8)
        if change == "trait_values":
            assert payload["IQ"][0] == 42
        elif change == "embedding_source":
            assert payload["umap_cell_x"][0] == 100
        elif change == "traits":
            assert "Height" in payload
            assert "IQ" not in payload
