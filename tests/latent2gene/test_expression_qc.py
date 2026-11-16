from types import SimpleNamespace

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from gsMap.latent2gene.rank_calculator import RankCalculator, filter_cells
from gsMap.utils.expression_qc import expression_row_metrics, h5_expression_row_metrics


def _noncanonical_counts():
    # Two bad rows have >=20 stored entries but only ONE actually expressed gene.
    return sp.csr_matrix(
        (
            np.concatenate(([1], np.zeros(24), np.ones(25 + 20 + 25))).astype(np.float32),
            np.concatenate((np.arange(25), np.zeros(25, dtype=int), np.arange(20), np.arange(25))),
            np.asarray([0, 25, 50, 70, 95]),
        ),
        shape=(4, 25),
    )


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_qc_counts_distinct_positive_genes_with_bounded_reads(tmp_path, encoding):
    x = _noncanonical_counts()
    converted = {"csr": lambda: x, "csc": x.tocsc, "dense": x.toarray}[encoding]()
    data = ad.AnnData(converted)
    path = tmp_path / "input.h5ad"
    data.write_h5ad(path)
    with h5py.File(path, "r") as handle:
        genes, totals = h5_expression_row_metrics(handle["X"], chunk_rows=1)
    np.testing.assert_array_equal(genes, [1, 1, 20, 25])
    np.testing.assert_array_equal(totals, [1, 25, 20, 25])
    obs = pd.DataFrame(index=["a", "b", "c", "d"])
    filtered, stats, mask = filter_cells(obs, X=x, min_genes=20)
    assert filtered.index.tolist() == ["c", "d"]
    assert stats["low_gene_count_removed"] == 2
    np.testing.assert_array_equal(mask, [False, False, True, True])


def test_row_metrics_never_request_more_than_the_chunk_size():
    class Reader:
        shape = (50, 30)

        def __getitem__(self, rows):
            assert rows.stop - rows.start <= 7
            return np.ones((rows.stop - rows.start, 30))

    genes, counts = expression_row_metrics(Reader(), chunk_rows=7)
    np.testing.assert_array_equal(genes, np.full(50, 30))
    np.testing.assert_array_equal(counts, np.full(50, 30))


@pytest.mark.parametrize("bad_value", [-1.0, np.nan, np.inf, -np.inf])
def test_default_count_qc_rejects_negative_and_nonfinite_values(bad_value):
    with pytest.raises(ValueError, match="Expression QC requires finite"):
        expression_row_metrics(np.asarray([[1.0, bad_value]]))


@pytest.mark.parametrize("encoding", ["csr", "csc", "dense"])
def test_explicit_noncounts_qc_counts_signed_nonzero_features(tmp_path, encoding):
    dense = np.asarray([[-2, 0, 4, -1], [0, 0, 0, 0]], dtype=np.float32)
    matrix = {"csr": sp.csr_matrix, "csc": sp.csc_matrix, "dense": np.asarray}[encoding](dense)
    path = tmp_path / "signed.h5ad"
    ad.AnnData(matrix).write_h5ad(path)
    with h5py.File(path, "r") as handle:
        features, row_sums = h5_expression_row_metrics(
            handle["X"], chunk_rows=1, allow_negative=True
        )
    np.testing.assert_array_equal(features, [3, 0])
    np.testing.assert_array_equal(row_sums, [1, 0])


def test_noncounts_qc_canonicalizes_duplicate_and_cancelled_features():
    matrix = sp.csr_matrix(
        (np.asarray([-2, 2, -3, -4, 0, 5]), np.asarray([0, 0, 1, 1, 2, 3]), [0, 6]),
        shape=(1, 4),
    )
    features, row_sums = expression_row_metrics(matrix, allow_negative=True)
    np.testing.assert_array_equal(features, [2])
    np.testing.assert_array_equal(row_sums, [-2])
    # Opt-in canonicalization must not weaken the default pre-sum count check.
    with pytest.raises(ValueError, match="nonnegative"):
        expression_row_metrics(matrix)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_noncounts_qc_still_rejects_nonfinite_values(bad_value):
    with pytest.raises(ValueError, match="finite"):
        expression_row_metrics(np.asarray([[-1.0, bad_value]]), allow_negative=True)


def test_standalone_latent2gene_does_not_trust_upstream_qc_or_sparse_storage_counts(tmp_path):
    counts = _noncanonical_counts()
    data = ad.AnnData(
        X=sp.csr_matrix(counts.shape),
        obs=pd.DataFrame(
            {"cell_qc_n_genes": [9999] * 4},
            index=["bad_zero", "bad_duplicate", "good_a", "good_b"],
        ),
        var=pd.DataFrame(index=[f"g{i}" for i in range(25)]),
    )
    data.layers["counts"] = counts
    data.obsm["emb_cell"] = np.ones((4, 2), dtype=np.float32)
    data.uns["cell_qc"] = {"n_retained": 4}  # Deliberately stale upstream information.
    path = tmp_path / "external_embeddings.h5ad"
    data.write_h5ad(path)
    out = tmp_path / "ranks"
    config = SimpleNamespace(
        latent_dir=tmp_path,
        latent2gene_dir=out,
        concatenated_latent_adata_path=out / "concatenated.h5ad",
        rank_memmap_path=out / "rank_memmap",
        mean_frac_path=out / "mean_frac.parquet",
        sample_h5ad_dict={"external": path},
        annotation=None,
        min_cells_per_type=1,
        rank_batch_size=2,
        rank_write_interval=1,
        mkscore_write_workers=1,
        memmap_tmp_dir=None,
    )
    outputs = RankCalculator(config).calculate_ranks_and_concatenate(data_layer="counts")
    outputs["rank_memmap_handle"].close()
    observed = ad.read_h5ad(config.concatenated_latent_adata_path)
    assert observed.obs_names.tolist() == ["good_a|external", "good_b|external"]
    assert observed.shape == (2, 25)
    assert not (tmp_path / "find_latent_representations").exists()


def test_noncounts_rank_stage_uses_feature_qc_and_log_ranks_without_changing_input(tmp_path):
    import hashlib

    from scipy.stats import rankdata

    dense = np.vstack((np.arange(-12, 13), np.ones(25), -np.ones(25), [1] + [0] * 24)).astype(
        np.float32
    )
    data = ad.AnnData(
        X=sp.csr_matrix(dense),
        obs=pd.DataFrame(index=["signed", "positive", "negative", "few_features"]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(25)]),
    )
    data.obsm["cell_emb_pca"] = np.ones((4, 2), dtype=np.float32)
    path = tmp_path / "external_pca.h5ad"
    data.write_h5ad(path)
    original_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    out = tmp_path / "ranks"
    config = SimpleNamespace(
        expression_data_type="noncounts",
        no_expression_fraction=True,
        latent_dir=tmp_path,
        latent2gene_dir=out,
        concatenated_latent_adata_path=out / "concatenated.h5ad",
        rank_memmap_path=out / "rank_memmap",
        mean_frac_path=out / "mean_frac.parquet",
        sample_h5ad_dict={"external": path},
        annotation=None,
        min_cells_per_type=1,
        rank_batch_size=2,
        rank_write_interval=1,
        mkscore_write_workers=1,
        memmap_tmp_dir=None,
    )
    outputs = RankCalculator(config).calculate_ranks_and_concatenate(data_layer="X")
    outputs["rank_memmap_handle"].close()
    observed = ad.read_h5ad(config.concatenated_latent_adata_path)
    assert observed.obs_names.tolist() == [
        "signed|external",
        "positive|external",
        "negative|external",
    ]
    assert observed.shape == (3, 25)
    actual = np.memmap(
        config.rank_memmap_path.with_suffix(".dat"), dtype=np.float16, mode="r", shape=(3, 25)
    )
    expected = np.log(rankdata(dense[:3], method="average", axis=1)).astype(np.float16)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original_sha
