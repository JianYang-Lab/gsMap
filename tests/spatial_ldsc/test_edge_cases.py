"""Edge cases in spatial LDSC, Cauchy aggregation and the LD weight matrix builder.

Valid inputs keep bit-identical results; these tests pin the behaviour of the
inputs that used to produce silently wrong values, hangs or hard exits.
"""

import os
import subprocess
import sys
import time
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.stats import norm

from gsMap.cauchy_combination_test import (
    _build_group_index,
    _load_trait_matrix,
    _matrix_positions,
    _read_metadata,
    _result_frame,
    _summarize_trait_groups,
    run_cauchy_on_dataframe,
)
from gsMap.spatial_ldsc.backends.ldscore import is_gpu_memory_error
from gsMap.spatial_ldsc.data.loading import generate_expected_output_filename, prepare_trait_data
from gsMap.spatial_ldsc.data.writer import ResultWriter
from gsMap.spatial_ldsc.execution.multi_gpu import wait_workers

# ---------------------------------------------------------------- result writer


def test_failed_regressions_are_written_as_missing_not_as_p_one(tmp_path):
    writer = ResultWriter(
        config=SimpleNamespace(project_name="p", cell_indices_range=None, sample_filter=None),
        trait_name="t",
        output_dir=tmp_path,
        n_spots=4,
        total_chunks=1,
        expected_cells=4,
    )
    writer.add_chunk_result(
        chunk_idx=0,
        betas=np.array([1.0, np.nan, 2.0, 0.0], dtype=np.float32),
        ses=np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        spot_names=pd.Index(["a", "b", "c", "d"]),
        abs_start=0,
        abs_end=4,
    )
    result = pd.read_csv(writer.finalize())

    # A valid spot is unchanged; NaN beta, SE = 0 and 0/0 have no p-value.
    assert result["p"].iloc[0] == pytest.approx(norm.sf(1.0), rel=1e-14)
    assert result["neg_log10_p"].iloc[0] == pytest.approx(-np.log10(norm.sf(1.0)), rel=1e-14)
    assert result["p"].iloc[1:].isna().all()
    assert result["neg_log10_p"].iloc[1:].isna().all()
    assert writer.failed_cells == 3
    assert writer.max_neg_log10_p == pytest.approx(-np.log10(norm.sf(1.0)))


# ---------------------------------------------------------------- Cauchy


def _write_ldsc(path, spots, p_values):
    with np.errstate(divide="ignore"):
        pd.DataFrame(
            {
                "spot": spots,
                "beta": 0.1,
                "se": 0.1,
                "z": 1.0,
                "p": p_values,
                "neg_log10_p": -np.log10(p_values),
            }  # fmt: skip
        ).to_csv(path, index=False, compression="gzip")


@pytest.fixture
def cauchy_inputs(tmp_path):
    spots = [f"s{i:02d}" for i in range(18)]
    obs = pd.DataFrame(
        {
            "annotation": pd.Categorical(["A"] * 6 + ["B"] * 6 + ["C"] * 6),
            "sample_name": pd.Categorical(["one"] * 9 + ["two"] * 9),
        },
        index=spots,
    )
    h5ad = tmp_path / "latent.h5ad"
    ad.AnnData(X=np.zeros((18, 1), np.float32), obs=obs).write_h5ad(h5ad)
    return h5ad, spots


def _compact(h5ad, paths, sample_level):
    metadata = _read_metadata(h5ad, ["annotation"])
    matrix = _load_trait_matrix(paths, metadata, max_workers=1)
    groups = _build_group_index(
        metadata.annotations["annotation"],
        _matrix_positions(matrix.valid_spots),
        sample_name=metadata.sample_name if sample_level else None,
    )
    rows = []
    for index, trait in enumerate(matrix.traits):
        rows += _summarize_trait_groups(
            trait, matrix.values[index], "annotation", groups, sample_name=metadata.sample_name
        )
    return matrix, _result_frame(rows, sample_level=sample_level)


def test_underflowed_p_values_do_not_turn_a_group_into_p_one(cauchy_inputs, tmp_path):
    h5ad, spots = cauchy_inputs
    p = np.geomspace(0.9, 1e-6, 18)
    p[:5] = 0.0  # z > ~38 underflows to p = 0 in group A
    path = tmp_path / "t.csv.gz"
    _write_ldsc(path, spots, p)

    matrix, result = _compact(h5ad, {"t": path}, sample_level=False)
    assert np.isfinite(matrix.values).all()
    group_a = result.set_index("annotation").loc["A"]
    # Previously the inf values made the outlier filter drop every spot and the group
    # reported p = 1. Now the legacy IQR filter treats the capped values as outliers
    # and the remaining spot is tested.
    assert group_a["total_spots"] == 1
    assert group_a["p_cauchy"] < 0.05


@pytest.mark.parametrize("sample_level", [False, True])
def test_one_traits_failed_spots_do_not_change_another_trait(
    cauchy_inputs, tmp_path, sample_level
):
    h5ad, spots = cauchy_inputs
    full = np.geomspace(0.9, 1e-8, 18)
    partial = np.geomspace(0.5, 1e-4, 18)
    partial[[0, 1, 2, 10]] = np.nan
    paths = {"full": tmp_path / "full.csv.gz", "partial": tmp_path / "partial.csv.gz"}
    _write_ldsc(paths["full"], spots, full)
    _write_ldsc(paths["partial"], spots, partial)

    matrix, together = _compact(h5ad, paths, sample_level)
    _, alone = _compact(h5ad, {"full": paths["full"]}, sample_level)
    assert matrix.valid_spots.all()  # every spot has at least one trait

    key = ["trait", "annotation", *(["sample_name"] if sample_level else [])]
    pd.testing.assert_frame_equal(
        together[together.trait == "full"].sort_values(key, ignore_index=True),
        alone.sort_values(key, ignore_index=True),
    )

    # The partial trait uses only its own spots, background included.
    frame = pd.DataFrame({"partial": -np.log10(partial)})
    metadata = _read_metadata(h5ad, ["annotation"])
    frame["annotation"] = metadata.annotations["annotation"]
    frame["sample_name"] = metadata.sample_name
    reference = run_cauchy_on_dataframe(
        frame[np.isfinite(frame["partial"])],
        annotation_col="annotation",
        trait_cols=["partial"],
        extra_group_col="sample_name" if sample_level else None,
    )
    pd.testing.assert_frame_equal(
        together[together.trait == "partial"].sort_values(key, ignore_index=True),
        reference.sort_values(key, ignore_index=True),
        check_dtype=False,
        rtol=1e-13,
    )


# ---------------------------------------------------------------- GWAS input


def test_snps_with_missing_n_are_dropped_instead_of_poisoning_every_spot(tmp_path):
    names = [f"rs{i}" for i in range(60)]
    rng = np.random.default_rng(3)
    n = rng.uniform(1e4, 2e4, 60)
    n[[3, 17]] = np.nan
    n[20] = 0
    path = tmp_path / "t.sumstats.gz"
    pd.DataFrame({"SNP": names, "N": n, "Z": rng.normal(size=60)}).to_csv(
        path, sep="\t", index=False
    )
    baseline = pd.DataFrame(rng.uniform(1, 5, (60, 2)), index=names)
    w_ld = pd.DataFrame({"LD_weights": rng.uniform(1, 5, 60)}, index=names)
    reference = ad.AnnData(sparse.eye(60, format="csr"))
    reference.obs_names = names
    config = SimpleNamespace(chisq_max=None)

    data, common = prepare_trait_data(config, "t", str(path), baseline, w_ld, reference)
    assert len(common) == 57
    assert not {"rs3", "rs17", "rs20"} & set(common)
    assert np.isfinite(data["N"]).all()
    assert np.isfinite(data["Nbar"])


# ---------------------------------------------------------------- execution helpers


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("CUDA out of memory. Tried to allocate 2.00 GiB", True),
        ("RESOURCE_EXHAUSTED: Out of memory while trying to allocate", True),
        ("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate", True),
        ("CUDA error: an illegal memory access was encountered", False),
        ("invalid CSR index", False),
    ],
)
def test_only_allocation_failures_count_as_gpu_out_of_memory(message, expected):
    assert is_gpu_memory_error(RuntimeError(message)) is expected


def test_hung_gpu_worker_fails_instead_of_waiting_forever(tmp_path):
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="appears hung"):
        wait_workers([process], heartbeat_paths=[[tmp_path / "never.json"]], stall_seconds=0.5)
    assert time.monotonic() - started < 20
    assert process.poll() is not None  # terminated and reaped


def test_active_gpu_worker_is_not_treated_as_hung(tmp_path):
    beat = tmp_path / "beat"
    code = (
        "import pathlib, time\n"
        f"p = pathlib.Path({str(beat)!r})\n"
        "for _ in range(15):\n    p.write_text('x'); time.sleep(0.1)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", code])
    wait_workers([process], heartbeat_paths=[[beat]], stall_seconds=0.5)
    assert process.returncode == 0


def test_skip_name_for_a_cell_range_matches_the_clamped_result_name():
    config = SimpleNamespace(
        project_name="p", cell_indices_range=(100, 10_000), sample_filter=None
    )
    assert (
        generate_expected_output_filename(config, "t", n_spots=500) == "p_t_cells_100_500.csv.gz"
    )
    assert generate_expected_output_filename(config, "t") == "p_t_cells_100_10000.csv.gz"


def test_importing_gsmap_leaves_jax_x64_alone_until_a_run_starts():
    code = (
        "import jax\n"
        "import gsMap.spatial_ldsc.spatial_ldsc_jax, gsMap.latent2gene.connectivity\n"
        "import gsMap.spatial_ldsc.kernels.jax_regression\n"
        "print(jax.config.jax_enable_x64)\n"
        "from gsMap.utils.jax_setup import use_float32_jax\n"
        "use_float32_jax(); print(jax.config.jax_enable_x64)\n"
    )
    env = dict(os.environ, JAX_ENABLE_X64="1", JAX_PLATFORMS="cpu", CUDA_VISIBLE_DEVICES="")
    output = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.split()
    assert output == ["True", "False"]


# ---------------------------------------------------------------- LD weight matrix


def test_missing_plink_files_for_a_requested_chromosome_raise(tmp_path):
    from gsMap.ldscore.pipeline import LDScorePipeline

    pipeline = LDScorePipeline.__new__(LDScorePipeline)
    pipeline.config = SimpleNamespace(
        bfile_root=str(tmp_path / "panel.{chr}"), snps_only=True, mac_min=0, maf_min=0
    )
    with pytest.raises(FileNotFoundError, match="chromosome 5"):
        pipeline._load_plink_reader(5)


def test_malformed_bed_mapping_raises_instead_of_exiting(tmp_path):
    from gsMap.ldscore.pipeline import LDScorePipeline

    bed = tmp_path / "map.bed"
    bed.write_text("chr1\t100\t200\n")
    pipeline = LDScorePipeline.__new__(LDScorePipeline)
    pipeline.config = SimpleNamespace(mapping_type="bed", mapping_file=bed)
    with pytest.raises(ValueError, match="BED"):
        pipeline._load_mapping_data()
