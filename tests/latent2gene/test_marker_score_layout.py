from __future__ import annotations

import multiprocessing
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from gsMap.latent2gene.marker_layout import (
    MARKER_SCORE_LAYOUT_VERSION,
    SequentialMarkerRowWriter,
    marker_obs_indices_path,
    open_marker_obs_indices,
)
from gsMap.latent2gene.marker_scores import MarkerScoreCalculator
from gsMap.latent2gene.memmap_io import (
    MemMapDense,
    ParallelMarkerScoreWriter,
    _automatic_lustre_stripe_count,
    _should_use_tmp_storage,
)
from gsMap.spatial_ldsc.data.loading import (
    _acquire_shared_marker_cache,
    load_marker_scores_memmap_format,
)

# Spawned children import torch/JAX/scanpy; this can exceed 10 s on slow filesystems.
PROCESS_TIMEOUT = 120


def _hold_marker_cache_lease(marker_path, cache_dir, metadata, ready, release):
    from gsMap.spatial_ldsc.data import loading as spatial_io

    spatial_io._paths_share_storage_device = lambda *_: False
    spatial_io.shutil.disk_usage = lambda _path: SimpleNamespace(free=10 * 1024**3)
    marker_lease = spatial_io._acquire_shared_marker_cache(
        Path(marker_path),
        Path(cache_dir),
        metadata,
    )
    ready.set()
    release.wait(timeout=PROCESS_TIMEOUT)
    marker_lease.close()


def test_sequential_marker_layout_appends_skipped_rows_and_is_atomic(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    writer = SequentialMarkerRowWriter(marker_path, n_obs=5)

    assert writer.append(np.asarray([2, 0], dtype=np.int32)) == slice(0, 2)
    assert writer.append(np.asarray([3], dtype=np.int64)) == slice(2, 3)
    metadata = writer.finalize()

    mapping = open_marker_obs_indices(marker_path, {"row_layout": metadata}, expected_obs=5)
    try:
        np.testing.assert_array_equal(mapping, [2, 0, 3, 1, 4])
    finally:
        mapping._mmap.close()

    assert metadata["scored_rows"] == 3
    assert metadata["unscored_rows"] == 2
    assert not list(tmp_path.glob("*.incomplete.*"))


def test_shared_marker_read_cache_is_reused_then_removed_by_final_reader(
    tmp_path,
    monkeypatch,
):
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    marker_path = source_dir / "marker_scores.dat"
    marker_values = np.arange(12, dtype=np.float16).reshape(4, 3)
    marker_values.tofile(marker_path)
    metadata = {"shape": [4, 3], "dtype": "float16", "complete": True}

    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading._paths_share_storage_device", lambda *_: False
    )
    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )

    first = _acquire_shared_marker_cache(marker_path, cache_dir, metadata)
    second = _acquire_shared_marker_cache(marker_path, cache_dir, metadata)

    assert first.path == second.path
    np.testing.assert_array_equal(
        np.fromfile(first.path.with_suffix(".dat"), dtype=np.float16).reshape(4, 3),
        marker_values,
    )
    assert first.path.with_suffix(".meta.json").is_file()
    assert len(list((cache_dir / "gsmap_read_cache").glob("*.dat"))) == 1

    first.close()
    assert first.path.with_suffix(".dat").is_file()
    second.close()
    assert not first.path.with_suffix(".dat").exists()
    assert not first.path.with_suffix(".meta.json").exists()
    assert len(list((cache_dir / "gsmap_read_cache").glob("*.lock"))) == 1


def test_shared_marker_read_cache_is_retained_across_processes(tmp_path):
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    marker_path = source_dir / "marker_scores.dat"
    np.arange(12, dtype=np.float16).tofile(marker_path)
    metadata = {"shape": [4, 3], "dtype": "float16", "complete": True}

    context = multiprocessing.get_context("spawn")
    first_ready = context.Event()
    first_release = context.Event()
    second_ready = context.Event()
    second_release = context.Event()
    first_process = context.Process(
        target=_hold_marker_cache_lease,
        args=(marker_path, cache_dir, metadata, first_ready, first_release),
    )
    second_process = context.Process(
        target=_hold_marker_cache_lease,
        args=(marker_path, cache_dir, metadata, second_ready, second_release),
    )

    try:
        first_process.start()
        assert first_ready.wait(timeout=PROCESS_TIMEOUT)
        second_process.start()
        assert second_ready.wait(timeout=PROCESS_TIMEOUT)
        cache_files = list((cache_dir / "gsmap_read_cache").glob("*.dat"))
        assert len(cache_files) == 1

        first_release.set()
        first_process.join(timeout=PROCESS_TIMEOUT)
        assert first_process.exitcode == 0
        assert cache_files[0].is_file()

        second_release.set()
        second_process.join(timeout=PROCESS_TIMEOUT)
        assert second_process.exitcode == 0
        assert not cache_files[0].exists()
    finally:
        first_release.set()
        second_release.set()
        for process in (first_process, second_process):
            if process.pid is None:  # never started because an earlier wait failed
                continue
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)


def test_shared_marker_read_cache_can_be_reused_without_populating(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    marker_path = source_dir / "marker_scores.dat"
    np.arange(12, dtype=np.float16).tofile(marker_path)
    metadata = {"shape": [4, 3], "dtype": "float16", "complete": True}
    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading._paths_share_storage_device", lambda *_: False
    )

    marker_lease = _acquire_shared_marker_cache(
        marker_path,
        cache_dir,
        metadata,
        populate=False,
    )

    assert marker_lease.path == marker_path
    assert not list((cache_dir / "gsmap_read_cache").glob("*.dat"))
    marker_lease.close()


def test_invalid_same_key_cache_is_removed_before_capacity_check(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    marker_path = source_dir / "marker_scores.dat"
    marker_values = np.arange(12, dtype=np.float16).reshape(4, 3)
    marker_values.tofile(marker_path)
    metadata = {"shape": [4, 3], "dtype": "float16", "complete": True}

    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading._paths_share_storage_device", lambda *_: False
    )
    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )
    initial_lease = _acquire_shared_marker_cache(marker_path, cache_dir, metadata)
    cache_base = initial_lease.path
    initial_lease.close()

    invalid_data = cache_base.with_suffix(".dat")
    invalid_metadata = cache_base.with_suffix(".meta.json")
    abandoned_partial = cache_base.parent / f".{cache_base.name}.dat.999.partial"
    invalid_data.write_bytes(b"truncated cache")
    invalid_metadata.write_text('{"complete": false}\n')
    abandoned_partial.write_bytes(b"abandoned partial copy")

    capacity_checks = 0

    def disk_usage_after_cleanup(_path):
        nonlocal capacity_checks
        capacity_checks += 1
        assert not invalid_data.exists()
        assert not invalid_metadata.exists()
        assert not abandoned_partial.exists()
        return SimpleNamespace(free=10 * 1024**3)

    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading.shutil.disk_usage",
        disk_usage_after_cleanup,
    )
    recovered_lease = _acquire_shared_marker_cache(marker_path, cache_dir, metadata)
    try:
        assert capacity_checks == 1
        np.testing.assert_array_equal(
            np.fromfile(recovered_lease.path.with_suffix(".dat"), dtype=np.float16).reshape(4, 3),
            marker_values,
        )
    finally:
        recovered_lease.close()


def test_nonpopulating_reader_removes_invalid_same_key_cache(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    marker_path = source_dir / "marker_scores.dat"
    np.arange(12, dtype=np.float16).tofile(marker_path)
    metadata = {"shape": [4, 3], "dtype": "float16", "complete": True}

    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading._paths_share_storage_device", lambda *_: False
    )
    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )
    initial_lease = _acquire_shared_marker_cache(marker_path, cache_dir, metadata)
    cache_base = initial_lease.path
    initial_lease.close()

    invalid_data = cache_base.with_suffix(".dat")
    invalid_metadata = cache_base.with_suffix(".meta.json")
    abandoned_partial = cache_base.parent / f".{cache_base.name}.dat.999.partial"
    invalid_data.write_bytes(b"truncated cache")
    invalid_metadata.write_text("not json\n")
    abandoned_partial.write_bytes(b"abandoned partial copy")

    report_lease = _acquire_shared_marker_cache(
        marker_path,
        cache_dir,
        metadata,
        populate=False,
    )
    try:
        assert report_lease.path == marker_path
        assert not invalid_data.exists()
        assert not invalid_metadata.exists()
        assert not abandoned_partial.exists()
    finally:
        report_lease.close()


def test_invalid_cache_is_removed_even_when_regeneration_does_not_fit(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    marker_path = source_dir / "marker_scores.dat"
    np.arange(12, dtype=np.float16).tofile(marker_path)
    metadata = {"shape": [4, 3], "dtype": "float16", "complete": True}

    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading._paths_share_storage_device", lambda *_: False
    )
    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )
    initial_lease = _acquire_shared_marker_cache(marker_path, cache_dir, metadata)
    cache_base = initial_lease.path
    initial_lease.close()

    invalid_data = cache_base.with_suffix(".dat")
    invalid_metadata = cache_base.with_suffix(".meta.json")
    abandoned_partial = cache_base.parent / f".{cache_base.name}.dat.999.partial"
    invalid_data.write_bytes(b"truncated cache")
    invalid_metadata.write_text('{"complete": false}\n')
    abandoned_partial.write_bytes(b"abandoned partial copy")

    def insufficient_capacity_after_cleanup(_path):
        assert not invalid_data.exists()
        assert not invalid_metadata.exists()
        assert not abandoned_partial.exists()
        return SimpleNamespace(free=0)

    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading.shutil.disk_usage",
        insufficient_capacity_after_cleanup,
    )
    fallback_lease = _acquire_shared_marker_cache(marker_path, cache_dir, metadata)
    try:
        assert fallback_lease.path == marker_path
        assert not invalid_data.exists()
        assert not invalid_metadata.exists()
        assert not abandoned_partial.exists()
    finally:
        fallback_lease.close()


def test_marker_cache_acquisition_removes_inactive_orphan(tmp_path, monkeypatch):
    source_dir = tmp_path / "source"
    cache_dir = tmp_path / "cache"
    shared_cache_dir = cache_dir / "gsmap_read_cache"
    source_dir.mkdir()
    cache_dir.mkdir()
    shared_cache_dir.mkdir()
    marker_path = source_dir / "marker_scores.dat"
    np.arange(12, dtype=np.float16).tofile(marker_path)
    metadata = {"shape": [4, 3], "dtype": "float16", "complete": True}

    orphan_base = shared_cache_dir / "marker_scores_orphaned"
    orphan_base.with_suffix(".dat").write_bytes(b"orphaned cache")
    orphan_base.with_suffix(".meta.json").write_text('{"complete": true}\n')
    orphan_base.with_suffix(".lock").touch()

    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading._paths_share_storage_device", lambda *_: False
    )
    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )

    marker_lease = _acquire_shared_marker_cache(marker_path, cache_dir, metadata)
    assert not orphan_base.with_suffix(".dat").exists()
    assert not orphan_base.with_suffix(".meta.json").exists()
    marker_lease.close()


@pytest.mark.parametrize(
    "rows",
    [
        np.asarray([1, 1], dtype=np.int64),
        np.asarray([0, 1, 2, 0], dtype=np.int64),
    ],
)
def test_sequential_marker_layout_rejects_duplicate_rows(tmp_path, rows):
    writer = SequentialMarkerRowWriter(tmp_path / "marker_scores.dat", n_obs=4)
    writer.append(rows)
    try:
        with pytest.raises(ValueError, match="duplicate"):
            writer.finalize()
    finally:
        writer.abort()


def test_sequential_marker_layout_rejects_invalid_indices(tmp_path):
    writer = SequentialMarkerRowWriter(tmp_path / "marker_scores.dat", n_obs=3)
    try:
        with pytest.raises(IndexError, match="outside"):
            writer.append(np.asarray([0, 3], dtype=np.int64))
        with pytest.raises(TypeError, match="integer"):
            writer.append(np.asarray([0.0, 1.0]))
    finally:
        writer.abort()


def _write_marker_memmap(path, values, row_order=None):
    marker = MemMapDense(
        path,
        shape=values.shape,
        dtype=values.dtype,
        mode="w",
        num_write_workers=0,
    )
    marker.memmap[:] = values
    metadata_updates = None
    if row_order is not None:
        layout_writer = SequentialMarkerRowWriter(path, n_obs=values.shape[0])
        layout_writer.append(np.asarray(row_order, dtype=np.int64))
        row_layout = layout_writer.finalize()
        metadata_updates = {
            "marker_score_layout_version": MARKER_SCORE_LAYOUT_VERSION,
            "row_layout": row_layout,
        }
    marker.close(metadata_updates=metadata_updates)


def _source_adata(path):
    obs_names = ["spot_a", "spot_b", "spot_c", "spot_d"]
    obs = pd.DataFrame(
        {
            "annotation": ["A", "B", "C", "D"],
            "sample_name": ["s1", "s2", "s1", "s2"],
        },
        index=obs_names,
    )
    source = ad.AnnData(
        X=np.zeros((4, 3), dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=["g1", "g2", "g3"]),
    )
    source.obsm["spatial"] = np.arange(12, dtype=np.float32).reshape(4, 3)
    source.obsm["frame"] = pd.DataFrame(
        np.arange(8, dtype=np.float32).reshape(4, 2),
        index=obs_names,
        columns=["x", "y"],
    )
    source.write_h5ad(path)
    return source


def test_loader_aligns_marker_rows_with_original_annotations_and_coordinates(
    tmp_path, monkeypatch
):
    from gsMap.utils import h5ad

    source_path = tmp_path / "concatenated_latent_adata.h5ad"
    source = _source_adata(source_path)
    source.obsm["homogeneous_neighbors"] = np.zeros((source.n_obs, 200), dtype=np.int32)
    source.write_h5ad(source_path)
    read_elem = h5ad.read_elem

    def selective_read(node):
        assert node.name not in {"/X", "/obsm", "/obsm/homogeneous_neighbors"}
        return read_elem(node)

    monkeypatch.setattr(h5ad, "read_elem", selective_read)
    marker_path = tmp_path / "marker_scores.dat"
    marker_values = np.asarray(
        [
            [20, 21, 22],
            [0, 1, 2],
            [30, 31, 32],
            [10, 11, 12],
        ],
        dtype=np.float16,
    )
    _write_marker_memmap(marker_path, marker_values, row_order=[2, 0, 3])

    config = SimpleNamespace(
        marker_scores_memmap_path=marker_path,
        concatenated_latent_adata_path=source_path,
        memmap_tmp_dir=None,
        spatial_key="spatial",
    )
    loaded = load_marker_scores_memmap_format(config)
    try:
        assert loaded.obs_names.tolist() == ["spot_c", "spot_a", "spot_d", "spot_b"]
        assert loaded.obs["annotation"].tolist() == ["C", "A", "D", "B"]
        np.testing.assert_array_equal(loaded.X, marker_values)
        np.testing.assert_array_equal(
            loaded.obsm["spatial"],
            source.obsm["spatial"][[2, 0, 3, 1]],
        )
        assert "frame" not in loaded.obsm
        assert loaded.uns["marker_score_row_layout"]["unscored_rows"] == 1
    finally:
        loaded.uns["memmap_manager"].close()

    # The mapping is applied only by the wrapper; the rank-indexed source stays
    # in its original order for safe latent-to-gene reruns.
    unchanged = ad.read_h5ad(source_path)
    assert unchanged.obs_names.tolist() == source.obs_names.tolist()


def test_loader_preserves_legacy_identity_layout(tmp_path):
    source_path = tmp_path / "concatenated_latent_adata.h5ad"
    source = _source_adata(source_path)
    marker_path = tmp_path / "legacy_marker_scores.dat"
    marker_values = np.arange(12, dtype=np.float16).reshape(4, 3)
    _write_marker_memmap(marker_path, marker_values)

    config = SimpleNamespace(
        marker_scores_memmap_path=marker_path,
        concatenated_latent_adata_path=source_path,
        memmap_tmp_dir=None,
        spatial_key="spatial",
    )
    loaded = load_marker_scores_memmap_format(config)
    try:
        assert loaded.obs_names.tolist() == source.obs_names.tolist()
        np.testing.assert_array_equal(loaded.X, marker_values)
        assert "marker_score_row_layout" not in loaded.uns
    finally:
        loaded.uns["memmap_manager"].close()


def test_loader_removes_scratch_cache_when_memmap_manager_closes(tmp_path, monkeypatch):
    source_path = tmp_path / "concatenated_latent_adata.h5ad"
    source = _source_adata(source_path)
    marker_path = tmp_path / "marker_scores.dat"
    marker_values = np.arange(12, dtype=np.float16).reshape(4, 3)
    _write_marker_memmap(marker_path, marker_values)
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading._paths_share_storage_device", lambda *_: False
    )
    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading.shutil.disk_usage",
        lambda _path: SimpleNamespace(free=10 * 1024**3),
    )
    config = SimpleNamespace(
        marker_scores_memmap_path=marker_path,
        concatenated_latent_adata_path=source_path,
        memmap_tmp_dir=scratch_dir,
        spatial_key="spatial",
    )

    loaded = load_marker_scores_memmap_format(config)
    cache_dir = scratch_dir / "gsmap_read_cache"
    assert len(list(cache_dir.glob("*.dat"))) == 1
    np.testing.assert_array_equal(loaded.X, marker_values)

    loaded.uns["memmap_manager"].close()
    assert not list(cache_dir.glob("*.dat"))
    assert not list(cache_dir.glob("*.meta.json"))
    assert source.obs_names.tolist() == loaded.obs_names.tolist()


def test_report_style_loader_does_not_populate_full_scratch_cache(tmp_path, monkeypatch):
    source_path = tmp_path / "concatenated_latent_adata.h5ad"
    _source_adata(source_path)
    marker_path = tmp_path / "marker_scores.dat"
    marker_values = np.arange(12, dtype=np.float16).reshape(4, 3)
    _write_marker_memmap(marker_path, marker_values)
    scratch_dir = tmp_path / "scratch"
    scratch_dir.mkdir()

    monkeypatch.setattr(
        "gsMap.spatial_ldsc.data.loading._paths_share_storage_device", lambda *_: False
    )
    config = SimpleNamespace(
        marker_scores_memmap_path=marker_path,
        concatenated_latent_adata_path=source_path,
        memmap_tmp_dir=scratch_dir,
        spatial_key="spatial",
    )

    loaded = load_marker_scores_memmap_format(config, populate_cache=False)
    try:
        np.testing.assert_array_equal(loaded.X, marker_values)
        assert not list((scratch_dir / "gsmap_read_cache").glob("*.dat"))
    finally:
        loaded.uns["memmap_manager"].close()


def test_complete_marker_memmap_requires_its_row_mapping(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    _write_marker_memmap(
        marker_path,
        np.zeros((3, 2), dtype=np.float16),
        row_order=[2, 0, 1],
    )
    assert MemMapDense.check_complete(marker_path)[0]

    marker_obs_indices_path(marker_path).unlink()

    assert not MemMapDense.check_complete(marker_path)[0]


@pytest.mark.parametrize(
    ("size_gib", "expected"),
    [(1, 1), (8, 4), (128, 8), (1024, 16)],
)
def test_automatic_lustre_stripe_count_scales_with_file_size(size_gib, expected):
    assert _automatic_lustre_stripe_count(size_gib * 1024**3) == expected


def test_optional_tmp_storage_falls_back_when_capacity_is_insufficient(
    tmp_path,
    monkeypatch,
):
    usage = type("Usage", (), {"free": 8 * 1024**3})()
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io._same_filesystem",
        lambda first, second: False,
    )
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io.shutil.disk_usage",
        lambda path: usage,
    )

    assert not _should_use_tmp_storage(
        tmp_path / "shared" / "marker_scores.dat",
        tmp_path / "scratch",
        file_bytes=10 * 1024**3,
    )


def _force_distinct_scratch_with_capacity(monkeypatch):
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io._same_filesystem",
        lambda first, second: False,
    )
    usage = type("Usage", (), {"free": 100 * 1024**3})()
    monkeypatch.setattr(
        "gsMap.latent2gene.memmap_io.shutil.disk_usage",
        lambda path: usage,
    )


def test_rank_publish_retains_scratch_and_syncs_project_once(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    rank_path = project_dir / "ranks.dat"
    values = np.arange(12, dtype=np.float16).reshape(4, 3)
    _force_distinct_scratch_with_capacity(monkeypatch)

    sync_calls = []
    original_sync = MemMapDense._sync_tmp_to_original

    def tracked_sync(memmap):
        sync_calls.append(memmap.data_path)
        original_sync(memmap)

    monkeypatch.setattr(MemMapDense, "_sync_tmp_to_original", tracked_sync)

    rank = MemMapDense(
        rank_path,
        shape=values.shape,
        dtype=values.dtype,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    assert rank.using_tmp
    assert rank.data_path.parent.parent == scratch_dir
    assert not rank_path.is_file()
    assert not rank_path.with_suffix(".meta.json").is_file()

    rank.memmap[:] = values
    rank.publish()

    assert sync_calls == [rank.data_path]
    assert MemMapDense.check_complete(rank_path)[0]
    assert rank.data_path.is_file()
    np.testing.assert_array_equal(rank.memmap, values)

    completed = np.memmap(rank_path, dtype=np.float16, mode="r", shape=values.shape)
    try:
        np.testing.assert_array_equal(completed, values)
    finally:
        completed._mmap.close()

    scratch_subdir = rank.tmp_subdir
    rank.close()
    rank.close()
    assert sync_calls == [rank.data_path]
    assert not scratch_subdir.exists()


def test_marker_input_reuses_live_rank_scratch_mapping(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    rank_path = project_dir / "ranks.dat"
    adata_path = project_dir / "concatenated_latent_adata.h5ad"
    _force_distinct_scratch_with_capacity(monkeypatch)

    rank = MemMapDense(
        rank_path,
        shape=(3, 2),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    rank.memmap[:] = np.arange(6, dtype=np.float16).reshape(3, 2)
    rank.publish()
    scratch_entries_before = set(scratch_dir.iterdir())

    ad.AnnData(
        X=np.zeros((3, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b", "c"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(adata_path)

    calculator = object.__new__(MarkerScoreCalculator)
    calculator.config = SimpleNamespace(
        high_quality_neighbor_filter=False,
        memmap_tmp_dir=scratch_dir,
    )
    calculator.load_global_stats = lambda path: (
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )

    loaded = calculator._load_input_data(adata_path, rank, "unused.parquet")
    assert loaded[1] is rank
    assert loaded[1].data_path == rank.data_path
    assert set(scratch_dir.iterdir()) == scratch_entries_before
    np.testing.assert_array_equal(loaded[-1], np.ones(3, dtype=bool))

    rank.close()


def test_invalid_marker_input_closes_live_rank_scratch_mapping(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    rank_path = project_dir / "ranks.dat"
    adata_path = project_dir / "concatenated_latent_adata.h5ad"
    _force_distinct_scratch_with_capacity(monkeypatch)

    rank = MemMapDense(
        rank_path,
        shape=(3, 2),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    rank.publish()
    scratch_subdir = rank.tmp_subdir

    ad.AnnData(
        X=np.zeros((2, 2), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g1", "g2"]),
    ).write_h5ad(adata_path)

    calculator = object.__new__(MarkerScoreCalculator)
    calculator.config = SimpleNamespace(
        high_quality_neighbor_filter=False,
        memmap_tmp_dir=scratch_dir,
    )
    calculator.load_global_stats = lambda path: (
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Cell count mismatch"):
        calculator._load_input_data(adata_path, rank, "unused.parquet")

    assert rank._closed
    assert not scratch_subdir.exists()


def test_staged_marker_output_publishes_data_metadata_and_mapping_together(
    tmp_path,
    monkeypatch,
):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    marker_path = project_dir / "marker_scores.dat"
    values = np.arange(12, dtype=np.float16).reshape(4, 3)

    # Exercise the cross-filesystem branch using two test directories on the
    # same CI filesystem.
    _force_distinct_scratch_with_capacity(monkeypatch)

    marker = MemMapDense(
        marker_path,
        shape=values.shape,
        dtype=values.dtype,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    assert marker.using_tmp
    marker.memmap[:] = values

    layout_writer = SequentialMarkerRowWriter(marker_path, n_obs=4)
    layout_writer.append(np.asarray([2, 0, 3, 1], dtype=np.int64))
    row_layout = layout_writer.finalize()
    marker.close(
        metadata_updates={
            "marker_score_layout_version": MARKER_SCORE_LAYOUT_VERSION,
            "row_layout": row_layout,
        }
    )

    assert MemMapDense.check_complete(marker_path)[0]
    completed = np.memmap(marker_path, dtype=np.float16, mode="r", shape=values.shape)
    try:
        np.testing.assert_array_equal(completed, values)
    finally:
        completed._mmap.close()
    assert marker_obs_indices_path(marker_path).is_file()
    assert not list(scratch_dir.glob("memmap_*"))


def test_check_complete_accepts_metadata_path(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    _write_marker_memmap(marker_path, np.zeros((2, 3), dtype=np.float16))

    metadata_path = marker_path.with_suffix(".meta.json")
    assert MemMapDense.check_complete(metadata_path)[0]


def test_memmap_open_rejects_metadata_dtype_mismatch(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    _write_marker_memmap(marker_path, np.zeros((2, 3), dtype=np.float32))

    with pytest.raises(ValueError, match="Dtype mismatch"):
        MemMapDense(marker_path, shape=(2, 3), dtype=np.float16, mode="r")


def test_zero_worker_memmap_write_batch_is_synchronous(tmp_path):
    marker_path = tmp_path / "marker_scores.dat"
    marker = MemMapDense(
        marker_path,
        shape=(3, 2),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
    )
    values = np.arange(6, dtype=np.float16).reshape(3, 2)

    marker.write_batch(values, slice(0, 3))
    marker.close()

    completed = np.memmap(marker_path, dtype=np.float16, mode="r", shape=(3, 2))
    try:
        np.testing.assert_array_equal(completed, values)
    finally:
        completed._mmap.close()


def test_parallel_marker_writer_rejects_zero_workers(tmp_path):
    marker = MemMapDense(
        tmp_path / "marker_scores.dat",
        shape=(1, 1),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
    )
    try:
        with pytest.raises(ValueError, match="at least one worker"):
            ParallelMarkerScoreWriter(marker, num_workers=0)
    finally:
        marker.abort()


def test_annotation_free_marker_mode_does_not_create_invalid_obs_column(tmp_path):
    calculator = object.__new__(MarkerScoreCalculator)
    calculator.config = SimpleNamespace(annotation=None)
    adata = ad.AnnData(
        X=np.zeros((2, 1), dtype=np.float32),
        obs=pd.DataFrame(index=["a", "b"]),
        var=pd.DataFrame(index=["g1"]),
    )

    assert calculator._get_cell_types(adata) == ["all"]
    assert None not in adata.obs.columns
    adata.write_h5ad(tmp_path / "annotation_free.h5ad")


def test_parallel_marker_writer_surfaces_write_failure_without_deadlock(tmp_path, monkeypatch):
    marker_path = tmp_path / "marker_scores.dat"
    marker = MemMapDense(
        marker_path,
        shape=(2, 2),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
    )
    writer = ParallelMarkerScoreWriter(marker, num_workers=1, queue_size=2)

    def fail_write(*args):
        raise OSError("synthetic marker write failure")

    monkeypatch.setattr(writer, "_write_rows", fail_write)
    writer.write_queue.put((0, np.ones((1, 2), dtype=np.float16), np.asarray([0], dtype=np.int64)))
    try:
        with pytest.raises(RuntimeError, match="synthetic marker write failure"):
            writer.close()
    finally:
        writer.abort()
        marker.abort()

    assert not MemMapDense.check_complete(marker_path)[0]


def test_memmap_abort_removes_scratch_without_publishing(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    marker_path = project_dir / "marker_scores.dat"
    _force_distinct_scratch_with_capacity(monkeypatch)

    marker = MemMapDense(
        marker_path,
        shape=(2, 3),
        dtype=np.float16,
        mode="w",
        num_write_workers=0,
        tmp_dir=scratch_dir,
    )
    scratch_subdir = marker.tmp_subdir
    marker.memmap[:] = 1
    marker.abort()

    assert not scratch_subdir.exists()
    assert not marker_path.exists()
    assert not marker_path.with_suffix(".meta.json").exists()


def test_memmap_constructor_failure_removes_staged_scratch(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    scratch_dir = tmp_path / "scratch"
    project_dir.mkdir()
    scratch_dir.mkdir()
    rank_path = project_dir / "ranks.dat"
    _write_marker_memmap(rank_path, np.zeros((3, 2), dtype=np.float16))
    _force_distinct_scratch_with_capacity(monkeypatch)

    with pytest.raises(ValueError, match="Dtype mismatch"):
        MemMapDense(
            rank_path,
            shape=(3, 2),
            dtype=np.float32,
            mode="r",
            num_write_workers=0,
            tmp_dir=scratch_dir,
        )

    assert not list(scratch_dir.glob("memmap_*"))
