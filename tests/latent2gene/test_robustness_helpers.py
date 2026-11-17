"""Latent-to-gene edge cases: neighbor index dtypes, embedding norms and writer back-pressure."""

import numpy as np

from gsMap.latent2gene.marker_scores import assign_rows_without_overflow, normalize_float16_rows
from gsMap.latent2gene.memmap_io import MemMapDense


def test_wide_neighbor_indices_widen_the_shared_array_instead_of_wrapping():
    stored = np.full((6, 2), -1, dtype=np.int16)
    stored = assign_rows_without_overflow(stored, [0, 1], np.array([[1, 2], [3, 4]], np.int16))
    assert stored.dtype == np.int16  # values that fit keep the original dtype

    wide = np.array([[40_000, 70_000]], dtype=np.int32)
    stored = assign_rows_without_overflow(stored, [5], wide)
    assert stored.dtype == np.int32
    np.testing.assert_array_equal(stored[5], [40_000, 70_000])
    np.testing.assert_array_equal(stored[:2], [[1, 2], [3, 4]])


def test_float16_embedding_norms_that_overflow_are_normalized_in_float32():
    source = np.array([[0.3, 0.4, 0.0], [300.0, 400.0, 0.0]], dtype=np.float32)
    embedding = source.astype(np.float16)
    result = normalize_float16_rows(embedding, source)

    reference = embedding[:1] / (np.linalg.norm(embedding[:1], axis=1, keepdims=True) + 1e-8)
    np.testing.assert_array_equal(result[:1], reference)  # ordinary rows: unchanged
    np.testing.assert_allclose(result[1].astype(np.float32), [0.6, 0.8, 0.0], atol=1e-3)
    assert result.dtype == np.float16


def test_matrix_writer_queue_is_bounded_by_the_requested_size(tmp_path):
    matrix = MemMapDense(str(tmp_path / "m"), shape=(4, 3), mode="w", queue_size=3)
    try:
        assert matrix.write_queue.maxsize == 3
    finally:
        matrix.close()
