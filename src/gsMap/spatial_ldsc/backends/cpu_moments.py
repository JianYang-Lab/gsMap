"""CPU four-column LDSC moments with bounded parallel jackknife solves."""

import numpy as np
from numba import config as numba_config
from numba import get_num_threads, njit, prange, set_num_threads

from gsMap.spatial_ldsc.backends.cpu_sparse import available_cpu_count
from gsMap.spatial_ldsc.backends.moment_design import gene_hsq, prepare_moment_design


@njit(nogil=True, parallel=True, cache=True)
def accumulate_partial(spatial, positions, lengths, features, nbar, hsq, n_blocks, partial=128):
    n_cells = spatial.shape[1]
    traits = len(lengths)
    terms = np.zeros((traits, n_blocks, 14, n_cells), np.float64)
    for task in prange(traits * n_blocks):
        trait = task // n_blocks
        block = task % n_blocks
        block_size = lengths[trait] // n_blocks
        acc = np.empty((14, n_cells), np.float32)
        for lo in range(block * block_size, (block + 1) * block_size, partial):
            for f in range(14):
                for c in range(n_cells):
                    acc[f, c] = np.float32(0)
            for row in range(lo, min(lo + partial, (block + 1) * block_size)):
                source_row = positions[trait, row]
                n = features[trait, 0, row]
                b0, b1, b2 = features[trait, 1, row], features[trait, 2, row], np.float32(1)
                y = features[trait, 3, row]
                wld_row = features[trait, 5, row]
                p00 = np.float32(b0 * b0)
                p01 = np.float32(b0 * b1)
                p02 = np.float32(b0 * b2)
                p11 = np.float32(b1 * b1)
                p12 = np.float32(b1 * b2)
                p22 = np.float32(b2 * b2)
                py0, py1, py2 = np.float32(b0 * y), np.float32(b1 * y), np.float32(b2 * y)
                for cell in range(n_cells):
                    value = spatial[source_row, cell]
                    ld = max(np.float32(value + features[trait, 4, row]), np.float32(1))
                    c = np.float32(np.float32(hsq[trait, cell] * n) / np.float32(10000))
                    term = np.float32(np.float32(1) + np.float32(c * ld))
                    denominator = np.float32(
                        np.float32(np.float32(2) * np.float32(term * term)) * wld_row
                    )
                    weight = np.float32(np.float32(1) / denominator)
                    x = np.float32(np.float32(value * n) / nbar[trait])
                    wx = np.float32(x * weight)
                    acc[0, cell] += np.float32(np.float32(np.float32(x * x) * weight))
                    acc[1, cell] += np.float32(np.float32(wx * b0))
                    acc[2, cell] += np.float32(np.float32(wx * b1))
                    acc[3, cell] += np.float32(np.float32(wx * b2))
                    acc[4, cell] += np.float32(np.float32(weight * p00))
                    acc[5, cell] += np.float32(np.float32(weight * p01))
                    acc[6, cell] += np.float32(np.float32(weight * p02))
                    acc[7, cell] += np.float32(np.float32(weight * p11))
                    acc[8, cell] += np.float32(np.float32(weight * p12))
                    acc[9, cell] += np.float32(np.float32(weight * p22))
                    acc[10, cell] += np.float32(np.float32(wx * y))
                    acc[11, cell] += np.float32(np.float32(weight * py0))
                    acc[12, cell] += np.float32(np.float32(weight * py1))
                    acc[13, cell] += np.float32(np.float32(weight * py2))
            for f in range(14):
                for c in range(n_cells):
                    terms[trait, block, f, c] += np.float64(acc[f, c])
    return terms


@njit(inline="always", error_model="numpy")
def coefficient(v):
    # Cholesky of the symmetric 4x4 normal equation; only beta[0] is needed.
    a = np.sqrt(v[0])
    b = v[1] / a
    c = v[2] / a
    d = v[3] / a
    e = np.sqrt(v[4] - b * b)
    f = (v[5] - c * b) / e
    g = (v[6] - d * b) / e
    h = np.sqrt(v[7] - c * c - f * f)
    i = (v[8] - d * c - g * f) / h
    j = np.sqrt(v[9] - d * d - g * g - i * i)
    u = v[10] / a
    vv = (v[11] - b * u) / e
    w = (v[12] - c * u - f * vv) / h
    z = (v[13] - d * u - g * vv - i * w) / j
    x3 = z / j
    x2 = (w - i * x3) / h
    x1 = (vv - f * x2 - g * x3) / e
    return (u - b * x1 - c * x2 - d * x3) / a


@njit(nogil=True, parallel=True, cache=True, error_model="numpy")
def solve_parallel(terms, nbar):
    traits, blocks, _, cells = terms.shape
    out = np.empty((traits, cells, 2), np.float64)
    for task in prange(traits * cells):
        t = task // cells
        c = task % cells
        total = np.zeros(14, np.float64)
        deleted = np.empty(14, np.float64)
        for b in range(blocks):
            for k in range(14):
                total[k] += terms[t, b, k, c]
        full = coefficient(total)
        mean = 0.0
        ss = 0.0
        for b in range(blocks):
            for k in range(14):
                deleted[k] = total[k] - terms[t, b, k, c]
            val = coefficient(deleted)
            delta = val - mean
            mean += delta / (b + 1)
            ss += delta * (val - mean)
        out[t, c, 0] = (blocks * full - (blocks - 1) * mean) / nbar[t]
        out[t, c, 1] = np.sqrt(ss * (blocks - 1) / blocks) / nbar[t]
    return out


@njit(nogil=True, parallel=True, cache=True)
def sparse_out(indptr, indices, values, marker, out):
    for row in prange(len(indptr) - 1):
        for c in range(marker.shape[1]):
            out[row, c] = np.float32(0)
        for entry in range(indptr[row], indptr[row + 1]):
            g = indices[entry]
            w = values[entry]
            for c in range(marker.shape[1]):
                out[row, c] += w * marker[g, c]


class CPUMomentRegression:
    """Synchronous shared-LD backend; buffers never escape to asynchronous users."""

    def __init__(self, sparse, prepared, n_blocks, threads=1, trait_batch=4):
        self.sparse = sparse
        self.design = prepare_moment_design(sparse, prepared, n_blocks)
        self.n_blocks = n_blocks
        self.threads = max(1, min(threads, numba_config.NUMBA_NUM_THREADS, available_cpu_count()))
        self.trait_batch = trait_batch
        self.output = None

    def compute(self, marker):
        marker = np.ascontiguousarray(marker, dtype=np.float32)
        if marker.ndim != 2 or marker.shape[0] != self.sparse.shape[1] or marker.shape[1] < 1:
            raise ValueError("Marker dimensions do not match SNP-gene matrix")
        if not np.isfinite(marker).all():
            raise ValueError("Non-finite marker scores")
        shape = (self.sparse.shape[0], marker.shape[1])
        if self.output is None or self.output.shape != shape:
            self.output = np.empty(shape, np.float32)
        old = get_num_threads()
        try:
            set_num_threads(self.threads)
            a = self.sparse
            p = self.design
            sparse_out(a.indptr, a.indices, a.data, marker, self.output)
            h = gene_hsq(p, marker)
            values = []
            for start in range(0, len(p["lengths"]), self.trait_batch):
                end = start + self.trait_batch
                terms = accumulate_partial(
                    self.output,
                    p["positions"][start:end],
                    p["lengths"][start:end],
                    p["features"][start:end],
                    p["nbar"][start:end],
                    h[start:end],
                    self.n_blocks,
                    128,
                )
                values.append(solve_parallel(terms, p["nbar"][start:end]))
            result = np.concatenate(values, axis=0).transpose(0, 2, 1)
            if not np.isfinite(result).all() or np.any(result[:, 1] <= 0):
                from gsMap.spatial_ldsc.backends.regression import invalid_moments_error

                raise invalid_moments_error(result)
            return result
        finally:
            set_num_threads(old)

    def close(self):
        self.output = None
