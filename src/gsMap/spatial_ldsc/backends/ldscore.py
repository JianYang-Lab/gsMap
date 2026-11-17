"""LD-score construction backends for spatial LDSC.

The core weight-matrix operation is:

    spatial_ld = snp_gene_weight_sparse @ marker_score_chunk

where the sparse matrix is SNP x gene, the dense marker chunk is gene x cell,
and the returned LD-score chunk must be usable by the JAX regression kernel.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

import jax
import numpy as np

logger = logging.getLogger("gsMap.spatial_ldsc.ldscore_backend")

CPU_BACKEND_VALUES = {"0", "false", "off", "disable", "disabled", "cpu"}
GPU_BACKEND_VALUES = {"auto", "torch"}


_GPU_MEMORY_ERROR_MARKERS = (
    "out of memory",  # torch.OutOfMemoryError, CUDA_ERROR_OUT_OF_MEMORY
    "resource_exhausted",  # XLA allocation failures
    "resource exhausted",
    "cudaerrormemoryallocation",
    "cublas_status_alloc_failed",
    "cusparse_status_alloc_failed",
    "failed to allocate",
)


def is_gpu_memory_error(exc: BaseException) -> bool:
    """Return True for GPU out-of-memory errors.

    Only allocation failures qualify. Errors such as "an illegal memory access was
    encountered" leave the CUDA context unusable and must propagate, not fall back.
    """
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _GPU_MEMORY_ERROR_MARKERS)


class LDScoreBackend(ABC):
    """Backend that turns a CPU marker-score chunk into a JAX LD-score array."""

    name: str

    @abstractmethod
    def compute(self, marker_score_chunk: np.ndarray):
        """Return a JAX array with shape SNP x cell."""

    def close(self):  # noqa: B027  optional hook; only GPU backends override it
        """Release backend-owned GPU state."""


class TorchLDScoreBackend(LDScoreBackend):
    """PyTorch CSR backend.

    This is the preferred GPU path for gsMap GPU installs because PyTorch is
    already required and the benchmarked sparse CSR kernel is the fastest path
    for the production LD-score chunk shape.
    """

    name = "torch"

    def __init__(self, sparse_weight):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is not available")

        self.torch = torch
        self.device = torch.device("cuda:0")
        self.sparse_weight = None
        self._last_output = None
        start_time = time.time()
        try:
            crow = torch.as_tensor(
                sparse_weight.indptr.astype(np.int64, copy=False), device=self.device
            )
            col = torch.as_tensor(
                sparse_weight.indices.astype(np.int64, copy=False), device=self.device
            )
            values = torch.as_tensor(
                sparse_weight.data.astype(np.float32, copy=False), device=self.device
            )
            self.sparse_weight = torch.sparse_csr_tensor(
                crow,
                col,
                values,
                size=sparse_weight.shape,
                dtype=torch.float32,
                device=self.device,
                check_invariants=True,
            )
            torch.cuda.synchronize()
        except Exception:
            self.close()
            raise
        logger.info(
            "Prepared PyTorch GPU SNP-gene sparse matrix for LD-score construction "
            f"in {time.time() - start_time:.2f}s"
        )

    def compute(self, marker_score_chunk: np.ndarray):
        ldscore_gpu, _ = self.compute_torch(marker_score_chunk)
        return jax.dlpack.from_dlpack(ldscore_gpu)

    def compute_torch(self, marker_score_chunk: np.ndarray):
        """Return fresh LD and marker tensors for same-stream fused consumers."""
        marker_score_gpu = self.torch.as_tensor(
            marker_score_chunk,
            dtype=self.torch.float32,
            device=self.device,
        )
        # beta=0 overwrites every output entry without reading its old value.
        # Avoid the zero-filled output allocated by the sparse.mm path. Keep a
        # fresh tensor per batch: an earlier result may still be used by JAX.
        # This SNP x cell product is independent of the baseline column count.
        ldscore_gpu = self.torch.empty(
            (self.sparse_weight.shape[0], marker_score_gpu.shape[1]),
            dtype=self.torch.float32,
            device=self.device,
        )
        self.torch.addmm(
            ldscore_gpu, self.sparse_weight, marker_score_gpu, beta=0, out=ldscore_gpu
        )
        # The DLPack object protocol passes JAX's consumer stream to PyTorch,
        # which inserts an event dependency on its producer stream. A device-
        # wide barrier here needlessly waits for the previous JAX regression.
        # Do not replace this with a raw capsule: that loses stream negotiation.
        self._last_output = ldscore_gpu
        return ldscore_gpu, marker_score_gpu

    def close(self):
        self._last_output = None
        self.sparse_weight = None
        if getattr(self, "torch", None) is not None and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def _try_backend(backend_cls: type[LDScoreBackend], sparse_weight) -> LDScoreBackend | None:
    try:
        return backend_cls(sparse_weight)
    except Exception as e:  # noqa: BLE001
        if is_gpu_memory_error(e):
            logger.warning(
                "Could not initialize %s LD-score backend because GPU memory is insufficient. "
                "Falling back to CPU/SciPy LD-score construction while keeping the shared "
                "--use-gpu JAX regression path. Reduce --spots-per-chunk-quick-mode, "
                "set GSMAP_DISABLE_GPU_LDSCORE=1, or use --no-gpu if regression also runs "
                "out of memory.",
                backend_cls.name,
            )
            return None
        logger.warning(
            "Could not initialize %s LD-score backend. Reason: %s: %s",
            backend_cls.name,
            type(e).__name__,
            e,
        )
        return None


def _backend_candidates(requested_backend: str) -> list[type[LDScoreBackend]]:
    if requested_backend == "auto":
        return [TorchLDScoreBackend]
    if requested_backend == "torch":
        return [TorchLDScoreBackend]
    return []


def select_gpu_ldscore_backend(sparse_weight) -> LDScoreBackend | None:
    """Select and initialize the GPU LD-score backend.

    Returns None when GPU LD-score construction is disabled or unavailable,
    leaving the caller on the CPU/SciPy construction path.
    """
    if os.environ.get("GSMAP_DISABLE_GPU_LDSCORE") == "1":
        logger.info(
            "GPU LD-score construction disabled by GSMAP_DISABLE_GPU_LDSCORE=1; "
            "using CPU sparse multiplication"
        )
        return None

    requested_backend = os.environ.get("GSMAP_GPU_LDSCORE_BACKEND", "auto").strip().lower()
    if requested_backend in CPU_BACKEND_VALUES:
        logger.info(
            "GPU LD-score construction disabled by GSMAP_GPU_LDSCORE_BACKEND=%r; "
            "using CPU sparse multiplication",
            requested_backend,
        )
        return None
    if requested_backend not in GPU_BACKEND_VALUES:
        logger.warning(
            "Unknown GSMAP_GPU_LDSCORE_BACKEND=%r; expected auto, torch, or cpu. Using auto.",
            requested_backend,
        )
        requested_backend = "auto"

    for backend_cls in _backend_candidates(requested_backend):
        backend = _try_backend(backend_cls, sparse_weight)
        if backend is not None:
            return backend

    if requested_backend == "auto":
        logger.warning(
            "No GPU LD-score backend could be initialized; using CPU sparse multiplication"
        )
    else:
        logger.warning(
            "Requested %s LD-score backend is unavailable; using CPU sparse multiplication",
            requested_backend,
        )
    return None
