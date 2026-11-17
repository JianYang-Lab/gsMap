"""Uniform shared-regression dispatch and explicit asynchronous result ownership.

This module contains runtime policy, not numerical kernels. CPU moments, GPU
moments and the general JAX path return the same RegressionBatch interface.
CUDA-only dependencies remain lazily imported after backend selection.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gsMap.spatial_ldsc.data.traits import PARAMETER_KEYS

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger("gsMap.spatial_ldsc.regression_backend")


def select_moment_backend(config, sparse, prepared, ld_backend, platform):
    """Choose a supported fast path without hiding native execution failures."""
    from gsMap.spatial_ldsc.backends.moment_design import supports_moments

    requested = getattr(config, "ldsc_regression_backend", "auto")
    if requested not in ("auto", "jax", "moments"):
        raise ValueError("Unknown LDSC regression backend")
    if not 1 <= config.ldsc_trait_batch_size <= 16:
        raise ValueError("ldsc_trait_batch_size must be 1..16")
    if requested == "jax":
        return None
    if supports_moments(sparse, prepared):
        if platform == "cpu":
            from gsMap.spatial_ldsc.backends.cpu_moments import CPUMomentRegression

            return CPUMomentRegression(
                sparse,
                prepared,
                config.n_blocks,
                threads=getattr(config, "ldsc_compute_workers", 10),
                trait_batch=config.ldsc_trait_batch_size,
            )
        if ld_backend is not None and hasattr(ld_backend, "compute_torch"):
            try:
                from gsMap.spatial_ldsc.backends.gpu_moments import GPUMomentRegression
            except ImportError:
                logger.warning("Optional Triton moment backend unavailable; using JAX")
            else:
                return GPUMomentRegression(
                    sparse,
                    prepared,
                    config.n_blocks,
                    ld_backend,
                    trait_batch=config.ldsc_trait_batch_size,
                )
    if requested == "moments":
        raise ValueError(
            "Moments require float32 CSR marker mode, two baseline annotations "
            "and a supported CPU or CUDA/Triton backend"
        )
    logger.info("Using general JAX regression for this input/backend")
    return None


@dataclass(slots=True)
class RegressionBatch:
    """Grouped [traits, beta/SE, cells] results with their producer lifetimes."""

    groups: list
    owners: Any = None
    validate_moments: bool = False

    def wait(self) -> None:
        """Finish all consumers before the caller releases producer owners."""
        import jax

        jax.block_until_ready(self.groups)

    def to_host(self) -> np.ndarray:
        import jax

        result = np.concatenate(jax.device_get(self.groups), axis=0)
        if self.validate_moments and (not np.isfinite(result).all() or np.any(result[:, 1] <= 0)):
            raise invalid_moments_error(result)
        return result


class SharedRegressionBackend(AbstractContextManager):
    """Own prepared compute state; the caller owns and drains submitted batches."""

    def __init__(self, config, sparse, prepared, n_snps, *, moment_selector=None):
        import jax
        import jax.numpy as jnp

        from gsMap.spatial_ldsc.backends.ldscore import select_gpu_ldscore_backend
        from gsMap.spatial_ldsc.kernels.jax_multitrait import prepare_mean_design

        self.config = config
        self.sparse = sparse
        self.platform = jax.default_backend()
        self.ld_backend = None
        self.moments = None
        self.design = None
        self.groups = []
        self._closed = False
        selector = moment_selector if moment_selector is not None else select_moment_backend
        try:
            if sparse is not None and self.platform == "gpu":
                self.ld_backend = select_gpu_ldscore_backend(sparse)
            self.moments = selector(config, sparse, prepared, self.ld_backend, self.platform)
            if self.moments is None:
                params = [
                    (
                        jnp.asarray(d["indices"]),
                        tuple(jnp.asarray(d[key]) for key in PARAMETER_KEYS),
                    )
                    for d in prepared
                ]
                self.design = prepare_mean_design(
                    n_snps,
                    [d["indices"] for d in prepared],
                    [d["N"] for d in prepared],
                    [d["baseline_sum"] for d in prepared],
                    [d["chisq"] for d in prepared],
                )
                size = config.ldsc_trait_batch_size
                self.groups = [
                    (i, tuple(params[i : i + size])) for i in range(0, len(params), size)
                ]
        except BaseException:
            self.close()
            raise

    @property
    def name(self) -> str:
        return type(self.moments).__name__ if self.moments is not None else "JAX"

    @property
    def cpu_moments(self) -> bool:
        return self.moments is not None and self.platform == "cpu"

    def thread_context(self):
        """Keep the small projection from oversubscribing the Numba workers."""
        if self.cpu_moments:
            from threadpoolctl import threadpool_limits

            return threadpool_limits(limits=1, user_api="blas")
        return nullcontext()

    def compute(self, *, marker=None, ldscore=None) -> RegressionBatch:
        """Submit one ordinary chunk without waiting for previous GPU batches."""
        import jax.numpy as jnp

        from gsMap.spatial_ldsc.kernels.jax_multitrait import mean_regression, shared_hsq

        if self._closed:
            raise RuntimeError("Shared regression backend is closed")
        if self.moments is not None:
            if self.cpu_moments:
                return RegressionBatch([self.moments.compute(marker)], validate_moments=True)
            groups, owners = self.moments.compute(marker)
            return RegressionBatch(groups, owners, validate_moments=True)
        if self.sparse is None:
            ld = jnp.asarray(ldscore)
        elif self.ld_backend is not None:
            ld = self.ld_backend.compute(marker)
        else:
            ld = jnp.asarray(self.sparse @ marker)
        hsq = shared_hsq(ld, *self.design)
        groups = [
            mean_regression(self.config.n_blocks, ld, group, hsq[lo : lo + len(group)])
            for lo, group in self.groups
        ]
        return RegressionBatch(groups, ld)

    def close(self) -> None:
        """Close all owned backends even if one cleanup raises; idempotent."""
        if self._closed:
            return
        self._closed = True
        try:
            if self.moments is not None:
                self.moments.close()
        finally:
            try:
                if self.ld_backend is not None:
                    self.ld_backend.close()
            finally:
                self.groups.clear()
                self.design = None

    def __exit__(self, *exc_info):
        self.close()
        return False


def invalid_moments_error(result) -> FloatingPointError:
    """Explain a rejected moments result (never published) and the exact fallback."""
    import numpy as np

    values = np.asarray(result)
    bad = ~np.isfinite(values).all(axis=1) | np.any(values[:, 1:2] <= 0, axis=1)
    return FloatingPointError(
        f"Singular or invalid spatial LDSC moment regression for {int(bad.sum()):,} "
        "trait-spot pairs; no results were written. Rerun with "
        "--ldsc-regression-backend jax, which fits these spots exactly and records "
        "singular spots as missing."
    )
