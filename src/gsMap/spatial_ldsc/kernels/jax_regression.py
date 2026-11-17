"""Single production regression kernel for spatial LDSC."""

from functools import partial

import jax
import jax.numpy as jnp
from jax import jit


def _packed_block_equations(spatial, baseline, chisq, weights_sq):
    """Share SNP-wise products across cells for small baseline designs.

    Pack the symmetric baseline products and response products into one
    contraction, and the spatial cross-products into another. The shared
    tensor has no cell axis; at most four baseline columns produce fourteen
    shared columns rather than a SNP-by-cell-by-feature design tensor.
    """
    n_baseline = baseline.shape[-1]
    pairs = [(i, j) for i in range(n_baseline) for j in range(i, n_baseline)]
    shared = jnp.stack(
        [baseline[..., i] * baseline[..., j] for i, j in pairs]
        + [baseline[..., i] * chisq for i in range(n_baseline)],
        axis=-1,
    )
    # Normal equations can amplify reduced-mantissa product errors. Do not
    # let GPU autotuning trade FP32 product precision for TF32 throughput.
    baseline_terms = jnp.einsum(
        "nbf,nbs->nsf", shared, weights_sq, precision=jax.lax.Precision.HIGHEST
    )
    cross_design = jnp.concatenate([baseline, chisq[..., None]], axis=-1)
    cross_terms = jnp.einsum(
        "nbf,nbs->nsf", cross_design, spatial * weights_sq, precision=jax.lax.Precision.HIGHEST
    )
    spatial_square = jnp.sum(spatial * spatial * weights_sq, axis=1)

    entries = {(i + 1, j + 1): baseline_terms[..., index] for index, (i, j) in enumerate(pairs)}
    entries.update({(0, i + 1): cross_terms[..., i] for i in range(n_baseline)})
    entries[0, 0] = spatial_square
    xtx_blocks = jnp.stack(
        [
            jnp.stack([entries[min(i, j), max(i, j)] for j in range(n_baseline + 1)], axis=-1)
            for i in range(n_baseline + 1)
        ],
        axis=-2,
    )
    xty_blocks = jnp.concatenate(
        [cross_terms[..., -1, None], baseline_terms[..., len(pairs) :]], axis=-1
    )
    return xtx_blocks, xty_blocks


@partial(jit, static_argnums=(0,))
def process_spatial_ldsc_batch(
    n_blocks: int,
    spatial_ld: jnp.ndarray,
    baseline_ld_sum: jnp.ndarray,
    chisq: jnp.ndarray,
    N: jnp.ndarray,
    baseline_ann: jnp.ndarray,
    w_ld: jnp.ndarray,
    Nbar: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Regress one SNP-by-cell LD-score batch with bounded intermediates.

    The kernel accumulates weighted normal equations by jackknife block rather
    than materializing a SNP-by-cell-by-feature design tensor. It is used on
    both CPU and GPU so both backends have identical numerical semantics.
    """
    n_snps, n_spots = spatial_ld.shape
    n_baseline = baseline_ann.shape[1]

    x_tot = spatial_ld + baseline_ld_sum.reshape(-1, 1)
    N_expanded = N.reshape(-1, 1)
    mean_chisq = jnp.mean(chisq)
    mean_x_tot_N = jnp.mean(x_tot * N_expanded, axis=0)
    hsq = 10000.0 * (mean_chisq - 1.0) / mean_x_tot_N
    hsq = jnp.clip(hsq, 0.0, 1.0)

    ld_clip = jnp.maximum(x_tot, 1.0)
    w_ld_clip = jnp.maximum(w_ld.reshape(-1, 1), 1.0)
    c = (hsq.reshape(1, -1) * N_expanded) / 10000.0
    # Normalizing sqrt(weights_sq) by its per-cell SNP sum multiplies every
    # normal equation (including each delete-block equation) by the same
    # positive scalar. It cancels from the estimates and their jackknife SE.
    # Use inverse-variance weights directly, avoiding a full-array sqrt and
    # global normalization pass. Reduction-order roundoff can still differ.
    weights_sq = 1.0 / (2 * jnp.square(1.0 + c * ld_clip) * w_ld_clip)

    block_size = n_snps // n_blocks
    n_snps_used = block_size * n_blocks
    spatial_ann = spatial_ld * N_expanded / Nbar
    spatial_blocks = spatial_ann[:n_snps_used].reshape(n_blocks, block_size, n_spots)
    baseline_blocks = baseline_ann[:n_snps_used].reshape(n_blocks, block_size, n_baseline)
    chisq_blocks = chisq[:n_snps_used].reshape(n_blocks, block_size)
    weights_sq_blocks = weights_sq[:n_snps_used].reshape(n_blocks, block_size, n_spots)

    if 1 <= n_baseline <= 4:
        xtx_blocks, xty_blocks = _packed_block_equations(
            spatial_blocks, baseline_blocks, chisq_blocks, weights_sq_blocks
        )
    else:
        # Avoid explicitly packing O(n_baseline**2) SNP-wise products for
        # larger custom baseline sets; retain the general contraction path.
        spatial_xty = jnp.einsum(
            "nbs,nb,nbs->ns",
            spatial_blocks,
            chisq_blocks,
            weights_sq_blocks,
            precision=jax.lax.Precision.HIGHEST,
        )
        baseline_xty = jnp.einsum(
            "nbk,nb,nbs->nsk",
            baseline_blocks,
            chisq_blocks,
            weights_sq_blocks,
            precision=jax.lax.Precision.HIGHEST,
        )
        xty_blocks = jnp.concatenate([spatial_xty[..., None], baseline_xty], axis=-1)

        spatial_spatial = jnp.einsum(
            "nbs,nbs,nbs->ns",
            spatial_blocks,
            spatial_blocks,
            weights_sq_blocks,
            precision=jax.lax.Precision.HIGHEST,
        )
        spatial_baseline = jnp.einsum(
            "nbs,nbk,nbs->nsk",
            spatial_blocks,
            baseline_blocks,
            weights_sq_blocks,
            precision=jax.lax.Precision.HIGHEST,
        )
        baseline_baseline = jnp.einsum(
            "nbk,nbl,nbs->nskl",
            baseline_blocks,
            baseline_blocks,
            weights_sq_blocks,
            precision=jax.lax.Precision.HIGHEST,
        )
        top_row = jnp.concatenate(
            [spatial_spatial[..., None, None], spatial_baseline[..., None, :]], axis=-1
        )
        bottom_rows = jnp.concatenate([spatial_baseline[..., :, None], baseline_baseline], axis=-1)
        xtx_blocks = jnp.concatenate([top_row, bottom_rows], axis=-2)

    xty_total = jnp.sum(xty_blocks, axis=0)
    xtx_total = jnp.sum(xtx_blocks, axis=0)
    est = jnp.linalg.solve(xtx_total, xty_total[..., None]).squeeze(-1)

    xty_del = xty_total - xty_blocks
    xtx_del = xtx_total - xtx_blocks
    delete_ests = jnp.linalg.solve(xtx_del, xty_del[..., None]).squeeze(-1)

    spatial_pseudovalues = n_blocks * est[:, 0] - (n_blocks - 1) * delete_ests[:, :, 0]
    spatial_jknife_est = jnp.mean(spatial_pseudovalues, axis=0)
    spatial_centered = spatial_pseudovalues - spatial_jknife_est
    spatial_jknife_se = jnp.sqrt(
        jnp.sum(jnp.square(spatial_centered), axis=0) / (n_blocks * (n_blocks - 1))
    )
    return spatial_jknife_est / Nbar, spatial_jknife_se / Nbar
