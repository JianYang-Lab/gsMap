"""
Simplified LD score computation without masking or padding.

Direct computation of unbiased L2 statistics from genotype matrices using NumPy and Scipy.
"""

import numpy as np
import scipy.sparse

from .constants import LDSC_BIAS_CORRECTION_DF


def compute_unbiased_l2_batch(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
) -> np.ndarray:
    """
    Compute unbiased LD scores (L2) for HM3 SNPs against reference block.

    The unbiased L2 estimator is:
        L2 = r^2 - (1 - r^2) / (N - 2)

    where r^2 is the squared correlation and N is the number of individuals.

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)

    Returns
    -------
    np.ndarray
        Unbiased LD scores, shape (n_hm3_snps, n_ref_snps)
    """
    n_individuals = X_hm3.shape[0]

    if n_individuals <= LDSC_BIAS_CORRECTION_DF:
        raise ValueError(f"At least {LDSC_BIAS_CORRECTION_DF + 1} individuals are required")

    # Keep one working array throughout. Algebraically,
    #   r² - (1-r²)/(N-2) = r²*(N-1)/(N-2) - 1/(N-2).
    # The in-place formulation avoids allocating separate r², correction, and
    # corrected-L2 matrices for every HM3 batch.
    l2_unbiased = np.matmul(X_hm3.T, X_ref_block)
    l2_unbiased *= 1.0 / n_individuals
    np.square(l2_unbiased, out=l2_unbiased)
    denominator = n_individuals - LDSC_BIAS_CORRECTION_DF
    l2_unbiased *= (n_individuals - 1) / denominator
    l2_unbiased -= 1.0 / denominator

    return l2_unbiased


def compute_batch_weights_sparse(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
    block_mapping_matrix: scipy.sparse.csr_matrix | np.ndarray,
    ld_window_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute LD score weight matrix using matrix multiplication.

    Works for both sparse and dense mapping matrices.
    Weights = L2_Unbiased @ Mapping_Matrix

    Parameters
    ----------
    X_hm3 : np.ndarray
        Standardized genotypes for HM3 SNPs, shape (n_individuals, n_hm3_snps)
    X_ref_block : np.ndarray
        Standardized genotypes for reference block, shape (n_individuals, n_ref_snps)
    block_mapping_matrix : Union[scipy.sparse.csr_matrix, np.ndarray]
        Mapping matrix for the reference block, shape (n_ref_snps, n_features).
        Can be a sparse CSR matrix (from creating_snp_feature_map) or a dense array (from annotations).

    Returns
    -------
    weights : np.ndarray
        Weight matrix, shape (n_hm3_snps, n_features)
    """
    # 1. Compute unbiased L2 matrix: (n_hm3_snps, n_ref_snps)
    l2_unbiased = compute_unbiased_l2_batch(X_hm3, X_ref_block)
    if ld_window_mask is not None:
        if ld_window_mask.shape != l2_unbiased.shape:
            raise ValueError("LD-window mask shape does not match the L2 matrix")
        np.multiply(l2_unbiased, ld_window_mask, out=l2_unbiased)

    # 2. Compute Weights: W = L2 @ M
    # (n_hm3, n_ref) @ (n_ref, n_features) -> (n_hm3, n_features)
    # numpy dot handles dense @ dense
    # scipy.sparse handles dense @ sparse
    if scipy.sparse.issparse(block_mapping_matrix):
        weights = l2_unbiased @ block_mapping_matrix
    else:
        weights = np.matmul(l2_unbiased, block_mapping_matrix)

    # Ensure output is a dense numpy array
    if scipy.sparse.issparse(weights):
        weights = weights.toarray()
    elif isinstance(weights, np.matrix):
        weights = np.asarray(weights)

    return np.asarray(weights, dtype=np.float32)


def compute_batch_weights_and_ld_scores(
    X_hm3: np.ndarray,
    X_ref_block: np.ndarray,
    block_mapping_matrix: scipy.sparse.csr_matrix | np.ndarray,
    ld_window_mask: np.ndarray,
    ld_score_reference_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute annotation weights and optional weight-LD scores from one L2 block.

    The expensive standardized-genotype matrix multiplication is shared.  The
    annotation result uses every broad-reference column in the exact LD window,
    while ``ld_score_reference_mask`` can restrict the LD-score sum to a second
    universe, such as HM3 regression SNPs excluding the MHC.
    """
    l2_unbiased = compute_unbiased_l2_batch(X_hm3, X_ref_block)
    if ld_window_mask.shape != l2_unbiased.shape:
        raise ValueError("LD-window mask shape does not match the L2 matrix")
    np.multiply(l2_unbiased, ld_window_mask, out=l2_unbiased)

    if scipy.sparse.issparse(block_mapping_matrix):
        weights = l2_unbiased @ block_mapping_matrix
    else:
        weights = np.matmul(l2_unbiased, block_mapping_matrix)
    if scipy.sparse.issparse(weights):
        weights = weights.toarray()
    elif isinstance(weights, np.matrix):
        weights = np.asarray(weights)

    ld_scores = None
    if ld_score_reference_mask is not None:
        reference_mask = np.asarray(ld_score_reference_mask, dtype=bool)
        if reference_mask.ndim != 1 or reference_mask.shape[0] != l2_unbiased.shape[1]:
            raise ValueError("LD-score reference mask does not match the reference block")
        if not np.any(reference_mask):
            raise ValueError("LD-score reference mask contains no regression SNPs")
        ld_scores = np.sum(l2_unbiased[:, reference_mask], axis=1, dtype=np.float64)

    return np.asarray(weights, dtype=np.float32), ld_scores
