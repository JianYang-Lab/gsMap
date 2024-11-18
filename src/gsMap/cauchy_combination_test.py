import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp

from gsMap.config import CauchyCombinationConfig

logger = logging.getLogger(__name__)

# The fun of cauchy combination
def acat_test(pvalues, weights=None):
    '''acat_test()
    Aggregated Cauchy Assocaition Test
    A p-value combination method using the Cauchy distribution.

    Inspired by: https://github.com/yaowuliu/ACAT/blob/master/R/ACAT.R
    Inputs:
        pvalues: <list or numpy array>
            The p-values you want to combine.
        weights: <list or numpy array>, default=None
            The weights for each of the p-values. If None, equal weights are used.

    Returns:
        pval: <float>
            The ACAT combined p-value.
    '''
    if any(np.isnan(pvalues)):
        raise Exception("Cannot have NAs in the p-values.")
    if any([(i > 1) | (i < 0) for i in pvalues]):
        raise Exception("P-values must be between 0 and 1.")
    if any([i == 1 for i in pvalues]) & any([i == 0 for i in pvalues]):
        raise Exception("Cannot have both 0 and 1 p-values.")
    if any([i == 0 for i in pvalues]):
        logger.info("Warn: p-values are exactly 0.")
        return 0
    if any([i == 1 for i in pvalues]):
        logger.info("Warn: p-values are exactly 1.")
        return 1
    if weights == None:
        weights = [1 / len(pvalues) for i in pvalues]
    elif len(weights) != len(pvalues):
        raise Exception("Length of weights and p-values differs.")
    elif any([i < 0 for i in weights]):
        raise Exception("All weights must be positive.")
    else:
        weights = [i / len(weights) for i in weights]

    pvalues = np.array(pvalues)
    weights = np.array(weights)

    if any([i < 1e-16 for i in pvalues]) == False:
        cct_stat = sum(weights * np.tan((0.5 - pvalues) * np.pi))
    else:
        is_small = [i < (1e-16) for i in pvalues]
        is_large = [i >= (1e-16) for i in pvalues]
        cct_stat = sum((weights[is_small] / pvalues[is_small]) / np.pi)
        cct_stat += sum(weights[is_large] * np.tan((0.5 - pvalues[is_large]) * np.pi))

    if cct_stat > 1e15:
        pval = (1 / cct_stat) / np.pi
    else:
        pval = 1 - sp.stats.cauchy.cdf(cct_stat)

    return pval


def run_Cauchy_combination(config: CauchyCombinationConfig):
    ldsc_list = []

    for sample_name in config.sample_name_list:
        # Load the LDSC results for the current sample
        logger.info(f'------Loading LDSC results for sample {sample_name}...')
        ldsc_input_file = config.get_ldsc_result_file(trait_name=config.trait_name, sample_name=sample_name)
        ldsc = pd.read_csv(ldsc_input_file, compression='gzip')
        ldsc['spot'] = ldsc['spot'].astype(str)
        ldsc.index = ldsc['spot']

        # Load the spatial transcriptomics (ST) data for the current sample
        logger.info(f'------Loading ST data for sample {sample_name}...')
        h5ad_file = config.get_hdf5_with_latent_path(sample_name=sample_name)
        adata = sc.read_h5ad(h5ad_file)

        # Identify common cells between LDSC results and ST data
        common_cells = np.intersect1d(ldsc.index, adata.obs_names)
        adata = adata[common_cells]
        ldsc = ldsc.loc[common_cells]

        # Add annotations to the LDSC dataframe
        ldsc['annotation'] = adata.obs.loc[ldsc.spot, config.annotation].to_list()
        ldsc_list.append(ldsc)

    # Concatenate all LDSC dataframes from different samples
    ldsc_all = pd.concat(ldsc_list)

    # Run the Cauchy combination
    p_cauchy = []
    p_median = []
    annotations = ldsc_all['annotation'].unique()

    for ct in annotations:
        p_values = ldsc_all.loc[ldsc_all['annotation'] == ct, 'p']

        # Handle extreme outliers to enhance robustness
        p_values_log = -np.log10(p_values)
        median_log = np.median(p_values_log)
        iqr_log = np.percentile(p_values_log, 75) - np.percentile(p_values_log, 25)

        p_values_filtered = p_values[p_values_log < median_log + 3 * iqr_log]
        n_removed = len(p_values) - len(p_values_filtered)

        # Remove outliers if the number is reasonable
        if 0 < n_removed < 20:
            logger.info(f'Removed {n_removed}/{len(p_values)} outliers (median + 3IQR) for {ct}.')
            p_cauchy_temp = acat_test(p_values_filtered)
        else:
            p_cauchy_temp = acat_test(p_values)

        p_median_temp = np.median(p_values)
        p_cauchy.append(p_cauchy_temp)
        p_median.append(p_median_temp)

    # Prepare the results dataframe
    results = pd.DataFrame({
        'annotation': annotations,
        'p_cauchy': p_cauchy,
        'p_median': p_median
    })

    # Save the results
    output_dir = Path(config.output_file)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    output_file = Path(config.output_file)
    results.to_csv(
        output_file,
        compression='gzip',
        index=False,
    )
    logger.info(f'Cauchy combination results saved at {output_file}.')
