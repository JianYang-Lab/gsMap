import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from scipy.stats import norm

logger = logging.getLogger(__name__)


def convert_z_to_p(gwas_data):
    """Convert Z-scores to P-values."""
    z_scores = pd.to_numeric(gwas_data["Z"], errors="coerce").to_numpy()
    gwas_data["Z"] = z_scores
    gwas_data["P"] = norm.sf(np.abs(z_scores)) * 2
    min_p_value = 1e-300
    gwas_data["P"] = gwas_data["P"].clip(lower=min_p_value)
    return gwas_data


def load_gwas_data(sumstats_file, *, usecols=None, calculate_p: bool = True, snp_filter=None):
    """Load GWAS columns with the fastest available parser."""

    started = time.perf_counter()
    source = Path(sumstats_file)
    parser = "pyarrow"
    input_rows = None
    try:
        options = {"dtype_backend": "pyarrow"} if snp_filter is not None else {}
        gwas_data = pd.read_csv(
            source,
            compression="gzip",
            sep="\t",
            usecols=usecols,
            engine="pyarrow",
            **options,
        )
        input_rows = len(gwas_data)
        if snp_filter is not None:
            # Delay Python string/object creation until irrelevant GWAS rows
            # have been removed. Keep all columns and the input row order.
            table = pa.Table.from_pandas(gwas_data, preserve_index=False)
            nullable_integers = [
                name
                for name in table.column_names
                if pa.types.is_integer(table[name].type) and table[name].null_count
            ]
            nullable_booleans = [
                name
                for name in table.column_names
                if pa.types.is_boolean(table[name].type) and table[name].null_count
            ]
            # A nearly reference-sized GWAS gains little from an extra filter.
            # This gate affects cost only: Manhattan still joins to its weights.
            if "SNP" in table.column_names:
                if len(snp_filter) == 0:
                    table = table.slice(0, 0)
                elif len(table) > 2 * len(snp_filter):
                    mask = pc.is_in(table["SNP"], value_set=pa.array(snp_filter))
                    table = table.filter(mask)
            gwas_data = table.replace_schema_metadata(None).to_pandas()
            # Match the full-table numpy conversion even when all nulls were
            # outside the retained rows (important for CSV dtype/format parity).
            for name in nullable_integers:
                gwas_data[name] = gwas_data[name].astype(np.float64)
            for name in nullable_booleans:
                gwas_data[name] = gwas_data[name].astype(object)
        if not {"SNP", "Z"}.issubset(gwas_data.columns):
            raise ValueError("Not a gzip-compressed tab-separated GWAS file")
    except (ImportError, ValueError, OSError, pa.ArrowException):
        # Accept what spatial LDSC accepts: plain, gzip or bz2 files with any
        # whitespace separator.
        parser = "pandas"
        logger.debug(f"PyArrow could not parse {source}; falling back to pandas")
        gwas_data = pd.read_csv(
            source,
            compression="infer",
            sep=r"\s+",
            na_values=".",
            usecols=usecols,
        )
        input_rows = len(gwas_data)
        if (
            snp_filter is not None
            and "SNP" in gwas_data.columns
            and len(gwas_data) > 2 * len(snp_filter)
        ):
            gwas_data = gwas_data[gwas_data["SNP"].isin(snp_filter)].reset_index(drop=True)

    required_columns = {"SNP", "Z"}
    missing_columns = required_columns.difference(gwas_data.columns)
    if missing_columns:
        raise ValueError(
            f"GWAS file {source} is missing required columns: {sorted(missing_columns)}"
        )

    logger.info(
        f"Loaded {len(gwas_data):,}/{input_rows:,} GWAS rows from {source.name} with {parser} "
        f"in {time.perf_counter() - started:.2f}s"
    )
    gwas_data.attrs["source_rows"] = input_rows
    return convert_z_to_p(gwas_data) if calculate_p else gwas_data


def filter_snps(gwas_data_with_gene_annotation_sort, SUBSAMPLE_SNP_NUMBER):
    """Filter the SNPs based on significance levels."""
    pass_suggestive_line_mask = gwas_data_with_gene_annotation_sort["P"] < 1e-5
    pass_suggestive_line_number = pass_suggestive_line_mask.sum()

    if pass_suggestive_line_number > SUBSAMPLE_SNP_NUMBER:
        snps2plot = gwas_data_with_gene_annotation_sort[pass_suggestive_line_mask].SNP
        logger.info(
            f"To reduce the number of SNPs to plot, only {snps2plot.shape[0]} SNPs with P < 1e-5 are plotted."
        )
    else:
        snps2plot = gwas_data_with_gene_annotation_sort.head(SUBSAMPLE_SNP_NUMBER).SNP
        logger.info(
            f"To reduce the number of SNPs to plot, {snps2plot.shape[0]} SNPs with "
            f"the smallest P-values are plotted."
        )

    return snps2plot
