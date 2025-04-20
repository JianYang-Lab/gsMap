import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import norm

from gsMap.config import DiagnosisConfig
from gsMap.utils.manhattan_plot import ManhattanPlot
from gsMap.visualize import draw_scatter, estimate_point_size_for_plot, load_ldsc, load_st_coord

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)


def convert_z_to_p(gwas_data):
    """Convert Z-scores to P-values."""
    gwas_data["P"] = norm.sf(abs(gwas_data["Z"])) * 2
    min_p_value = 1e-300
    gwas_data["P"] = gwas_data["P"].clip(lower=min_p_value)
    return gwas_data


def load_gene_diagnostic_info(config: DiagnosisConfig):
    """Load or compute gene diagnostic info."""
    gene_diagnostic_info_save_path = config.get_gene_diagnostic_info_save_path(config.trait_name)
    if gene_diagnostic_info_save_path.exists():
        logger.info(
            f"Loading gene diagnostic information from {gene_diagnostic_info_save_path}..."
        )
        return pd.read_csv(gene_diagnostic_info_save_path)
    else:
        logger.info(
            "Gene diagnostic information not found. Calculating gene diagnostic information..."
        )
        return compute_gene_diagnostic_info(config)


def compute_gene_diagnostic_info(config: DiagnosisConfig):
    """Calculate gene diagnostic info and save it to adata."""
    logger.info("Loading ST data and LDSC results...")
    # adata = sc.read_h5ad(config.hdf5_with_latent_path, backed='r')
    mk_score = pd.read_feather(config.mkscore_feather_path)
    mk_score.set_index("HUMAN_GENE_SYM", inplace=True)
    mk_score = mk_score.T
    trait_ldsc_result = load_ldsc(config.get_ldsc_result_file(config.trait_name))

    # Align marker scores with trait LDSC results
    mk_score = mk_score.loc[trait_ldsc_result.index]

    # Filter out genes with no variation
    has_variation = (~mk_score.eq(mk_score.iloc[0], axis=1)).any()
    mk_score = mk_score.loc[:, has_variation]

    logger.info("Calculating correlation between gene marker scores and trait logp-values...")
    corr = mk_score.corrwith(trait_ldsc_result["logp"])
    corr.name = "PCC"

    grouped_mk_score = mk_score.groupby(adata.obs[config.annotation]).median()
    max_annotations = grouped_mk_score.idxmax()

    high_GSS_Gene_annotation_pair = pd.DataFrame(
        {
            "Gene": max_annotations.index,
            "Annotation": max_annotations.values,
            "Median_GSS": grouped_mk_score.max().values,
        }
    )

    high_GSS_Gene_annotation_pair = high_GSS_Gene_annotation_pair.merge(
        corr, left_on="Gene", right_index=True
    )

    # Prepare the final gene diagnostic info dataframe
    gene_diagnostic_info_cols = ["Gene", "Annotation", "Median_GSS", "PCC"]
    gene_diagnostic_info = (
        high_GSS_Gene_annotation_pair[gene_diagnostic_info_cols]
        .drop_duplicates()
        .dropna(subset=["Gene"])
    )
    gene_diagnostic_info.sort_values("PCC", ascending=False, inplace=True)

    # Save gene diagnostic info to a file
    gene_diagnostic_info_save_path = config.get_gene_diagnostic_info_save_path(config.trait_name)
    gene_diagnostic_info.to_csv(gene_diagnostic_info_save_path, index=False)
    logger.info(f"Gene diagnostic information saved to {gene_diagnostic_info_save_path}.")

    return gene_diagnostic_info.reset_index()


def load_gwas_data(config: DiagnosisConfig):
    """Load and process GWAS data."""
    logger.info("Loading and processing GWAS data...")
    gwas_data = pd.read_csv(config.sumstats_file, compression="gzip", sep="\t")
    return convert_z_to_p(gwas_data)


def load_snp_gene_pairs(config: DiagnosisConfig):
    """Load SNP-gene pairs from multiple chromosomes."""
    ldscore_save_dir = Path(config.ldscore_save_dir)
    return pd.concat(
        [
            pd.read_feather(ldscore_save_dir / f"SNP_gene_pair/SNP_gene_pair_chr{chrom}.feather")
            for chrom in range(1, 23)
        ]
    )


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
            f"To reduce the number of SNPs to plot, only {SUBSAMPLE_SNP_NUMBER} SNPs with the smallest P-values are plotted."
        )

    return snps2plot


def generate_manhattan_plot(config: DiagnosisConfig):
    """Generate Manhattan plot."""
    # report_save_dir = config.get_report_dir(config.trait_name)
    gwas_data = load_gwas_data(config)
    snp_gene_pair = load_snp_gene_pairs(config)
    gwas_data_with_gene = snp_gene_pair.merge(gwas_data, on="SNP", how="inner").rename(
        columns={"gene_name": "GENE"}
    )
    gene_diagnostic_info = load_gene_diagnostic_info(config)
    gwas_data_with_gene_annotation = gwas_data_with_gene.merge(
        gene_diagnostic_info, left_on="GENE", right_on="Gene", how="left"
    )

    gwas_data_with_gene_annotation = gwas_data_with_gene_annotation[
        ~gwas_data_with_gene_annotation["Annotation"].isna()
    ]
    gwas_data_with_gene_annotation_sort = gwas_data_with_gene_annotation.sort_values("P")

    snps2plot = filter_snps(gwas_data_with_gene_annotation_sort, SUBSAMPLE_SNP_NUMBER=100_000)
    gwas_data_to_plot = gwas_data_with_gene_annotation[
        gwas_data_with_gene_annotation["SNP"].isin(snps2plot)
    ].reset_index(drop=True)
    gwas_data_to_plot["Annotation_text"] = (
        "PCC: "
        + gwas_data_to_plot["PCC"].round(2).astype(str)
        + "<br>"
        + "Annotation: "
        + gwas_data_to_plot["Annotation"].astype(str)
    )

    # Verify data integrity
    if gwas_data_with_gene_annotation_sort.empty:
        logger.error("Filtered GWAS data is empty, cannot create Manhattan plot")
        return

    if len(gwas_data_to_plot) == 0:
        logger.error("No SNPs passed filtering criteria for Manhattan plot")
        return

    # Log some diagnostic information
    logger.info(f"Creating Manhattan plot with {len(gwas_data_to_plot)} SNPs")
    logger.info(f"Columns available: {list(gwas_data_to_plot.columns)}")
    logger.info(f"Chromosome column values: {gwas_data_to_plot['CHR'].unique()}")

    fig = ManhattanPlot(
        dataframe=gwas_data_to_plot,
        title="gsMap Diagnosis Manhattan Plot",
        point_size=3,
        highlight_gene_list=config.selected_genes
        or gene_diagnostic_info.Gene.iloc[: config.top_corr_genes].tolist(),
        suggestiveline_value=-np.log10(1e-5),
        annotation="Annotation_text",
    )

    save_manhattan_plot_path = config.get_manhattan_html_plot_path(config.trait_name)
    fig.write_html(save_manhattan_plot_path)
    logger.info(f"Diagnostic Manhattan Plot saved to {save_manhattan_plot_path}.")


def generate_GSS_distribution(config: DiagnosisConfig):
    """Generate GSS distribution plots."""
    # logger.info('Loading ST data...')
    # adata = sc.read_h5ad(config.hdf5_with_latent_path)
    mk_score = pd.read_feather(config.mkscore_feather_path).set_index("HUMAN_GENE_SYM").T

    plot_genes = (
        config.selected_genes
        or load_gene_diagnostic_info(config).Gene.iloc[: config.top_corr_genes].tolist()
    )
    if config.selected_genes is not None:
        logger.info(
            f"Generating GSS & Expression distribution plot for selected genes: {plot_genes}..."
        )
    else:
        logger.info(
            f"Generating GSS & Expression distribution plot for top {config.top_corr_genes} correlated genes..."
        )

    if config.customize_fig:
        pixel_width, pixel_height, point_size = (
            config.fig_width,
            config.fig_height,
            config.point_size,
        )
    else:
        (pixel_width, pixel_height), point_size = estimate_point_size_for_plot(
            adata.obsm["spatial"]
        )
    sub_fig_save_dir = config.get_GSS_plot_dir(config.trait_name)

    # save plot gene list
    config.get_GSS_plot_select_gene_file(config.trait_name).write_text("\n".join(plot_genes))

    for selected_gene in plot_genes:
        expression_series = pd.Series(
            adata[:, selected_gene].X.toarray().flatten(), index=adata.obs.index, name="Expression"
        )
        threshold = np.quantile(expression_series, 0.9999)
        expression_series[expression_series > threshold] = threshold
        generate_and_save_plots(
            adata,
            mk_score,
            expression_series,
            selected_gene,
            point_size,
            pixel_width,
            pixel_height,
            sub_fig_save_dir,
            config.sample_name,
            config.annotation,
        )


def generate_and_save_plots(
    adata,
    mk_score,
    expression_series,
    selected_gene,
    point_size,
    pixel_width,
    pixel_height,
    sub_fig_save_dir,
    sample_name,
    annotation,
):
    """Generate and save the plots."""
    select_gene_expression_with_space_coord = load_st_coord(adata, expression_series, annotation)
    sub_fig_1 = draw_scatter(
        select_gene_expression_with_space_coord,
        title=f"{selected_gene} (Expression)",
        annotation="annotation",
        color_by="Expression",
        point_size=point_size,
        width=pixel_width,
        height=pixel_height,
    )
    save_plot(sub_fig_1, sub_fig_save_dir, sample_name, selected_gene, "Expression")

    select_gene_GSS_with_space_coord = load_st_coord(
        adata, mk_score[selected_gene].rename("GSS"), annotation
    )
    sub_fig_2 = draw_scatter(
        select_gene_GSS_with_space_coord,
        title=f"{selected_gene} (GSS)",
        annotation="annotation",
        color_by="GSS",
        point_size=point_size,
        width=pixel_width,
        height=pixel_height,
    )
    save_plot(sub_fig_2, sub_fig_save_dir, sample_name, selected_gene, "GSS")

    # combined_fig = make_subplots(rows=1, cols=2,
    #                              subplot_titles=(f'{selected_gene} (Expression)', f'{selected_gene} (GSS)'))
    # for trace in sub_fig_1.data:
    #     combined_fig.add_trace(trace, row=1, col=1)
    # for trace in sub_fig_2.data:
    #     combined_fig.add_trace(trace, row=1, col=2)
    #


def save_plot(sub_fig, sub_fig_save_dir, sample_name, selected_gene, plot_type):
    """Save the plot to HTML and PNG."""
    save_sub_fig_path = (
        sub_fig_save_dir / f"{sample_name}_{selected_gene}_{plot_type}_Distribution.html"
    )
    # sub_fig.write_html(str(save_sub_fig_path))
    sub_fig.update_layout(showlegend=False)
    sub_fig.write_image(str(save_sub_fig_path).replace(".html", ".png"))


def generate_gsMap_plot(config: DiagnosisConfig):
    """Generate gsMap plot."""
    logger.info("Creating gsMap plot...")

    trait_ldsc_result = load_ldsc(config.get_ldsc_result_file(config.trait_name))
    space_coord_concat = load_st_coord(adata, trait_ldsc_result, annotation=config.annotation)

    if config.customize_fig:
        pixel_width, pixel_height, point_size = (
            config.fig_width,
            config.fig_height,
            config.point_size,
        )
    else:
        (pixel_width, pixel_height), point_size = estimate_point_size_for_plot(
            adata.obsm["spatial"]
        )
    fig = draw_scatter(
        space_coord_concat,
        title=f"{config.trait_name} (gsMap)",
        point_size=point_size,
        width=pixel_width,
        height=pixel_height,
        annotation=config.annotation,
    )

    output_dir = config.get_gsMap_plot_save_dir(config.trait_name)
    output_file_html = config.get_gsMap_html_plot_save_path(config.trait_name)
    output_file_png = output_file_html.with_suffix(".png")
    output_file_csv = output_file_html.with_suffix(".csv")

    fig.write_html(output_file_html)
    fig.write_image(output_file_png)
    space_coord_concat.to_csv(output_file_csv)

    logger.info(f"gsMap plot created and saved in {output_dir}.")


def run_Diagnosis(config: DiagnosisConfig):
    """Main function to run the diagnostic plot generation."""
    global adata
    adata = sc.read_h5ad(config.hdf5_with_latent_path)
    if "log1p" not in adata.uns.keys() and adata.X.max() > 14:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    if config.plot_type in ["gsMap", "all"]:
        generate_gsMap_plot(config)
    if config.plot_type in ["manhattan", "all"]:
        generate_manhattan_plot(config)
    if config.plot_type in ["GSS", "all"]:
        generate_GSS_distribution(config)
