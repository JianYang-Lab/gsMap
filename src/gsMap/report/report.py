import logging
import os
import shutil

import pandas as pd
from jinja2 import Environment, FileSystemLoader

import gsMap
from gsMap.cauchy_combination_test import run_Cauchy_combination
from gsMap.config import CauchyCombinationConfig, ReportConfig
from gsMap.diagnosis import run_Diagnosis

logger = logging.getLogger(__name__)

# Load the Jinja2 environment
try:
    from importlib.resources import files

    template_dir = files("gsMap").joinpath("templates")
except (ImportError, FileNotFoundError):
    # Fallback to a relative path if running in development mode
    template_dir = os.path.join(os.path.dirname(__file__), "templates")

# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader(template_dir))

# Load the template
template = env.get_template("report_template.html")


def copy_files_to_report_dir(result_dir, report_dir, files_to_copy):
    """Copy specified files (HTML or PNG) to the report directory."""
    os.makedirs(report_dir, exist_ok=True)
    for file in files_to_copy:
        shutil.copy2(file, os.path.join(report_dir, os.path.basename(file)))


def load_cauchy_table(csv_file):
    """Load the Cauchy combination table from a compressed CSV file using Pandas."""
    df = pd.read_csv(csv_file, compression="gzip")
    table_data = df[["annotation", "p_cauchy", "p_median"]].to_dict(orient="records")
    return table_data


def load_gene_diagnostic_info(csv_file):
    """Load the Gene Diagnostic Info CSV file and return the top 50 rows."""
    df = pd.read_csv(csv_file)
    top_50 = df.head(50).to_dict(orient="records")
    return top_50


def embed_html_content(file_path):
    """Read the content of an HTML file and return it as a string."""
    with open(file_path) as f:
        return f.read()


def check_and_run_cauchy_combination(config):
    cauchy_result_file = config.get_cauchy_result_file(config.trait_name)
    if cauchy_result_file.exists():
        logger.info(
            f"Cauchy combination already done for trait {config.trait_name}. Results saved at {cauchy_result_file}. Skipping..."
        )
    else:
        logger.info(f"Running Cauchy combination for trait {config.trait_name}...")
        cauchy_config = CauchyCombinationConfig(
            workdir=config.workdir,
            sample_name=config.sample_name,
            annotation=config.annotation,
            trait_name=config.trait_name,
        )
        run_Cauchy_combination(cauchy_config)

    df = pd.read_csv(cauchy_result_file, compression="gzip")
    table_data = df[["annotation", "p_cauchy", "p_median"]].to_dict(orient="records")

    return table_data


def run_report(config: ReportConfig, run_parameters=None):
    logger.info("Running gsMap Diagnosis Module")
    run_Diagnosis(config)
    logger.info("gsMap Diagnosis running successfully")

    report_dir = config.get_report_dir(config.trait_name)
    gene_diagnostic_info_file = config.get_gene_diagnostic_info_save_path(config.trait_name)
    gene_diagnostic_info = load_gene_diagnostic_info(gene_diagnostic_info_file)

    # Load data (Cauchy table and gene diagnostic info)
    cauchy_table = check_and_run_cauchy_combination(config)

    # Paths to PNGs for gene expression and GSS distribution
    gss_distribution_dir = config.get_GSS_plot_dir(config.trait_name)

    gene_plots = []
    plot_select_gene_list = (
        config.get_GSS_plot_select_gene_file(config.trait_name).read_text().splitlines()
    )
    for gene_name in plot_select_gene_list:
        expression_png = (
            gss_distribution_dir / f"{config.sample_name}_{gene_name}_Expression_Distribution.png"
        )
        gss_png = gss_distribution_dir / f"{config.sample_name}_{gene_name}_GSS_Distribution.png"
        # check if expression and GSS plots exist
        if not os.path.exists(expression_png) or not os.path.exists(gss_png):
            print(f"Skipping gene {gene_name} as expression or GSS plot is missing.")
            continue
        gene_plots.append(
            {
                "name": gene_name,
                "expression_plot": expression_png.relative_to(
                    report_dir
                ),  # Path for gene expression plot
                "gss_plot": gss_png.relative_to(report_dir),  # Path for GSS distribution plot
            }
        )

    # # Copy PNG files to the report directory
    # copy_files_to_report_dir(result_dir, report_dir, [gene['expression_plot'] for gene in gene_plots] + [gene['gss_plot'] for gene in gene_plots])

    # Update paths to point to copied images inside the report folder
    # for gene in gene_plots:
    #     gene['expression_plot'] = os.path.join(os.path.basename(gene['expression_plot']))
    #     gene['gss_plot'] = os.path.join(os.path.basename(gene['gss_plot']))

    # Sample data for other report components
    title = f"{config.sample_name} Genetic Spatial Mapping Report"

    genetic_mapping_plot = embed_html_content(
        config.get_gsMap_html_plot_save_path(config.trait_name)
    )
    manhattan_plot = embed_html_content(config.get_manhattan_html_plot_path(config.trait_name))

    gsmap_version = gsMap.__version__
    # Render the template with dynamic content, including the run parameters

    trait_name = config.trait_name
    default_run_parameters = {
        "Sample Name": config.sample_name,
        "Trait Name": trait_name,
        "Summary Statistics File": config.sumstats_file,
        "HDF5 Path": config.hdf5_with_latent_path,
        "Annotation": config.annotation,
        "Spatial LDSC Save Directory": config.ldsc_save_dir,
        "Cauchy Directory": config.cauchy_save_dir,
        "Report Directory": config.get_report_dir(trait_name),
        "gsMap Report File": config.get_gsMap_report_file(trait_name),
        "Gene Diagnostic Info File": config.get_gene_diagnostic_info_save_path(trait_name),
        "Report Generation Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if run_parameters is not None:
        default_run_parameters.update(run_parameters)

    output_html = template.render(
        title=title,
        genetic_mapping_plot=genetic_mapping_plot,  # Inlined genetic mapping plot
        manhattan_plot=manhattan_plot,  # Inlined Manhattan plot
        cauchy_table=cauchy_table,
        gene_plots=gene_plots,  # List of PNG paths for gene plots
        gsmap_version=gsmap_version,
        parameters=default_run_parameters,  # Pass the run parameters to the template
        gene_diagnostic_info=gene_diagnostic_info,  # Include top 50 gene diagnostic info rows
    )

    # Save the generated HTML report in the 'report' directory
    report_file = config.get_gsMap_report_file(config.trait_name)
    with open(report_file, "w") as f:
        f.write(output_html)

    logger.info(f"Report generated successfully! Saved at {report_file}.")
    logger.info(
        "Copy the report directory to your local PC and open the HTML report file in a web browser to view the report."
    )
