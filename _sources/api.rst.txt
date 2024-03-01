.. _api-documentation:

GPS Command Line Alphabets
===========================


.. program:: GPS

Synopsis
--------

.. code-block:: shell

    usage: GPS [-h] [--version] {run_find_latent_representations,run_latent_to_gene,run_generate_ldscore,run_spatial_ldsc,run_cauchy_combination,run_visualize,run_all_mode} ...

Description
-----------

GPS: Genetics-informed pathogenic spatial mapping.

Options
-------

.. option:: -h, --help

    Show this help message and exit.

.. option:: --version, -v

    Show program's version number and exit.

Subcommands
-----------

.. option:: format_sumstats

    Run Format_sumstats.
    Convert GWAS summary statistics into the format that GPS can recognize.

.. option:: run_find_latent_representations

    Run Find_latent_representations.
    Find the latent representations of each spot by running GNN-VAE.

.. option:: run_latent_to_gene

    Run Latent_to_gene.
    Estimate gene marker gene scores for each spot by using latent representations from nearby spots.

.. option:: run_generate_ldscore

    Run Generate_ldscore.
    Generate LD scores for each spot.

.. option:: run_spatial_ldsc

    Run Spatial_ldsc.
    Run spatial LDSC for each spot.

.. option:: run_cauchy_combination

    Run Cauchy_combination for each annotation.

.. option:: run_visualize

    Visualize the GPS results.

.. option:: run_all_mode

    Run GPS Pipeline.
    GPS Pipeline (Run Find_latent_representations, Latent_to_gene, and Generate_ldscore) in order.

-----

.. toctree::
    :maxdepth: 1
    :caption: Subcommands Documentation

    api/format_sumstats
    api/find_latent_representations
    api/latent_to_gene
    api/generate_ldscore
    api/spatial_ldsc
    api/cauchy_combination
    api/visualization

