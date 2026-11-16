Welcome to gsMap's documentation!
===================================

.. warning::

   **gsMap 1.x is deprecated.** This documentation describes gsMap 1.x (1.73.x), which
   receives bug fixes only. Pin the version when installing:
   ``pip install "gsMap<2"`` or ``conda install "bioconda::gsmap<2"``. Without a version,
   ``pip install gsMap`` installs gsMap 2.x on Python 3.12 or later.

   New analyses should use gsMap 2.x: see the `gsMap 2.x documentation <https://jianyang-lab.github.io/gsMap/>`_ and
   `Upgrading to gsMap 2.0 <https://jianyang-lab.github.io/gsMap/upgrading.html>`_.


Introduction
------------

``gsMap`` (genetically informed spatial mapping of cells for complex traits) integrates spatial transcriptomics (ST) data with genome-wide association study (GWAS) summary statistics to map cells to human complex traits, including diseases, in a spatially resolved manner.


How to Cite
------------
If you use ``gsMap`` in your studies, please cite:

   Song, L., Chen, W., Hou, J., Guo, M. & Yang, J. `Spatially resolved mapping of cells associated with human complex traits <https://doi.org/10.1038/s41586-025-08757-x>`_. Nature (2025).

Key Features
------------

- **Spatially-aware High-Resolution Trait Mapping**: Maps trait-associated cells at single-cell resolution, offering insights into their spatial distributions.
- **Spatial Region Identification**: Aggregates trait-cell association p-values into trait-tissue region association p-values, prioritizing tissue regions relevant to traits of interest.
- **Putative Causal Genes Identification**: Prioritizes putative causal genes by associating gene expression levels with cell-trait relevance.


Overview of ``gsMap`` Method
-----------------------------

``gsMap`` operates on a four-step process:

1. **Gene Specificity Assessment in Spatial Contexts**: To address technical noise and capture spatial correlations of gene expression profiles in ST data, ``gsMap`` leverages GNNs to identify homogeneous spots for each spot and estimates gene specificity scores by aggregating information from those homogeneous spots.
2. **Linking Gene Specificity to SNPs**: ``gsMap`` assigns gene specificity scores to single nucleotide polymorphisms (SNPs) based on their proximity to gene transcription start sites (TSS) and SNP-to-gene epigenetic linking maps.
3. **Spatial S-LDSC**: To estimate the relevance of spots to traits, ``gsMap`` associates stratified LD scores of individual spots with GWAS summary statistics using the S-LDSC framework.
4. **Spatial Region Identification**: To evaluate the association of a specific spatial region with traits, ``gsMap`` employs the Cauchy combination test to aggregate p-values from individual spots within that spatial region.

.. image:: _static/schematic.svg
   :width: 600
   :alt: Model architecture

Schmatics of ``gsMap`` method. For more details about the ``gsMap``, please check out our `publication <URL>`__.

Installation
------------

``gsMap`` is available on `gsMap GitHub <https://github.com/JianYang-Lab/gsMap>`__.


How to install ``gsMap``, check out the `installation guide <install.rst>`__

Tutorials
---------
How to use ``gsMap``, check out the `tutorials <tutorials.rst>`__


Online Analysis Service (coming soon)
--------------------------------------
Users could upload their own GWAS summary statistics data to perform the analysis.


.. toctree::
    :maxdepth: 2
    :caption: Contents:

    install
    tutorials
    data
    api
    release
