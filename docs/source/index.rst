Welcome to gsMap's documentation!
=================================

**gsMap**: **g**\ enetically informed **s**\ patial **Map**\ ping of cells for complex traits.

**gsMap** integrates spatial transcriptomics (ST) data, from 2D tissue sections to 3D tissues, with genome-wide association study (GWAS) summary statistics to map cells associated with human complex traits and diseases.

.. note::

   This documentation describes gsMap 2.x, currently in beta. gsMap 1.x users should read
   :doc:`upgrading`. The gsMap 1.x documentation remains at
   `gsMap 1.x Documentation <https://yanglab.westlake.edu.cn/gsmap/document/software>`_.


What's New in gsMap 2.0
-----------------------

.. important::

   - Support for **3D** ST data mapping of registered sections and thick tissues
   - **Dual-embedding** strategy for more accurate gene specificity scoring
   - **GPU support** for embeddings, gene specificity scoring and spatial LDSC
   - **Multi-trait** and multi-GPU spatial LDSC with interactive reports


Features
--------

- **Spatially-aware High-Resolution Trait Mapping**: Maps trait-associated cells at single-cell resolution, offering insights into their spatial distributions.
- **Spatial Region Identification**: Aggregates trait-cell association p-values into trait-tissue region association p-values, prioritizing tissue regions relevant to traits of interest.
- **Putative Causal Genes Identification**: Prioritizes putative causal genes by associating gene expression levels with cell-trait relevance.
- **Scalability**: Supports CPU/GPU computation and disk-backed arrays for large spatial datasets.


Explore gsMap
-------------

.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - **Installation**

       Get gsMap running on your system using uv, pip, or conda.

       :doc:`Go to Installation <install>`
     - **Key Concepts**

       Learn basic CLI/Python usage, input formats, key settings and the biological workflow.

       :doc:`Go to Key Concepts <key_concepts>`

   * - **Tutorials**

       Step-by-step guides for 2D and 3D spatial data.

       :doc:`2D Tutorial <2d_tutorial/index>` | :doc:`3D Tutorial <3d_tutorial/index>`
     - **Output Files**

       Understand the output directory structure and file formats.

       :doc:`Go to Output Files Reference <output_files>`

   * - **Customization Guide**

       Change input fields, embeddings, graph/matching settings and LD references.

       :doc:`Go to Customization Guide <advanced_usage>`
     - **Scalability**

       E16.5 examples for CPU/GPU, SSD storage, memory tuning and cluster jobs.

       :doc:`Go to Scalability Guide <scalability>`

Overview of the ``gsMap`` Method
--------------------------------

``gsMap`` operates on a four-step process:

1. **Gene Specificity Assessment in 3D Spatial Contexts**:
   To address technical noise and capture spatial correlations of gene expression across consecutive ST sections,
   ``gsMap`` constructs batch-corrected dual embeddings of molecular and spatial features.
   These embeddings jointly model transcriptomic similarity and local spatial context to identify homogeneous cells
   in 3D space. Gene specificity scores (GSS) are then computed by aggregating normalized gene expression ranks
   across these 3D homogeneous cells, enabling robust identification of genes that are both highly and specifically
   expressed in focal cells.

2. **Linking Gene Specificity to Genetic Variants**:
   ``gsMap`` links gene specificity scores to single nucleotide polymorphisms (SNPs) by assigning GSS to SNPs
   based on their proximity to gene transcription start sites (TSS) and SNP-to-gene epigenetic linking maps,
   thereby connecting spatially resolved gene expression patterns with trait-associated genetic variation.

3. **3D Spatial S-LDSC for Cell–Trait Association**:
   To quantify cell–trait associations in 3D space, ``gsMap`` integrates the 3D gene specificity scores with
   genome-wide association study (GWAS) summary statistics using stratified LD score regression (S-LDSC).
   This framework enables the estimation of trait relevance for individual cells by associating their stratified
   LD scores with GWAS signals.

4. **Spatial Region–Trait Association Analysis**:
   To evaluate associations between traits and spatial regions, ``gsMap`` aggregates p-values from cells
   within a given 3D spatial region using the Cauchy combination test, yielding region-level association
   statistics that reflect coordinated genetic effects across spatially organized cell populations.

.. figure:: /_static/gsMap3D_Method_schematic.png
   :align: center
   :width: 100%
   :alt: gsMap Method Schematic


How to Cite
------------
If you use ``gsMap`` in your studies, please cite:
   gsMap: Song, L., Chen, W., Hou, J., Guo, M. & Yang, J. "**Spatially resolved mapping of cells associated with human complex traits.**" *Nature* (2025). `doi:10.1038/s41586-025-08757-x <https://doi.org/10.1038/s41586-025-08757-x>`_

   A citation for the gsMap 2.0 methods will be added here when available.


.. toctree::
    :maxdepth: 2
    :hidden:

    install
    upgrading
    downloads
    key_concepts
    2D Mapping <2d_tutorial/index>
    3D Mapping <3d_tutorial/index>
    output_files
    advanced_usage
    scalability
    ldscore_weight_matrix
    cli_reference
    release
