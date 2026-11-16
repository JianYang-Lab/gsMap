# gsMap

|               |                                                                                                      |                |                                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------- |
| __Version__   | [![PyPI version][pypi-badge]][pypi-url] [![Python][python-badge]][python-url]                        | __Status__     | [![Project Status][status-badge]][status-url] [![Maintenance][maintenance-badge]][maintenance-url] |
| __Activity__  | [![GitHub commits][commits-badge]][commits-url] [![Last Commit][last-commit-badge]][last-commit-url] | __Quality__    | [![codecov][codecov-badge]][codecov-url] [![Ruff][ruff-badge]][ruff-url]                           |
| __CI/CD__     | [![Docs][docs-badge]][docs-url] [![test][test-badge]][test-url]                                      | __Community__  | [![GitHub stars][stars-badge]][stars-url] [![GitHub forks][forks-badge]][forks-url]                |
| __Downloads__ | [![Downloads][downloads-badge]][downloads-url]                                                       | __License__    | [![License: MIT][license-badge]][license-url] [![DOI][doi-badge]][doi-url]                         |
| __Platform__  | [![Linux][linux-badge]][linux-url]                                                                   | __Contribute__ | [![Issues][issues-badge]][issues-url] [![PRs Welcome][pr-badge]][pr-url]                           |

`gsMap` (**g**enetically informed **s**patial **map**ping of cells for complex traits)
integrates spatial transcriptomics (ST) data with genome-wide association study (GWAS)
summary statistics to map cells associated with human complex traits and diseases,
in 2D tissue sections and in 3D tissues.

> **Note — gsMap 2.0 is in beta.** Install it with an explicit version, as shown in
> [Installation](#%EF%B8%8F-installation). Until 2.0.0 is released, `pip install gsMap` without a
> version still installs gsMap 1.x. Looking for gsMap 1.x? See [gsMap 1.x](#gsmap-1x).

______________________________________________________________________

## ✨ What's new in gsMap 2.0

- **3D spatial mapping** of registered multi-section data and single thick-tissue volumes,
    with optional matching of homogeneous cells across adjacent sections.
- **Dual embeddings** of molecular and spatial features for more accurate gene specificity scores.
- **GPU acceleration** of embeddings, gene specificity scores and spatial LDSC, including
    multi-trait and single-node multi-GPU spatial LDSC.
- **Interactive reports** with trait comparison views and SpatialVista export for 3D data.
- **Ancestry-specific GRCh38 LD references** with matching regression weights and checksums.
- **A new command-line interface and Python API.** See [Upgrading](#-upgrading).

______________________________________________________________________

## 🚀 Features

- **Spatially-aware High-Resolution Trait Mapping**\
    Maps trait-associated cells at single-cell resolution, offering insights into their spatial distributions.

- **Spatial Region Identification**\
    Aggregates trait–cell association p-values into trait–tissue region association p-values, prioritizing tissue regions relevant to traits of interest.

- **Putative Causal Genes Identification**\
    Prioritizes putative causal genes by associating gene expression levels with cell–trait relevance.

- **Scalability**\
    Uses [JAX](https://github.com/jax-ml/jax) JIT compilation, GPU acceleration and disk-backed arrays to scale to spatial datasets with millions of cells.

______________________________________________________________________

## 🧠 Overview

`gsMap` operates in four steps:

1. **Gene specificity assessment in spatial context.** To address technical noise and capture
    spatial correlations of gene expression across consecutive ST sections, `gsMap` constructs
    batch-corrected dual embeddings of molecular and spatial features. These embeddings identify
    homogeneous cells in 2D or 3D space. Gene specificity scores (GSS) are computed by aggregating
    normalized gene expression ranks across these homogeneous cells.
1. **Linking gene specificity to genetic variants.** GSS are assigned to SNPs based on their
    proximity to gene transcription start sites (TSS) and SNP-to-gene epigenetic linking maps.
1. **Spatial S-LDSC for cell–trait association.** Stratified LD score regression (S-LDSC) relates
    the stratified LD scores of each cell to GWAS summary statistics, estimating the trait relevance
    of individual cells.
1. **Spatial region–trait association.** The Cauchy combination test aggregates the p-values of
    cells within a spatial region into region-level association statistics.

![gsMap method schematic](https://raw.githubusercontent.com/JianYang-Lab/gsMap/main/docs/source/_static/gsMap3D_Method_schematic.png)

______________________________________________________________________

## 🛠️ Installation

gsMap 2.x requires Python 3.12 or later on Linux. Choose exactly one compute backend:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.13
source .venv/bin/activate

# Choose exactly one command.
uv pip install "gsMap[cpu]==2.0.0b1" --torch-backend=cpu
uv pip install "gsMap[cuda128]==2.0.0b1" --torch-backend=cu128
uv pip install "gsMap[cuda130]==2.0.0b1" --torch-backend=cu130

gsmap --version
```

The [Installation guide](https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/install.html)
covers the CUDA 12.6 and TPU backends, Conda environments and installation from source.

## ▶️ Quick start

Download the shared resources and a dataset from
[Downloads](https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/downloads.html),
then run the whole pipeline with one command:

```bash
gsmap quick-mode \
    --workdir "results" \
    --project-name "mouse-e16-mosta" \
    --dataset-type "spatial2D" \
    --h5ad-path "example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad" \
    --homolog-file "resources/homologs/mouse_human_homologs.txt" \
    --species MOUSE_GENE_SYM \
    --w-ld-dir "resources/1000g-hc-grch38-v50-eur/w_ld" \
    --snp-gene-weight-adata-path "resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad" \
    --annotation "annotation" \
    --spatial-key "spatial" \
    --data-layer "count" \
    --trait-name "IQ" \
    --sumstats-file "example_data/example-traits/IQ.sumstats.gz"

gsmap report-view results/mouse-e16-mosta/gsmap_web_report
```

Follow the [2D and 3D tutorials](https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/3d_tutorial/human_gastrulation.html)
for complete examples. The
[Customization Guide](https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/advanced_usage.html)
explains how to use your own data, reference weights, embeddings and analysis settings.

______________________________________________________________________

## 🔄 Upgrading

### From gsMap 1.x

- **Python 3.12 or later is required.** On Python 3.10 and 3.11, `pip install gsMap` keeps
    resolving to gsMap 1.x.

- **The command-line interface changed:**

    | gsMap 1.x                               | gsMap 2.x                                                           |
    | --------------------------------------- | ------------------------------------------------------------------- |
    | `gsmap quick_mode`                      | `gsmap quick-mode`                                                  |
    | `gsmap run_find_latent_representations` | `gsmap find-latent`                                                 |
    | `gsmap run_latent_to_gene`              | `gsmap latent-to-gene`                                              |
    | `gsmap run_generate_ldscore`            | `gsmap ldscore-weight-matrix` (run once per reference)              |
    | `gsmap run_spatial_ldsc`                | `gsmap spatial-ldsc`                                                |
    | `gsmap run_cauchy_combination`          | `gsmap cauchy-combination`                                          |
    | `gsmap run_report`                      | report step of `gsmap quick-mode`; view it with `gsmap report-view` |
    | `gsmap format_sumstats`                 | `gsmap format-sumstats`                                             |

    Run `gsmap COMMAND --help` or see the
    [CLI reference](https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/cli_reference.html)
    for the current options.

- **Reference resources changed.** gsMap 2.x reads a precomputed SNP-by-gene weight matrix
    (`--snp-gene-weight-adata-path`) and matching regression weights (`--w-ld-dir`). Download them
    from Downloads or build your own with `gsmap ldscore-weight-matrix`.

- Start 2.x analyses in a new working directory; the output layout changed (see
    [Output Files](https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1/docs/r1/output_files.html)).

### From the gsMap3D preview (0.1.x)

gsMap 2.0 continues the code previewed on PyPI as `gsMap3D`. Only the package name changed. Both
packages install the same `gsMap` module, so remove the preview first:

```bash
pip uninstall gsMap3D
```

### gsMap 1.x

gsMap 1.x remains available for existing analyses and receives bug fixes only:

```bash
pip install "gsMap<2"
```

Its source is on the [`1.x` branch](https://github.com/JianYang-Lab/gsMap/tree/1.x) and its
documentation is at [gsMap 1.x Documentation](https://yanglab.westlake.edu.cn/gsmap/document/software).

______________________________________________________________________

## 📚 Documentation

See the [Documentation](https://yanglab.westlake.edu.cn/gsmap3d-portal/docs) and
[Website](https://yanglab.westlake.edu.cn/gsmap3d-portal). Release notes are in
[CHANGELOG.md](https://github.com/JianYang-Lab/gsMap/blob/main/CHANGELOG.md).

______________________________________________________________________

## 📝 How to Cite

If you use `gsMap` in your studies, please cite:

Song, L., Chen, W., Hou, J., Guo, M. & Yang, J.
[Spatially resolved mapping of cells associated with human complex traits.](https://doi.org/10.1038/s41586-025-08757-x)
*Nature* (2025).

A citation for the gsMap 2.0 methods will be added here when available.

## ✨ Research Highlights

gsMap was highlighted in [Nature Methods](https://www.nature.com/articles/s41592-025-02711-5)
and [Nature Reviews Genetics](https://www.nature.com/articles/s41576-025-00877-4).

<!-- Badge links -->

[codecov-badge]: https://codecov.io/gh/JianYang-Lab/gsMap/graph/badge.svg?token=NFZFXZIEUU
[codecov-url]: https://codecov.io/gh/JianYang-Lab/gsMap
[commits-badge]: https://img.shields.io/github/commit-activity/m/JianYang-Lab/gsMap
[commits-url]: https://github.com/JianYang-Lab/gsMap/commits/main
[docs-badge]: https://github.com/JianYang-Lab/gsMap/actions/workflows/docs.yml/badge.svg
[docs-url]: https://github.com/JianYang-Lab/gsMap/actions/workflows/docs.yml
[doi-badge]: https://img.shields.io/badge/DOI-10.1038%2Fs41586--025--08757--x-blue
[doi-url]: https://doi.org/10.1038/s41586-025-08757-x
[downloads-badge]: https://static.pepy.tech/badge/gsMap
[downloads-url]: https://pepy.tech/project/gsMap
[forks-badge]: https://img.shields.io/github/forks/JianYang-Lab/gsMap
[forks-url]: https://github.com/JianYang-Lab/gsMap/network/members
[issues-badge]: https://img.shields.io/github/issues/JianYang-Lab/gsMap
[issues-url]: https://github.com/JianYang-Lab/gsMap/issues
[last-commit-badge]: https://img.shields.io/github/last-commit/JianYang-Lab/gsMap
[last-commit-url]: https://github.com/JianYang-Lab/gsMap/commits/main
[license-badge]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://opensource.org/licenses/MIT
[linux-badge]: https://img.shields.io/badge/Linux-%E2%9C%93-success
[linux-url]: https://github.com/JianYang-Lab/gsMap/actions/workflows/test_linux.yml
[maintenance-badge]: https://img.shields.io/badge/Maintained%3F-yes-green.svg
[maintenance-url]: https://github.com/JianYang-Lab/gsMap/graphs/commit-activity
[pr-badge]: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
[pr-url]: https://github.com/JianYang-Lab/gsMap/pulls
[pypi-badge]: https://img.shields.io/pypi/v/gsMap
[pypi-url]: https://pypi.org/project/gsMap/
[python-badge]: https://img.shields.io/pypi/pyversions/gsMap
[python-url]: https://www.python.org
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff-url]: https://github.com/astral-sh/ruff
[stars-badge]: https://img.shields.io/github/stars/JianYang-Lab/gsMap
[stars-url]: https://github.com/JianYang-Lab/gsMap/stargazers
[status-badge]: https://www.repostatus.org/badges/latest/active.svg
[status-url]: https://www.repostatus.org/#active
[test-badge]: https://github.com/JianYang-Lab/gsMap/actions/workflows/test_linux.yml/badge.svg
[test-url]: https://github.com/JianYang-Lab/gsMap/actions/workflows/test_linux.yml
