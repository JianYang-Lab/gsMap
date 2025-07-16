# gsMap

|               |                                                                                                      |                |                                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------------------- | -------------- | -------------------------------------------------------------------------------------------------- |
| __Version__   | [![PyPI version][pypi-badge]][pypi-url] [![Python][python-badge]][python-url]                        | __Status__     | [![Project Status][status-badge]][status-url] [![Maintenance][maintenance-badge]][maintenance-url] |
| __Activity__  | [![GitHub commits][commits-badge]][commits-url] [![Last Commit][last-commit-badge]][last-commit-url] | __Quality__    | [![codecov][codecov-badge]][codecov-url] [![Ruff][ruff-badge]][ruff-url]                           |
| __CI/CD__     | [![Docs][docs-badge]][docs-url] [![test][test-badge]][test-url]                                      | __Community__  | [![GitHub stars][stars-badge]][stars-url] [![GitHub forks][forks-badge]][forks-url]                |
| __Downloads__ | [![Downloads][downloads-badge]][downloads-url]                                                       | __License__    | [![License: MIT][license-badge]][license-url] [![DOI][doi-badge]][doi-url]                         |
| __Platform__  | [![Linux][linux-badge]][linux-url]                                                                   | __Contribute__ | [![Issues][issues-badge]][issues-url] [![PRs Welcome][pr-badge]][pr-url]                           |

## Introduction

`gsMap` (genetically informed spatial mapping of cells for complex traits)
integrates spatial transcriptomics (ST) data with genome-wide association study (GWAS)
summary statistics to map cells to human complex traits, including diseases,
in a spatially resolved manner.

## Key Features

- __Spatially-aware High-Resolution Trait Mapping__
- __Spatial Region Identification__
- __Putative Causal Genes Identification__

![Model Architecture](schematic.png)

## 🛠️ Installation

Install using pip:

```bash
conda create -n gsMap python>=3.10
conda activate gsMap
pip install gsMap
```

Install using conda:

```bash
conda create -n gsMap python>=3.10
conda activate gsMap
conda install bioconda::gsmap
```

Install from source:

```bash
git clone https://github.com/JianYang-Lab/gsMap
cd gsMap
pip install -e .
```

Verify the installation by running the following command:

```bash
gsmap --help
```

## 📘 Usage

Please check out the documentation and tutorials at [gsMap Documentation](https://yanglab.westlake.edu.cn/gsmap/document/software).

## 🌐 Online Visualization

To visualize the traits-cell association spatial maps,
please refer to [gsMap Visualization](https://yanglab.westlake.edu.cn/gsmap/visualize).

## 📖 Citation

Song, L., Chen, W., Hou, J., Guo, M. & Yang, J.
[Spatially resolved mapping of cells associated with human complex traits.](https://doi.org/10.1038/s41586-025-08757-x)
Nature (2025).

Please cite the paper and give us a STAR if you find gsMap useful for your research.

## ✨ Research Highlight
gsMap was highlighted in [Nature Methods](https://www.nature.com/articles/s41592-025-02711-5).  
gsMap was highlighted in [Nature Review Genetics](https://www.nature.com/articles/s41576-025-00877-4).

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
