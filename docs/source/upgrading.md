(upgrading)=

# Upgrading to gsMap 2.0

gsMap 2.0 is a major release. It adds 3D spatial transcriptomics support and replaces the
command-line interface, the reference resources and the output layout of gsMap 1.x.
gsMap 1.x analyses are not affected: install `"gsMap<2"` to keep using them.

```{note}
gsMap 2.0 was previewed on PyPI as `gsMap3D` (0.1.x). The code is the same; only the
package name changed. Both packages install the `gsMap` module, so run
`pip uninstall gsMap3D` before installing gsMap 2.x.
```

## Requirements

- **Python 3.12 or later.** On Python 3.10 and 3.11, `pip install gsMap` keeps resolving to
    gsMap 1.x.
- **One compute backend.** PyTorch and JAX must use compatible runtimes; pick exactly one of
    the `cpu`, `cuda126`, `cuda128`, `cuda130` or `tpu` extras ({doc}`install`).

## Commands

Commands use hyphenated names. Run `gsmap COMMAND --help` or see {doc}`cli_reference`.

| gsMap 1.x                               | gsMap 2.x                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------- |
| `gsmap quick_mode`                      | `gsmap quick-mode`                                                        |
| `gsmap run_find_latent_representations` | `gsmap find-latent`                                                       |
| `gsmap run_latent_to_gene`              | `gsmap latent-to-gene`                                                    |
| `gsmap run_generate_ldscore`            | `gsmap ldscore-weight-matrix`, run once per LD reference (see below)      |
| `gsmap run_spatial_ldsc`                | `gsmap spatial-ldsc`                                                      |
| `gsmap run_cauchy_combination`          | `gsmap cauchy-combination`                                                |
| `gsmap run_report`                      | the `report` step of `gsmap quick-mode`; open it with `gsmap report-view` |
| `gsmap format_sumstats`                 | `gsmap format-sumstats`                                                   |
| `gsmap create_slice_mean`               | not needed: pass all sections to one run ({ref}`multiple-sections`)       |

To run part of the pipeline, use `gsmap quick-mode --start-step STEP --stop-step STEP`
with the steps `find_latent`, `latent2gene`, `spatial_ldsc`, `cauchy` and `report`.

## Common quick-mode options

| Purpose             | gsMap 1.x                         | gsMap 2.x                                                           |
| ------------------- | --------------------------------- | ------------------------------------------------------------------- |
| Output directory    | `--workdir`                       | `--workdir`                                                         |
| Run name            | `--sample_name`                   | `--project-name`                                                    |
| ST input            | `--hdf5_path` (one file)          | `--h5ad-path` (repeatable), `--h5ad-list-file` or `--h5ad-yaml`     |
| Data type           | 2D only                           | `--dataset-type spatial2D`, `spatial3D` or `scRNA`                  |
| Reference resources | `--gsMap_resource_dir`            | `--w-ld-dir` and `--snp-gene-weight-adata-path`                     |
| Expression layer    | `--data_layer`                    | `--data-layer`                                                      |
| Cell annotation     | `--annotation`                    | `--annotation`                                                      |
| Homolog mapping     | `--homolog_file`                  | `--homolog-file`, with `--species` naming the species-symbol column |
| One trait           | `--trait_name`, `--sumstats_file` | `--trait-name`, `--sumstats-file`                                   |
| Several traits      | `--sumstats_config_file`          | `--sumstats-config-file`                                            |

Embedding and neighbor-selection options were redesigned for the dual-embedding model. Start
from the defaults and see {doc}`advanced_usage` before tuning them.

## Reference resources

gsMap 1.x computed LD scores for every sample with `run_generate_ldscore`. gsMap 2.x instead
reads a precomputed SNP-by-gene weight matrix (`--snp-gene-weight-adata-path`) and matching
LD regression weights (`--w-ld-dir`), and combines them with gene specificity scores during
`spatial-ldsc`.

- Download the GRCh38 1000 Genomes high-coverage references (several ancestries) from
    {doc}`downloads`.
- To use another LD panel or gene annotation, build the matrix once with
    `gsmap ldscore-weight-matrix` ({doc}`ldscore_weight_matrix`).

(multiple-sections)=

## Multiple sections

gsMap 1.x analyzed sections one at a time and could share gene ranks across sections with
`create_slice_mean`. gsMap 2.x analyzes all sections in one run:

- **Separate 2D sections:** `--dataset-type spatial2D` with one `--h5ad-path` per section.
- **Registered 3D sections:** `--dataset-type spatial3D` with XYZ coordinates and the H5AD
    files listed in anatomical order along Z. Cross-section matching of homogeneous cells is
    controlled by `--n-adjacent-slices`.
- **One thick-tissue volume:** `--dataset-type spatial3D` with a single XYZ H5AD file.

See {doc}`key_concepts` for the coordinate requirements.

## Python API

```python
from gsMap.config import DatasetType, QuickModeConfig
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="results",
    project_name="my-project",
    dataset_type=DatasetType.SPATIAL_2D,
    h5ad_path=["data/section.h5ad"],
    w_ld_dir="resources/1000g-hc-grch38-v50-eur/w_ld",
    snp_gene_weight_adata_path="resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad",
    annotation="cell_type",
    trait_name="IQ",
    sumstats_file="gwas/IQ.sumstats.gz",
)
run_quick_mode(config)
```

## Outputs

Results are written to `{workdir}/{project_name}/`, with one folder per step and an
interactive report in `gsmap_web_report/`. Start 2.x analyses in a new working directory.
See {doc}`output_files` for the layout and file formats.
