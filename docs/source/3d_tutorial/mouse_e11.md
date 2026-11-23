# Mouse embryo E11.5

Analyze 84 sections (6,993,667 cells) with ordinal cross-section matching using
IQ, Kazuo atrial fibrillation (AF) and LDL GWAS.

[Qiu et al. · 2024 · Cell](https://doi.org/10.1016/j.cell.2024.10.011) ·
[gsMap Report preview](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/e11-ordinal/r1/index.html)

The preview shows IQ, AF and LDL results from this tutorial: 6,609,635 cells
across all 84 sections after quality control.

## Files needed for this tutorial

Run from `gsmap_tutorials/`. Reuse the shared reference and GWAS files from
other tutorials.

```text
gsmap_tutorials/
├── resources/
│   ├── 1000g-hc-grch38-v50-eur/
│   │   ├── snp_gene_weights.h5ad
│   │   └── w_ld/
│   └── homologs/mouse_human_homologs.txt
└── example_data/
    ├── example-traits/
    │   ├── IQ.sumstats.gz
    │   ├── AF.sumstats.gz
    │   └── LDL.sumstats.gz
    └── mouse-e11/
        ├── ST/                         # 84 serial sections
        ├── sample_paths.txt
        └── gwas.yaml
```

## 1. Download

Download the shared reference, homologs and GWAS once from {doc}`../downloads`.
Then download this dataset. **Skip this step if its folder already exists.**

```bash
mkdir -p example_data
wget -O example_data/mouse-e11.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/3d/mouse-e11/r1/bundle.tar.gz && \
    tar -xzf example_data/mouse-e11.tar.gz -C example_data
```

The archive includes all 84 serial sections, `sample_paths.txt` and `gwas.yaml`;
no preparation or configuration-file editing is needed.

Raw counts are stored once in `adata.X`; use `--data-layer X` as shown below.

```{tip}
E11.5 is a large dataset; start with
{doc}`human_gastrulation` for a smaller first run.
```

## 2. Run gsMap

The first and last entries of `sample_paths.txt` are:

```text
ST/Mouse_E11.5_embryo_S00.h5ad
ST/Mouse_E11.5_embryo_S01.h5ad
ST/Mouse_E11.5_embryo_S02.h5ad
...
ST/Mouse_E11.5_embryo_S81.h5ad
ST/Mouse_E11.5_embryo_S82.h5ad
ST/Mouse_E11.5_embryo_S83.h5ad
```

```{important}
**Path order matters:** preserve anatomical slice order along the z axis.
gsMap uses this list order for matching; it does not sort sections by coordinates.
`n_adjacent_slices=8` searches eight sections on each side, with boundary
compensation near the first and last sections. This is a section-count setting,
not a physical-distance cutoff.
```

Choose **CLI or Python**. One quick-mode call runs embedding, GSS, spatial LDSC
and report generation:

````{tab} CLI
```bash
mkdir -p results
gsmap quick-mode \
    --workdir results \
    --project-name e11-ordinal \
    --dataset-type spatial3D \
    --h5ad-list-file example_data/mouse-e11/sample_paths.txt \
    --data-layer X \
    --spatial-key z_correction \
    --annotation spatial_domain \
    --cauchy-annotations ManualAnnotation \
    --homolog-file resources/homologs/mouse_human_homologs.txt \
    --species MOUSE_GENE_SYM \
    --n-adjacent-slices 8 \
    --w-ld-dir resources/1000g-hc-grch38-v50-eur/w_ld \
    --snp-gene-weight-adata-path resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad \
    --sumstats-config-file example_data/mouse-e11/gwas.yaml
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="results",
    project_name="e11-ordinal",
    dataset_type="spatial3D",
    h5ad_list_file="example_data/mouse-e11/sample_paths.txt",
    data_layer="X",
    spatial_key="z_correction",
    annotation="spatial_domain",
    cauchy_annotations=["ManualAnnotation"],
    homolog_file="resources/homologs/mouse_human_homologs.txt",
    species="MOUSE_GENE_SYM",
    n_adjacent_slices=8,
    w_ld_dir="resources/1000g-hc-grch38-v50-eur/w_ld",
    snp_gene_weight_adata_path="resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad",
    sumstats_config_file="example_data/mouse-e11/gwas.yaml",
)
run_quick_mode(config)
```
````

This example uses `spatial_domain` for both embedding QC and matching.
Cell-type labels (`ManualAnnotation`) are also included in the enrichment summaries
and report, without changing the embeddings or matching.
Other
settings use quick-mode defaults, including the bidirectional KNN graph (10 external neighbors)
and 200 spatial candidates per adjacent section. Graph propagation stays within
sections; matching connects information across sections.

## 3. View results

[gsMap Report preview](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/e11-ordinal/r1/index.html)

```bash
gsmap report-view results/e11-ordinal/gsmap_web_report --port 8080 --no-browser
```

Open the printed HTTP address to explore IQ, AF and LDL association maps and
annotation-level enrichment. Outputs and logs are in `results/e11-ordinal/`;
see {doc}`../output_files` for file details.

### Open in SpatialVista

The report also creates `results/e11-ordinal/e11-ordinal.spatialvista.h5ad`
automatically. It contains 3D coordinates, section IDs, both annotations and
IQ, AF and LDL −log10(P) values; no separate export command is needed.

With [SpatialVista](https://github.com/JianYang-Lab/spatial-vista-py) installed:

```bash
spatialvista --input results/e11-ordinal/e11-ordinal.spatialvista.h5ad
```

Select **Query Traits** to search IQ, AF or LDL. Coordinates, annotations and
sections are detected automatically. For remote use, add
`--no-browser --port 8765` and forward the port to your computer.

```{tip}
The H5AD includes all result cells by default, independently of HTML sampling.
For a smaller E11.5 preview, add `--spatialvista-max-spots 20000` to the
quick-mode command, or `spatialvista_max_spots=20000` to `QuickModeConfig`.
This samples only the viewer export, not the analysis. To disable the H5AD
output, use `--no-export-spatialvista` or `export_spatialvista=False`.
```
