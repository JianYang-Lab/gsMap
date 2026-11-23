# CS8 human gastrulation

Analyze a 3D reconstruction from 62 aligned human CS8 sections with IQ, Height and MCHC GWAS using the default 3D settings: full XYZ distances and matching to two neighboring sections on each side.

[Xiao et al. · 2024 · Cell](https://doi.org/10.1016/j.cell.2024.03.041) ·
[gsMap Report preview](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/human-gastrulation-cs8/r1/index.html)

## Files needed for this tutorial

Run all commands from `gsmap_tutorials/`. Shared resources are downloaded once
and reused across tutorials.

```text
gsmap_tutorials/
├── resources/1000g-hc-grch38-v50-eur/
│   ├── snp_gene_weights.h5ad
│   └── w_ld/                           # 22 chromosome weight files
└── example_data/
    ├── example-traits/
    │   ├── IQ.sumstats.gz
    │   ├── Height.sumstats.gz
    │   └── MCHC.sumstats.gz
    └── human-gastrulation-cs8/
        ├── ST/                         # 62 H5AD sections
        ├── sample_paths.txt
        └── gwas.yaml
```

## 1. Download

First time using gsMap? Follow {doc}`../install` and download the shared
reference and example GWAS from {doc}`../downloads`.
Then download this dataset. **Skip this step if its folder already exists.**

```bash
mkdir -p example_data
wget -O example_data/human-gastrulation-cs8.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/3d/human-gastrulation-cs8/r1/bundle.tar.gz && \
    tar -xzf example_data/human-gastrulation-cs8.tar.gz -C example_data
```

The archive includes `sample_paths.txt` and `gwas.yaml`; no configuration files
need to be created. Download links and checksum instructions are on
{doc}`../downloads`.

## 2. Run gsMap

The first and last entries of `sample_paths.txt` are shown below:

```text
ST/cs8_section_001.h5ad
ST/cs8_section_002.h5ad
ST/cs8_section_003.h5ad
...
ST/cs8_section_060.h5ad
ST/cs8_section_061.h5ad
ST/cs8_section_062.h5ad
```

```{important}
**Path order matters:** keep the supplied section order along the z axis.
gsMap uses this order for ordinal cross-section matching.
```

Choose **CLI or Python** to run the same analysis:

````{tab} CLI
```bash
mkdir -p results
gsmap quick-mode \
    --workdir results \
    --project-name human-gastrulation-cs8 \
    --dataset-type spatial3D \
    --h5ad-list-file example_data/human-gastrulation-cs8/sample_paths.txt \
    --w-ld-dir resources/1000g-hc-grch38-v50-eur/w_ld \
    --snp-gene-weight-adata-path resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad \
    --sumstats-config-file example_data/human-gastrulation-cs8/gwas.yaml \
    --annotation clusters \
    --spatial-key spatial_3d \
    --data-layer counts
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig, DatasetType
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="results",
    project_name="human-gastrulation-cs8",
    dataset_type=DatasetType.SPATIAL_3D,
    h5ad_list_file="example_data/human-gastrulation-cs8/sample_paths.txt",
    w_ld_dir="resources/1000g-hc-grch38-v50-eur/w_ld",
    snp_gene_weight_adata_path="resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad",
    sumstats_config_file="example_data/human-gastrulation-cs8/gwas.yaml",
    annotation="clusters",
    spatial_key="spatial_3d",
    data_layer="counts",
)
run_quick_mode(config)
```
````

This example uses `counts` for expression, `clusters` for annotation and
`spatial_3d` for coordinates. The default ordinal search uses two sections on
each side, with boundary compensation. Human genes need no homolog conversion.
See {doc}`../advanced_usage` to change these settings.

## 3. View results

[gsMap Report preview](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/human-gastrulation-cs8/r1/index.html)

```bash
gsmap report-view results/human-gastrulation-cs8/gsmap_web_report --port 8080 --no-browser
```

Open the printed HTTP address. Select a trait to explore its spatial association
map and annotation-level enrichment. Outputs and logs are in
`results/human-gastrulation-cs8/`; see {doc}`../output_files` for file details.

### Open in SpatialVista

The report automatically creates a SpatialVista H5AD with all result spots,
annotations and trait −log10(P) values. With
[SpatialVista](https://github.com/JianYang-Lab/spatial-vista-py) installed,
open it directly:

```bash
spatialvista --input results/human-gastrulation-cs8/human-gastrulation-cs8.spatialvista.h5ad
```

Select **Query Traits** to search IQ, Height or MCHC. No coordinate or
annotation flags are needed. For remote use, add `--no-browser --port 8765`
and forward the port to your computer.
