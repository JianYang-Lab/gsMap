# Mouse Embryo 2D (Quick Mode)

Quick mode runs embedding, gene specificity scoring, spatial LDSC, Cauchy
combination and reporting using precomputed LD resources. This example analyzes
the MOSTA E16.5 E1S1 section (121,767 input spots) in `spatial2D` mode; cross-section
matching is disabled. The reference is GENCODE v50 / 1000G high-coverage EUR.

[Chen et al. · 2022 · Cell](https://doi.org/10.1016/j.cell.2022.04.003) ·
[gsMap Report preview](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/mouse-e16-mosta/r1/index.html)

## Files needed for this tutorial

Run from `gsmap_tutorials/`. Shared dependencies come from {doc}`../downloads`;
if you already completed the Visium tutorial, only the E16.5 ST file is new.

```text
gsmap_tutorials/
├── resources/                                    # shared; reuse
│   ├── 1000g-hc-grch38-v50-eur/
│   │   ├── snp_gene_weights.h5ad
│   │   └── w_ld/weights.{1..22}.l2.ldscore.gz     # 22 autosomal files
│   └── homologs/mouse_human_homologs.txt
└── example_data/
    ├── example-traits/IQ.sumstats.gz             # shared GWAS; reuse
    └── mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad  # this tutorial's ST data
```

`{1..22}` denotes the 22 chromosome files.

## 1. Download

First time using gsMap? Follow {doc}`../install` and download the shared
reference, homologs and example GWAS from {doc}`../downloads`.
Then download this dataset. **Skip this step if its folder already exists.**

```bash
mkdir -p example_data
wget -O example_data/mouse-e16-mosta.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/datasets/st/2d/mouse-e16-mosta/r1/bundle.tar.gz && \
    tar -xzf example_data/mouse-e16-mosta.tar.gz -C example_data
```

Download links and checksum instructions are on {doc}`../downloads`.

## 2. Run gsMap

Choose the CLI or Python tab. Both specify the same inputs and keep the default
corrected symmetric KNN graph (10 external neighbors). Counts are in
`layers['count']`, annotations in `obs['annotation']`, and coordinates in
`obsm['spatial']`.

:::{note}
Use `--plot-origin lower` (Python: `plot_origin="lower"`) for this MOSTA
section. It places the origin at the bottom left, with Y increasing upward;
`upper` reverses the Y-axis. This setting controls the orientation of the
report's spatial maps and gene expression/GSS plots, not the analysis results.
:::

````{tab} CLI
```bash
# Create output directory
mkdir -p results

# Run gsMap in quick mode
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
    --sumstats-file "example_data/example-traits/IQ.sumstats.gz" \
    --plot-origin "lower"
```
````

````{tab} Python
```python
from gsMap.config import QuickModeConfig, DatasetType
from gsMap.pipeline import run_quick_mode

config = QuickModeConfig(
    workdir="results",
    project_name="mouse-e16-mosta",
    dataset_type=DatasetType.SPATIAL_2D,
    h5ad_path=["example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad"],
    homolog_file="resources/homologs/mouse_human_homologs.txt",
    species="MOUSE_GENE_SYM",
    w_ld_dir="resources/1000g-hc-grch38-v50-eur/w_ld",
    snp_gene_weight_adata_path="resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad",
    annotation="annotation",
    spatial_key="spatial",
    data_layer="count",
    trait_name="IQ",
    sumstats_file="example_data/example-traits/IQ.sumstats.gz",
    plot_origin="lower",
)

run_quick_mode(config)
```
````

Use `count` for expression, `annotation` for labels and `spatial` for coordinates.
The homolog file maps mouse gene symbols to human genes. See
{doc}`../advanced_usage` for other flags and multiple-trait configurations.

To run these same data on multiple CPU cores or GPUs, share LD scores across traits,
or display every analyzed spot, follow the E16.5 examples in
{doc}`../scalability`. The biological workflow is explained in
{doc}`../key_concepts`.

## 3. View results

[gsMap Report preview](https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reports/mouse-e16-mosta/r1/index.html)

```bash
gsmap report-view results/mouse-e16-mosta/gsmap_web_report --port 8080 --no-browser
```

Open the HTTP address printed by the server. Results and per-stage logs are in
`results/mouse-e16-mosta/`; see {doc}`../output_files`. Select IQ in the report to explore the spatial map and annotation-level enrichment.
