(key_concepts)=

# Key Concepts

This guide introduces the basic workflow, input formats and key settings.
For a complete walkthrough, follow the {doc}`2d_tutorial/mouse_embryo` or
{doc}`3d_tutorial/index`. Use {doc}`advanced_usage` for customization and
{doc}`scalability` for hardware, I/O and performance tuning.

## Basic usage

gsMap can be run from the command line or Python. Choose one interface and
replace the **placeholder paths, project name and trait name** below with your own.

This example assumes human gene symbols, counts in `adata.X`, coordinates in
`obsm["spatial"]` and annotations in `obs["cell_type"]`. Change these fields to
match your data; non-human data also need a homolog file.

````{tab} CLI
```bash
mkdir -p ./output

gsmap quick-mode \
    --workdir "./output" \
    --project-name "my_project" \
    --dataset-type spatial2D \
    --h5ad-path "sample.h5ad" \
    --annotation "cell_type" \
    --spatial-key "spatial" \
    --data-layer "X" \
    --trait-name "Trait1" \
    --sumstats-file "trait1.sumstats.gz" \
    --snp-gene-weight-adata-path "./resources/snp_gene_weights.h5ad" \
    --w-ld-dir "./resources/w_ld"
```
````

````{tab} Python
```python
from pathlib import Path

from gsMap.config import QuickModeConfig
from gsMap.pipeline import run_quick_mode

Path("./output").mkdir(parents=True, exist_ok=True)

config = QuickModeConfig(
    workdir="./output",
    project_name="my_project",
    dataset_type="spatial2D",
    h5ad_path=["sample.h5ad"],
    annotation="cell_type",
    spatial_key="spatial",
    data_layer="X",
    trait_name="Trait1",
    sumstats_file="trait1.sumstats.gz",
    snp_gene_weight_adata_path="./resources/snp_gene_weights.h5ad",
    w_ld_dir="./resources/w_ld",
)
run_quick_mode(config)
```
````

Results are saved in `output/my_project/`. For a ready-to-run example with
downloadable data, follow the {doc}`2d_tutorial/mouse_embryo`.

The sections below explain the arguments. Short flag excerpts illustrate
settings to add to a complete command; they are not standalone commands.

(pipeline-overview)=

## Pipeline overview

Start with **Simplified** for the main stages. Switch to **Detailed** to see
the inputs to each calculation; arrow labels describe what each connection
contributes. Use **Expand** for a larger view.

````{tab} Simplified
```{mermaid}
:name: pipeline-simple

flowchart TB
    ST("Spatial transcriptomics")
    GWAS("GWAS + LD resources")

    subgraph PIPELINE["Integrated pipeline"]
        direction TB
        S1("1 · Latent representation<br/>Cell + niche embeddings")
        S2("2 · Homogeneous neighbors<br/>Similar cells in spatial context")
        S3("3 · Gene specificity score<br/>GSS calculation")
        S4("4 · Spatial LDSC<br/>Cell–trait associations")
        S5("5 · Cauchy combination<br/>Region / cell-type associations")
        S1 --> S2 --> S3 --> S4 --> S5
    end

    Q("Interactive report")
    ST --> S1
    GWAS --> S4
    S5 --> Q

    classDef input fill:#eaf2fa,stroke:#809ab7,color:#304962;
    classDef process fill:#f0edfa,stroke:#9284b5,color:#453a61;
    classDef output fill:#eaf5ef,stroke:#79a58e,color:#2f5140;
    class ST,GWAS input;
    class S1,S2,S3,S4,S5 process;
    class Q output;
    style PIPELINE fill:#faf9fd,stroke:#d9d3e7,color:#453a61
```
````

````{tab} Detailed
Scroll horizontally to see the full diagram, or choose **Expand**.

```{mermaid}
:name: pipeline-detailed

flowchart TB
    E["Gene expression"]
    S["Spatial coordinates<br/>+ section information"]
    A["Annotation · optional"]

    subgraph NEIGHBORS["1 · Choose comparable cells"]
        C["Cell embedding"]
        N["Niche embedding · optional"]
        P["Spatial constraints<br/>Allowed cells / sections"]
        H["Homogeneous neighbors<br/>+ similarity weights"]
        C -->|"Cell similarity"| H
        N -.->|"Niche similarity"| H
        P -->|"Eligible candidates"| H
    end
    E -->|"Expression features"| C
    E -.->|"Propagated expression"| N
    S -.->|"Spatial graph"| N
    S -->|"Distances / section order"| P
    A -.->|"Same-group restriction"| H

    subgraph SCORES["2 · Calculate gene specificity"]
        R["Expression ranks<br/>Within each cell"]
        F["Global reference<br/>Geometric-mean rank<br/>+ expression fraction"]
        M["Marker scores · GSS<br/>Gene specificity"]
        R -->|"Neighbor ranks"| M
        F -->|"Normalize / filter"| M
    end
    E -->|"Rank genes per cell"| R
    E -->|"Summarize across cells"| F
    H -->|"Neighbor IDs + weights"| M
    %% Invisible links keep the three stages vertically ordered.
    A ~~~ C
    H ~~~ R
    H ~~~ F

    subgraph ASSOCIATION["3 · Test trait associations"]
        D["Cell-specific LD scores"]
        L["Spatial LDSC"]
        O["Cell–trait associations"]
        D -->|"Cell-specific annotations"| L
        L -->|"Regression test"| O
    end
    M -->|"Gene scores"| D
    W["SNP × gene<br/>weight matrix"] -->|"LD-weighted mapping"| D
    G["GWAS + w_ld<br/>Baseline covariates"] -->|"Trait statistics + reference"| L
    O -->|"Cell-level p-values"| Q["Cauchy enrichment<br/>+ interactive report"]

    classDef compute fill:#edf4fc,stroke:#517ba5,color:#19324d;
    classDef data fill:#eef2f6,stroke:#8798ad,color:#33465e;
    classDef result fill:#edf7ef,stroke:#71876b,color:#253621;
    class C,N,P,R,F,D,L compute;
    class E,S,A,W,G data;
    class H,M,O,Q result;
```

Dashed arrows indicate optional inputs or paths. The global geometric mean is
calculated from **expression ranks**, not raw expression values. Expression
fraction means the proportion of cells expressing a gene.
````

1. **Embeddings:** cell embeddings describe expression similarity; optional
    niche embeddings add spatial context.
2. **Homogeneous neighbors:** choose similar cells subject to spatial and
    optional annotation constraints. Spatial2D stays within each section;
    Spatial3D can also match cells in allowed neighboring sections.
3. **Gene specificity:** aggregate neighbors' expression ranks using their
    similarity weights and compare with the global reference to calculate GSS.
4. **Spatial LDSC:** combine GSS, SNP × gene weights and GWAS statistics to
    test cell–trait associations.
5. **Group summaries and report:** combine cell-level p-values by annotation
    using the Cauchy test, then explore the spatial maps and enrichment results.

```{note}
Graph propagation and cross-section matching are separate. Allowing matches
across sections does not itself create cross-section graph edges.
```

## Key Configurations

The following options are used in `gsmap quick-mode`. In Python, replace hyphens
with underscores: for example, `--project-name` becomes `project_name=...`.

### Project Setup

- **`--workdir`**:
    The root directory where all output files will be saved.
- **`--project-name`**:
    Name of the project. A subdirectory with this name will be created inside `workdir`.

### Spatial Transcriptomics Input

The input ST data must be an h5ad file containing at least the gene expression matrix and spatial coordinates.

- **Counts**: Raw UMI counts are preferred (typically in `adata.X` or `adata.layers['counts']`).
- **Coordinates**: Spatial coordinates stored in `adata.obsm['spatial']`.
- **Gene Names**: Must match the species of your data. Use `--homolog-file` to map non-human genes to human orthologs if necessary.
- **Annotations**: Optionally, the h5ad file may include spot (cell) annotations in the `obs` attribute.

```python
import scanpy as sc

adata = sc.read_h5ad("example_data/mouse-e16-mosta/ST/E16.5_E1S1.MOSTA.h5ad")

print(adata.layers["count"].shape)
print(adata.obsm["spatial"].shape)
print(adata.obs["annotation"].value_counts().head())
```

- **`--dataset-type`**:
    Specifies the data modality.
    - `spatial2D`: Standard 2D spatial transcriptomics (e.g., Visium, Stereo-seq).
    - `spatial3D`: 3D spatial transcriptomics (aligned sections or thick tissue).

#### H5ad Path

There are three ways to provide input AnnData (.h5ad) files:

1. **Single or Multiple Files** (`--h5ad-path`):
    Specify each file with its own flag. Sample names are derived from filenames (e.g., `sample1.h5ad` -> `sample1`).

    ```text
    --h5ad-path data/sample1.h5ad \
    --h5ad-path data/sample2.h5ad
    ```

2. **YAML Configuration** (`--h5ad-yaml`):
    A YAML file where keys are sample names and values are file paths.

    ```yaml
    sample1: /path/to/sample1.h5ad
    sample2: /path/to/sample2.h5ad
    ```

3. **List File** (`--h5ad-list-file`):
    A text file where each line is a file path. Sample names are derived from filenames.

    ```text
    /path/to/sample1.h5ad
    /path/to/sample2.h5ad
    ```

For YAML and list files, relative paths are resolved from the directory containing
the configuration file, not the shell working directory. Use one input form.

```{important}
For multiple `spatial3D` inputs, list H5AD files in anatomical order along Z (ascending or descending). Each file identifies one section; distinct coordinate values do not create new sections.
```

#### Data Layers & Coordinates

- **`--data-layer`**: The key in `adata.layers` containing raw counts (e.g., 'counts'). If not provided, defaults to `adata.X`.
- **`--spatial-key`**: The key in `adata.obsm` containing spatial coordinates (default: 'spatial').

```{important}
Use columns **[X, Y, Z]**, with Z in the third column, for `spatial3D`.
Multiple sections must share an **aligned coordinate system**, with XY in the
section plane and Z along the stack. Within each section, the sampled Z span
must be smaller than both X and Y spans; otherwise configuration fails.
A single XYZ file is a volume and is exempt from this section-thickness check.
In `spatial2D`, XYZ input emits a warning and only X and Y are used.
```

#### Annotation

- **`--annotation`**: Key in `adata.obs` containing cell type or spatial domain annotations. Restricts the homogeneous-neighbor search to the same annotation group.
- **`--two-stage`**: If provided, this annotation is used to refine the latent embedding in the find latent representation stage.

```{note}
If both cell type and spatial domain are available, **cell type** is usually
appropriate for restricting homogeneous-neighbor matching. Quick mode also
needs annotation groups for enrichment/reporting; use `--annotation` or
`--cauchy-annotations`. The E16.5 example supplies `--annotation annotation`.
```

### GWAS Summary Statistics

The input GWAS data is a text file containing at least the columns for SNP (rs number), Z (Z-statistics), and N (sample size). Use these exact column names. Alleles `A1` (effect allele) and `A2` (other allele)
are retained by the formatter for checking effect orientation.

An illustrative formatted file looks like this (not real association results):

```text
SNP          A1  A2  Z       N
rs12184267   T   C   0.916   225955
rs12184277   G   A   0.656   226215
```

- **SNP:** variant rsID matching the LD reference.
- **Z:** signed association Z statistic, not a p-value.
- **N:** positive sample size for that variant.

Inspect a formatted example without decompressing the entire file:

```python
import pandas as pd

print(pd.read_csv("example_data/example-traits/IQ.sumstats.gz", sep=r"\s+", nrows=5))
```

#### Formatting your own GWAS

Provide rsIDs, effect/non-effect alleles, signed Z statistics (or beta and SE),
and sample size. Confirm the ancestry and genome build from the original study.
Column names and QC thresholds are study-specific; automatic detection is not
substitute for checking effect direction, sample size and variant identifiers.

The downloadable example GWAS are already formatted and do not need reformatting.
For example, suppose your raw file `my_trait.tsv` has columns `rsid`,
`effect_allele`, `other_allele`, `beta`, `se` and `n_samples`. Map them explicitly:

```bash
gsmap format-sumstats \
    --sumstats my_trait.tsv \
    --out my_trait \
    --snp rsid \
    --a1 effect_allele \
    --a2 other_allele \
    --beta beta \
    --se se \
    --n n_samples
```

This writes `my_trait.sumstats.gz`, with signed `Z = beta / se`.
If the study supplies Z directly, use `--z COLUMN` instead of `--beta` and `--se`.
For other fields and QC options:

```bash
gsmap format-sumstats --help
```

#### Providing GWAS Data

There are two ways to provide GWAS summary statistics:

1. **Single Trait** (`--trait-name` + `--sumstats-file`):
    Provide a single trait name and its corresponding summary statistics file.

    ```text
    --trait-name "IQ" --sumstats-file "example_data/example-traits/IQ.sumstats.gz"
    ```

2. **Multiple Traits** (`--sumstats-config-file`):
    Provide a YAML configuration file mapping trait names to file paths.

    ```yaml
    # Save as inputs/gwas_config.yaml; paths are relative to this YAML file.
    IQ: ../example_data/example-traits/IQ.sumstats.gz
    Height: ../example_data/example-traits/Height.sumstats.gz
    MCHC: ../example_data/example-traits/MCHC.sumstats.gz
    ```

    ```text
    --sumstats-config-file "inputs/gwas_config.yaml"
    ```

### SNP to Gene & LD Weights

These files define how SNPs are linked to genes and the LD structure used for heritability partitioning. You can use the pre-calculated weights provided by gsMap, or build your own using the `gsmap ldscore-weight-matrix` command with a customized SNP-to-gene BED file and PLINK reference panel. See [Computing Custom LD Score Weight Matrix](ldscore_weight_matrix.md) for detailed instructions.

- **`--snp-gene-weight-adata-path`**:
    Path to the pre-calculated SNP-to-gene weight matrix (.h5ad format).

- **`--w-ld-dir`**:
    Directory containing the per-chromosome regression-weight LD scores.

Download the matrix and `w_ld/` as a **matched pair** from {doc}`downloads`.
Choose ancestry according to the **GWAS population**, not the tissue species.
The SNP × gene matrix converts GSS into LD scores; `w_ld` weights the regression.

### Homogeneous Neighbor & GSS

- **`--spatial-neighbors`**:
    Number of within-section spatial candidates for homogeneous-neighbor selection (default: 301).
- **`--homogeneous-neighbors`**:
    Number of similar neighbors selected for GSS (default: 21); default 3D pooling selects neighbors per section.

#### Graph neighbors versus homogeneous neighbors

These are different settings: `--n-neighbors` requests **10 external neighbors**
per spot for the default KNN graph, before bidirectional union. It controls
embedding propagation, not the number of homogeneous neighbors used for GSS.
Batch and chunk sizes control execution, not the biological neighborhood.

#### 3D Specific Options

For multiple `spatial3D` sections, gsMap can search for homogeneous neighbors across aligned adjacent slices. Set `n_adjacent_slices=0` to disable this matching while retaining full XYZ distances. Single-file thick tissue always uses within-volume neighbors. Cross-section matching and the embedding graph are independent.

- **`--n-adjacent-slices`**:
    Default: `2` sections above and below. Ordinal selection follows input order
    and can fill its quota from farther sections near boundaries. Two-section
    inputs must explicitly set `1`. The {doc}`3d_tutorial/mouse_e11` uses `8`.
    Ordinal section counts are not physical distances; see {doc}`cli_reference`
    for calibrated physical-distance options.
- **`--cross-slice-marker-score-strategy`**:
    Strategy for calculating Gene Specificity Scores across slices:
    - `hierarchical_pool` (Default): Select *K* homogeneous neighbors from each adjacent slice independently, compute a per-slice GSS, then average across slices.
    - `global_pool`: Select the top K × (2 × n_adjacent_slices + 1) homogeneous neighbors globally across all adjacent slices. Each slice may contribute a variable number of neighbors.
    - `per_slice_pool`: Select *K* homogeneous neighbors from each adjacent slice, then compute a single GSS from the pooled neighbors.

### Advanced Options

- **`--latent-representation-cell`**:
    Key in `adata.obsm` for cell identity embedding (default: `emb_cell`).
- **`--latent-representation-niche`**:
    Key in `adata.obsm` for the optional spatial niche embedding. Spatial quick mode
    uses `emb_niche` by default; the standalone latent-to-gene step defaults to none.
- **`--use-gpu` / `--no-gpu`**:
    Enable or disable JAX GPU acceleration for marker scoring and spatial LDSC.
    For a fully CPU-only run, including embeddings, see {doc}`scalability`.

Changing graph or matching settings can change the scientific result; use a new
project for such changes. GSS can be reused for new GWAS traits when the inputs
and scientific settings stay the same.

## See Also

- {doc}`Output Files Reference <output_files>`: Detailed documentation of all output files and their formats
- {doc}`2D Tutorial <2d_tutorial/index>`: Step-by-step guide for 2D spatial transcriptomics
- {doc}`3D Tutorial <3d_tutorial/index>`: Step-by-step guide for 3D spatial transcriptomics
- {doc}`Performance and scalability <scalability>`: CPU/GPU, SSD and batch-size examples
- {doc}`Customization Guide <advanced_usage>`: Adapt inputs and analysis settings
- {doc}`CLI Reference <cli_reference>`: Complete command-line reference
