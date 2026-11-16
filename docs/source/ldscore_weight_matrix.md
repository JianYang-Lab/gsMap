# Compute custom LD-weight matrices

Build a SNP × gene LD-weight matrix from gene coordinates or enhancer–gene links.
For routine analyses, use a precomputed reference from {doc}`downloads`.

Both examples use chromosome 22 for a short first run. Use
`--chromosomes all` (Python: `chromosomes="all"`) for a whole-genome reference.

## Files needed for this tutorial

Run from `gsmap_tutorials/`, as in the other tutorials. This tutorial uses a
PLINK panel rather than a precomputed SNP × gene matrix.

```text
gsmap_tutorials/
└── resources/
    └── ldscore-tutorial/
        ├── LD_Reference_Panel/1000G_EUR_Phase3_plink/
        │   ├── 1000G.EUR.QC.1.bed
        │   ├── 1000G.EUR.QC.1.bim
        │   ├── 1000G.EUR.QC.1.fam
        │   └── ...                     # chromosomes 1–22
        ├── hapmap3_snps/
        │   ├── hm.1.snp
        │   └── ...                     # hm.1.snp through hm.22.snp
        └── genome_annotation/
            ├── gencode/gencode_v46lift37_protein_coding.bed
            └── enhancer/by_tissue/
                ├── BRN/ABC_roadmap_merged.bed
                └── ...                 # other tissues listed below
```

The bundle contains the **GRCh37 1000 Genomes Phase 3 low-coverage EUR panel**
(489 individuals), HM3 SNP lists, GENCODE v46 gene intervals and tissue-specific
enhancer–gene links. All example coordinates are GRCh37.

## 1. Download

After {doc}`install`, download the inputs once for both examples.
**Skip this step if the extracted folder already exists.**

```bash
mkdir -p resources
wget -O resources/ldscore-tutorial.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_panels/ldscore-tutorial/r1/bundle.tar.gz && \
    tar -xzf resources/ldscore-tutorial.tar.gz -C resources
```

Archive checksums are on {doc}`downloads`.

## 2. Compute the matrices

Choose the CLI or Python tab for each example. Both interfaces use the same
settings and outputs. `--calculate-w-ld` also computes the matching HM3
regression weights.

### Example 1: TSS-based mapping with GENCODE annotation

Extend each gene interval by 50 kb on either side. If a SNP overlaps more than
one extended interval, assign it to the gene with the closest transcription
start site (TSS). The extension applies to the **gene interval**, not just its TSS.

````{tab} CLI
```bash
gsmap ldscore-weight-matrix \
    --bfile-root "resources/ldscore-tutorial/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chr}" \
    --hm3-snp-dir resources/ldscore-tutorial/hapmap3_snps \
    --mapping-file resources/ldscore-tutorial/genome_annotation/gencode/gencode_v46lift37_protein_coding.bed \
    --mapping-type bed \
    --strategy tss \
    --feature-window-size 50000 \
    --ld-unit CM --ld-wind 1 --mac-min 5 \
    --mhc-start-bp 25000000 --mhc-end-bp 34000000 \
    --chromosomes 22 \
    --output-dir results/gencode-tss \
    --output-filename gencode_v46_tss_50kb_weights \
    --calculate-w-ld \
    --w-ld-dir results/gencode-tss/w_ld
```
````

````{tab} Python
```python
from pathlib import Path
from gsMap.config import LDScoreConfig
from gsMap.ldscore.pipeline import LDScorePipeline

config = LDScoreConfig(
    bfile_root="resources/ldscore-tutorial/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chr}",
    hm3_snp_dir=Path("resources/ldscore-tutorial/hapmap3_snps"),
    mapping_file=Path(
        "resources/ldscore-tutorial/genome_annotation/gencode/gencode_v46lift37_protein_coding.bed"
    ),
    mapping_type="bed",
    strategy="tss",
    feature_window_size=50000,
    ld_unit="CM",
    ld_wind=1,
    mac_min=5,
    mhc_start_bp=25000000,
    mhc_end_bp=34000000,
    chromosomes="22",
    output_dir=Path("results/gencode-tss"),
    output_filename="gencode_v46_tss_50kb_weights",
    calculate_w_ld=True,
    w_ld_dir=Path("results/gencode-tss/w_ld"),
)
LDScorePipeline(config).run()
```
````

### Example 2: Enhancer-based mapping with center strategy

Use the brain (`BRN`) enhancer–gene links from the combined ABC and Roadmap
annotations. A SNP must overlap an enhancer interval. If it overlaps several,
`center` selects the interval whose center is closest to that SNP and assigns
the SNP to the linked gene. It does not search for enhancers outside the intervals.

The fourth BED column contains **gene symbols**, so the output remains SNP ×
gene. `feature_window_size=0` leaves intervals unchanged; `center` uses binary
assignments, not ABC scores.

````{tab} CLI
```bash
gsmap ldscore-weight-matrix \
    --bfile-root "resources/ldscore-tutorial/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chr}" \
    --hm3-snp-dir resources/ldscore-tutorial/hapmap3_snps \
    --mapping-file resources/ldscore-tutorial/genome_annotation/enhancer/by_tissue/BRN/ABC_roadmap_merged.bed \
    --mapping-type bed \
    --strategy center \
    --feature-window-size 0 \
    --ld-unit CM --ld-wind 1 --mac-min 5 \
    --mhc-start-bp 25000000 --mhc-end-bp 34000000 \
    --chromosomes 22 \
    --output-dir results/enhancer-brain \
    --output-filename brain_enhancer_abc_weights \
    --calculate-w-ld \
    --w-ld-dir results/enhancer-brain/w_ld
```
````

````{tab} Python
```python
from pathlib import Path
from gsMap.config import LDScoreConfig
from gsMap.ldscore.pipeline import LDScorePipeline

config = LDScoreConfig(
    bfile_root="resources/ldscore-tutorial/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC.{chr}",
    hm3_snp_dir=Path("resources/ldscore-tutorial/hapmap3_snps"),
    mapping_file=Path(
        "resources/ldscore-tutorial/genome_annotation/enhancer/by_tissue/BRN/ABC_roadmap_merged.bed"
    ),
    mapping_type="bed",
    strategy="center",
    feature_window_size=0,
    ld_unit="CM",
    ld_wind=1,
    mac_min=5,
    mhc_start_bp=25000000,
    mhc_end_bp=34000000,
    chromosomes="22",
    output_dir=Path("results/enhancer-brain"),
    output_filename="brain_enhancer_abc_weights",
    calculate_w_ld=True,
    w_ld_dir=Path("results/enhancer-brain/w_ld"),
)
LDScorePipeline(config).run()
```
````

#### Other tissues

Replace `BRN` in the mapping-file path to use another tissue:

| Code   | Tissue           | Code   | Tissue   |
| ------ | ---------------- | ------ | -------- |
| `ALL`  | All tissues      | `BLD`  | Blood    |
| `BRN`  | Brain            | `FAT`  | Adipose  |
| `GI`   | Gastrointestinal | `HRT`  | Heart    |
| `KID`  | Kidney           | `LIV`  | Liver    |
| `LNG`  | Lung             | `PANC` | Pancreas |
| `SKIN` | Skin             |        |          |

Each folder provides `roadmap.bed` and `ABC_roadmap_merged.bed`; `ABC.bed` is
also included where available. The kidney folder has no separate ABC annotation.

## 3. View and use the outputs

```text
results/
├── gencode-tss/
│   ├── gencode_v46_tss_50kb_weights.h5ad
│   └── w_ld/weights.22.l2.ldscore.gz
└── enhancer-brain/
    ├── brain_enhancer_abc_weights.h5ad
    └── w_ld/weights.22.l2.ldscore.gz
```

Each H5AD contains a sparse matrix in `X`, HM3 SNP rows in `obs`, and gene columns
in `var`, including an `Unmapped` column. The `w_ld/` files are separate
regression weights, **not** row sums of the SNP × gene matrix.

For a whole-genome run, change `--chromosomes 22` to `--chromosomes all` and use a
new output directory. In a [quick-mode command](2d_tutorial/mouse_embryo.md), set
`--snp-gene-weight-adata-path` to the resulting H5AD and `--w-ld-dir` to its
accompanying `w_ld/` directory.

```{important}
The examples produce chromosome-22-only references for a short test.
Do not use a chromosome-22-only matrix for whole-genome spatial LDSC.
```

## Customize the mapping

| Flag                      | Use                                                                                                                         |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `--mapping-file`          | BED file; column 4 supplies output gene/feature names. Use gene symbols compatible with your GSS matrix for spatial LDSC.   |
| `--strategy tss`          | Resolve overlapping intervals by the closest TSS; requires BED strand information.                                          |
| `--strategy center`       | Resolve overlapping intervals by the closest interval center.                                                               |
| `--strategy score`        | Select the highest-scoring overlap and use its score; requires numeric BED column 5.                                        |
| `--strategy allow_repeat` | Retain all overlaps; uses BED scores when present, otherwise 1. Multiple intervals for the same SNP–gene pair add together. |
| `--feature-window-size`   | Extend annotation intervals by this many base pairs on each side. This is separate from the LD window.                      |
| `--batch-size-hm3`        | Number of target SNPs processed together; reduce it if memory is limited. It does not change the intended LD window.        |

See {doc}`cli_reference` for all CLI and Python parameters.

### Reference settings

```{warning}
Use a reference matched to the GWAS ancestry and keep the PLINK positions,
annotation BED and MHC interval in the **same genome build**. The examples use
GRCh37, so they explicitly override the GRCh38 MHC boundaries below.
```

| CLI flag                          | Default                 | Meaning and tutorial setting                                                                                                                                      |
| --------------------------------- | ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--snps-only`                     | Enabled                 | Keep only biallelic A/C/G/T SNPs; exclude indels and other allele types.                                                                                          |
| `--mac-min`                       | `5`                     | Keep variants with minor-allele count **MAC > 5**, not ≥5. MAC is calculated from nonmissing genotypes in the input panel.                                        |
| `--maf-min`                       | Unset                   | No additional minor-allele frequency filter. For example, `--maf-min 0.01` requires **MAF ≥ 0.01** as well as passing the MAC filter.                             |
| `--ld-unit` / `--ld-wind`         | `CM` / `1`              | Sum LD within **±1 cM of each SNP**. For a physical ±1 Mb window, use `--ld-unit KB --ld-wind 1000`; this is not equivalent to ±1 cM.                             |
| `--exclude-mhc-from-regression`   | Enabled                 | Exclude chr6 MHC SNPs from HM3 output rows and from both sides of the `w_ld` calculation. Keep MHC variants in the broad reference used for annotation LD scores. |
| `--mhc-start-bp` / `--mhc-end-bp` | `28510120` / `33480577` | Inclusive chr6 BP boundaries for GRCh38. **For this GRCh37 tutorial, use `25000000` / `34000000`.** These values are not inferred from the input files.           |
| `--calculate-w-ld`                | Disabled                | Enable to generate matching HM3 regression weights alongside the SNP × gene matrix, as in both examples.                                                          |
| `--w-ld-dir`                      | Unset                   | Directory for those regression-weight files; set it explicitly when enabling `--calculate-w-ld`.                                                                  |

For Python, replace hyphens with underscores: for example, `mac_min=5`,
`maf_min=None`, and `calculate_w_ld=True`. MAC and MAF are different filters:
MAC counts minor alleles, whereas MAF is their fraction among called alleles.
Changing these settings cannot restore variants already removed from the PLINK
panel by upstream QC.

The MHC setting distinguishes three SNP sets:

| SNP set             | Role                                          | MHC      |
| ------------------- | --------------------------------------------- | -------- |
| Broad reference     | Variants summed over for annotation LD scores | Retained |
| HM3 regression SNPs | GWAS observations used in regression          | Excluded |
| Weight-LD SNPs      | Both targets and summed variants for `w_ld`   | Excluded |

For cM windows, the BIM genetic positions must be finite, monotonic and
nonconstant. An all-zero cM column is not a usable genetic map; an arbitrary
nonzero origin, including negative starting values, is valid.
