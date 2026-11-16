# Downloads

[Start a tutorial](#start-a-tutorial) · [Browse all resources](#browse-all-resources)

Download a dataset, reuse the shared reference, and explore its report preview.

## Start a tutorial

After {doc}`install`, download the shared resources once. Every tutorial reuses
them; only the spatial dataset changes.

```text
gsmap_tutorials/
├── resources/
│   ├── 1000g-hc-grch38-v50-eur/
│   │   ├── snp_gene_weights.h5ad
│   │   └── w_ld/
│   └── homologs/
└── example_data/
    ├── example-traits/            # shared GWAS
    └── <dataset>/                # download from your chosen tutorial
```

Create a workspace, or enter the one you already use:

```bash
mkdir -p gsmap_tutorials
cd gsmap_tutorials
```

### 1. Download shared resources

```{tip}
**Skip a download if its extracted folder already exists.** These commands use
the EUR reference required by the tutorials; other ancestries are in the catalog.
```

**LD reference** — the SNP × gene matrix and matching regression weights:

```bash
mkdir -p resources
wget -O resources/1000g-hc-grch38-v50-eur.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/ld_weights/1000g-hc-grch38-v50-eur/r1/bundle.tar.gz && \
    tar -xzf resources/1000g-hc-grch38-v50-eur.tar.gz -C resources
```

**Homolog tables** — needed for mouse/macaque data; skip for human data:

```bash
mkdir -p resources
wget -O resources/homologs.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/reference/homologs/homologs/r1/bundle.tar.gz && \
    tar -xzf resources/homologs.tar.gz -C resources
```

**Example GWAS** — IQ, AF, LDL, Height, MCHC and serum creatinine:

```bash
mkdir -p example_data
wget -O example_data/example-traits.tar.gz \
    https://yanglab.westlake.edu.cn/data/gsMap/gsMap3D-r1/gwas/example-traits/r1/bundle.tar.gz && \
    tar -xzf example_data/example-traits.tar.gz -C example_data
```

### 2. Choose a tutorial

Each tutorial downloads only its own dataset, then shows how to run gsMap and
open the report. Continue using this workspace.

| Dataset | Tutorial |
|---|---|
| Mouse brain and kidney, Visium | {doc}`2D Visium <2d_tutorial/10x>` |
| Mouse E16.5 embryo, MOSTA | {doc}`2D embryo <2d_tutorial/mouse_embryo>` |
| Human CS8 embryo, 62 sections | {doc}`3D gastrulation <3d_tutorial/human_gastrulation>` |
| Mouse E11.5 embryo, 84 sections | {doc}`E11.5 tutorial <3d_tutorial/mouse_e11>` |

<details>
<summary>Verify your download</summary>

Every archive has a **SHA-256** link in the catalog. Print your archive's checksum
and compare the 64-character value with that link:

```bash
sha256sum resources/1000g-hc-grch38-v50-eur.tar.gz
```

Local archive filenames may differ; the checksum value must match.
After extraction, check the files inside the bundle:

```bash
(cd resources/1000g-hc-grch38-v50-eur && sha256sum --check SHA256SUMS)
```

All entries should report `OK`. Use the same check from any extracted bundle
folder. Do not use files that fail verification.

</details>

Using your own data or storing resources elsewhere? See {doc}`advanced_usage`
for paths and CLI flags.

## Browse all resources

Click a dataset name to open its tutorial. Publication links open the original
study or provider page; report previews open the interactive gsMap results.

<!-- RESOURCE_CATALOG -->
