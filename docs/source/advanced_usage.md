# Customization Guide

Configure gsMap for your own data, GWAS traits and reference weights. This guide
also covers custom embeddings, custom GSS and spatial-neighborhood settings.

Examples show CLI options and their Python equivalents. Apply option-only
snippets to a complete command from the {doc}`2d_tutorial/10x` or
{doc}`3d_tutorial/human_gastrulation` tutorial.

For the underlying concepts, see {doc}`key_concepts`; for execution tuning, see
{doc}`scalability`.

## File paths and configuration flags

| File or folder                   | CLI flag                              | Python field                  |
| -------------------------------- | ------------------------------------- | ----------------------------- |
| `snp_gene_weights.h5ad`          | `--snp-gene-weight-adata-path`        | `snp_gene_weight_adata_path`  |
| Matching `w_ld/` folder          | `--w-ld-dir`                          | `w_ld_dir`                    |
| `mouse_human_homologs.txt`       | `--homolog-file`                      | `homolog_file`                |
| One ST H5AD                      | `--h5ad-path`                         | `h5ad_path=["..."]`           |
| Ordered `sample_paths.txt`       | `--h5ad-list-file`                    | `h5ad_list_file`              |
| One GWAS summary-statistics file | `--sumstats-file` with `--trait-name` | `sumstats_file`, `trait_name` |
| Multiple GWAS in `gwas.yaml`     | `--sumstats-config-file`              | `sumstats_config_file`        |
| Output parent folder             | `--workdir`                           | `workdir`                     |

The tutorial folder names are a convenience, not a requirement. Use absolute paths
when shared resources are stored elsewhere. Paths inside `sample_paths.txt` and
`gwas.yaml` are relative to each file's folder; keep the dataset and
`example-traits/` folders together when moving an example. Preserve anatomical
section order along the z axis in a 3D section list.

```{important}
For a different ancestry, change both the matrix and `w_ld/` paths. For a newer
resource revision, extract into a separate parent folder, check its metadata and
update paths explicitly; do not overwrite an existing installation. Leave
checksum-listed bundle files unchanged and make separate configuration copies
when customizing an example.
```

## GWAS names in reports

Use stable trait IDs as the keys in `gwas.yaml`. An optional `label` provides a
readable name in report selectors, titles, heatmaps and tables:

```yaml
IQ:
  path: ../example-traits/IQ.sumstats.gz
  label: "Intelligence — cognitive performance"
SCZ:
  path: /data/GWAS/SCZ.sumstats.gz
  label: "Schizophrenia"
```

Pass this file with `--sumstats-config-file gwas.yaml`, or
`sumstats_config_file="gwas.yaml"` in Python. Relative `path` values are resolved
from the YAML file's directory. Searches match both the ID and its label; spaces
and punctuation in labels are fine. IDs remain unchanged in results and caches.
After editing a label, regenerate the report with `--start-step report`; LDSC
does not need to run again. Use unique IDs, without path separators.

The original path-only format is still supported, including mixed entries:

```yaml
IQ: ../example-traits/IQ.sumstats.gz
SCZ:
  path: /data/GWAS/SCZ.sumstats.gz
  label: "Schizophrenia"
```

Without a label, the report displays the ID. In the Python API,
`sumstats_config_dict` also accepts these entries, or use a path dictionary with
`trait_labels={"IQ": "Intelligence"}`.

## Customize SNP-to-gene linking

Current tutorials use GENCODE v50 / 1000G high-coverage EUR. For custom gene BED
files or another LD panel, see {doc}`ldscore_weight_matrix`. For prebuilt AFR,
AMR, EAS, EUR, SAS and historical v46 EUR choices, see {doc}`downloads`.

### Using a different or custom weight matrix

Change **both** the matrix and its accompanying regression weights. For example,
after downloading the EAS bundle, replace these options:

```text
--snp-gene-weight-adata-path resources/1000g-hc-grch38-v50-eas/snp_gene_weights.h5ad
--w-ld-dir resources/1000g-hc-grch38-v50-eas/w_ld
```

Python uses `snp_gene_weight_adata_path` and `w_ld_dir` with the same paths in
`QuickModeConfig`. Choose ancestry from the GWAS study, not the tissue species.
Ancestry matching does not guarantee larger association statistics.

This is a **complete command** using the downloaded EUR pair and Visium data.
To use your reference, replace the two reference paths; for an EAS GWAS, also
replace the GWAS file and choose the appropriate reference pair.

```bash
gsmap quick-mode \
    --workdir results \
    --project-name custom-reference \
    --dataset-type spatial2D \
    --h5ad-path example_data/mouse-visium/ST/V1_Adult_Mouse_Brain_Coronal_Section.h5ad \
    --annotation domain \
    --spatial-key spatial \
    --data-layer count \
    --species MOUSE_GENE_SYM \
    --homolog-file resources/homologs/mouse_human_homologs.txt \
    --snp-gene-weight-adata-path resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad \
    --w-ld-dir resources/1000g-hc-grch38-v50-eur/w_ld \
    --trait-name IQ \
    --sumstats-file example_data/example-traits/IQ.sumstats.gz \
    --plot-origin lower
```

Do not mix v46/v50, builds or ancestry pairs. `w_ld` is not the row sums of the
SNP × gene matrix. Absolute paths work if shared resources are stored elsewhere.

## Use your own embedding

Store cell-state and niche embeddings in `adata.obsm`, then skip gsMap embedding
training with these options:

```text
--start-step latent2gene
--latent-representation-cell X_pca
--latent-representation-niche X_spatial_pca
--data-layer counts
```

Python equivalents: `start_step="latent2gene"`,
`latent_representation_cell="X_pca"`, `latent_representation_niche="X_spatial_pca"`
and `data_layer="counts"` in `QuickModeConfig`.

For example, first create a **human-gene-symbol** input with:

```text
inputs/my_spatial_with_embeddings.h5ad
├── obs_names                    unique cell/spot IDs
├── obs['cell_type']              annotations
├── var_names                    human gene symbols
├── layers['counts']              raw expression counts
├── obsm['spatial']               coordinates
├── obsm['X_pca']                 cell-state embedding
└── obsm['X_spatial_pca']          niche embedding
```

Then run this complete command on your prepared input:

```bash
gsmap quick-mode \
    --workdir results \
    --project-name custom-embedding \
    --dataset-type spatial2D \
    --h5ad-path inputs/my_spatial_with_embeddings.h5ad \
    --start-step latent2gene \
    --latent-representation-cell X_pca \
    --latent-representation-niche X_spatial_pca \
    --annotation cell_type \
    --spatial-key spatial \
    --data-layer counts \
    --snp-gene-weight-adata-path resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad \
    --w-ld-dir resources/1000g-hc-grch38-v50-eur/w_ld \
    --trait-name IQ \
    --sumstats-file example_data/example-traits/IQ.sumstats.gz
```

Both embeddings must have one finite-valued row per observation. If raw counts
are in `adata.X`, use `--data-layer X`; they do **not** have to be in X when a
count layer is provided. Scaled/log-transformed values are not raw counts.
For mouse or macaque genes, also specify `--species` and `--homolog-file` as in
the tutorials. For multiple XYZ sections, use an anatomically ordered `--h5ad-list-file`
and `--dataset-type spatial3D`. For one thick-tissue XYZ volume, use `--h5ad-path`.

## Use your customized GSS

Run spatial LDSC directly on external Gene Specificity Scores to skip embedding
and latent-to-gene. Create the input described below before executing:

```bash
gsmap spatial-ldsc \
    --workdir results \
    --project-name custom-gss \
    --marker-score-format h5ad \
    --marker-score-h5ad-path inputs/custom_marker_scores.h5ad \
    --snp-gene-weight-adata-path resources/1000g-hc-grch38-v50-eur/snp_gene_weights.h5ad \
    --w-ld-dir resources/1000g-hc-grch38-v50-eur/w_ld \
    --trait-name IQ \
    --sumstats-file example_data/example-traits/IQ.sumstats.gz
```

### Format requirements

- **Rows:** cells/spots, with unique IDs in `.obs_names`.
- **Columns:** human gene symbols in `.var_names`.
- **Values in X:** your gene-specificity/marker scores, not ordinary expression counts.
- **Gene overlap:** only overlapping genes with the reference are used. Inspect
    the overlap; the presence of a few common genes is not sufficient validation.

Python uses `marker_score_format="h5ad"` and
`marker_score_h5ad_path="inputs/custom_marker_scores.h5ad"` in `SpatialLDSCConfig`,
with the same GWAS, reference and output fields shown above.

This standalone command produces LDSC results, not missing spatial metadata for
reports or Cauchy. Those steps require annotations/coordinates identifying the
same spot IDs; matching row positions is insufficient. For native gsMap memmaps,
keep `marker_scores.obs_indices.dat` with the matrix and metadata: it maps marker
rows to canonical AnnData rows.

## Choose spatial geometry

| Input                               | Mode        | Homogeneous-neighbor search                    | Spatial report                    |
| ----------------------------------- | ----------- | ---------------------------------------------- | --------------------------------- |
| One or more XY sections             | `spatial2D` | XY, within each section                        | 2D                                |
| One or more XYZ sections in 2D mode | `spatial2D` | First two coordinates only; a warning is shown | 2D                                |
| Multiple registered XYZ sections    | `spatial3D` | Full XYZ; optional cross-section matching      | 3D and section views              |
| One XYZ H5AD (thick tissue)         | `spatial3D` | Full XYZ within the volume                     | 3D, without section-style 2D maps |

For a thick-tissue volume, replace the tutorial input flags with:

```text
--dataset-type spatial3D
--h5ad-path data/thick_tissue.h5ad
--spatial-key spatial
```

Python uses `dataset_type="spatial3D", h5ad_path=["data/thick_tissue.h5ad"]`.
The coordinate array must be N×3. A single input is automatically treated as a
volume; do not split it into artificial Z sections. Cross-section matching is
disabled, while UMAP, enrichment, GWAS and gene statistics remain in its report.

For multiple registered XYZ sections that do not need cross-section matching,
keep `spatial3D` and set `--n-adjacent-slices 0` (Python:
`n_adjacent_slices=0`). Distances still use all three coordinates. This does
not disable edges in an explicitly selected `section_aware_3d` embedding graph.

Always provide columns **[X, Y, Z]**; the third column is Z. For multiple
sections, XY must describe the section plane and Z the stacking direction.
Reorient other coordinate conventions consistently before running gsMap.

At configuration time, gsMap checks shape, dtype and at most 4,096 deterministic
rows per file without reading expression data. For each section, its 1st–99th
percentile Z span must be **strictly smaller than both X and Y spans**.
Equal or larger spans (including degenerate XY geometry) raise an error naming
the file and spans, even with `n_adjacent_slices=0`. Thick-tissue volumes and
`spatial2D` inputs are exempt.

This is a sampled geometry check, not proof of anatomical registration; sparse
outliers or unsampled problems can be missed. All used coordinates are checked
for finite values during computation. XY input is rejected in 3D mode; no
artificial Z is added.

## Customize the graph and cross-section matching

### Graph neighbors

Replace/add these flags in the embedding or full quick-mode command:

```text
--graph-mode symmetric_knn
--n-neighbors 20
```

This requests 20 **external** neighbors before bidirectional union. The final
undirected degree can be larger. Python uses `graph_mode="symmetric_knn"` and
`n_neighbors=20`.

For coordinates measured in micrometres, an **illustrative** 50 µm radius is:

```text
--graph-mode radius
--graph-radius 50
```

The Gaussian-weighted alternative with illustrative bandwidth 25 µm is:

```text
--graph-mode weighted_radius
--graph-radius 50
--graph-sigma 25
```

Python uses `graph_mode`, `graph_radius` and `graph_sigma` with the same values.
These distances are not universal defaults. Changing graph flags after starting
at `latent2gene` cannot change embeddings that have already been computed.

### Cross-section matching

For multiple XYZ sections in `spatial3D`, the default ordinal setting searches two sections on each side,
with boundary compensation: near the first and last sections, farther sections
can fill the neighbor quota. This is a section-order rule, not a physical-distance
cutoff. To use E11.5 ordinal matching:

```text
--n-adjacent-slices 8
```

Python uses `n_adjacent_slices=8` in `QuickModeConfig`. Keep the section paths in
anatomical order along the z axis. See {doc}`3d_tutorial/mouse_e11` for a complete
example and {doc}`cli_reference` for all available options.
This setting controls **matching**, not graph edges between sections.
`spatial2D` keeps matching within each section.

### Resume an analysis

If embedding completed successfully and inputs/settings are unchanged, add
`--start-step latent2gene` to the same quick-mode command, or
`start_step="latent2gene"` to its Python configuration.
If embedding used high-quality cell QC, also pass `--high-quality-neighbor-filter`
(`high_quality_neighbor_filter=True` in Python) to preserve that filtering.
Do not rerun embedding upstream of existing GSS/LDSC outputs.

```{warning}
Use a new `project_name` when changing inputs, graph/matching settings or the LD
reference. Completion checks do not establish scientific cache compatibility.
Do not run the CLI and Python interfaces concurrently into the same project.
```

## Specify your own fields and traits

| Input                 | CLI flag excerpt                                                   | Python field                    |
| --------------------- | ------------------------------------------------------------------ | ------------------------------- |
| Counts in a layer     | `--data-layer counts`                                              | `data_layer="counts"`           |
| Counts in X           | `--data-layer X`                                                   | `data_layer="X"`                |
| Labels in `obs`       | `--annotation cell_type`                                           | `annotation="cell_type"`        |
| Coordinates in `obsm` | `--spatial-key spatial_aligned`                                    | `spatial_key="spatial_aligned"` |
| One GWAS              | `--trait-name MyTrait --sumstats-file inputs/my_trait.sumstats.gz` | `trait_name`, `sumstats_file`   |
| Multiple GWAS         | `--sumstats-config-file inputs/gwas.yaml`                          | `sumstats_config_file`          |

For multiple GWAS, replace the single-trait options; do not supply both forms.
Paths in a GWAS YAML are relative to the **YAML directory**. For example, save
this as `inputs/gwas.yaml` in the tutorial workspace:

```yaml
IQ: ../example_data/example-traits/IQ.sumstats.gz
Height: ../example_data/example-traits/Height.sumstats.gz
```

## Scalability and performance

See {doc}`scalability` for E16.5 examples covering CPU/GPU selection, SSD storage,
shared multi-trait execution, CPU/multi-GPU scaling, memory tuning and report
display limits.
For all available options, consult {doc}`cli_reference` or run:

```bash
gsmap quick-mode --help
gsmap spatial-ldsc --help
```
