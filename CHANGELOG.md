# Changelog

All notable changes to gsMap are recorded here. gsMap 2.x is published on PyPI as `gsMap`.
Release notes on GitHub are generated from the section that matches each version tag.

<!-- release-notes-start -->

## Unreleased

- Documentation is published to GitHub Pages (<https://jianyang-lab.github.io/gsMap/>) from `main` by the docs
    workflow.
- The upgrade guide lists the gsMap 2.x Python API next to each command.
- The README is shorter and links to the documentation for installation and upgrade
    details.
- Interactive reports: the 3D viewer draws its points from typed arrays in one WebGL layer
    instead of Plotly point traces. Large 3D reports load without long freezes, use less
    memory, and respond faster to orbiting and to trait, opacity, threshold, point-size and
    category changes.
- Interactive reports: the rest of the page is faster too. Tailwind CSS is compiled ahead of
    time and inlined, instead of being compiled in the browser on every load (the bundled
    `tailwindcss.js` is gone). Large UMAP and Manhattan payloads stay outside Alpine's reactive
    state, the Manhattan plot draws one trace per chromosome run instead of a colour per SNP,
    and 2D maps are drawn once at their final point size. Trait and annotation switches take
    about half as long and the page uses less memory.
- Reports look the same. Existing reports pick up both changes by re-rendering `index.html`;
    the data files are unchanged.
- Spatial LDSC: a spot whose regression fails (non-finite beta or SE, or SE = 0) is written
    with a missing p-value instead of p = 1, and Cauchy combination skips it. SNPs with a
    missing or non-positive N are dropped with a warning; one such SNP used to make every
    spot's p-value 1. Results for valid inputs are unchanged.
- Cauchy combination: p-values that underflow to 0 no longer make a whole annotation report
    p = 1, and each trait now skips only its own missing spots, so one trait's result no
    longer depends on which other traits run with it.
- Spatial LDSC robustness: only real GPU out-of-memory errors fall back to the CPU; a hung
    multi-GPU worker fails after an hour without progress
    (`GSMAP_LDSC_WORKER_STALL_SECONDS`); resources are released when setup fails; a skipped
    trait is no longer re-read; the moments backend explains how to continue when it rejects
    a singular fit. Importing gsMap no longer changes JAX settings.
- `gsmap ldscore-weight-matrix`: a missing PLINK or annotation file for a requested
    chromosome, or inconsistent features, now raises an error instead of silently leaving
    the chromosome out; the matrix is written atomically, and errors raise exceptions
    instead of exiting the Python process.
- `gsmap format-sumstats`:
    - Odds-ratio input: SE comes from SE_OR / OR (delta method) instead of log(SE_OR), and
        OR alone (or LOG_ODDS) is enough to sign Z.
    - P = 0 (underflow) SNPs are kept for gsMap output when BETA/SE or Z give a finite Z.
    - N_CAS + N_CON is used when there is no N column.
    - A missing value only drops a SNP when it is in a column that is written out or filtered
        on; missing values in unrelated columns no longer remove SNPs.
    - COJO output keeps full numeric precision (it was rounded to 3 decimals).
    - Position-to-rsID conversion works whether or not chromosome names start with "chr", for
        a single chromosome, and at multi-allelic sites (alleles must match).
    - gsMap-format output for ordinary input is unchanged.
- Reruns: each step records the settings and inputs that produced its outputs. When a rerun
    skips a step because outputs exist, a warning names any setting or input that changed
    (for example `--annotation`, a GWAS file, or upstream results that were recomputed) and
    how to recompute; the outputs are still reused.
- Reruns: outputs that later runs trust are written atomically (temporary file, fsync,
    rename), including the latent-to-gene sidecar AnnData that is rewritten after its
    checkpoint, and `report_meta.json` is written only after every report payload, so an
    interrupted run is never mistaken for a complete one. Scratch copies left by killed
    runs on the same host are removed.
- Quick mode: `force_report_re_run` now regenerates the report; a run resumed at
    `--start-step latent2gene` from this project's find_latent outputs uses the same
    dual-embedding settings as a full run (it used to switch the niche similarity off); a
    deleted rank checkpoint no longer blocks reuse of completed latent-to-gene outputs.
- Latent-to-gene warns when explicit `--h5ad-*` inputs are replaced by the project's
    find_latent outputs, and the report fails early when two sample names map to the same
    report file name.
- Latent-to-gene: neighbor indices of a large cell type no longer wrap when stored after a
    smaller type's narrower index array; embeddings whose float16 norm overflows are
    normalised in float32 instead of becoming zeros; the rank matrix is published only if
    every expected row was written; the rank writer buffers at most about 2 GiB while the
    filesystem catches up; embeddings are uploaded to the accelerator once instead of
    once per cell type.
- Quick mode checks the settings of later steps before step 1: spatial LDSC options, the
    weights directory and weight matrix, trait names that clash with metadata columns, and
    Cauchy annotation columns. Errors that used to appear hours into a run (after
    find_latent and latent-to-gene) now appear at once; finished steps are not rechecked.
- No more global side effects: importing gsMap no longer silences every warning in the
    process, the report no longer reseeds NumPy's global random generator (same samples),
    and a run leaves the `gsMap` logger's level and propagation as it found them, so
    notebook logging keeps working after a run.
- `gsmap report-view` serves the report without changing the working directory, answers
    requests in parallel and can restart on the same port at once; the report reads GWAS
    files in every format spatial LDSC accepts (plain, gzip or bz2; any whitespace).

## 2.0.0b1

First beta of gsMap 2.0, which adds 3D spatial transcriptomics mapping. gsMap 2.0 continues
the code previewed on PyPI as `gsMap3D` (0.1.x); install this pre-release explicitly with
`uv pip install "gsMap[cpu]==2.0.0b1" --torch-backend=cpu` (or another backend extra).

### Highlights

- 3D mapping of registered multi-section data and single thick-tissue volumes, with optional
    cross-section matching of homogeneous cells.
- Dual embeddings of molecular and spatial features for gene specificity scoring.
- GPU-accelerated embeddings, gene specificity scores and spatial LDSC, including multi-trait
    and single-node multi-GPU spatial LDSC.
- Interactive reports with comparison views and SpatialVista export for 3D data.
- Ancestry-specific GRCh38 1000 Genomes high-coverage LD references with matching regression
    weights and SHA-256 manifests.

### Breaking changes from gsMap 1.x

- Python 3.12 or later is required. On Python 3.10 and 3.11, `pip` keeps installing gsMap 1.x.
- The command-line interface was replaced, for example `gsmap quick_mode` became
    `gsmap quick-mode` and `gsmap run_spatial_ldsc` became `gsmap spatial-ldsc`.
- Spatial LDSC reads a precomputed SNP-by-gene weight matrix and regression weights instead of
    per-sample LD scores; `gsmap ldscore-weight-matrix` builds the matrix for a new reference.
- Output directories and file formats changed. Start 2.x analyses in a new working directory.

See [Upgrading to gsMap 2.0](https://github.com/JianYang-Lab/gsMap/blob/main/docs/source/upgrading.md)
for the full command and option mapping.

### Changes since gsMap3D 0.1.4a2

- The package is published as `gsMap`. `gsmap --version` and run metadata report the `gsMap`
    version, and every command warns when the superseded `gsMap3D` package is also installed.
- The top-level `gsmap --help` no longer lists a nonexistent `gsmap report` command.
- Smaller installation: unused report code paths were removed together with the dependencies
    only they needed (`pyvista`/VTK and `kaleido`), and the unused direct dependencies
    `bitarray`, `zarr` and `xarray` were dropped. Analysis results are unchanged.
- The Docker image installs CPU PyTorch next to CPU JAX instead of CUDA PyTorch from PyPI.
- `click` is pinned below 8.5: with the pinned `typer` 0.20, click 8.5 drops the help text of
    positional arguments from `gsmap COMMAND --help`.
- The embedding model architecture is written to the debug log instead of standard output.
- Documentation: new upgrade guide, pre-release installation instructions and a regenerated
    CLI reference.
- CI and test tooling: resource downloads send an explicit User-Agent (the resource server
    rejects the default `Python-urllib` agent), the 3D integration test uses registered
    XYZ sections, and the 2D integration fixture no longer leaks CLI logging into later tests.

## gsMap3D preview releases

These pre-releases of the 2.0 code were published on PyPI as `gsMap3D`.

### 0.1.4a2

- Explicit `expression_data_type=noncounts` support for rank-only marker scoring with an
    existing cell embedding. This mode skips `find_latent`, requires
    `no_expression_fraction=True`, and rejects count-based QC options. Counts remain the default;
    automatic expression-type detection is not yet included.
- A GWAS-independent `DataPreparationConfig` for configuring the latent representation and
    marker-score stages.
- Clearer 2D/3D coordinate validation and support for single-volume thick-tissue inputs, with
    cross-section matching disabled for a single 3D input.
- Numeric pandas DataFrame spatial coordinates are accepted.
- Updated report headers, spatial cell banners, plotting guidance, tutorial resources,
    documentation navigation, and resource-mirror configuration.

### 0.1.4a1

- Shared multi-trait spatial LDSC, optimized CPU/GPU regression, and single-node multi-GPU
    execution.
- Faster GWAS preparation and per-stage resource and throughput logging.
- Improved interactive reports and automatic SpatialVista export for 3D data.
- Expression-based cell filtering before latent modeling and HVG provenance.
- Ordinal matching defaults to two sections on each side.
- Quick mode exposes the exact external-neighbor count, radius and bandwidth.
- Executable 2D/3D tutorials use paired V50 LD references and report previews.
- The E11.5 example runs ordinal ±8-section matching in one quick-mode call.
- Cauchy aggregation aligns reordered LDSC rows by unique spot ID rather than assuming the
    canonical AnnData order. Duplicate or mismatched IDs are rejected.
- LD calculation accepts finite, ordered cM maps with a negative starting origin; constant,
    non-finite and unsorted maps remain invalid.
- Resource bundles include SHA-256 manifests and matching regression weights.

### 0.1.0

- First public preview release.

## gsMap 1.x

gsMap 1.x (1.0 to 1.73.8) is maintained on the
[`1.x` branch](https://github.com/JianYang-Lab/gsMap/tree/1.x) and receives bug fixes only.
Its release notes are on the
[GitHub releases page](https://github.com/JianYang-Lab/gsMap/releases?q=v1.&expanded=true).
