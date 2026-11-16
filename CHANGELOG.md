# Changelog

All notable changes to gsMap are recorded here. gsMap 2.x is published on PyPI as `gsMap`.
Release notes on GitHub are generated from the section that matches each version tag.

<!-- release-notes-start -->

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
