# Documentation tools

Run from the repository root after installing the documentation dependencies:

```bash
python -m docs.tools.generate_cli_reference
python -m docs.tools.generate_downloads --catalog docs/resource_catalog.json
sphinx-build -W --keep-going -b html docs/source docs/build/html
```

These tools generate public documentation only. The resource catalogue contains
public download metadata, not local source mappings or server configuration.
Tutorial checks live in `tests/helpers/`; see `tests/README.md`.

## Resource URL prefix

`gsmap_resource_base_url` in `docs/source/conf.py` controls download URLs,
tutorial `wget` commands and report-preview links in the built documentation.
Its default is
`https://yanglab-obs.lab.westlake.edu.cn/public/gsMap/gsMap3D-r1`.
To build for another server without editing the source pages:

```bash
sphinx-build -W --keep-going -b html \
    -D gsmap_resource_base_url=https://your-server.example/data/gsMap \
    docs/source docs/build/html
```

Alternatively, change the value directly in `conf.py`. Trailing slashes are
removed. Publication links and relative documentation links are unchanged.
The resource directory layout must be the same under the new prefix.
This configures documentation links only: it does not move files, update the
published catalogue or change the software's resource-download settings.

## Publishing

Maintainers publish the built HTML and resource bundles to the OBS mirror under the
`gsmap_resource_base_url` prefix. Build into a fresh directory so obsolete pages are
not carried into a release:

```bash
sphinx-build -E -a -W --keep-going -b html docs/source docs/build/obs-public-review
```

Upload credentials, endpoints and the step-by-step publication checklist are kept in
the maintainers' private notes, not in this repository. Check the public page and
asset URLs after an upload; a build or dry run alone is not a publication.

## Performance figures

Regenerate the CPU and GPU scalability figures from the checked-in measurement
CSVs without running a new benchmark:

```bash
python -m docs.tools.plot_trait_scalability
python -m docs.tools.plot_scalability
```

`plot_trait_scalability` draws the IQ-only/four-trait CPU and GPU comparisons.
`plot_scalability` redraws the separate CPU batch-size sweep and the earlier
four-trait GPU sweep; the latter is retained as historical measurement data,
not mixed into the new curves. Both tools read local CSVs and need no GWAS files.

To import a new **complete, numerically validated** single-/four-trait sweep,
pass its results directory to `plot_trait_scalability --benchmark-root PATH`.
It requires all 72 measurements: CPU 1/2/4/8/16/32 cores (three repetitions),
and A100/V100/RTX 2080 Ti 1/2/4 GPUs (two repetitions), for both trait counts.
Both platforms must use the same 1,000,037 cells. CPU summaries are read from
`cpu/runs_million/`; GPU summaries from `gpu/scaling_results/`. Source and input
provenance must match across all summaries.
Incomplete or failed summaries cannot replace the public measurement CSV.

Measurement scope and provenance are recorded in
`docs/source/_static/benchmarks/README.md`. Do not mix different timing protocols
in a CPU-versus-GPU speedup comparison.
