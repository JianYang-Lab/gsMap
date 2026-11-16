# Test suite

Run the default suite from the repository root:

```bash
JAX_PLATFORMS=cpu python -m pytest
```

Pytest uses the local `src/` tree. Tests requiring downloaded data are opt-in;
tests requiring optional libraries or Node.js report a skip when unavailable.
The default suite does not download resources or submit scheduler jobs.

## Organization

| Directory       | Coverage                                                                  |
| --------------- | ------------------------------------------------------------------------- |
| `config/`       | CLI entry points, option defaults, and configuration validation           |
| `find_latent/`  | Spatial graphs, inference, training, and embedding inputs                 |
| `latent2gene/`  | Rank readers, caches, neighbors, marker scores, and row mappings          |
| `ldscore/`      | Reference filters, LD windows, and SNP × gene matrix generation           |
| `spatial_ldsc/` | CPU/GPU backends, regression, pipeline concurrency, and output            |
| `report/`       | Metadata I/O, UMAP caching, payloads, labels, and JavaScript behavior     |
| `docs/`         | Direct tutorial/API equivalence, documented flags, and reference accuracy |
| `resources/`    | Verified test-data downloads and archive safety                           |
| `integration/`  | End-to-end 2D, 3D, single-cell, and custom-input workflows                |
| `logging/`      | Log layout, context, progress, and handler cleanup                        |

Run one area, for example:

```bash
python -m pytest tests/spatial_ldsc tests/report
```

Some module suites also contain real-data tests. Select them by marker rather
than assuming that all real-data cases live in `integration/`.

## Real-data integration tests

Download and extract the CI test bundle using the resource downloader, then run:

```bash
python tests/tools/fetch_resources.py ci-test-data --destination .test-data
JAX_PLATFORMS=cpu python -m pytest tests -m real_data --run-real-data \
  --test-data .test-data/ci-test-data \
  --work-dir .test-results
```

Use a fresh output directory when verifying a new implementation, so cached
analysis products cannot hide regressions. CI runs the real-data cases too.
When explicitly enabled, missing data files fail rather than silently skip.
GPU-specific cases need a compatible CUDA environment and an allocated GPU.

## Shared helpers and coverage

- `conftest.py` owns downloaded-data fixtures, CLI runners, and log-state isolation.
- `paths.py` locates the repository independently of the test module's depth.
- `docs/helpers.py` inspects CLI options without executing an analysis or validating
    installation-specific data paths. Real execution is tested separately.
- `helpers/` contains generic tutorial extraction, execution and browser checks.
    Run them as modules, for example `python -m tests.helpers.audit_commands --help`.
- `tools/fetch_resources.py` verifies downloaded test data. Resource preparation
    and server publication are not dependencies of the public test suite.
- Report runtime tests execute JavaScript with Node.js test doubles; these are
    not substitutes for a browser visual inspection.

Prefer behavioral assertions over source-string checks. Keep numerical parity,
row/allele alignment, failure propagation, and concurrency tests even when their
implementations overlap. Remove checks for retired UI text or duplicated help
output, rather than weakening scientific or failure-handling coverage.
