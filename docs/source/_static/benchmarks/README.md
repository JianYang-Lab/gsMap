# Spatial LDSC scalability measurements

These files support the Performance and scalability guide. They contain only
aggregate timing and numerical-validation statistics, not GWAS or cell-level data.

## Single-trait / four-trait sweeps (current figures)

- Date: 2026-09-13. Frozen implementation under test:
    `a157625fc133d70c751e734d2ca2bcf833f69e65`.
- IQ-only traits: `[64]`; four-trait set: `[64, 92, 2, 59]` in the prepared
    manifest, corresponding to IQ, schizophrenia, age at menarche and height.
- Public data: `trait_scaling_measurements.csv`, 72 actual timed measurements.
- Figures: `cpu_trait_scaling.png/.svg`, `gpu_trait_scaling.png/.svg`.
- Each current figure contains one throughput axis (cell-traits/s), with
    independently measured IQ-only and four-trait curves. Endpoint labels are
    approximate means. CPU uses only the default batch 400; GPU uses batch 100
    per device. Extra axes and batch-size comparisons are not shown in the guide.
- Every point is measured, including IQ-only. Four-trait cells/s is not reused
    as an estimate of running a single trait. Trait-specific SNP masks are kept:
    1,029,600 SNPs for IQ alone, 1,073,133 union SNPs for four traits, 17,528 genes.
- Output audits check row count/order/identity, finite coefficients and positive
    standard errors, and uncapped absolute delta in -log10(p) against validated
    reference output. All 72 cases passed tolerance \<0.1. Maximum observed delta:
    CPU 0.033071106540, GPU 0.050043881144 (rounded upward).

### CPU

- Job 6289748 on `amd-ep5-099`; source runs `t1_6289748`, `t4_6289748`
    under `cpu/runs_million/`. Completed with exit 0:0 in 08:35:54.
- 1,000,037 distinct HESTA CS20 cells; batch 400; 1/2/4/8/16/32 physical cores;
    three repetitions for each trait set / core count (36 measurements).
- Exactly the GPU workload: same prepared manifest, marker prefix (row offset 0),
    frozen source modules and independently validated million-cell references.
    Staged marker-prefix SHA256:
    `ed724a47438bd02a79de82d13e66f8e7cad5b5c27a0e66f0e701aeac07eb1180`.
    Both CPU summaries record `matched_gpu.same_cells_as_gpu=true`.
- This replaces the earlier 30,037-cell curves from job 6288124. Those small-run
    timings are not mixed into the current CSV. All 36 GPU measurements are unchanged.
- Both trait sets use the same allocation. Recorded topology puts all selected
    cores on one NUMA node. Numba threads and CPU affinity increase together;
    four reader threads share that affinity; BLAS=1.
- Timed: actual production shared pipeline, including backend setup, marker
    reading, LD construction, all regressions, gzip writing and finalization.
- Excluded: imports, JIT warmup, initial staging, validation, shared-storage
    copy-back. Input and output use node-local SSD during timing.

### GPU

| Job     | Hardware            | IQ source run         | Four-trait source run         |
| ------- | ------------------- | --------------------- | ----------------------------- |
| 6288159 | A100-SXM4-40GB      | A100_t64_6288159      | A100_t64x92x2x59_6288159      |
| 6288160 | V100-SXM2-32GB      | V100_t64_6288160      | V100_t64x92x2x59_6288160      |
| 6288161 | GeForce RTX 2080 Ti | RTX2080Ti_t64_6288161 | RTX2080Ti_t64x92x2x59_6288161 |

- 1,000,037 distinct HESTA CS20 cells; batch 100 per GPU, FP32; 1/2/4 GPUs
    on one node with eight physical CPU cores per worker. Two repetitions with
    reversed GPU-count order, 36 measurements total. Each GPU model's IQ/four-trait
    sweeps share the allocation. Distinct physical GPU identities were checked.
- Timed: warm production compute components behind a benchmark start barrier,
    marker reading, LD construction, regression, gzip output and ordered merge.
- Excluded: worker setup/JIT, initial staging, validation, shared-storage
    copy-back. Input and output use node-local SSD during timing. These are
    throughput measurements, not full CLI startup-time measurements.
- `kernel_hashes` records the frozen implementation's modules. The reused
    harness's `source_head` field describes the *reference* implementation;
    it is not the implementation under test. Workers import `source-a157625/src`.
- The prepared manifest was relocated without changing analytical inputs.
    `metadata_relocation_verified=true` records successful reconstruction of the
    original manifest checksum by restoring only path strings and config hash.
    Initial preflight failures produced no timing points and are not in the CSV.

Rebuild current figures without private data:

```bash
python -m docs.tools.plot_trait_scalability
```

An internal results directory can be supplied with `--benchmark-root PATH` to
refresh the CSV. The importer requires complete, passed summaries for the entire
72-case grid and refuses missing, duplicate or failed measurements.
It imports CPU summaries from `cpu/runs_million/` and GPU summaries from
`gpu/scaling_results/`, and requires matching cell counts, row offsets,
implementation hashes and prepared-manifest provenance across all summaries.

## Earlier CPU batch-size sweep (separate tuning experiment)

- Source run: `cpu_batch_scaling/runs/full_6281577`, job 6281577, 2026-09-13.
- Frozen production source: `a157625fc133d70c751e734d2ca2bcf833f69e65`.
- 30,037 HESTA CS20 cells, four GWAS (IQ, schizophrenia, age at menarche, height).
- 1/2/4/8/16/32 physical AMD EPYC cores on one NUMA node. Numba threads and
    affinity increase together; four prefetch readers share that affinity; BLAS=1.
- Five batches: 100/400/800/1600/3200, three repetitions each, 90 measurements.
- `cpu_batch_measurements.csv` and `cpu_batch_scaling.csv` are unchanged copies
    of the run's raw measurements and aggregates.
- Timed: actual production shared pipeline, backend setup, reads, LD,
    regressions, gzip writing/finalization on node-local SSD.
- Excluded: imports, JIT warmup, initial staging, validation, shared-storage copy-back.
- All cases passed; maximum uncapped absolute log-p delta=0 versus validated
    batch-100 CPU output. This does not mean equality to every other backend.

## Earlier four-trait GPU sweep (retained, not used in current curves)

- Source runs: `scale4_A100_6277179` and `scale4_RTX_6277175`, 2026-09-12.
- Production integration's frozen gene-projection/fused-moment implementation;
    numerical integration subsequently committed as `c8c41c8` and reorganized
    through `a157625`. Final CLI/orchestration tests were separate from timing.
- 1,000,037 distinct HESTA CS20 cells, the same four traits, batch=100 per GPU.
- One/two/four A100-SXM4-40GB or RTX 2080 Ti GPUs, eight physical CPU cores per
    GPU; two reversed-order repetitions. No eight-GPU measurement.
- `gpu_scaling_measurements.csv` extracts only `method=moments` cases from
    completed source summaries. No legacy-control measurements are plotted.
- Timed: warm production components behind a benchmark start barrier, reads,
    LD, regression, gzip writing and ordered merge, all on node-local SSD.
- Excluded: worker setup/JIT, initial staging, validation, shared-storage copy-back.
- Source-run numerical audits against independently validated references are
    included as `max_abs_logp`; a separate paired implementation audit established
    maximum uncapped absolute log-p delta=0.0472214 across 429 trait files.

## Shared interpretation

All sweeps use the GRCh37 1000 Genomes Phase 3 / GENCODE v46 SNP × gene matrix.
These are not the E16.5/v50 tutorial inputs.
The current CPU and GPU curves use identical cells and analytical inputs, but
slightly different setup timing boundaries: CPU includes per-run backend setup;
GPU excludes worker setup behind a start barrier. They show practical throughput,
not a strictly setup-matched hardware comparison or a cold-start runtime prediction.

One cell-trait is one cell tested for one GWAS. Four-trait cells/s is
cell-traits/s divided by four. Current plots show mean cell-traits/s only;
all repetitions remain available in the CSV. In the earlier archived plots,
speedup is relative to the same series with one core/GPU and error bars show
observed min–max, not confidence intervals.
The two trait sets differ in both trait count and retained SNP universe;
their ratio does not isolate the benefit of sharing versus running all four
traits separately. CPU batch tuning used a different allocation from the
single-/four-trait sweep; do not combine their timings into one scaling curve.

Rebuild earlier PNG/SVG figures with `python -m docs.tools.plot_scalability`.
Its CPU plot shows batches 100/400/1600; all five batches remain in the CSVs.
