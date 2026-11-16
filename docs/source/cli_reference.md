# CLI reference

Generated from the current `gsMap.cli` application; do not edit by hand.

## `gsmap`

gsMap: genetically informed spatial mapping of cells for complex traits

| Option | Description |
|---|---|
| `-v, --version` | Show version and exit |
| `--help` | Show this message and exit. |

## `gsmap quick-mode`

Run the complete gsMap pipeline with all steps.

This command runs the gsMap analysis pipeline including:
- Data loading and preprocessing (Find Latent)
- Gene expression analysis (Latent to Gene)
- GWAS integration (Spatial LDSC)
- Cauchy Combination Test
- Result generation (Report)

Requires pre-generated SNP-gene weight matrix and LD weights.

| Option | Description |
|---|---|
| `--workdir DIRECTORY` | Path to the working directory  \[required] |
| `--project-name TEXT` | Name of the project  \[required] |
| `--dataset-type [scrna\|spatial2d\|spatial3d]` | Type of dataset: scRNA, spatial2D (XY), or spatial3D (XYZ sections or a single thick-tissue volume)  \[default: spatial2D] |
| `--h5ad-path PATH` | Space-separated list of h5ad file paths. Sample names are derived from file names without suffix. |
| `--h5ad-yaml FILE` | YAML file with sample names and h5ad paths |
| `--h5ad-list-file FILE` | Each row is a h5ad file path, sample name is the file name without suffix |
| `--data-layer TEXT` | Expression data layer in h5ad layers, e.g., 'count', 'counts'; use 'X' for adata.X  \[default: X] |
| `--cell-qc-min-genes INTEGER RANGE` | Minimum positive-expression genes before HVG, training, graph and inference  \[default: 20; x>=0] |
| `--cell-qc-min-counts FLOAT RANGE` | Optional minimum total raw counts for front-end cell QC  \[x>=0] |
| `--cell-qc-chunk-size INTEGER RANGE` | Rows per bounded expression QC/read block  \[default: 8192; x>=1] |
| `--spatial-key TEXT` | Spatial key in adata.obsm  \[default: spatial] |
| `--annotation TEXT` | Name of the annotation in adata.obs to use |
| `--homolog-file FILE` | Path to homologous gene conversion file |
| `--species TEXT` | Species-symbol column in the homolog file (for example MOUSE_GENE_SYM) |
| `--latent-representation-niche TEXT` | Key for spatial niche embedding in obsm |
| `--latent-representation-cell TEXT` | Key for cell identity embedding in obsm  \[default: emb_cell] |
| `--high-quality-cell-qc / --no-high-quality-cell-qc` | Opt in to annotation module-score QC (not sequencing-quality QC). Computes DEGs and module scores; disabled by default.  \[default: no-high-quality-cell-qc] |
| `--n-neighbors INTEGER RANGE` | Number of external neighbors for corrected LGCN graphs  \[default: 10; 1<=x<=50] |
| `--graph-mode TEXT` | Spatial graph mode: symmetric_knn (default), radius, weighted_radius, sectionwise_2d, section_aware_3d, no_graph, or revision-only legacy_bug/legacy_symmetric  \[default: symmetric_knn] |
| `--graph-radius FLOAT RANGE` | Physical radius for radius graph modes  \[x>=0.0] |
| `--graph-sigma FLOAT RANGE` | Gaussian sigma for weighted-radius edges  \[x>=0.0] |
| `--graph-z-scale FLOAT RANGE` | Multiplier converting the stored z coordinate to the physical XY units for section_aware_3d graphs  \[default: 1.0; x>=0.0] |
| `--inference-chunk-size INTEGER RANGE` | Target number of focal spots per section-aligned inference chunk. Physical sections remain intact, all spots are retained, and section-aware chunks include graph halos.  \[default: 250000; x>=1] |
| `--two-stage / --single-stage` | Tune the cell embeddings based on the provided annotation  \[default: single-stage] |
| `--n-cell-training INTEGER RANGE` | Number of cells used for training  \[default: 100000; 1000<=x<=1000000] |
| `--expression-data-type [counts\|noncounts]` | Expression QC contract: counts requires finite nonnegative values; noncounts permits finite signed values with external embeddings, counts distinct nonzero features for min_genes, and requires --no-expression-fraction.  \[default: counts] |
| `--spatial-neighbors INTEGER RANGE` | k1: Number of spatial neighbors in it's own slice for spatial dataset  \[default: 301; 10<=x<=5000] |
| `--homogeneous-neighbors INTEGER RANGE` | k3: Number of homogeneous neighbors per cell (for spatial) or KNN neighbors (for scRNA-seq)  \[default: 21; 1<=x<=200] |
| `--cell-embedding-similarity-threshold FLOAT RANGE` | Minimum similarity threshold for cell embedding.  \[default: 0.0; 0.0<=x<=1.0] |
| `--spatial-domain-similarity-threshold FLOAT RANGE` | Minimum similarity threshold for spatial domain embedding.  \[default: 0.6; 0.0<=x<=1.0] |
| `--homogeneous-selection-mode [soft_priority\|hard_niche_filter]` | How niche similarity is used: soft_priority ranks passing neighbors first and fills remaining slots; hard_niche_filter reproduces the historical rule by assigning zero marker-score weight below the niche threshold.  \[default: soft_priority] |
| `--no-expression-fraction` | Skip expression fraction filtering |
| `--adjacent-slice-spatial-neighbors INTEGER RANGE` | Number of spatial neighbors to find on each adjacent slice for 3D data  \[default: 200; 10<=x<=2000] |
| `--n-adjacent-slices INTEGER RANGE` | Adjacent sections on each side (default 2). Set 0 to disable cross-section homogeneous matching while retaining XYZ distances. Automatically 0 for a single spatial3D volume. Does not disable an explicit section-aware embedding graph.  \[default: 2; 0<=x<=100] |
| `--cross-slice-selection-mode [ordinal\|physical_window]` | How cross-section candidates are chosen: by ordinal input order (legacy) or by a measured physical z-distance window.  \[default: ordinal] |
| `--cross-slice-max-distance FLOAT RANGE` | Maximum absolute physical z distance for physical-window section selection. Required when cross_slice_selection_mode=physical_window.  \[x>=0.0] |
| `--cross-slice-z-scale FLOAT RANGE` | Physical units per registered z-coordinate unit. For example, use 4 for the E11.5 z_correction coordinates to express the window in micrometres.  \[default: 1.0; x>=0.0] |
| `--cross-slice-marker-score-strategy [global_pool\|per_slice_pool\|hierarchical_pool]` | Strategy for computing marker scores across slices in spatial3D datasets. 'global_pool': Select the top K most similar neighbors globally across all slices combined. 'per_slice_pool': Select a fixed number of neighbors (K) from each slice independently, then compute a single weighted average score from all selected neighbors. 'hierarchical_pool': Compute an independent marker score for each slice using its top K neighbors, then take the average of these per-slice scores.  \[default: hierarchical_pool] |
| `--high-quality-neighbor-filter / --no-high-quality-filter` | Only find neighbors within high quality cells (requires High_quality column in obs)  \[default: no-high-quality-filter] |
| `--use-gpu / --no-gpu` | Use GPU rather than CPU for spatial LDSC  \[default: use-gpu] |
| `--memmap-tmp-dir DIRECTORY` | Optional fast scratch directory. Concurrent spatial-LDSC processes stage and reuse one content-addressed, read-only marker cache per directory/node. If it does not fit, gsMap streams directly from project storage. |
| `--mkscore-batch-size INTEGER RANGE` | Cells per marker-score batch; 0 selects automatically  \[default: 0; 0<=x<=1000] |
| `--rank-read-workers INTEGER RANGE` | Threads used to materialize distinct rank rows for each cache window  \[default: 120; 1<=x<=128] |
| `--mkscore-compute-workers INTEGER RANGE` | Marker-score compute workers  \[default: 4; 1<=x<=16] |
| `--mkscore-write-workers INTEGER RANGE` | Marker-score and rank write workers  \[default: 2; 1<=x<=50] |
| `--compute-input-queue-size INTEGER RANGE` | Maximum pending marker-score batches per compute worker  \[default: 1; 1<=x<=10] |
| `--writer-queue-size INTEGER RANGE` | Writer-input queue capacity  \[default: 20; 10<=x<=500] |
| `--ldsc-num-gpus INTEGER RANGE` | Number of visible GPUs for spatial LDSC on this node. Cells are split between GPUs; each GPU processes all traits. Requires --use-gpu.  \[default: 1; x>=1] |
| `--ldsc-share-ld / --no-ldsc-share-ld` | Share each LD-score chunk across GWAS traits. Required for multiple GPUs.  \[default: ldsc-share-ld] |
| `--ldsc-regression-backend [auto\|jax\|moments]` | Regression backend: auto uses fused moments for two baseline annotations plus intercept, otherwise JAX. jax disables this fast path.  \[default: auto] |
| `--ldsc-trait-batch-size INTEGER RANGE` | Traits per regression group in shared-LD mode; all requested traits are processed, regardless of this group size.  \[default: 4; 1<=x<=16] |
| `--ldsc-read-workers INTEGER RANGE` | Reader/preparation workers. Shared-LD mode uses at most four readers for NumPy/memmap markers and one for shared file handles; non-shared mode uses this for sparse-transform or marker-preparation workers.  \[default: 10; x>=1] |
| `--ldsc-gwas-workers INTEGER RANGE` | Concurrent GWAS file preparation in shared-LD mode (bounded prefetch). Use fewer workers to reduce temporary memory; at most 16 files are in flight.  \[default: 16; 1<=x<=16] |
| `--ldsc-compute-workers INTEGER RANGE` | Requested CPU compute threads for the automatic moment backend, capped by available cores and thread limits; regression workers in non-shared mode. Does not select the number of GPUs.  \[default: 10; x>=1] |
| `--spots-per-chunk-quick-mode INTEGER RANGE` | Spots per spatial-LDSC compute chunk. Omit to select automatically for the execution device; an explicit value overrides both CPU and GPU defaults.  \[default: (auto: CPU=400, GPU=100); x>=1] |
| `--trait-name TEXT` | Name of the trait for GWAS analysis |
| `--sumstats-file FILE` | Path to GWAS summary statistics file |
| `--sumstats-config-file FILE` | Path to sumstats config file |
| `--w-ld-dir DIRECTORY` | Directory containing the weights files (w_ld) |
| `--additional-baseline-h5ad-path-list FILE` | List of additional baseline h5ad paths |
| `--chisq-max INTEGER` | Maximum chi-square value |
| `--cell-indices-range <INTEGER INTEGER>...` | 0-based range [start, end) of cell indices to process |
| `--sample-filter TEXT` | Filter processing to a specific sample |
| `--n-blocks INTEGER RANGE` | Number of jackknife blocks  \[default: 200; x>=2] |
| `--snp-gene-weight-adata-path FILE` | Path to the SNP-gene weight matrix (H5AD format) |
| `--ldscore-h5ad-path FILE` | Path to a pre-computed SNP × Cell LD score h5ad file (obs=SNPs, var=cells). When provided, skips on-the-fly LD score computation from the SNP-gene weight matrix. additional_baseline_h5ad_path_list must also be provided to supply all baseline annotations. |
| `--marker-score-feather-path FILE` | Path to marker score feather file |
| `--marker-score-h5ad-path FILE` | Path to marker score h5ad file |
| `--marker-score-format [memmap\|feather\|h5ad]` | Format of marker scores |
| `--cauchy-annotations TEXT` | List of annotations in adata.obs to use |
| `--export-spatialvista / --no-export-spatialvista` | Automatically export a SpatialVista H5AD for spatial3D reports; ignored for spatial2D and scRNA  \[default: export-spatialvista] |
| `--spatialvista-max-spots INTEGER RANGE` | Optional deterministic sampling limit for the SpatialVista H5AD; default exports all result spots, independently of HTML sampling  \[x>=1] |
| `--downsampling-n-spots-pcc INTEGER RANGE` | Number of spots to downsample for PCC calculation if n_spots > this value  \[default: 20000; 1000<=x<=100000] |
| `--downsampling-n-spots-umap INTEGER RANGE` | Maximum representative spots used to calculate and display each report UMAP. Valid precomputed UMAP coordinates are sampled to the same limit.  \[default: 50000; 1000<=x<=500000] |
| `--downsampling-n-spots-3d INTEGER RANGE` | Maximum representative spots rendered in the browser's 3D view. Larger datasets use deterministic annotation-aware sampling.  \[default: 200000; 1000<=x<=2000000] |
| `--downsampling-n-spots-2d INTEGER RANGE` | Maximum representative spots per sample in the browser's 2D view. Larger samples use deterministic annotation-aware sampling.  \[default: 300000; 10000<=x<=500000] |
| `--top-corr-genes INTEGER RANGE` | Number of top correlated genes per trait for expression/GSS diagnostic plots and Manhattan highlights  \[default: 10; 1<=x<=500] |
| `--plot-origin TEXT` | Plot origin for spatial plots ('upper' or 'lower'). 'upper' will flip the y-axis (standard for images).  \[default: upper] |
| `--start-step TEXT` | Step to start execution from (find_latent, latent2gene, spatial_ldsc, cauchy, report)  \[default: find_latent] |
| `--stop-step TEXT` | Step to stop execution at (inclusive) |
| `--help` | Show this message and exit. |

## `gsmap find-latent`

Find latent representations of each spot using Graph Neural Networks.

This step:
- Loads spatial transcriptomics data
- Builds neighborhood graphs
- Learns latent representations using GNN
- Saves the model and embeddings

| Option | Description |
|---|---|
| `--workdir DIRECTORY` | Path to the working directory  \[required] |
| `--project-name TEXT` | Name of the project  \[required] |
| `--dataset-type [scrna\|spatial2d\|spatial3d]` | spatial2D (XY) or spatial3D (XYZ sections or a thick-tissue volume)  \[default: spatial2D] |
| `--h5ad-path PATH` | Space-separated list of h5ad file paths. Sample names are derived from file names without suffix. |
| `--h5ad-yaml FILE` | YAML file with sample names and h5ad paths |
| `--h5ad-list-file FILE` | Each row is a h5ad file path, sample name is the file name without suffix |
| `--data-layer TEXT` | Gene expression raw counts data layer in h5ad layers, e.g., 'count', 'counts'. Other wise use 'X' for adata.X  \[default: X] |
| `--cell-qc-min-genes INTEGER RANGE` | Minimum positive-expression genes before HVG, training, graph and inference  \[default: 20; x>=0] |
| `--cell-qc-min-counts FLOAT RANGE` | Optional minimum total raw counts for front-end cell QC  \[x>=0] |
| `--cell-qc-chunk-size INTEGER RANGE` | Rows per bounded expression QC/read block  \[default: 8192; x>=1] |
| `--spatial-key TEXT` | Spatial key in adata.obsm storing spatial coordinates  \[default: spatial] |
| `--annotation TEXT` | Annotation of cell type in adata.obs to use |
| `--homolog-file FILE` | Path to homologous gene conversion file |
| `--species TEXT` | Species-symbol column in the homolog file (for example MOUSE_GENE_SYM) |
| `--latent-representation-niche TEXT` | Key for spatial niche embedding in obsm  \[default: emb_niche] |
| `--latent-representation-cell TEXT` | Key for cell identity embedding in obsm  \[default: emb_cell] |
| `--high-quality-cell-qc / --no-high-quality-cell-qc` | Opt in to annotation module-score QC (not sequencing-quality QC). Computes DEGs and module scores; disabled by default.  \[default: no-high-quality-cell-qc] |
| `--feat-cell INTEGER RANGE` | Number of top variable features to retain  \[default: 2000; 100<=x<=10000] |
| `--n-neighbors INTEGER RANGE` | Number of external neighbors for corrected LGCN graphs  \[default: 10; 1<=x<=50] |
| `--graph-mode TEXT` | Spatial graph mode: symmetric_knn (default), radius, weighted_radius, sectionwise_2d, section_aware_3d, no_graph, or revision-only legacy_bug/legacy_symmetric  \[default: symmetric_knn] |
| `--graph-radius FLOAT RANGE` | Physical radius for radius graph modes  \[x>=0.0] |
| `--graph-sigma FLOAT RANGE` | Gaussian sigma for weighted-radius edges  \[x>=0.0] |
| `--graph-z-scale FLOAT RANGE` | Multiplier converting the stored z coordinate to the physical XY units for section_aware_3d graphs  \[default: 1.0; x>=0.0] |
| `--k INTEGER RANGE` | Graph convolution depth for LGCN  \[default: 3; 1<=x<=10] |
| `--inference-chunk-size INTEGER RANGE` | Target number of focal spots per section-aligned inference chunk. Physical sections remain intact, all spots are retained, and section-aware chunks include graph halos.  \[default: 250000; x>=1] |
| `--hidden-size INTEGER RANGE` | Units in the first hidden layer  \[default: 128; 32<=x<=512] |
| `--embedding-size INTEGER RANGE` | Size of the latent embedding layer  \[default: 32; 8<=x<=128] |
| `--use-tf` | Enable transformer module |
| `--module-dim INTEGER RANGE` | Dimensionality of transformer modules  \[default: 30; 10<=x<=100] |
| `--hidden-gmf INTEGER RANGE` | Hidden units for global mean feature extractor  \[default: 128; 32<=x<=512] |
| `--n-modules INTEGER RANGE` | Number of transformer modules  \[default: 16; 4<=x<=64] |
| `--nhead INTEGER RANGE` | Number of attention heads in transformer  \[default: 4; 1<=x<=16] |
| `--n-enc-layer INTEGER RANGE` | Number of transformer encoder layers  \[default: 2; 1<=x<=8] |
| `--distribution TEXT` | Distribution type for loss calculation  \[default: nb] |
| `--batch-size INTEGER RANGE` | Batch size for training  \[default: 1024; 32<=x<=4096] |
| `--itermax INTEGER RANGE` | Maximum number of training iterations  \[default: 100; 10<=x<=1000] |
| `--patience INTEGER RANGE` | Early stopping patience  \[default: 10; 1<=x<=50] |
| `--random-seed INTEGER RANGE` | Random seed for training and sampling  \[default: 2024; x>=0] |
| `--two-stage / --single-stage` | Tune the cell embeddings based on the provided annotation  \[default: single-stage] |
| `--do-sampling / --no-sampling` | Down-sample cells used by the training loss; spatial features are still propagated on each complete section graph  \[default: do-sampling] |
| `--n-cell-training INTEGER RANGE` | Number of cells used for training  \[default: 100000; 1000<=x<=1000000] |
| `--help` | Show this message and exit. |

## `gsmap latent-to-gene`

Estimate gene marker scores for each spot using latent representations.

This step:
- Loads latent representations
- Estimates gene marker scores
- Performs spatial smoothing
- Saves marker scores for LDSC

| Option | Description |
|---|---|
| `--workdir DIRECTORY` | Path to the working directory  \[required] |
| `--project-name TEXT` | Name of the project  \[required] |
| `--dataset-type [scrna\|spatial2d\|spatial3d]` | Type of dataset: scRNA, spatial2D (XY), or spatial3D (XYZ sections or a single thick-tissue volume)  \[default: spatial2D] |
| `--h5ad-path PATH` | Space-separated list of h5ad file paths. Sample names are derived from file names without suffix. |
| `--h5ad-yaml FILE` | YAML file with sample names and h5ad paths |
| `--h5ad-list-file FILE` | Each row is a h5ad file path, sample name is the file name without suffix |
| `--annotation TEXT` | Cell type annotation in adata.obs to use. This would constrain finding homogeneous spots within each cell type |
| `--data-layer TEXT` | Expression data layer in h5ad layers, e.g., 'count', 'counts'; use 'X' for adata.X  \[default: X] |
| `--expression-data-type [counts\|noncounts]` | Expression QC contract: counts requires finite nonnegative values; noncounts permits finite signed values with external embeddings, counts distinct nonzero features for min_genes, and requires --no-expression-fraction.  \[default: counts] |
| `--latent-representation-niche TEXT` | Key for spatial niche embedding in obsm |
| `--latent-representation-cell TEXT` | Key for cell identity embedding in obsm  \[default: emb_cell] |
| `--spatial-key TEXT` | Spatial key in adata.obsm  \[default: spatial] |
| `--spatial-neighbors INTEGER RANGE` | k1: Number of spatial neighbors in it's own slice for spatial dataset  \[default: 301; 10<=x<=5000] |
| `--homogeneous-neighbors INTEGER RANGE` | k3: Number of homogeneous neighbors per cell (for spatial) or KNN neighbors (for scRNA-seq)  \[default: 21; 1<=x<=200] |
| `--cell-embedding-similarity-threshold FLOAT RANGE` | Minimum similarity threshold for cell embedding.  \[default: 0.0; 0.0<=x<=1.0] |
| `--spatial-domain-similarity-threshold FLOAT RANGE` | Minimum similarity threshold for spatial domain embedding.  \[default: 0.6; 0.0<=x<=1.0] |
| `--homogeneous-selection-mode [soft_priority\|hard_niche_filter]` | How niche similarity is used: soft_priority ranks passing neighbors first and fills remaining slots; hard_niche_filter reproduces the historical rule by assigning zero marker-score weight below the niche threshold.  \[default: soft_priority] |
| `--no-expression-fraction` | Skip expression fraction filtering |
| `--adjacent-slice-spatial-neighbors INTEGER RANGE` | Number of spatial neighbors to find on each adjacent slice for 3D data  \[default: 200; 10<=x<=2000] |
| `--n-adjacent-slices INTEGER RANGE` | Adjacent sections on each side (default 2). Set 0 to disable cross-section homogeneous matching while retaining XYZ distances. Automatically 0 for a single spatial3D volume. Does not disable an explicit section-aware embedding graph.  \[default: 2; 0<=x<=100] |
| `--cross-slice-selection-mode [ordinal\|physical_window]` | How cross-section candidates are chosen: by ordinal input order (legacy) or by a measured physical z-distance window.  \[default: ordinal] |
| `--cross-slice-max-distance FLOAT RANGE` | Maximum absolute physical z distance for physical-window section selection. Required when cross_slice_selection_mode=physical_window.  \[x>=0.0] |
| `--cross-slice-z-scale FLOAT RANGE` | Physical units per registered z-coordinate unit. For example, use 4 for the E11.5 z_correction coordinates to express the window in micrometres.  \[default: 1.0; x>=0.0] |
| `--cross-slice-marker-score-strategy [global_pool\|per_slice_pool\|hierarchical_pool]` | Strategy for computing marker scores across slices in spatial3D datasets. 'global_pool': Select the top K most similar neighbors globally across all slices combined. 'per_slice_pool': Select a fixed number of neighbors (K) from each slice independently, then compute a single weighted average score from all selected neighbors. 'hierarchical_pool': Compute an independent marker score for each slice using its top K neighbors, then take the average of these per-slice scores.  \[default: hierarchical_pool] |
| `--high-quality-neighbor-filter / --no-high-quality-filter` | Only find neighbors within high quality cells (requires High_quality column in obs)  \[default: no-high-quality-filter] |
| `--use-gpu / --no-gpu` | Use GPU for JAX computations (requires sufficient GPU memory)  \[default: use-gpu] |
| `--memmap-tmp-dir DIRECTORY` | Optional fast scratch directory for memory-mapped files. A file is staged only when the full logical memmap plus safety headroom fits; otherwise gsMap uses the project storage directly. |
| `--mkscore-batch-size INTEGER RANGE` | Cells per marker-score batch; 0 selects automatically  \[default: 0; 0<=x<=1000] |
| `--rank-read-workers INTEGER RANGE` | Threads used to materialize distinct rank rows for each cache window  \[default: 120; 1<=x<=128] |
| `--mkscore-compute-workers INTEGER RANGE` | Marker-score compute workers  \[default: 4; 1<=x<=16] |
| `--mkscore-write-workers INTEGER RANGE` | Marker-score and rank write workers  \[default: 2; 1<=x<=50] |
| `--compute-input-queue-size INTEGER RANGE` | Maximum pending marker-score batches per compute worker  \[default: 1; 1<=x<=10] |
| `--writer-queue-size INTEGER RANGE` | Writer-input queue capacity  \[default: 20; 10<=x<=500] |
| `--help` | Show this message and exit. |

## `gsmap spatial-ldsc`

Run spatial LDSC analysis for genetic association.

This step:
- Loads LD scores and GWAS summary statistics
- Performs spatial LDSC regression
- Computes enrichment statistics
- Saves results for downstream analysis

| Option | Description |
|---|---|
| `--workdir DIRECTORY` | Path to the working directory  \[required] |
| `--project-name TEXT` | Name of the project  \[required] |
| `--use-gpu / --no-gpu` | Use GPU rather than CPU for spatial LDSC  \[default: use-gpu] |
| `--ldsc-num-gpus INTEGER RANGE` | Number of visible GPUs for spatial LDSC on this node. Cells are split between GPUs; each GPU processes all traits. Requires --use-gpu.  \[default: 1; x>=1] |
| `--ldsc-share-ld / --no-ldsc-share-ld` | Share each LD-score chunk across GWAS traits. Required for multiple GPUs.  \[default: ldsc-share-ld] |
| `--ldsc-regression-backend [auto\|jax\|moments]` | Regression backend: auto uses fused moments for two baseline annotations plus intercept, otherwise JAX. jax disables this fast path.  \[default: auto] |
| `--ldsc-trait-batch-size INTEGER RANGE` | Traits per regression group in shared-LD mode; all requested traits are processed, regardless of this group size.  \[default: 4; 1<=x<=16] |
| `--memmap-tmp-dir DIRECTORY` | Optional fast scratch directory. Concurrent spatial-LDSC processes stage and reuse one content-addressed, read-only marker cache per directory/node. If it does not fit, gsMap streams directly from project storage. |
| `--ldsc-read-workers INTEGER RANGE` | Reader/preparation workers. Shared-LD mode uses at most four readers for NumPy/memmap markers and one for shared file handles; non-shared mode uses this for sparse-transform or marker-preparation workers.  \[default: 10; x>=1] |
| `--ldsc-gwas-workers INTEGER RANGE` | Concurrent GWAS file preparation in shared-LD mode (bounded prefetch). Use fewer workers to reduce temporary memory; at most 16 files are in flight.  \[default: 16; 1<=x<=16] |
| `--ldsc-compute-workers INTEGER RANGE` | Requested CPU compute threads for the automatic moment backend, capped by available cores and thread limits; regression workers in non-shared mode. Does not select the number of GPUs.  \[default: 10; x>=1] |
| `--spots-per-chunk-quick-mode INTEGER RANGE` | Spots per spatial-LDSC compute chunk. Omit to select automatically for the execution device; an explicit value overrides both CPU and GPU defaults.  \[default: (auto: CPU=400, GPU=100); x>=1] |
| `--trait-name TEXT` | Name of the trait for GWAS analysis |
| `--sumstats-file FILE` | Path to GWAS summary statistics file |
| `--sumstats-config-file FILE` | Path to sumstats config file |
| `--w-ld-dir DIRECTORY` | Directory containing the weights files (w_ld) |
| `--additional-baseline-h5ad-path-list FILE` | List of additional baseline h5ad paths |
| `--chisq-max INTEGER` | Maximum chi-square value |
| `--cell-indices-range <INTEGER INTEGER>...` | 0-based range [start, end) of cell indices to process |
| `--sample-filter TEXT` | Filter processing to a specific sample |
| `--n-blocks INTEGER RANGE` | Number of jackknife blocks  \[default: 200; x>=2] |
| `--snp-gene-weight-adata-path FILE` | Path to the SNP-gene weight matrix (H5AD format) |
| `--ldscore-h5ad-path FILE` | Path to a pre-computed SNP × Cell LD score h5ad file (obs=SNPs, var=cells). When provided, skips on-the-fly LD score computation from the SNP-gene weight matrix. additional_baseline_h5ad_path_list must also be provided to supply all baseline annotations. |
| `--marker-score-feather-path FILE` | Path to marker score feather file |
| `--marker-score-h5ad-path FILE` | Path to marker score h5ad file |
| `--marker-score-format [memmap\|feather\|h5ad]` | Format of marker scores |
| `--help` | Show this message and exit. |

## `gsmap cauchy-combination`

Run Cauchy combination test to combine spatial LDSC results across spots.

This step:
- Loads spatial LDSC results for a trait
- Removes outliers
- Performs Cauchy combination test for each annotation
- Computes Fisher's exact test for enrichment

| Option | Description |
|---|---|
| `--workdir DIRECTORY` | Path to the working directory  \[required] |
| `--project-name TEXT` | Name of the project  \[required] |
| `--trait-name TEXT` | Name of the trait for GWAS analysis |
| `--sumstats-file FILE` | Path to GWAS summary statistics file |
| `--sumstats-config-file FILE` | Path to sumstats config file |
| `--annotation TEXT` | Name of the annotation in adata.obs to use |
| `--cauchy-annotations TEXT` | List of annotations in adata.obs to use |
| `--help` | Show this message and exit. |

## `gsmap ldscore-weight-matrix`

Compute LD score weight matrices for features.

This command runs the LDScorePipeline to:
- Load genotypes (PLINK)
- Load feature mappings (BED/Dict) or Annotations
- Compute LD-based weights between SNPs and Features
- Save results as AnnData (.h5ad)

| Option | Description |
|---|---|
| `--bfile-root TEXT` | Reference panel prefix template (e.g., 'data/1000G.{chr}')  \[required] |
| `--hm3-snp-dir DIRECTORY` | Directory containing per-chromosome HM3 SNP lists (e.g., hm.{chr}.snp). Typically set to '<ldscore_data_dir>/hapmap3_snps'.  \[required] |
| `--output-dir PATH` | Output directory. If None, uses {workdir}/{project_name}/generate_ldscore |
| `--output-filename TEXT` | Prefix for output files  \[default: ld_score_weights] |
| `--omics-h5ad-path FILE` | Path to omics H5AD file |
| `--mapping-type TEXT` | Mapping type: 'bed' or 'dict'  \[default: bed] |
| `--mapping-file FILE` | Path to mapping file |
| `--annot-file TEXT` | Template for annotation files (e.g., 'baseline.{chr}.annot.gz') |
| `--feature-window-size INTEGER` | bp window for mapping (e.g. TSS window)  \[default: 0] |
| `--strategy TEXT` | Strategy: 'score', 'tss', 'center', 'allow_repeat'  \[default: score] |
| `--ld-wind FLOAT` | LD window size  \[default: 1.0] |
| `--ld-unit TEXT` | LD unit: 'SNP', 'KB', 'CM'  \[default: CM] |
| `--snps-only / --no-snps-only` | Restrict the broad reference set to biallelic single-nucleotide A/C/G/T variants before decoding the BED payload.  \[default: snps-only] |
| `--mac-min INTEGER RANGE` | Minimum minor-allele count for the broad reference SNP set. Variants are retained when MAC is strictly greater than this value; the default therefore implements MAC > 5.  \[default: 5; x>=0] |
| `--maf-min FLOAT RANGE` | Optional minimum MAF filter for the broad reference SNP set. Disabled by default because MAC > 5 is the default QC rule.  [0.0<=x<=0.5] |
| `--exclude-mhc-from-regression / --no-exclude-mhc-from-regression` | Exclude extended-MHC SNPs from HM3 regression targets and the HM3-only weight-LD universe. MHC variants remain in the broad reference SNP set used in annotation LD-score sums.  \[default: exclude-mhc-from-regression] |
| `--mhc-start-bp INTEGER` | GRCh38 extended-MHC start coordinate (inclusive)  \[default: 28510120] |
| `--mhc-end-bp INTEGER` | GRCh38 extended-MHC end coordinate (inclusive)  \[default: 33480577] |
| `--chromosomes TEXT` | Chromosomes to process. 'all' uses 1-22 autosomes, or provide a comma-separated list of chromosomes (e.g., '1,2,3')  \[default: all] |
| `--batch-size-hm3 INTEGER` | Batch size for HM3 SNPs  \[default: 50] |
| `--calculate-w-ld / --no-calculate-w-ld` | Whether to calculate w_ld  \[default: no-calculate-w-ld] |
| `--w-ld-dir PATH` | Directory for w_ld outputs |
| `--help` | Show this message and exit. |

## `gsmap format-sumstats`

Format GWAS summary statistics for gsMap or COJO.

This command:
- Filters SNPs based on INFO, MAF, and P-value
- Converts SNP positions to RSID (if dbsnp is provided)
- Saves formatted summary statistics

| Option | Description |
|---|---|
| `--sumstats TEXT` | Path to gwas summary data  \[required] |
| `--out TEXT` | Path to save the formatted gwas data  \[required] |
| `--snp TEXT` | Name of snp column |
| `--a1 TEXT` | Name of effect allele column |
| `--a2 TEXT` | Name of none-effect allele column |
| `--info TEXT` | Name of info column |
| `--beta TEXT` | Name of gwas beta column. |
| `--se TEXT` | Name of gwas standar error of beta column |
| `--p TEXT` | Name of p-value column |
| `--frq TEXT` | Name of A1 ferquency column |
| `--n TEXT` | Name of sample size column |
| `--z TEXT` | Name of gwas Z-statistics column |
| `--or TEXT` | Name of gwas OR column |
| `--se-or TEXT` | Name of standar error of OR column |
| `--chr TEXT` | Name of SNP chromosome column  \[default: Chr] |
| `--pos TEXT` | Name of SNP positions column  \[default: Pos] |
| `--dbsnp TEXT` | Path to reference dnsnp file |
| `--chunksize INTEGER` | Chunk size for loading dbsnp file  \[default: 1000000] |
| `--format [gsmap\|cojo]` | Format of output data  \[default: gsMap] |
| `--info-min FLOAT` | Minimum INFO score.  \[default: 0.9] |
| `--maf-min FLOAT` | Minimum MAF.  \[default: 0.01] |
| `--keep-chr-pos / --no-keep-chr-pos` | Keep SNP chromosome and position columns in the output data  \[default: no-keep-chr-pos] |
| `--help` | Show this message and exit. |

## `gsmap report-view`

Launch a local web server to view the gsMap report.

This command starts a simple HTTP server to serve the report files,
which is necessary for proper loading of JavaScript modules.

Example:
    gsmap report-view /path/to/project/gsmap_web_report
    gsmap report-view /path/to/project/gsmap_web_report --port 9000

| Option | Description |
|---|---|
| `REPORT_PATH` | Path to gsmap_web_report directory containing index.html  \[required] |
| `--port INTEGER` | Port to serve the report on  \[default: 8080] |
| `--no-browser` | Don't automatically open browser |
| `--help` | Show this message and exit. |
