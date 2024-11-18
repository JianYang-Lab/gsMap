# Mouse Embryo (Step by Step)

This tutorial guides you through running `gsMap` step by step, with user-defined parameters and resources, granting greater flexibility and control over the analysis. This mode is suited for users who require detailed customization of their pipeline. For a faster, one-command execution, please see the {doc}`Quick Mode <quick_mode>` tutorial.

## Preparation

Please ensure you have {doc}`installed <install>` the `gsMap`. This tutorial guides you through using gsMap in a step-by-step manner.

### 1. Download dependencies

`gsMap` requires specific reference files:
- **Gene transfer format (GTF) file**, for gene coordinates on the genome.
- **LD reference panel (PLINK bfile)**, for computing LD scores.
- **SNP weight file**, to adjust correlations between SNP-trait association statistics.
- **Homologous gene transformations file** (optional), to map genes between species.
- **Enhancer-gene mapping file** (optional), for linking SNPs to genes based on enhancer annotations.

To download the resources:
```bash
wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_resource.tar.gz
tar -xvzf gsMap_resource.tar.gz
```

Directory structure:
```bash
tree -L 2

gsMap_resource
    ├── genome_annotation
    │   ├── enhancer
    │   └── gtf
    ├── homologs
    │   ├── macaque_human_homologs.txt
    │   └── mouse_human_homologs.txt
    ├── LD_Reference_Panel
    │   └── 1000G_EUR_Phase3_plink
    ├── LDSC_resource
    │   ├── hapmap3_snps
    │   └── weights_hm3_no_hla
    └── quick_mode
        ├── baseline
        ├── SNP_gene_pair
        └── snp_gene_weight_matrix.h5ad
```
If you want to use your own reference files, please ensure that the genome build versions (e.g., Hg37 or Hg38) are consistent between the GTF file and the LD reference panel.

### 2. Download example data

```bash
wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_example_data.tar.gz
tar -xvzf gsMap_example_data.tar.gz
```

Directory structure:
```bash
tree -L 2

gsMap_example_data/
├── GWAS
│   ├── BCX2_MCHC_EA_GWAMA.sumstats.gz
│   ├── GIANT_EUR_Height_2022_Nature.sumstats.gz
│   ├── gwas_config.yaml
│   └── IQ_NG_2018.sumstats.gz
└── ST
    └── E16.5_E1S1.MOSTA.h5ad
```

## Running `gsMap`

### 1. find latent representations (optional)

**Objective**: Learn latent representations for spots. The latent embedding learned from this step will be store in the AnnData object `obsm` field under the key `latent_GVAE`.

```{note}
The `--workdir` parameter specifies the working directory for gsMap, where all outputs will be saved.
```

**Execution**: <span style="color:#31a354"> required memory: ~80G (120K cells) </span>

```bash
gsmap run_find_latent_representations \
    --workdir './example/Mouse_Embryo' \
    --sample_name 'E16.5_E1S1.MOSTA' \
    --input_hdf5_path 'gsMap_example_data/ST/E16.5_E1S1.MOSTA.h5ad' \
    --annotation 'annotation' \
    --data_layer 'count'
```

### 2. generate gene specificity scores

**Objective**: Identify homogeneous spots for each spot based on their latent representations specified by `--latent_representation`, and then generate gene specificity scores (GSS) for each spot by aggregating information from its homogeneous spots.

```{note}
If your ST data is not from a human species but you want to map human GWAS data to it, please provide a homologous transformation file to convert gene names. The first column should list gene names from the ST data species, and the second column from the GWAS data species.
```

**Execution**: <span style="color:#31a354"> required memory: ~45G (120K cells) </span>

```bash
gsmap run_latent_to_gene \
    --workdir './example/Mouse_Embryo' \
    --sample_name 'E16.5_E1S1.MOSTA' \
    --annotation 'annotation' \
    --latent_representation 'latent_GVAE' \
    --num_neighbour 51 \
    --num_neighbour_spatial 201 \
    --homolog_file 'gsMap_resource/homologs/mouse_human_homologs.txt'
```

### 3. generate ldscore

**Objective**: Assign gene specificity scores (GSS) to SNPs and compute the stratified LD score.

**Execution**: <span style="color:#31a354"> required memory: ~40G </span>

**Three SNP to gene linking strategies are available:**

````{tab} 1. Use TSS
This strategy uses TSS to assign GSS to SNPs. The --`gene_window_size parameter` defines the window size around the gene body for this assignment. If a SNP falls within the window of multiple genes, the GSS from the nearest gene will be used.

```bash
for CHROM in {1..22}
do
    gsmap run_generate_ldscore \
        --workdir './example/Mouse_Embryo' \
        --sample_name 'E16.5_E1S1.MOSTA' \
        --chrom $CHROM \
        --bfile_root 'gsMap_resource/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC' \
        --keep_snp_root 'gsMap_resource/LDSC_resource/hapmap3_snps/hm' \
        --gtf_annotation_file 'gsMap_resource/genome_annotation/gtf/gencode.v39lift37.annotation.gtf' \
        --gene_window_size 50000
done
```
````

`````{tab} 2. Use Enhancer-Gene Linking
This strategy uses enhancer-gene linking to assign GSS to SNPs. When a SNP maps to multiple enhancers, the GSS for the SNP is determined by the `--snp_multiple_enhancer_strategy` parameter. By default, this is set to `max_mkscore`, which assigns the SNP the maximum GSS among the enhancers it maps to. Another option is `nearest_TSS`.

```bash
for CHROM in {1..22}
do
    gsmap run_generate_ldscore \
        --workdir './example/Mouse_Embryo' \
        --sample_name 'E16.5_E1S1.MOSTA' \
        --chrom $CHROM \
        --bfile_root 'gsMap_resource/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC' \
        --keep_snp_root 'gsMap_resource/LDSC_resource/hapmap3_snps/hm' \
        --gtf_annotation_file 'gsMap_resource/genome_annotation/gtf/gencode.v39lift37.annotation.gtf' \
        --enhancer_annotation_file 'gsMap_resource/genome_annotation/enhancer/by_tissue/ALL/ABC_roadmap_merged.bed' \
        --snp_multiple_enhancer_strategy 'max_mkscore' \
        --gene_window_enhancer_priority 'enhancer_only'
done
```
`````

`````{tab} 3. Use TSS and Enhancer-Gene Linking
This strategy uses both TSS and enhancer-gene linking to assign GSS to SNPs. If a SNP maps to both a gene TSS window and an enhancer linked to a different gene, the `--gene_window_enhancer_priority` parameter decides which gene the SNP is assigned to. The options are `gene_window_first` or `enhancer_first`.

```bash
for CHROM in {1..22}
do
    gsmap run_generate_ldscore \
        --workdir './example/Mouse_Embryo' \
        --sample_name 'E16.5_E1S1.MOSTA' \
        --chrom $CHROM \
        --bfile_root 'gsMap_resource/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC' \
        --keep_snp_root 'gsMap_resource/LDSC_resource/hapmap3_snps/hm' \
        --gtf_annotation_file 'gsMap_resource/genome_annotation/gtf/gencode.v39lift37.annotation.gtf' \
        --gene_window_size 50000 \
        --enhancer_annotation_file 'gsMap_resource/genome_annotation/enhancer/by_tissue/ALL/ABC_roadmap_merged.bed' \
        --snp_multiple_enhancer_strategy 'max_mkscore' \
        --gene_window_enhancer_priority 'gene_window_first'
done
```
`````

```{caution}
If you run out of memory during this step or the next, you can reduce the `--spots_per_chunk` parameter to a smaller value. Generally, 40GB of memory is required when `--spots_per_chunk` is set to 1000.
```

### 4. spatial ldsc

**Objective**: Run spatial LDSC to associate spots with traits. 

**Execution**: <span style="color:#31a354"> required memory: ~40G </span>

```bash
gsmap run_spatial_ldsc \
    --workdir './example/Mouse_Embryo' \
    --sample_name 'E16.5_E1S1.MOSTA' \
    --trait_name 'IQ' \
    --sumstats_file 'gsMap_example_data/GWAS/IQ_NG_2018.sumstats.gz' \
    --w_file 'gsMap_resource/LDSC_resource/weights_hm3_no_hla/weights.' \
    --num_processes 4
```


### 5. cauchy combination (optional)

**Objective**: Aggregate P values of individual spots within specific spatial regions (cell types) to evaluate the association of these regions (cell types) with the trait.

**Execution**: <span style="color:#31a354"> required memory: ~12G </span>

```bash
gsmap run_cauchy_combination \
    --workdir './example/Mouse_Embryo' \
    --sample_name 'E16.5_E1S1.MOSTA' \
    --trait_name 'IQ' \
    --annotation 'annotation'
```

### 6. report generation (optional)

**Objective**: Generate gsMap reports, including visualizations of mapping results and diagnostic plots.

```{note}
The default genes for visualization are the top 50 genes whose GSS shows the highest correlation with the -log10 p-values of the trait-cell associations. To select specific genes for visualization, use the `--selected_genes` parameter.
```

**Execution**: <span style="color:#31a354"> required memory: ~50G </span>

```bash
gsmap run_report \
    --workdir './example/Mouse_Embryo' \
    --sample_name 'E16.5_E1S1.MOSTA' \
    --trait_name 'IQ' \
    --annotation 'annotation' \
    --sumstats_file 'gsMap_example_data/GWAS/IQ_NG_2018.sumstats.gz' \
    --top_corr_genes 50
```


### 7. conditional analysis (optional)
**Objective**: Perform conditional analysis by adjusting for other functional annotations or cell-type-level annotations. 

This step extends `step 3: generate ldscore`, by adding additional functional annotations to the baseline with the aim of conducting a conditional analysis. The directory of additional annotations can be specified using the parameter `--additional_baseline_annotation`. The other steps are same to the tutorials above. 

Download the additional annotations:
```bash
wget https://yanglab.westlake.edu.cn/data/gsMap/gsMap_additional_annotation.tar.gz
tar -xvzf gsMap_additional_annotation.tar.gz
```
The format of the additional annotation files is such that each line represents a SNP, with columns indicating the annotation values for that SNP. These values can be either binary or continuous.
```bash
zless -S gsMap_additional_annotation/baseline.1.annot.gz
```

**Execution**: <span style="color:#31a354"> required memory: ~50G </span>


```bash
for CHROM in {1..22}
do
    gsmap run_generate_ldscore \
        --workdir './example/Mouse_Embryo' \
        --sample_name 'E16.5_E1S1.MOSTA' \
        --chrom $CHROM \
        --bfile_root 'gsMap_resource/LD_Reference_Panel/1000G_EUR_Phase3_plink/1000G.EUR.QC' \
        --keep_snp_root 'gsMap_resource/LDSC_resource/hapmap3_snps/hm' \
        --gtf_annotation_file 'gsMap_resource/genome_annotation/gtf/gencode.v39lift37.annotation.gtf' \
        --gene_window_size 50000 \
        --additional_baseline_annotation 'gsMap_additional_annotation'
done
```

### 8. create slice mean for multiple samples (optional)
**Objective**: Generate slice mean for multiple samples, and then use this gene expression mean to calculate the GSS. 

```bash
gsmap create_slice_mean \
    --workdir './workdir' \
    --sample_name_list 'sample1' 'sample2' 'sample3' \
    --h5ad_list 'sample1.h5ad' 'sample2.h5ad' 'sample3.h5ad' \
    --slice_mean_output_file  './workdir/sample_slice_mean.parquet' \
    --annotation 'annotation'

# Use the slice mean to calculate the GSS
gsmap run_latent_to_gene \
    --workdir './workdir' \
    --sample_name 'sample1' \
    --annotation 'annotation' \
    --gM_slices './workdir/sample_slice_mean.parquet' \
    --latent_representation 'latent_GVAE'
```
