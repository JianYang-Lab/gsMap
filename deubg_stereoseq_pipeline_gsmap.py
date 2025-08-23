#!/usr/bin/env python3
"""
Simplified Python script for local debugging of gsMap pipeline.
This directly calls the Python functions from gsMap modules.
"""

import sys
import glob
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
import argparse
import yaml

# Add gsMap to path
sys.path.append('/mnt/d/01_Project/01_Research/202312_gsMap/src/gsMap_develop/src')
sys.path.append('/storage/yangjianLab/chenwenhao/01_Project/01_Research/202312_gsMap/src/gsMap_develop/src')

from gsMap.config import (
    FindLatentRepresentationsConfig,
    LatentToGeneConfig,
    MaxPoolingConfig,
    RunLinkModeConfig,
    ThreeDCombineConfig,
)
from gsMap.find_latent_representation import run_find_latent_representation
from gsMap.latent_to_gene_gnn import run_latent_to_gene
from gsMap.latent_to_gene_gnn_zarr_refactored import MarkerScoreCalculator
from gsMap.latent_to_gene_gnn_zarr_refactored import LatentToGeneConfig as RefactoredLatentToGeneConfig
from gsMap.max_pooling import run_max_pooling
from gsMap.run_link_mode import run_pipeline_link
from gsMap.three_d_combine import three_d_combine


@dataclass
class PipelineConfig:
    """Configuration for the gsMap pipeline"""

    # Base paths
    workdir: str | Path = "/mnt/d/01_Project/01_Research/202312_gsMap/experiment/20250807_refactor_for_gsmap3d/02_latent2gene_optmization_max_pooling/01_mouse_E9.5_dev_v1"
    data_root: str = "/mnt/d/01_Project/01_Research/202312_gsMap/experiment/20250807_refactor_for_gsmap3d/02_latent2gene_optmization_max_pooling/Mouse_E9.5"
    gsmap_resource: str = "/mnt/d/01_Project/01_Research/202312_gsMap/data/gsMap_dev_data/online_resource/gsMap_resource"
    gwas_summary: str = "/mnt/d/01_Project/01_Research/202312_gsMap/experiment/20250807_refactor_for_gsmap3d/02_latent2gene_optmization_max_pooling/mouse_e9_5_gwas_config.yaml"

    # Project settings
    project_name: str = "202508115_Mouse_E9.5_dev_v5"
    annotation: str = "mapped_celltype"
    spatial_key: str = "spatial"

    # Processing parameters
    n_cell_training: int = 100000
    data_layer: str = "X"
    homolog_file: str = "/mnt/d/01_Project/01_Research/202312_gsMap/data/gsMap_dev_data/online_resource/gsMap_resource/homologs/mouse_human_homologs.txt"

    # 3D visualization
    adata_3d: str = "/storage/yangjianLab/songliyang/SpatialData/gsMap_analysis/MouseBrain_StereoSeq/MouseBrain_StereoSeq.meta.parquet"
    background_color: str = "white"
    spatial_key_3d: str = "3d_align_spatial"
    st_id: str = "st_id"

    # Additional parameters
    max_processes: int = 3
    
    # Refactored latent_to_gene parameters
    use_refactored_latent_to_gene: bool = True
    batch_size: int = 1000
    num_read_workers: int = 4
    gpu_batch_size: int = 300  # Smaller batch size for GPU to avoid OOM


def setup_directories(config: PipelineConfig):
    """Create necessary directories"""
    Path(config.workdir).mkdir(parents=True, exist_ok=True)
    Path(f"{config.workdir}/list").mkdir(parents=True, exist_ok=True)
    Path(f"{config.workdir}/{config.project_name}").mkdir(parents=True, exist_ok=True)


def get_sample_list(config: PipelineConfig) -> List[str]:
    """Get list of h5ad files for processing"""
    pattern = f"{config.data_root}/*.h5ad"
    files = glob.glob(pattern)
    return files


def step1_find_latent_representations(config: PipelineConfig):
    """Find latent representations using LGCN"""
    print("=" * 80)
    print("Step 1: Finding Latent Representations")
    print("=" * 80)

    # Create file list
    file_list_path = f"{config.workdir}/list/{config.project_name}_list"
    # files = get_sample_list(config)
    #
    # with open(file_list_path, 'w') as f:
    #     for file in files:
    #         f.write(f"{file}\n")

    # Create config for FindLatentRepresentations
    latent_config = FindLatentRepresentationsConfig(
        spe_file_list=file_list_path,
        workdir=config.workdir,
        project_name=config.project_name,
        annotation=config.annotation,
        spatial_key=config.spatial_key,
        data_layer=config.data_layer,
        homolog_file=config.homolog_file,
        n_cell_training=config.n_cell_training,
        sample_name="all",
        use_tf=False,
    )

    # Run the function
    run_find_latent_representation(latent_config)
    print("Latent representations completed!")


def step2_calculate_gss(config: PipelineConfig, sample_name: Optional[str] = None):
    """Calculate Gene Signature Scores (GSS)"""
    print("=" * 80)
    print("Step 2: Calculating GSS")
    print("=" * 80)

    if config.use_refactored_latent_to_gene:
        # Use the refactored version that processes all samples at once
        print("Using refactored JAX-accelerated latent_to_gene implementation")
        
        latent_dir = Path(config.workdir) / config.project_name / "find_latent_representations"
        rank_zarr_path = latent_dir / "ranks.zarr"
        output_path = Path(config.workdir) / config.project_name / "latent_to_gene" / "marker_scores.zarr"
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create refactored config
        refactored_config = RefactoredLatentToGeneConfig(
            latent_dir=str(latent_dir),
            rank_zarr_path=str(rank_zarr_path),
            output_path=str(output_path),
            latent_representation="emb_gcn",
            latent_representation_indv="emb",
            spatial_key=config.spatial_key,
            annotation_key=config.annotation,
            num_neighbour_spatial=201,
            num_anchor=51,
            num_neighbour=21,
            batch_size=config.batch_size,
            num_read_workers=config.num_read_workers,
            gpu_batch_size=config.gpu_batch_size
        )

        # Run the refactored calculator
        calculator = MarkerScoreCalculator(refactored_config)
        calculator.run()
        print("GSS calculation completed for all samples!")
        
    else:
        # Use the original version
        file_list_path = f"{config.workdir}/list/{config.project_name}_list"

        with open(file_list_path, 'r') as f:
            samples = [Path(line.strip()).stem for line in f]

        # If specific sample is provided, only process that one
        if sample_name:
            samples = [sample_name] if sample_name in samples else []

        for sample in samples:
            print(f"Processing sample: {sample}")
            mk_file = f"{config.workdir}/{config.project_name}/latent_to_gene/mk_score/{sample}_gene_marker_score.feather"

            if not Path(mk_file).exists():
                # Create config for LatentToGene
                gss_config = LatentToGeneConfig(
                    workdir=config.workdir,
                    project_name=config.project_name,
                    sample_name=sample,
                    annotation=config.annotation,
                    spatial_key=config.spatial_key
                )

                # Run the function
                run_latent_to_gene(gss_config)
                print(f"GSS calculation completed for {sample}!")


def step3_max_pooling(config: PipelineConfig, sample_name: Optional[str] = None):
    """Apply max pooling to GSS"""
    print("=" * 80)
    print("Step 3: Max Pooling")
    print("=" * 80)

    file_list_path = f"{config.workdir}/list/{config.project_name}_list"

    with open(file_list_path, 'r') as f:
        samples = [Path(line.strip()).stem for line in f]

    # If specific sample is provided, only process that one
    if sample_name:
        samples = [sample_name] if sample_name in samples else []

    for sample in samples:
        print(f"Processing sample: {sample}")

        # Create config for MaxPooling
        pooling_config = MaxPoolingConfig(
            workdir=config.workdir,
            project_name=config.project_name,
            sample_name=sample,
            spe_file_list=file_list_path,
            annotation=config.annotation,
            spatial_key=config.spatial_key
        )

        # Run the function
        run_max_pooling(pooling_config)
        print(f"Max pooling completed for {sample}!")


def step4_spatial_ldsc(config: PipelineConfig, sample_name: Optional[str] = None,
                       trait_name: Optional[str] = None):
    """Run spatial LDSC analysis"""
    print("=" * 80)
    print("Step 4: Spatial LDSC")
    print("=" * 80)

    # Load GWAS summary
    with open(config.gwas_summary, 'r') as f:
        gwas_data = yaml.safe_load(f)

    # Filter traits if specified
    if trait_name:
        gwas_data = {k: v for k, v in gwas_data.items() if k == trait_name}
    else:
        # Default filter for MB_h traits
        gwas_data = {k: v for k, v in gwas_data.items()}

    file_list_path = f"{config.workdir}/list/{config.project_name}_list"

    with open(file_list_path, 'r') as f:
        h5ad_files = [line.strip() for line in f]

    # Filter samples if specified
    if sample_name:
        h5ad_files = [f for f in h5ad_files if Path(f).stem == sample_name]
    else:
        with open(file_list_path, 'r') as f:
            h5ad_files = [line.strip() for line in f]

    for trait, sumstats_path in gwas_data.items():
        gwas_file = Path(sumstats_path).stem

        for h5ad_file in h5ad_files:
            sample = Path(h5ad_file).stem
            print(f"Processing {sample} for trait {trait}")

            out_file = f"{config.workdir}/{config.project_name}/spatial_ldsc/{sample}/{sample}_{gwas_file}.csv.gz"

            if not Path(out_file).exists():
                # Create config for RunLinkMode
                link_config = RunLinkModeConfig(
                    workdir=config.workdir,
                    project_name=config.project_name,
                    sample_name=sample,
                    spatial_key=config.spatial_key,
                    annotation=config.annotation,
                    gsMap_resource_dir=config.gsmap_resource,
                    trait_name=gwas_file,
                    sumstats_file=sumstats_path,
                    max_processes=config.max_processes
                )

                # Run the function
                run_pipeline_link(link_config)
                print(f"LDSC completed for {sample} - {trait}!")


def step5_3d_visualization(config: PipelineConfig, trait_name: Optional[str] = None):
    """Generate 3D visualizations"""
    print("=" * 80)
    print("Step 5: 3D Visualization")
    print("=" * 80)

    # Load GWAS summary
    with open(config.gwas_summary, 'r') as f:
        gwas_data = yaml.safe_load(f)

    # Filter traits if specified
    if trait_name:
        traits = [trait_name] if trait_name in gwas_data else []
    else:
        # Default filter for MB_h traits
        traits = [k for k in gwas_data.keys()]

    for trait in traits:
        print(f"Generating 3D visualization for {trait}")

        # Create config for 3D combine
        viz_config = ThreeDCombineConfig(
            workdir=config.workdir,
            project_name=config.project_name,
            trait_name=trait,
            adata_3d=config.adata_3d,
            annotation="major_brain_region",
            background_color=config.background_color,
            spatial_key=config.spatial_key_3d,
            st_id=config.st_id
        )

        # Run the function
        three_d_combine(viz_config)
        print(f"3D visualization completed for {trait}!")


def run_full_pipeline(config: PipelineConfig):
    """Run the complete pipeline"""
    print("\n" + "=" * 80)
    print("Starting Full Pipeline")
    print("=" * 80 + "\n")

    # Step 1: Find latent representations
    step1_find_latent_representations(config)

    # Step 2: Calculate GSS
    step2_calculate_gss(config)

    # Step 3: Max pooling
    step3_max_pooling(config)

    # Step 4: Spatial LDSC
    step4_spatial_ldsc(config)

    # Step 5: 3D visualization
    step5_3d_visualization(config)

    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)


def main():
    """Main function for debugging"""
    parser = argparse.ArgumentParser(description="gsMap Pipeline for Local Debugging")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--step", type=str, choices=[
        "latent", "gss", "pooling", "ldsc", "3d", "all"
    ], default="all", help="Which step to run")
    parser.add_argument("--sample", type=str, help="Specific sample to process")
    parser.add_argument("--trait", type=str, help="Specific trait to process")
    parser.add_argument("--project", type=str, help="Override project name")
    parser.add_argument("--workdir", type=str, help="Override working directory")

    args = parser.parse_args()

    # Load config
    config = PipelineConfig()

    if args.config:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # Override with command line arguments
    if args.project:
        config.project_name = args.project
    if args.workdir:
        config.workdir = args.workdir

    # Setup directories
    setup_directories(config)

    # Run requested step(s)
    if args.step == "all":
        run_full_pipeline(config)
    elif args.step == "latent":
        step1_find_latent_representations(config)
    elif args.step == "gss":
        step2_calculate_gss(config, args.sample)
    elif args.step == "pooling":
        step3_max_pooling(config, args.sample)
    elif args.step == "ldsc":
        step4_spatial_ldsc(config, args.sample, args.trait)
    elif args.step == "3d":
        step5_3d_visualization(config, args.trait)


if __name__ == "__main__":
    # main()
    # # get h5ad files
    config = PipelineConfig()
    # run_full_pipeline(config)
    # step1_find_latent_representations(config)
    step2_calculate_gss(config, )

    # step4_spatial_ldsc(config, )
    # step5_3d_visualization(config, )

    # h5ad_files = Path(config.data_root).glob("*.h5ad")
    # import scanpy as sc
    # import numpy as np
    # for h5ad_file  in h5ad_files:
    #     adata = sc.read_h5ad(h5ad_file)
    #     # adata.X = adata.X.astype(np.int32)
    #     adata.layers["count"] = adata.X.copy()
    #     adata.write_h5ad(h5ad_file)
