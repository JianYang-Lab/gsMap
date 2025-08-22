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
from gsMap.max_pooling import run_max_pooling
from gsMap.run_link_mode import run_pipeline_link
from gsMap.three_d_combine import three_d_combine


@dataclass
class PipelineConfig:
    """Configuration for the gsMap3D pipeline"""

    # Base paths
    workdir: str = "/storage/yangjianLab/chenwenhao/01_Project/01_Research/202312_gsMap/experiment/20250807_refactor_for_gsmap3d/02_latent2gene_optmization_max_pooling"
    data_root: str = "/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/ST_data_Collection/01_MERFISH/mouse_brain/01_cell_atlas_of_whole_mouse_brain/01_processed"
    gsmap_resource: str = "/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/resource/gsMap_resource"
    gwas_summary: str = "/storage/yangjianLab/songliyang/GWAS_trait/GWAS_brain_use.yaml"

    # Project settings
    project_name: str = "MERFISH_BRAIN_V1"
    annotation: str = "cell_type"
    spatial_key: str = "spatial"

    # Processing parameters
    n_cell_training: int = 100000
    data_layer: str = "X"
    homolog_file: str = "/storage/yangjianLab/songliyang/SpatialData/homologs/mouse_human_homologs.txt"

    # 3D visualization
    adata_3d: str = "/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/ST_data_Collection/01_MERFISH/mouse_brain/01_cell_atlas_of_whole_mouse_brain/01_processed"
    background_color: str = "white"
    spatial_key_3d: str = "X_CCF"
    st_id: str = "st_id"

    # Additional parameters
    max_processes: int = 3


def setup_directories(config: PipelineConfig):
    """Create necessary directories"""
    Path(config.workdir).mkdir(parents=True, exist_ok=True)
    Path(f"{config.workdir}/list").mkdir(parents=True, exist_ok=True)
    Path(f"{config.workdir}/{config.project_name}").mkdir(parents=True, exist_ok=True)


def get_sample_list(config: PipelineConfig) -> List[str]:
    """Get list of h5ad files for processing"""
    pattern = f"{config.data_root}/WB_imputation_animal3_sagittal_C57BL6J*.h5ad"
    files = glob.glob(pattern)
    return files


def step1_find_latent_representations(config: PipelineConfig):
    """Find latent representations using LGCN"""
    print("=" * 80)
    print("Step 1: Finding Latent Representations")
    print("=" * 80)

    # Create file list
    file_list_path = f"{config.workdir}/{config.project_name}/sample_list.txt"
    files = get_sample_list(config)

    with open(file_list_path, 'w') as f:
        for file in files:
            f.write(f"{file}\n")

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
    run_full_pipeline(config)
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
