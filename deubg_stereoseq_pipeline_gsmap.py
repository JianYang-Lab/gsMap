#!/usr/bin/env python3
"""
Test script for gsMap with migrated GNN features from gsMap.
This script tests the complete migration using the same data structure as stereoseq_brain_pipeline.py
"""

import sys
import glob
import time
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
import argparse
import yaml
import numpy as np
import pandas as pd
import scanpy as sc

# Add gsMap to path
sys.path.insert(0, '/mnt/d/01_Project/01_Research/202312_gsMap/src/gsMap/src')

from gsMap.config import (
    FindLatentRepresentationsConfig,
    LatentToGeneConfig,
    GenerateLDScoreConfig,
    SpatialLDSCConfig,
    CauchyCombinationConfig,
    ReportConfig
)
from gsMap.find_latent_representation_gnn import run_find_latent_representation
from gsMap.latent_to_gene import run_latent_to_gene
from gsMap.generate_ldscore import run_generate_ldscore
from gsMap.spatial_ldsc_multiple_sumstats import run_spatial_ldsc
from gsMap.cauchy_combination_test import run_Cauchy_combination
from gsMap.report import run_report

# Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

@dataclass
class PipelineConfig:
    """Configuration for testing migrated gsMap functionality"""
    
    # Base paths - same as stereoseq_brain_pipeline.py
    workdir: str = "/mnt/d/01_Project/01_Research/202312_gsMap/experiment/gsmap_gnn_test/Mouse_E9.5"
    data_root: str = "/mnt/d/01_Project/01_Research/202312_gsMap/experiment/20250807_refactor_for_gsmap3d/02_latent2gene_optmization_max_pooling/Mouse_E9.5"
    gsmap_resource: str = "/mnt/d/01_Project/01_Research/202312_gsMap/data/gsMap_dev_data/online_resource/gsMap_resource"
    gwas_summary: str = "/mnt/d/01_Project/01_Research/202312_gsMap/experiment/20250807_refactor_for_gsmap3d/02_latent2gene_optmization_max_pooling/mouse_e9_5_gwas_config.yaml"
    
    # Project settings
    project_name: str = "gsMap_migration_0815_v3"
    annotation: str = "mapped_celltype"
    spatial_key: str = "spatial"
    data_layer: str = "count"
    
    # Species conversion
    species: str = "mouse"
    homolog_file: str = "/mnt/d/01_Project/01_Research/202312_gsMap/data/gsMap_dev_data/online_resource/gsMap_resource/homologs/mouse_human_homologs.txt"
    
    # GNN model parameters (matching gsMap config.py defaults)
    hidden_size: int = 128  # Will be converted to [128, 128] for multi-encoder
    embedding_size: int = 32
    batch_embedding_size: int = 32  # Not in config but kept for compatibility
    module_dim: int = 30
    hidden_gmf: int = 128
    n_modules: int = 16
    nhead: int = 4
    n_enc_layer: int = 2
    
    # Training parameters (matching gsMap config.py defaults)
    itermax: int = 100
    patience: int = 10
    batch_size: int = 1024
    lr: float = 1e-3  # Not in config but standard value
    
    # Data processing (matching gsMap config.py defaults)
    n_cell_training: int = 100000
    feat_cell: int = 2000
    do_sampling: bool = True
    pearson_residual: bool = False  # Added from config
    
    # GNN specific (matching gsMap config.py defaults)
    K: int = 3  # Number of GCN hops
    n_neighbors: int = 10  # For spatial graph
    
    # Distribution settings (matching gsMap config.py defaults)
    distribution: str = "nb"  # nb, zinb, or gaussian
    use_tf: bool = False  # Use transformer (action="store_true" defaults to False)
    two_stage: bool = True  # Two-stage training
    
    # Latent to gene parameters (matching gsMap config.py defaults)
    latent_representation: str = "emb_gcn"  # Matching config default
    latent_representation_indv: str = "emb"  # Matching config default
    num_neighbour: int = 21
    num_neighbour_spatial: int = 201
    num_anchor: int = 51
    no_expression_fraction: bool = False
    use_gcn_smoothing: bool = True  # Enable GNN features in latent_to_gene
    gcn_K: int = 1  # Default for GCN smoothing in latent_to_gene
    n_neighbors_gcn: int = 10  # For GCN smoothing
    use_w: bool = False  # Section-specific weights for batch effect
    
    # Zarr storage
    use_zarr: bool = True
    zarr_group_path: Optional[str] = None
    
    # Processing parameters
    num_processes: int = 4
    spots_per_chunk: int = 1000
    
    # Timing and comparison
    enable_timing: bool = True
    compare_with_gsmap3d: bool = False  # If true, compare with gsMap results


class GsMapMigrationTester:
    """Test runner for migrated gsMap functionality"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.timing_results = {}
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories"""
        Path(self.config.workdir).mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.workdir}/list").mkdir(parents=True, exist_ok=True)
        Path(f"{self.config.workdir}/{self.config.project_name}").mkdir(parents=True, exist_ok=True)
        
        # Set zarr path if using zarr
        if self.config.use_zarr:
            self.config.zarr_group_path = f"{self.config.workdir}/{self.config.project_name}/zarr_storage"
            Path(self.config.zarr_group_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created working directory: {self.config.workdir}")
    
    def get_sample_list(self) -> List[str]:
        """Get list of h5ad files for processing"""
        pattern = f"{self.config.data_root}/*.h5ad"
        files = sorted(glob.glob(pattern))
        
        if not files:
            logger.warning(f"No h5ad files found in {self.config.data_root}")
            logger.warning("Creating example data for testing...")
            files = self.create_example_data()
        else:
            logger.info(f"Found {len(files)} h5ad files")
            for f in files[:5]:  # Show first 5 files
                logger.info(f"  - {Path(f).name}")
        
        return files
    
    def create_example_data(self) -> List[str]:
        """Create example Mouse E9.5-like data if real data not available"""
        logger.info("Creating synthetic Mouse E9.5 data for testing...")
        
        example_dir = Path(self.config.workdir) / "example_data"
        example_dir.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        n_samples = 3
        
        # Mouse E9.5 cell types
        cell_types = [
            "Neural_progenitor", "Mesenchymal", "Endothelial",
            "Epithelial", "Blood", "Cardiac", "Muscle"
        ]
        
        for i in range(n_samples):
            n_spots = 500
            n_genes = 2000
            
            # Create expression matrix
            X = np.random.negative_binomial(5, 0.3, size=(n_spots, n_genes))
            
            # Create AnnData object
            adata = sc.AnnData(X=X.astype(np.float32))
            
            # Add gene names (common mouse genes that would appear in homologs)
            # Use a mix of real mouse gene names and generic names
            mouse_genes = [
                "Actb", "Gapdh", "Ppia", "Hprt1", "Pgk1", "Ldha", "Rplp0", "Rpl13a",
                "B2m", "Ywhaz", "Sdha", "Tbp", "Hmbs", "Gusb", "Tfrc", "Atp5b",
                "Ubc", "Eef1a1", "Rps18", "Rps27a", "Hspa8", "Tubb5", "Eno1", "Pkm",
                "Aldoa", "Tpi1", "Pgam1", "Gpi1", "Mdh1", "Cs", "Idh1", "Ogdh",
                "Sucla2", "Sdhb", "Fh1", "Mdh2", "Got1", "Got2", "Gpt", "Glul",
                "Pcna", "Top2a", "Mki67", "Ccnb1", "Cdk1", "Aurkb", "Plk1", "Bub1",
                "Mad2l1", "Cenpa", "Cenpf", "Ndc80", "Smc2", "Smc4", "Kif11", "Kif23"
            ]
            # Ensure unique gene names and pad with generic names if needed
            gene_set = set(mouse_genes)
            idx = 0
            while len(gene_set) < n_genes:
                gene_set.add(f"Gene_{idx}")
                idx += 1
            adata.var_names = list(gene_set)[:n_genes]
            adata.obs_names = [f"Spot_{j}" for j in range(n_spots)]
            
            # Add spatial coordinates
            adata.obsm["spatial"] = np.random.randn(n_spots, 2) * 100
            
            # Add cell type annotations
            adata.obs["mapped_celltype"] = np.random.choice(cell_types, n_spots)
            
            # Store count layer
            adata.layers["count"] = X.astype(np.float32)
            
            # Add batch information
            adata.obs["batch"] = f"E9.5_section_{i+1:02d}"
            
            # Save
            output_file = example_dir / f"E9.5_section_{i+1:02d}.h5ad"
            adata.write_h5ad(output_file)
            created_files.append(str(output_file))
            
            logger.info(f"  Created: {output_file}")
        
        # Update data_root to example directory
        self.config.data_root = str(example_dir)
        
        return created_files
    
    def create_file_list(self, sample_files: List[str]) -> str:
        """Create a file list for multi-sample processing"""
        file_list_path = f"{self.config.workdir}/list/{self.config.project_name}_list.txt"
        with open(file_list_path, 'w') as f:
            for file in sample_files:
                f.write(f"{file}\n")
        logger.info(f"Created file list: {file_list_path}")
        return file_list_path
    
    def step1_find_latent_representations_multi(self, sample_files: List[str]):
        """Test finding latent representations with multiple files (gsMap style)"""
        logger.info("=" * 80)
        logger.info("Step 1: Finding Latent Representations (Multi-file Mode)")
        logger.info("=" * 80)
        
        # Create file list
        file_list_path = self.create_file_list(sample_files)
        
        # Create config for FindLatentRepresentations using gsMap parameters
        latent_config = FindLatentRepresentationsConfig(
            workdir=self.config.workdir,
            sample_name=self.config.project_name,
            spe_file_list=file_list_path,  # Use file list for multi-sample
            annotation=self.config.annotation,
            data_layer=self.config.data_layer,
            spatial_key=self.config.spatial_key,
            # GNN architecture
            hidden_size=self.config.hidden_size,
            embedding_size=self.config.embedding_size,
            batch_embedding_size=self.config.batch_embedding_size,
            module_dim=self.config.module_dim,
            hidden_gmf=self.config.hidden_gmf,
            n_modules=self.config.n_modules,
            nhead=self.config.nhead,
            n_enc_layer=self.config.n_enc_layer,
            # Training
            itermax=self.config.itermax,
            patience=self.config.patience,
            batch_size=self.config.batch_size,
            lr=self.config.lr,
            # Data processing
            n_cell_training=self.config.n_cell_training,
            feat_cell=self.config.feat_cell,
            do_sampling=self.config.do_sampling,
            pearson_residual=self.config.pearson_residual,
            # GNN specific
            K=self.config.K,
            n_neighbors=self.config.n_neighbors,
            # Distribution
            distribution=self.config.distribution,
            use_tf=self.config.use_tf,
            two_stage=self.config.two_stage,
            # Species (optional for testing)
            homolog_file=self.config.homolog_file if Path(self.config.homolog_file).exists() else None,
            species=self.config.species if self.config.homolog_file and Path(self.config.homolog_file).exists() else None,
            # Zarr
            zarr_group_path=self.config.zarr_group_path
        )
        
        try:
            start_time = time.time()
            run_find_latent_representation(latent_config)
            elapsed = time.time() - start_time
            
            self.timing_results["find_latent_multi"] = elapsed
            logger.info(f"✓ Latent representations completed in {elapsed:.2f}s")
            
            # Check output files
            latent_dir = Path(self.config.workdir) / self.config.project_name / "find_latent_representations" / "latent"
            if latent_dir.exists():
                latent_files = list(latent_dir.glob("*_with_latent.h5ad"))
                logger.info(f"✓ Generated {len(latent_files)} latent files")
                return True
            else:
                logger.warning(f"Latent directory not found: {latent_dir}")
                return False
                
        except Exception as e:
            logger.error(f"✗ Failed to find latent representations: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def step2_test_latent_to_gene_gnn(self, sample_files: List[str]):
        """Test GNN-enhanced latent to gene"""
        logger.info("=" * 80)
        logger.info("Step 2: Testing Latent to Gene with GNN")
        logger.info("=" * 80)
        
        success_count = 0
        
        for sample_file in sample_files[:2]:  # Test first 2 samples
            sample_name = Path(sample_file).stem
            
            # Find the latent h5ad file
            latent_file = Path(self.config.workdir) / self.config.project_name / \
                         "find_latent_representations" / "latent" / f"{sample_name}_with_latent.h5ad"
            
            if not latent_file.exists():
                logger.warning(f"Latent file not found for {sample_name}")
                continue
            
            logger.info(f"Processing {sample_name}...")
            
            # Create config for LatentToGene with GNN
            gss_config = LatentToGeneConfig(
                workdir=self.config.workdir,
                sample_name=f"{self.config.project_name}_{sample_name}",
                input_hdf5_path=str(latent_file),
                annotation=self.config.annotation,
                spatial_key=self.config.spatial_key,
                latent_representation=self.config.latent_representation,
                latent_representation_indv=self.config.latent_representation_indv,
                num_neighbour=self.config.num_neighbour,
                num_neighbour_spatial=self.config.num_neighbour_spatial,
                num_anchor=self.config.num_anchor,
                no_expression_fraction=self.config.no_expression_fraction,
                homolog_file=self.config.homolog_file,
                species=self.config.species,
                # GNN parameters
                use_gcn_smoothing=self.config.use_gcn_smoothing,
                gcn_K=self.config.gcn_K,
                n_neighbors_gcn=self.config.n_neighbors_gcn,
                zarr_group_path=self.config.zarr_group_path,
                use_w=self.config.use_w if hasattr(self.config, 'use_w') else False
            )
            
            try:
                start_time = time.time()
                run_latent_to_gene(gss_config)
                elapsed = time.time() - start_time
                
                self.timing_results[f"gss_{sample_name}"] = elapsed
                logger.info(f"  ✓ GSS completed for {sample_name} in {elapsed:.2f}s")
                success_count += 1
                
            except Exception as e:
                logger.error(f"  ✗ Failed GSS for {sample_name}: {e}")
        
        return success_count > 0
    
    def step3_verify_outputs(self):
        """Verify the outputs are correctly generated"""
        logger.info("=" * 80)
        logger.info("Step 3: Verifying Outputs")
        logger.info("=" * 80)
        
        verification_results = {}
        
        # Check latent representations
        latent_dir = Path(self.config.workdir) / self.config.project_name / "find_latent_representations" / "latent"
        if latent_dir.exists():
            latent_files = list(latent_dir.glob("*_with_latent.h5ad"))
            verification_results["latent_files"] = len(latent_files)
            
            # Check one latent file
            if latent_files:
                adata = sc.read_h5ad(latent_files[0])
                has_latent = "latent_GVAE" in adata.obsm
                has_latent_indv = "latent_GVAE_indv" in adata.obsm
                verification_results["has_latent_GVAE"] = has_latent
                verification_results["has_latent_GVAE_indv"] = has_latent_indv
                
                if has_latent:
                    latent_shape = adata.obsm["latent_GVAE"].shape
                    logger.info(f"  Latent shape: {latent_shape}")
                    verification_results["latent_shape"] = latent_shape
        
        # Check marker scores
        mk_score_files = list(Path(self.config.workdir).glob("**/mk_score/*_gene_marker_score.feather"))
        verification_results["mk_score_files"] = len(mk_score_files)
        
        if mk_score_files:
            mk_df = pd.read_feather(mk_score_files[0])
            verification_results["mk_score_shape"] = mk_df.shape
            logger.info(f"  Marker score shape: {mk_df.shape}")
        
        # Check zarr storage if enabled
        if self.config.use_zarr and self.config.zarr_group_path:
            zarr_path = Path(self.config.zarr_group_path)
            if zarr_path.exists():
                import zarr
                try:
                    zarr_group = zarr.open(str(zarr_path), mode='r')
                    zarr_arrays = list(zarr_group.array_keys())
                    verification_results["zarr_arrays"] = len(zarr_arrays)
                    logger.info(f"  Zarr arrays: {len(zarr_arrays)}")
                except:
                    pass
        
        # Print verification summary
        logger.info("\nVerification Summary:")
        for key, value in verification_results.items():
            status = "✓" if value else "✗"
            logger.info(f"  {status} {key}: {value}")
        
        return verification_results
    
    def compare_with_gsmap3d_results(self):
        """Compare results with gsMap if available"""
        logger.info("=" * 80)
        logger.info("Step 4: Comparing with gsMap Results (if available)")
        logger.info("=" * 80)
        
        # Path to gsMap results (adjust as needed)
        gsmap3d_dir = Path("/mnt/d/01_Project/01_Research/202312_gsMap/experiment/20250807_refactor_for_gsmap3d/02_latent2gene_optmization_max_pooling/01_mouse_E9.5_dev_v1")
        
        if not gsmap3d_dir.exists():
            logger.info("gsMap results not found for comparison")
            return
        
        # Compare latent representations
        # This is a placeholder - implement actual comparison logic based on your needs
        logger.info("Comparison with gsMap would go here...")
    
    def run_full_test(self, test_samples: Optional[int] = None):
        """Run complete migration test"""
        logger.info("\n" + "=" * 80)
        logger.info("Starting gsMap Migration Test")
        logger.info("=" * 80 + "\n")
        
        # Get sample list
        sample_files = self.get_sample_list()
        
        if not sample_files:
            logger.error("No sample files found or created")
            return False
        
        # Limit samples for testing
        if test_samples:
            sample_files = sample_files[:test_samples]
            logger.info(f"Testing with {len(sample_files)} samples")
        
        results = {}
        
        # Step 1: Find latent representations (multi-file mode)
        logger.info("\n>>> PHASE 1: Testing Multi-file Latent Representation Finding")
        results["find_latent"] = self.step1_find_latent_representations_multi(sample_files)
        
        # Step 2: Test GNN-enhanced latent to gene
        if results["find_latent"]:
            logger.info("\n>>> PHASE 2: Testing GNN-enhanced Latent to Gene")
            results["latent_to_gene"] = self.step2_test_latent_to_gene_gnn(sample_files)
        
        # Step 3: Verify outputs
        logger.info("\n>>> PHASE 3: Verifying Outputs")
        verification = self.step3_verify_outputs()
        results["verification"] = verification
        
        # Step 4: Compare with gsMap if requested
        if self.config.compare_with_gsmap3d:
            logger.info("\n>>> PHASE 4: Comparing with gsMap")
            self.compare_with_gsmap3d_results()
        
        # Print timing summary
        if self.config.enable_timing and self.timing_results:
            logger.info("\n" + "=" * 80)
            logger.info("Timing Summary:")
            logger.info("=" * 80)
            
            total_time = 0
            for key, time_val in sorted(self.timing_results.items()):
                logger.info(f"  {key}: {time_val:.2f}s")
                total_time += time_val
            logger.info(f"  Total: {total_time:.2f}s")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("Test Results Summary")
        logger.info("=" * 80)
        
        all_passed = True
        for test, result in results.items():
            if test == "verification":
                passed = result.get("has_latent_GVAE", False) and result.get("has_latent_GVAE_indv", False)
            else:
                passed = result
            
            status = "✓ PASSED" if passed else "✗ FAILED"
            logger.info(f"{test}: {status}")
            all_passed = all_passed and passed
        
        if all_passed:
            logger.info("\n🎉 All migration tests passed successfully!")
            logger.info("The gsMap functionality has been successfully migrated to gsMap.")
        else:
            logger.info("\n⚠️ Some tests failed. Please check the logs for details.")
        
        return all_passed


def main():
    """Main test function"""
    parser = argparse.ArgumentParser(description="Test gsMap Migration")
    parser.add_argument("--config", type=str, help="Path to config YAML file")
    parser.add_argument("--workdir", type=str, help="Override working directory")
    parser.add_argument("--data_root", type=str, help="Override data root directory")
    parser.add_argument("--test_samples", type=int, help="Number of samples to test")
    parser.add_argument("--use_example_data", action="store_true", help="Force use of example data")
    parser.add_argument("--compare_gsmap3d", action="store_true", help="Compare with gsMap results")
    parser.add_argument("--skip_training", action="store_true", help="Skip training, use existing model")
    
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
    if args.workdir:
        config.workdir = args.workdir
    if args.data_root:
        config.data_root = args.data_root
    if args.compare_gsmap3d:
        config.compare_with_gsmap3d = True
    if args.use_example_data:
        config.data_root = "force_example"  # Force example data creation
    
    # Reduce iterations for quick testing if requested
    if args.skip_training:
        config.itermax = 10
        config.patience = 2
    
    # Initialize tester
    tester = GsMapMigrationTester(config)
    
    # Run tests
    success = tester.run_full_test(test_samples=args.test_samples)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    config = PipelineConfig()

    tester = GsMapMigrationTester(config)
    tester.run_full_test(test_samples=2)  # Limit to 2 samples for quick testing
    # main()