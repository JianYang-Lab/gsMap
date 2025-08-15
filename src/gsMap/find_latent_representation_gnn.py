"""
Complete migration of find_latent_representation from gsMap3D with GNN training
"""
import os
import torch
import numpy as np
import logging
import random
from pathlib import Path
from torch.utils.data import (
    DataLoader,
    random_split,
    TensorDataset,
    SubsetRandomSampler,
)
import scanpy as sc  
from gsMap.GNN.TrainStep import ModelTrain
from gsMap.GNN.STmodel import StEmbeding
from gsMap.GNN.st_process import TrainingData, find_common_hvg, InferenceData
from gsMap.config import FindLatentRepresentationsConfig
from gsMap.slice_mean import process_slice_mean

logger = logging.getLogger(__name__)


def _parse_spe_file_list(spe_file_list):
    """Parse spatial file list from various input formats"""
    if isinstance(spe_file_list, str):
        spe_file_list = Path(spe_file_list)
        if not spe_file_list.exists():
            logger.error(f"Path not found: {spe_file_list}")
            raise FileNotFoundError(f"Path not found: {spe_file_list}")
        
        if spe_file_list.is_dir():
            final_list = [
                str(spe_file_list / file_path) for file_path in os.listdir(spe_file_list)
                if file_path.endswith('.h5ad')
            ]
        elif spe_file_list.is_file():
            with open(spe_file_list, "r") as f:
                final_list = f.readlines()
            final_list = [file.strip() for file in final_list]
    elif isinstance(spe_file_list, list):
        final_list = spe_file_list
    else:
        raise ValueError(
            "Invalid type for spe_file_list,"
            f"expected str or list, got {type(spe_file_list)}"
        )
    
    return final_list


def set_seed(seed_value):
    """
    Set seed for reproducibility in PyTorch and other libraries.
    """
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        logger.info("Using GPU for computations.")
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    else:
        logger.info("Using CPU for computations.")


def index_splitter(n, splits):
    """Split indices for train/validation"""
    idx = torch.arange(n)
    splits_tensor = torch.as_tensor(splits)
    multiplier = n / splits_tensor.sum()
    splits_tensor = (multiplier * splits_tensor).long()
    diff = n - splits_tensor.sum()
    splits_tensor[0] += diff
    return random_split(idx, splits_tensor)


def run_find_latent_representation_gnn(args: FindLatentRepresentationsConfig):
    """
    Main function for finding latent representations using GNN
    This is the complete migration from gsMap3D
    """
    # Set up project directory
    project_dir = Path(args.workdir) / args.sample_name / 'find_latent_representations'
    project_dir.mkdir(parents=True, exist_ok=True)
    latent_dir = project_dir / 'latent'
    latent_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Project dir: {project_dir}')
    set_seed(2024)
    
    # Add latent_dir to args for InferenceData
    args.latent_dir = latent_dir
    
    # Handle single file or multiple files
    if hasattr(args, 'input_hdf5_path') and args.input_hdf5_path:
        # Single file mode
        spe_file_list = [args.input_hdf5_path]
    elif hasattr(args, 'spe_file_list') and args.spe_file_list:
        # Multiple files mode
        spe_file_list = _parse_spe_file_list(args.spe_file_list)
    else:
        raise ValueError("Either input_hdf5_path or spe_file_list must be provided")
    
    # Find the highly variable genes
    logger.info("Finding highly variable genes...")
    hvg, n_cell_used, percent_annotation = find_common_hvg(spe_file_list, args)
    
    # Prepare the training data
    logger.info("Preparing training data...")
    get_training_data = TrainingData(args)
    get_training_data.prepare(spe_file_list, n_cell_used, hvg, percent_annotation)
    
    # Configure the distribution based on data layer
    if args.data_layer in ["count", "counts"]:
        distribution = getattr(args, 'distribution', 'nb')
        variational = True
        use_tf = getattr(args, 'use_tf', False)
    else:
        distribution = "gaussian"
        variational = False
        use_tf = False

    # Instantiate the LGCN VAE
    input_size = [
        get_training_data.expression_merge.size(1),
        get_training_data.expression_gcn_merge.size(1),
    ]
    class_size = len(torch.unique(get_training_data.label_merge))
    batch_size = get_training_data.batch_size
    cell_size, out_size = get_training_data.expression_merge.shape
    label_name = get_training_data.label_name
    
    # Configure the batch embedding dim
    batch_embedding_size = getattr(args, 'batch_embedding_size', 64)
    
    # Configure model parameters
    hidden_size = getattr(args, 'hidden_size', [512, 256])
    embedding_size = getattr(args, 'embedding_size', 64)
    module_dim = getattr(args, 'module_dim', 8)
    hidden_gmf = getattr(args, 'hidden_gmf', 256)
    n_modules = getattr(args, 'n_modules', 16)
    nhead = getattr(args, 'nhead', 8)
    n_enc_layer = getattr(args, 'n_enc_layer', 3)

    # Configure the model
    gsmap_lgcn_model = StEmbeding(
        # parameter of VAE
        input_size=input_size,
        hidden_size=hidden_size,
        embedding_size=embedding_size,
        batch_embedding_size=batch_embedding_size,
        out_put_size=out_size,
        batch_size=batch_size,
        class_size=class_size,
        # parameter of transformer
        module_dim=module_dim,
        hidden_gmf=hidden_gmf,
        n_modules=n_modules,
        nhead=nhead,
        n_enc_layer=n_enc_layer,
        # parameter of model structure
        distribution=distribution,
        use_tf=use_tf,
        variational=variational,
    )

    # Configure the optimizer
    lr = getattr(args, 'lr', 1e-3)
    optimizer = torch.optim.Adam(gsmap_lgcn_model.parameters(), lr=lr)
    logger.info(
        f"gsMap-LGCN parameters: {sum(p.numel() for p in gsmap_lgcn_model.parameters())}."
    )
    logger.info(f"Number of cells used in training: {cell_size}.")

    # Split the data to training (80%) and validation (20%).
    train_idx, val_idx = index_splitter(
        get_training_data.expression_gcn_merge.size(0), [80, 20]
    )
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Configure the data loader
    batch_size_train = getattr(args, 'batch_size', 256)
    dataset = TensorDataset(
        get_training_data.expression_gcn_merge,
        get_training_data.batch_merge,
        get_training_data.expression_merge,
        get_training_data.label_merge,
    )
    train_loader = DataLoader(
        dataset=dataset, batch_size=batch_size_train, sampler=train_sampler
    )
    val_loader = DataLoader(
        dataset=dataset, batch_size=batch_size_train, sampler=val_sampler
    )

    # Set model path
    model_path = project_dir / 'model.pt'
    
    # Model training
    gsMap_embedding_finder = ModelTrain(
        gsmap_lgcn_model,
        optimizer,
        distribution,
        mode="reconstruction",
        lr=lr,
        model_path=str(model_path),
    )
    gsMap_embedding_finder.set_loaders(train_loader, val_loader)
    
    if not os.path.exists(model_path):
        # Train for reconstruction
        itermax = getattr(args, 'itermax', 5000)
        patience = getattr(args, 'patience', 100)
        
        logger.info("Training model for reconstruction...")
        gsMap_embedding_finder.train(itermax, patience=patience)

        # Classification stage if annotations are provided
        two_stage = getattr(args, 'two_stage', False)
        if two_stage and args.annotation is not None:
            logger.info("Training model for classification...")
            gsMap_embedding_finder.model.load_state_dict(torch.load(model_path))
            gsMap_embedding_finder.mode = "classification"
            gsMap_embedding_finder.train(itermax, patience=patience)
    else:
        logger.info(f"Model found at {model_path}. Loading existing model.")
        
    # Load the best model
    gsMap_embedding_finder.model.load_state_dict(torch.load(model_path))
    gsmap_embedding_model = gsMap_embedding_finder.model

    # Configure the inference
    logger.info("Starting inference on spatial data...")
    infer = InferenceData(hvg, batch_size, gsmap_embedding_model, label_name, args)
    
    # Process each spatial file
    for st_id, st_file in enumerate(spe_file_list):
        logger.info(f"Processing file {st_id + 1}/{len(spe_file_list)}: {st_file}")
        
        # Infer embeddings for this file
        output_path = infer.infer_embedding_single(st_id, st_file)
        logger.info(f"Saved embeddings to: {output_path}")
        
        # Process slice mean if zarr path is provided
        if hasattr(args, 'zarr_group_path') and args.zarr_group_path:
            st_name = Path(st_file).stem
            adata = sc.read_h5ad(output_path)
            process_slice_mean(args, st_name, adata)
    
    logger.info("Latent representation finding completed successfully!")
    
    # Return the path to the processed file(s)
    if len(spe_file_list) == 1:
        return project_dir / 'latent' / f"{Path(spe_file_list[0]).stem}_with_latent.h5ad"
    else:
        return project_dir / 'latent'