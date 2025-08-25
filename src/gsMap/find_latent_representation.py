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
from gsMap.GNN.TrainStep import ModelTrain
from gsMap.GNN.STmodel import StEmbeding
from gsMap.ST_process import TrainingData, find_common_hvg, InferenceData
from gsMap.config import FindLatentRepresentationsConfig
from gsMap.slice_mean import process_slice_mean

from operator import itemgetter

logger = logging.getLogger(__name__)


def _parse_spe_file_list(spe_file_list: str | list[str]):
    if isinstance(spe_file_list, str):
        spe_file_list = Path(spe_file_list)
        if not spe_file_list.exists():
            logger.error(f"Path not found: {spe_file_list}")
            raise FileNotFoundError(f"Path not found: {spe_file_list}")
        # -
        if spe_file_list.is_dir():
            final_list = [
                spe_file_list / file_path for file_path in os.listdir(spe_file_list)
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
    # -
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
    idx = torch.arange(n)
    splits_tensor = torch.as_tensor(splits)
    multiplier = n / splits_tensor.sum()
    splits_tensor = (multiplier * splits_tensor).long()
    diff = n - splits_tensor.sum()
    splits_tensor[0] += diff
    return random_split(idx, splits_tensor)


def run_find_latent_representation(args: FindLatentRepresentationsConfig):
    logger.info(f'Project dir: {args.project_dir}')
    set_seed(2024)

    # Find the hvg
    spe_file_list = args.h5ad_list_file
    spe_file_list = _parse_spe_file_list(spe_file_list)
    hvg, n_cell_used, percent_annotation, gene_name_dict = find_common_hvg(spe_file_list, args)
    common_genes = np.array(list(gene_name_dict.keys()))

    # Prepare the trainning data
    get_trainning_data = TrainingData(args)
    get_trainning_data.prepare(spe_file_list, n_cell_used, hvg, percent_annotation)

    # Configure the distribution
    if args.data_layer in ["count", "counts"]:
        distribution = args.distribution
        variational = True
        use_tf = args.use_tf
    else:
        distribution = "gaussian"
        variational = False
        use_tf = False

    # Instantiation the LGCN VAE
    input_size = [
        get_trainning_data.expression_merge.size(1),
        get_trainning_data.expression_gcn_merge.size(1),
    ]
    class_size = len(torch.unique(get_trainning_data.label_merge))
    batch_size = get_trainning_data.batch_size
    cell_size, out_size = get_trainning_data.expression_merge.shape
    label_name = get_trainning_data.label_name

    # Configure the batch embedding dim
    batch_embedding_size = 64

    # Configure the model
    gsmap_lgcn_model = StEmbeding(
        # parameter of VAE
        input_size=input_size,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        batch_embedding_size=batch_embedding_size,
        out_put_size=out_size,
        batch_size=batch_size,
        class_size=class_size,
        # parameter of transformer
        module_dim=args.module_dim,
        hidden_gmf=args.hidden_gmf,
        n_modules=args.n_modules,
        nhead=args.nhead,
        n_enc_layer=args.n_enc_layer,
        # parameter of model structure
        distribution=distribution,
        use_tf=use_tf,
        variational=variational,
    )

    # Configure the optimizer
    optimizer = torch.optim.Adam(gsmap_lgcn_model.parameters(), lr=1e-3)
    logger.info(
        f"gsMap-LGCN parameters: {sum(p.numel() for p in gsmap_lgcn_model.parameters())}."
    )
    logger.info(f"Number of cells used in trainning: {cell_size}.")

    # Split the data to trainning (80%) and validation (20%).
    train_idx, val_idx = index_splitter(
        get_trainning_data.expression_gcn_merge.size(0), [80, 20]
    )
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Configure the data loader
    dataset = TensorDataset(
        get_trainning_data.expression_gcn_merge,
        get_trainning_data.batch_merge,
        get_trainning_data.expression_merge,
        get_trainning_data.label_merge,
    )
    train_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, sampler=train_sampler
    )
    val_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, sampler=val_sampler
    )

    # Model trainning
    gsMap_embedding_finder = ModelTrain(
        gsmap_lgcn_model,
        optimizer,
        distribution,
        mode="reconstruction",
        lr=1e-3,
        model_path=args.model_path,
    )
    gsMap_embedding_finder.set_loaders(train_loader, val_loader)
    print(gsMap_embedding_finder.model)

    if not os.path.exists(args.model_path):
        # reconstruction
        gsMap_embedding_finder.train(args.itermax, patience=args.patience)

        # classification
        if args.two_stage and args.annotation is not None:
            gsMap_embedding_finder.model.load_state_dict(torch.load(args.model_path))
            gsMap_embedding_finder.mode = "classification"
            gsMap_embedding_finder.train(args.itermax, patience=args.patience)
    else:
        logger.info(f"Model found at {args.model_path}. Skipping training.")

    # Load the best model
    gsMap_embedding_finder.model.load_state_dict(torch.load(args.model_path))
    gsmap_embedding_model = gsMap_embedding_finder.model

    # Configure the inference
    infer = InferenceData(hvg, batch_size, gsmap_embedding_model, label_name, args)

    print(args.zarr_group_path)

    for st_id, st_file in enumerate(spe_file_list):
        st_name = (Path(st_file).name).split(".h5ad")[0]

        output_path = args.latent_dir / f"{st_name}_add_latent.h5ad"

        # Infer the embedding
        adata = infer.infer_embedding_single(st_id, st_file)
        # adata.obs_names = st_name + '_' + adata.obs_names

        # Transfer the gene name
        common_genes = np.array(list(gene_name_dict.keys()))
        common_genes_transfer = np.array(itemgetter(*common_genes)(gene_name_dict))
        adata = adata[:, common_genes].copy()
        adata.var_names = common_genes_transfer


        # Compute the depth
        if args.data_layer in ["count", "counts"]:
            adata.obs['depth'] = np.array(adata.layers[args.data_layer].sum(axis=1)).flatten()

        # Save the ST data with embeddings
        adata.write_h5ad(output_path)
