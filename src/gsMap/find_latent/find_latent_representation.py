import logging
import os
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import (
    DataLoader,
    SubsetRandomSampler,
    TensorDataset,
    random_split,
)

from gsMap.config import FindLatentRepresentationsConfig

from .checkpoint import (
    load_checkpoint_state,
    make_feature_schema,
    read_checkpoint,
    validate_input_contract,
)
from .gnn.st_model import StEmbeding
from .gnn.train_step import ModelTrain
from .section_data import cell_qc_summary
from .st_process import (
    InferenceData,
    TrainingData,
    apply_module_score_qc,
    calculate_module_score,
    calculate_module_scores_from_degs,
    convert_to_human_genes,
    create_subsampled_adata,
    find_common_hvg,
)

logger = logging.getLogger(__name__)


def _load_compatible_model_state(
    model: StEmbeding, model_path: str | Path, feature_schema
) -> bool:
    return load_checkpoint_state(model, model_path, feature_schema)


def _resolve_expression_model(is_count_data, requested_distribution, use_tf):
    """Choose likelihood and variational behavior from expression values."""

    if is_count_data:
        return requested_distribution, True, use_tf
    return "gaussian", False, False


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


def _serialize_label_names(label_name: Any) -> list[Any]:
    """Return annotation labels in a YAML-safe list for run metadata."""

    if label_name is None:
        return []
    if isinstance(label_name, list):
        return label_name
    if hasattr(label_name, "tolist"):
        values = label_name.tolist()
        return values if isinstance(values, list) else [values]
    return list(label_name)


def write_latent_representation_outputs(
    adata,
    *,
    sample_name: str,
    config: FindLatentRepresentationsConfig,
) -> OrderedDict[str, Path]:
    """Write one latent H5AD per downstream sample or physical section."""

    output_path = config.latent_dir / f"{sample_name}_add_latent.h5ad"
    adata.write_h5ad(output_path)
    logger.info("Saved latent representation to %s", output_path)
    return OrderedDict({sample_name: output_path})


def run_find_latent_representation(config: FindLatentRepresentationsConfig) -> dict[str, Any]:
    """
    Run the find latent representation pipeline.

    Args:
        config: FindLatentRepresentationsConfig object with all necessary parameters

    Returns:
        Dictionary containing metadata about the run including config, model info,
        training info, outputs, and annotation info
    """
    from gsMap.config.spatial import guard_spatial_contract

    guard_spatial_contract(config, write=True)
    logger.info(f"Project dir: {config.project_dir}")
    set_seed(config.random_seed)

    # Refuse unverifiable legacy checkpoints before expensive feature/graph work.
    feature_schema = None
    if os.path.exists(config.model_path):
        feature_schema = read_checkpoint(config.model_path)["feature_schema"]
        validate_input_contract(feature_schema, config)
    hvg_audit = {}
    hvg, n_cell_used, gene_homolog_dict, is_count_data = find_common_hvg(
        config.sample_h5ad_dict,
        config,
        saved_hvg=feature_schema["hvg_order"] if feature_schema else None,
        audit=hvg_audit,
    )
    if feature_schema is None:
        feature_schema = make_feature_schema(hvg, config, is_count_data, hvg_audit)
    elif feature_schema["input_is_count_data"] != bool(is_count_data):
        raise ValueError("Checkpoint expression data type differs from the current inputs")
    elif feature_schema.get("cell_qc") != cell_qc_summary(config):
        raise ValueError("Checkpoint cell-QC population differs from the current inputs")
    common_genes = np.array(list(gene_homolog_dict.keys()))

    # Create subsampled concatenated adata with sample-specific stratified sampling
    training_adata = create_subsampled_adata(config.sample_h5ad_dict, n_cell_used, config)
    training_adata.var["highly_variable"] = training_adata.var_names.isin(hvg)
    training_adata.uns["hvg_order"] = list(hvg)
    training_adata.uns["hvg_feature_sha256"] = feature_schema["feature_sha256"]

    # Prepare the trainning data
    get_trainning_data = TrainingData(config)
    get_trainning_data.prepare(
        training_adata,
        hvg,
        source_h5ad_dict=config.sample_h5ad_dict,
    )

    distribution, variational, use_tf = _resolve_expression_model(
        is_count_data,
        config.distribution,
        config.use_tf,
    )
    logger.info(
        "Detected %s expression; using distribution=%s, variational=%s, use_tf=%s.",
        "count" if is_count_data else "continuous",
        distribution,
        variational,
        use_tf,
    )

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
    batch_embedding_size = 16

    # Configure the model
    gsmap_lgcn_model = StEmbeding(
        # parameter of VAE
        input_size=input_size,
        hidden_size=config.hidden_size,
        embedding_size=config.embedding_size,
        batch_embedding_size=batch_embedding_size,
        out_put_size=out_size,
        batch_size=batch_size,
        class_size=class_size,
        # parameter of transformer
        module_dim=config.module_dim,
        hidden_gmf=config.hidden_gmf,
        n_modules=config.n_modules,
        nhead=config.nhead,
        n_enc_layer=config.n_enc_layer,
        # parameter of model structure
        distribution=distribution,
        use_tf=use_tf,
        variational=variational,
        batch_label_classes=get_trainning_data.batch_label_encoder.classes_,
    )

    # Configure the optimizer
    optimizer = torch.optim.Adam(gsmap_lgcn_model.parameters(), lr=1e-3)
    logger.info(f"gsMap-LGCN parameters: {sum(p.numel() for p in gsmap_lgcn_model.parameters())}.")
    logger.info(f"Number of cells used in trainning: {cell_size}.")

    # Split the data to trainning (80%) and validation (20%).
    train_idx, val_idx = index_splitter(get_trainning_data.expression_gcn_merge.size(0), [80, 20])
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # Configure the data loader
    dataset = TensorDataset(
        get_trainning_data.expression_gcn_merge,
        get_trainning_data.batch_merge,
        get_trainning_data.expression_merge,
        get_trainning_data.label_merge,
    )
    train_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset=dataset, batch_size=config.batch_size, sampler=val_sampler)

    # Model trainning
    gsMap_embedding_finder = ModelTrain(
        gsmap_lgcn_model,
        optimizer,
        distribution,
        mode="reconstruction",
        lr=1e-3,
        model_path=config.model_path,
        feature_schema=feature_schema,
    )
    gsMap_embedding_finder.set_loaders(train_loader, val_loader)
    logger.debug("Embedding model architecture:\n%s", gsMap_embedding_finder.model)

    model_is_compatible = os.path.exists(config.model_path) and _load_compatible_model_state(
        gsMap_embedding_finder.model,
        config.model_path,
        feature_schema,
    )
    if not model_is_compatible:
        if os.path.exists(config.model_path):
            raise ValueError(
                "Existing model has incompatible encoder or batch-label semantics. "
                "Use a new output directory; refusing to overwrite the checkpoint."
            )
        # reconstruction
        gsMap_embedding_finder.train(config.itermax, patience=config.patience)

        # classification
        if config.two_stage and config.annotation is not None:
            if not _load_compatible_model_state(
                gsMap_embedding_finder.model,
                config.model_path,
                feature_schema,
            ):
                raise RuntimeError("Training produced an incompatible model checkpoint")
            gsMap_embedding_finder.mode = "classification"
            gsMap_embedding_finder.train(config.itermax, patience=config.patience)
    else:
        logger.info(f"Model found at {config.model_path}. Skipping training.")

    # Load the best model
    if not _load_compatible_model_state(
        gsMap_embedding_finder.model, config.model_path, feature_schema
    ):
        raise RuntimeError("Model checkpoint has incompatible encoder or batch-label semantics")
    gsmap_embedding_model = gsMap_embedding_finder.model

    # Configure the inference
    infer = InferenceData(
        hvg,
        batch_size,
        gsmap_embedding_model,
        label_name,
        config,
        get_trainning_data.batch_label_encoder,
    )

    output_h5ad_path_dict: OrderedDict[str, Path] = OrderedDict()
    # Do the DEG in the training adata if annotation is provided and high quality cell QC is enabled
    module_score_threshold_dict = {}
    if config.high_quality_cell_qc and config.annotation is not None:
        # Calculate module scores for training data
        training_adata = calculate_module_score(training_adata, config.annotation)

        # Calculate thresholds for each annotation
        for label in training_adata.obs[config.annotation].cat.categories:
            scores = training_adata.obs.loc[
                training_adata.obs[config.annotation] == label, f"{label}_module_score"
            ]
            Q1 = np.percentile(scores, 25)
            Q2 = np.median(scores)
            Q3 = np.percentile(scores, 75)
            IQR = Q3 - Q1
            threshold = max(0, Q2 - 1 * IQR)  # Ensure threshold is not negative
            module_score_threshold_dict[label] = threshold
            logger.info(f"High quality module score threshold for {label}: {threshold:.3f}")

        training_adata.uns["module_score_thresholds"] = module_score_threshold_dict

        # Apply QC to training data as well
        training_adata = apply_module_score_qc(
            training_adata, config.annotation, module_score_threshold_dict
        )

    # save the training adata
    training_adata_path = config.find_latent_metadata_path.parent / "training_adata.h5ad"
    training_adata.write_h5ad(training_adata_path)
    logger.info(f"Saved training adata to {training_adata_path}")

    if config.graph_mode == "section_aware_3d":
        inferred_adatas = infer.infer_embedding_sections(config.sample_h5ad_dict)
    else:
        inferred_adatas = (
            (sample_name, infer.infer_embedding_single(sample_name, st_file))
            for sample_name, st_file in config.sample_h5ad_dict.items()
        )

    for sample_name, adata in inferred_adatas:
        # Calculate module scores and apply QC for each annotation
        if config.high_quality_cell_qc and config.annotation is not None:
            # Calculate module scores for this sample using the same DEGs from training
            logger.info(f"Calculating module scores for {sample_name}...")

            # Get DEG results from training data
            deg_results = training_adata.uns["rank_genes_groups"]

            # Keep the same gene list as training data, because the module score is based on the training data.
            # This is critical for proper normalization: without this filter, normalize_total()
            # would compute different scaling factors compared to training data.
            adata = adata[:, training_adata.var_names].copy()

            # Calculate module scores using existing DEG results
            adata = calculate_module_scores_from_degs(adata, deg_results, config.annotation)

            # Apply QC based on module score thresholds
            if config.annotation in adata.obs.columns:
                adata = apply_module_score_qc(
                    adata, config.annotation, module_score_threshold_dict
                )
            else:
                logger.warning(
                    f"Annotation '{config.annotation}' not found in {sample_name}, skipping QC"
                )

        # Transfer to human gene names
        adata = convert_to_human_genes(adata, gene_homolog_dict, species=config.species)

        # Compute the depth
        if config.data_layer in ["count", "counts"]:
            adata.obs["depth"] = np.array(adata.layers[config.data_layer].sum(axis=1)).flatten()

        output_h5ad_path_dict.update(
            write_latent_representation_outputs(
                adata,
                sample_name=sample_name,
                config=config,
            )
        )

    # Convert config to dict with all Path objects as strings
    config_dict = config.to_dict_with_paths_as_strings()

    # Convert output_h5ad_path_dict to strings
    output_h5ad_path_dict_str = {k: str(v) for k, v in output_h5ad_path_dict.items()}

    # Save metadata
    metadata = {
        "config": config_dict,
        "feature_schema": feature_schema,
        "model_info": {
            "model_path": str(config.model_path),
            "n_parameters": int(sum(p.numel() for p in gsmap_embedding_model.parameters())),
            "input_size": [int(x) for x in input_size],
            "hidden_size": int(config.hidden_size),
            "embedding_size": int(config.embedding_size),
            "batch_embedding_size": int(batch_embedding_size),
            "class_size": int(class_size),
            "distribution": distribution,
            "input_is_count_data": bool(is_count_data),
            "encoder_input_transform": "identity",
            "latent_embedding_statistic": gsmap_embedding_model.latent_embedding_statistic,
            "batch_conditioning": "decoder_only",
            "variational": variational,
            "use_tf": use_tf,
        },
        "training_info": {
            "n_cells_used": int(cell_size),
            "n_genes_used": int(len(hvg)),
            "n_common_genes": int(len(common_genes)),
            "batch_size": int(config.batch_size),
            "n_epochs": int(config.itermax),
            "patience": int(config.patience),
            "two_stage": config.two_stage,
            "graph_context_mode": get_trainning_data.graph_context_mode,
            "batch_label_classes": [
                str(label) for label in get_trainning_data.batch_label_encoder.classes_
            ],
        },
        "outputs": {
            "latent_files": output_h5ad_path_dict_str,
            "n_sections": len(output_h5ad_path_dict),
        },
        "annotation_info": {
            "annotation_key": config.annotation,
            "n_classes": int(class_size),
            "label_names": _serialize_label_names(label_name),
        },
    }

    # Save metadata to YAML file
    metadata_path = config.find_latent_metadata_path
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved metadata to {metadata_path}")

    return metadata
