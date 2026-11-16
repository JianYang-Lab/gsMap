"""
Utility functions for gsMap configuration and data validation.
"""

import logging
from collections import OrderedDict
from pathlib import Path

import h5py
import yaml

logger = logging.getLogger("gsMap.config")


def _add_h5ad_path(sample_h5ad_dict: OrderedDict, h5ad_path: Path) -> None:
    """Append an H5AD path without dropping files that share a basename."""

    base_name = h5ad_path.stem
    sample_name = base_name
    duplicate_ordinal = 2
    while sample_name in sample_h5ad_dict:
        sample_name = f"{base_name}__{duplicate_ordinal}"
        duplicate_ordinal += 1
    if sample_name != base_name:
        logger.warning(
            "Duplicate H5AD sample name %r; preserving %s as %r",
            base_name,
            h5ad_path,
            sample_name,
        )
    sample_h5ad_dict[sample_name] = h5ad_path


def configure_jax_platform(use_accelerator: bool = True):
    """Configure JAX platform based on availability of accelerators.

    Args:
        use_accelerator: If True, try to use GPU then TPU if available,
                        otherwise fall back to CPU. If False, force CPU usage.

    Raises:
        ImportError: If JAX is not installed.
    """
    try:
        import os

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)

        import jax
        from jax import config as jax_config

        if not use_accelerator:
            # Recent CUDA-enabled JAX wheels discover every installed PJRT plugin,
            # even when CPU is explicitly selected. On a CPU-only node that CUDA
            # version probe logs a full cuInit traceback before falling back. Force
            # discovery while the irrelevant CUDA constraint check is disabled,
            # then restore the caller's environment immediately.
            skip_check = "JAX_SKIP_CUDA_CONSTRAINTS_CHECK"
            previous_skip_check = os.environ.get(skip_check)
            os.environ[skip_check] = "1"
            try:
                jax_config.update("jax_platform_name", "cpu")
                jax.devices("cpu")
            finally:
                if previous_skip_check is None:
                    os.environ.pop(skip_check, None)
                else:
                    os.environ[skip_check] = previous_skip_check
            logger.info("JAX configured to use CPU for computations (accelerators disabled)")
            return

        # Priority list for accelerators
        platforms = ["gpu", "tpu"]
        configured = False

        for platform in platforms:
            try:
                devices = jax.devices(platform)
                if len(devices) > 0:
                    jax_config.update("jax_platform_name", platform)
                    logger.info(f"JAX configured to use {platform.upper()} for computations.")
                    logger.info(
                        f" ({len(devices)} device(s) detected). Device {devices[0]} will be used."
                    )
                    configured = True
                    break
            except (RuntimeError, ValueError):
                continue

        if not configured:
            jax_config.update("jax_platform_name", "cpu")
            logger.info("No GPU or TPU detected, JAX configured to use CPU for computations")

    except ImportError:
        raise ImportError(
            "JAX is required but not installed. Please install JAX by running: "
            "pip install jax jaxlib (for CPU) or see JAX documentation for GPU/TPU installation."
        ) from None


def get_anndata_shape(h5ad_path: str):
    """Get the shape (n_obs, n_vars) of an AnnData file without loading it."""
    with h5py.File(h5ad_path, "r") as f:
        # 1. Verify it's a valid AnnData file by checking metadata
        if f.attrs.get("encoding-type") != "anndata":
            logger.error(f"File '{h5ad_path}' does not appear to be a valid AnnData file.")
            return None

        # 2. Determine n_obs and n_vars from the primary metadata sources
        if "obs" not in f or "var" not in f:
            logger.error("AnnData file is missing 'obs' or 'var' group.")
            return None

        # Get the name of the index column from attributes
        obs_index_key = f["obs"].attrs.get("_index", None)
        var_index_key = f["var"].attrs.get("_index", None)

        if not obs_index_key or obs_index_key not in f["obs"]:
            logger.error("Could not determine index for 'obs'.")
            return None
        if not var_index_key or var_index_key not in f["var"]:
            logger.error("Could not determine index for 'var'.")
            return None

        # The shape is the length of these index arrays
        obs_obj = f["obs"][obs_index_key]
        if isinstance(obs_obj, h5py.Group):
            obs_obj = obs_obj["categories"]
        n_obs = obs_obj.shape[0]

        var_obj = f["var"][var_index_key]
        if isinstance(var_obj, h5py.Group):
            var_obj = var_obj["categories"]
        n_vars = var_obj.shape[0]

        return n_obs, n_vars


def inspect_h5ad_structure(filename):
    """
    Inspect the structure of an h5ad file without loading data.

    Returns dict with keys present in each slot.
    """
    structure = {}

    with h5py.File(filename, "r") as f:
        # Check main slots
        slots = ["obs", "var", "obsm", "varm", "obsp", "varp", "uns", "layers", "X", "raw"]

        for slot in slots:
            if slot in f:
                if slot in ["obsm", "varm", "obsp", "varp", "layers", "uns"]:
                    # These are groups containing multiple keys
                    structure[slot] = list(f[slot].keys())
                elif slot in ["obs", "var"]:
                    # These are dataframes - get column names
                    if "column-order" in f[slot].attrs:
                        structure[slot] = list(f[slot].attrs["column-order"])
                    else:
                        structure[slot] = list(f[slot].keys())
                else:
                    # X, raw - just note they exist
                    structure[slot] = True

    return structure


def validate_h5ad_structure(sample_h5ad_dict, required_fields, optional_fields=None):
    """
    Validate h5ad files have required structure.

    Args:
        sample_h5ad_dict: OrderedDict of {sample_name: h5ad_path}
        required_fields: Dict of {field_name: (slot, field_key, error_msg_template)}
            e.g., {'spatial': ('obsm', 'spatial', 'Spatial key')}
        optional_fields: Dict of {field_name: (slot, field_key)} for fields to warn about

    Returns:
        None, raises ValueError if required fields are missing
    """
    for sample_name, h5ad_path in sample_h5ad_dict.items():
        if not h5ad_path.exists():
            raise FileNotFoundError(f"H5AD file not found for sample '{sample_name}': {h5ad_path}")

        # Inspect h5ad structure
        structure = inspect_h5ad_structure(h5ad_path)

        # Check required fields
        for field_name, (slot, field_key, error_msg) in required_fields.items():
            if field_key is None:  # Skip if field not specified
                continue

            # Special handling for data_layer
            if field_name == "data_layer" and field_key != "X":
                if "layers" not in structure or field_key not in structure.get("layers", []):
                    raise ValueError(
                        f"Data layer '{field_key}' not found in layers for sample '{sample_name}'. "
                        f"Available layers: {structure.get('layers', [])}"
                    )
            elif field_name == "data_layer" and field_key == "X":
                if "X" not in structure:
                    raise ValueError(f"X matrix not found in h5ad file for sample '{sample_name}'")
            else:
                # Standard validation for obsm, obs, etc.
                if slot not in structure or field_key not in structure.get(slot, []):
                    available = structure.get(slot, [])
                    raise ValueError(
                        f"{error_msg} '{field_key}' not found in {slot} for sample '{sample_name}'. "
                        f"Available keys in {slot}: {available}"
                    )

        # Check optional fields (warn only)
        if optional_fields:
            for field_name, (slot, field_key) in optional_fields.items():
                if field_key is None:  # Skip if field not specified
                    continue

                if slot not in structure or field_key not in structure.get(slot, []):
                    available = structure.get(slot, [])
                    logger.warning(
                        f"Optional field '{field_key}' not found in {slot} for sample '{sample_name}'. "
                        f"Available keys in {slot}: {available}"
                    )


def process_h5ad_inputs(config, input_options):
    """
    Process h5ad input options and create sample_h5ad_dict.

    Args:
        config: Configuration object with h5ad input fields
        input_options: Dict mapping option names to (field_name, processing_type)
            e.g., {'h5ad_yaml': ('h5ad_yaml', 'yaml'),
                   'h5ad': ('h5ad', 'list'),
                   'h5ad_list_file': ('h5ad_list_file', 'file')}

    Returns:
        OrderedDict of {sample_name: h5ad_path}
    """

    if config.sample_h5ad_dict is not None and len(config.sample_h5ad_dict) > 0:
        logger.info("Using pre-defined sample_h5ad_dict from configuration")
        return OrderedDict(config.sample_h5ad_dict)

    sample_h5ad_dict = OrderedDict()

    # Check which options are provided
    options_provided = []
    for option_name, (field_name, _) in input_options.items():
        if hasattr(config, field_name) and getattr(config, field_name):
            options_provided.append(option_name)

    # Ensure at most one option is provided
    if len(options_provided) > 1:
        raise ValueError(
            f"At most one input option can be provided. Got {len(options_provided)}: {', '.join(options_provided)}. "
            f"Please provide only one of: {', '.join(input_options.keys())}"
        )

    # Process the provided input option
    for option_name, (field_name, processing_type) in input_options.items():
        field_value = getattr(config, field_name, None)
        if not field_value:
            continue

        if processing_type == "yaml":
            logger.info(f"Using {option_name}: {field_value}")
            yaml_file_path = Path(field_value)
            yaml_parent_dir = yaml_file_path.parent
            with open(field_value) as f:
                h5ad_data = yaml.safe_load(f)
                if not isinstance(h5ad_data, dict) or not h5ad_data:
                    raise ValueError(f"H5AD YAML must contain a non-empty mapping: {field_value}")
                for sample_name, h5ad_path in h5ad_data.items():
                    h5ad_path = Path(h5ad_path)
                    # Resolve relative paths relative to yaml file location
                    if not h5ad_path.is_absolute():
                        h5ad_path = yaml_parent_dir / h5ad_path
                    sample_h5ad_dict[sample_name] = h5ad_path

        elif processing_type == "list":
            # Handle single string or Path input by wrapping in a list
            if isinstance(field_value, str | Path):
                field_value = [field_value]

            logger.info(f"Using {option_name} with {len(field_value)} files")
            for h5ad_path in field_value:
                h5ad_path = Path(h5ad_path)
                _add_h5ad_path(sample_h5ad_dict, h5ad_path)

        elif processing_type == "file":
            logger.info(f"Using {option_name}: {field_value}")
            list_file_path = Path(field_value)
            list_file_parent_dir = list_file_path.parent
            with open(field_value) as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        h5ad_path = Path(line)
                        # Resolve relative paths relative to list file location
                        if not h5ad_path.is_absolute():
                            h5ad_path = list_file_parent_dir / h5ad_path
                        _add_h5ad_path(sample_h5ad_dict, h5ad_path)
        break

    return sample_h5ad_dict


def verify_homolog_file_format(config):
    if config.homolog_file is not None:
        logger.info(
            f"User provided homolog file to map gene names to human: {config.homolog_file}"
        )
        # check the format of the homolog file
        with open(config.homolog_file) as f:
            first_line = f.readline().strip()
            _n_col = len(first_line.split())
            if _n_col != 2:
                raise ValueError(
                    f"Invalid homolog file format. Expected 2 columns, first column should be other species gene name, second column should be human gene name. "
                    f"Got {_n_col} columns in the first line."
                )
            else:
                first_col_name, second_col_name = first_line.split()
                config.species = first_col_name
                logger.info(
                    f"Homolog file provided and will map gene name from column1:{first_col_name} to column2:{second_col_name}"
                )
    else:
        logger.info("No homolog file provided. Run in human mode.")
