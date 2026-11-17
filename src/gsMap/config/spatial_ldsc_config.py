"""
Configuration for spatial LD score regression.
"""

import logging
import os
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Annotated, Literal

import typer

from gsMap.config.base import ConfigWithAutoPaths
from gsMap.config.gwas import load_gwas_mapping, normalize_gwas_mapping

logger = logging.getLogger("gsMap.config")


def resolve_spatial_ldsc_batch_size(batch_size: int | None, platform: str) -> int:
    """Resolve an automatic batch on the execution device, preserving overrides."""
    if batch_size is not None:
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError(
                "spots_per_chunk_quick_mode must be a positive integer or None (auto)"
            )
        return int(batch_size)
    return 400 if platform == "cpu" else 100


@dataclass
class SpatialLDSCComputeConfig:
    """Compute configuration for spatial LDSC step."""

    __display_in_quick_mode_cli__ = True

    use_gpu: Annotated[
        bool,
        typer.Option("--use-gpu/--no-gpu", help="Use GPU rather than CPU for spatial LDSC"),
        {"__display_in_quick_mode_cli__": True},
    ] = True

    ldsc_num_gpus: Annotated[
        int,
        typer.Option(
            help="Number of visible GPUs for spatial LDSC on this node. Cells are split "
            "between GPUs; each GPU processes all traits. Requires --use-gpu.",
            min=1,
        ),
        {"__display_in_quick_mode_cli__": True},
    ] = 1

    ldsc_share_ld: Annotated[
        bool,
        typer.Option(
            "--ldsc-share-ld/--no-ldsc-share-ld",
            help="Share each LD-score chunk across GWAS traits. Required for multiple GPUs.",
        ),
    ] = True

    ldsc_regression_backend: Annotated[
        Literal["auto", "jax", "moments"],
        typer.Option(
            help="Regression backend: auto uses fused moments for two baseline "
            "annotations plus intercept, otherwise JAX. jax disables this fast path."
        ),
    ] = "auto"

    ldsc_trait_batch_size: Annotated[
        int,
        typer.Option(
            help="Traits per regression group in shared-LD mode; all requested traits are "
            "processed, regardless of this group size.",
            min=1,
            max=16,
        ),
    ] = 4

    memmap_tmp_dir: Annotated[
        Path | None,
        typer.Option(
            help="Optional fast scratch directory. Concurrent spatial-LDSC processes stage and "
            "reuse one content-addressed, read-only marker cache per directory/node. If it does "
            "not fit, gsMap streams directly from project storage.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None

    ldsc_read_workers: Annotated[
        int,
        typer.Option(
            help="Reader/preparation workers. Shared-LD mode uses at most four readers for "
            "NumPy/memmap markers and one for shared file handles; non-shared mode uses "
            "this for sparse-transform or marker-preparation workers.",
            min=1,
        ),
    ] = 10

    ldsc_gwas_workers: Annotated[
        int,
        typer.Option(
            help="Concurrent GWAS file preparation in shared-LD mode (bounded prefetch). "
            "Use fewer workers to reduce temporary memory; at most 16 files are in flight.",
            min=1,
            max=16,
        ),
    ] = 16

    ldsc_compute_workers: Annotated[
        int,
        typer.Option(
            help="Requested CPU compute threads for the automatic moment backend, capped by "
            "available cores and thread limits; regression workers in non-shared mode. "
            "Does not select the number of GPUs.",
            min=1,
        ),
    ] = 10

    spots_per_chunk_quick_mode: Annotated[
        int | None,
        typer.Option(
            help="Spots per spatial-LDSC compute chunk. Omit to select automatically for the "
            "execution device; an explicit value overrides both CPU and GPU defaults.",
            min=1,
            show_default="auto: CPU=400, GPU=100",
        ),
    ] = None


@dataclass
class GWASSumstatsConfig:
    """Configuration for GWAS summary statistics."""

    trait_name: Annotated[str | None, typer.Option(help="Name of the trait for GWAS analysis")] = (
        None
    )

    sumstats_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to GWAS summary statistics file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    sumstats_config_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to sumstats config file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    sumstats_config_dict: dict[str, Path] = field(default_factory=dict)
    trait_labels: dict[str, str] = field(default_factory=dict)

    def trait_label(self, trait_id: str) -> str:
        """Human-readable report label; never an output filename or data key."""
        return self.trait_labels.get(trait_id, trait_id)

    @property
    def trait_name_list(self) -> list[str]:
        """Return the list of trait names to process."""
        return list(self.sumstats_config_dict.keys())

    def __post_init__(self):
        self._init_sumstats()

    def _init_sumstats(self):
        """
        Process sumstats input options and populate sumstats_config_dict.

        Priority:
        1. If ``sumstats_config_dict`` is already provided (non-empty), use it directly.
        2. Otherwise resolve from ``sumstats_file`` / ``sumstats_config_file``.
        """
        # If the caller already built the dict, validate and accept it directly.
        if self.sumstats_config_dict:
            self.sumstats_config_dict, self.trait_labels = normalize_gwas_mapping(
                self.sumstats_config_dict, labels=self.trait_labels
            )
            return

        sumstats_config_dict = {}

        if self.sumstats_file is None and self.sumstats_config_file is None:
            raise ValueError(
                "One of sumstats_file, sumstats_config_file, or sumstats_config_dict must be provided."
            )
        if self.sumstats_file is not None and self.sumstats_config_file is not None:
            raise ValueError(
                "Only one of sumstats_file and sumstats_config_file must be provided."
            )
        if self.sumstats_file is not None and self.trait_name is None:
            raise ValueError("trait_name must be provided if sumstats_file is provided.")
        if self.sumstats_config_file is not None and self.trait_name is not None:
            raise ValueError(
                "trait_name must not be provided if sumstats_config_file is provided."
            )
        # load the sumstats config file
        if self.sumstats_config_file is not None:
            self.sumstats_config_dict, self.trait_labels = load_gwas_mapping(
                self.sumstats_config_file, labels=self.trait_labels
            )
            return
        # load the sumstats file
        else:
            sumstats_config_dict[self.trait_name] = Path(self.sumstats_file)

        self.sumstats_config_dict, self.trait_labels = normalize_gwas_mapping(
            sumstats_config_dict, labels=self.trait_labels
        )


@dataclass
class SpatialLDSCCoreConfig(GWASSumstatsConfig):
    """Core configuration for spatial LDSC."""

    w_ld_dir: Annotated[
        Path | None,
        typer.Option(
            help="Directory containing the weights files (w_ld)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None

    additional_baseline_h5ad_path_list: Annotated[
        list[Path],
        typer.Option(
            help="List of additional baseline h5ad paths",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = field(default_factory=list)

    chisq_max: Annotated[int | None, typer.Option(help="Maximum chi-square value")] = None

    cell_indices_range: Annotated[
        tuple[int, int] | None,
        typer.Option(help="0-based range [start, end) of cell indices to process"),
    ] = None

    sample_filter: Annotated[
        str | None, typer.Option(help="Filter processing to a specific sample")
    ] = None

    n_blocks: Annotated[int, typer.Option(help="Number of jackknife blocks", min=2)] = 200

    # spots_per_chunk_quick_mode is inherited from SpatialLDSCComputeConfig

    snp_gene_weight_adata_path: Annotated[
        Path | None,
        typer.Option(
            help="Path to the SNP-gene weight matrix (H5AD format)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    ldscore_h5ad_path: Annotated[
        Path | None,
        typer.Option(
            help="Path to a pre-computed SNP × Cell LD score h5ad file (obs=SNPs, var=cells). "
            "When provided, skips on-the-fly LD score computation from the SNP-gene weight matrix. "
            "additional_baseline_h5ad_path_list must also be provided to supply all baseline annotations.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    # use_gpu is inherited from SpatialLDSCComputeConfig

    marker_score_feather_path: Annotated[
        Path | None,
        typer.Option(
            help="Path to marker score feather file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    marker_score_h5ad_path: Annotated[
        Path | None,
        typer.Option(
            help="Path to marker score h5ad file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    marker_score_format: Annotated[
        Literal["memmap", "feather", "h5ad"] | None, typer.Option(help="Format of marker scores")
    ] = None


@dataclass
class SpatialLDSCConfig(SpatialLDSCCoreConfig, SpatialLDSCComputeConfig, ConfigWithAutoPaths):
    """Spatial LDSC Configuration"""

    def __post_init__(self):
        super().__post_init__()
        validate_spatial_ldsc_settings(self)

        # Import here to avoid circular imports
        from gsMap.config.utils import configure_jax_platform, get_anndata_shape

        # Multi-GPU children select their allocated device before initializing
        # either framework. The coordinator must not allocate GPU contexts.
        if self.ldsc_num_gpus == 1:
            if self.use_gpu and self.ldsc_share_ld:
                os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
            configure_jax_platform(self.use_gpu)

        # Auto-detect marker_score_format if not specified
        if self.marker_score_format is None:
            if self.marker_score_feather_path is not None:
                self.marker_score_format = "feather"
                logger.info(
                    "Auto-detected marker_score_format as 'feather' based on marker_score_feather_path"
                )
            elif self.marker_score_h5ad_path is not None:
                self.marker_score_format = "h5ad"
                logger.info(
                    "Auto-detected marker_score_format as 'h5ad' based on marker_score_h5ad_path"
                )
            else:
                self.marker_score_format = "memmap"
                logger.info("Using default marker_score_format 'memmap'")

        # Validate cell_indices_range is 0-based
        if self.cell_indices_range is not None:
            # Validate exclusivity between sample_filter and cell_indices_range

            if self.sample_filter is not None:
                raise ValueError(
                    "Only one of sample_filter or cell_indices_range can be provided, not both. "
                    "Use sample_filter to filter by sample, or cell_indices_range to process specific cell indices."
                )

            start, end = self.cell_indices_range

            if start < 0:
                raise ValueError(f"cell_indices_range start must be >= 0, got {start}")

            # Check that start < end
            if start >= end:
                raise ValueError(
                    f"cell_indices_range start ({start}) must be less than end ({end})"
                )

            # Validate against actual data shape.
            # Direct ldscore mode takes priority: cells are var (columns) in the ldscore h5ad.
            if self.ldscore_h5ad_path is not None:
                ldscore_path = Path(self.ldscore_h5ad_path)
                if not ldscore_path.is_file():
                    raise FileNotFoundError(f"ldscore_h5ad_path not found at {ldscore_path}")
                shape = get_anndata_shape(str(ldscore_path))
                if shape is not None:
                    _, n_vars = shape  # n_vars = number of cells
                    if end > n_vars:
                        logger.warning(
                            f"cell_indices_range end ({end}) exceeds number of cells ({n_vars}) in ldscore h5ad. "
                            f"Setting end to {n_vars}."
                        )
                        end = n_vars
            elif self.marker_score_format == "memmap":
                # For memmap format, check the concatenated latent adata
                adata_path = self.concatenated_latent_adata_path
                shape = get_anndata_shape(str(adata_path))
                if shape is not None:
                    n_obs, _ = shape
                    if end > n_obs:
                        logger.warning(
                            f"cell_indices_range end ({end}) exceeds number of observations ({n_obs}). "
                            f"Setting end to {n_obs}."
                        )
                        end = n_obs
            elif self.marker_score_format == "h5ad":
                # For h5ad format, check the provided h5ad path
                adata_path = Path(self.marker_score_h5ad_path)
                if not adata_path.is_file():
                    raise FileNotFoundError(f"Marker score h5ad not found at {adata_path}")
                shape = get_anndata_shape(str(adata_path))
                if shape is not None:
                    n_obs, _ = shape
                    if end > n_obs:
                        logger.warning(
                            f"cell_indices_range end ({end}) exceeds number of observations ({n_obs}). "
                            f"Setting end to {n_obs}."
                        )
                        end = n_obs
            elif self.marker_score_format == "feather":
                # For feather format, validate the path exists
                feather_path = Path(self.marker_score_feather_path)
                if not feather_path.is_file():
                    raise FileNotFoundError(
                        f"Marker score feather file not found at {feather_path}"
                    )

                # Use pyarrow to get the number of rows and validate end
                try:
                    import pyarrow.feather as feather

                    # Read metadata without loading full data
                    feather_table = feather.read_table(
                        str(feather_path), memory_map=True, columns=[]
                    )
                    n_obs = feather_table.num_rows
                    if end > n_obs:
                        logger.warning(
                            f"cell_indices_range end ({end}) exceeds number of rows ({n_obs}) in feather file. "
                            f"Setting end to {n_obs}."
                        )
                        end = n_obs
                except ImportError:
                    logger.warning(
                        "pyarrow not available. Cannot validate cell_indices_range against feather file. "
                        "Install pyarrow to enable validation."
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"Could not read feather file metadata: {e}")

            # Update cell_indices_range with validated values
            self.cell_indices_range = (start, end)
            logger.info(f"Processing cell_indices_range: [{start}, {end})")

        # Handle w_ld_dir
        if self.w_ld_dir is None:
            self.w_ld_dir = resolve_w_ld_dir(self)
            logger.info(
                f"Using weights directory generated in the generate_ldscore step: {self.w_ld_dir}"
            )
        else:
            logger.info(f"Using provided weights directory: {self.w_ld_dir}")

        self.show_config(SpatialLDSCConfig)


def validate_spatial_ldsc_settings(config) -> None:
    """Static spatial LDSC checks that need no data; used by quick mode before step 1."""
    from gsMap.spatial_ldsc.execution.trait_loading import resolve_gwas_workers

    get = lambda name, default=None: getattr(config, name, default)  # noqa: E731
    if get("n_blocks", 200) < 2:
        raise ValueError("n_blocks must be at least 2 for jackknife standard errors")
    resolve_gwas_workers(get("ldsc_gwas_workers", 4), 1)
    backend = get("ldsc_regression_backend", "auto")
    if backend not in ("auto", "jax", "moments"):
        raise ValueError("ldsc_regression_backend must be auto, jax, or moments")
    if backend == "moments" and not get("ldsc_share_ld", True):
        raise ValueError("The moments backend requires ldsc_share_ld=True")
    num_gpus, trait_batch = get("ldsc_num_gpus", 1), get("ldsc_trait_batch_size", 1)
    if num_gpus < 1 or not 1 <= trait_batch <= 16:
        raise ValueError("ldsc_num_gpus must be positive; ldsc_trait_batch_size must be 1..16")
    if num_gpus > 1 and (not get("use_gpu", True) or not get("ldsc_share_ld", True)):
        raise ValueError("Multiple LDSC GPUs require use_gpu=True and ldsc_share_ld=True")
    if get("snp_gene_weight_adata_path") is None and get("ldscore_h5ad_path") is None:
        raise ValueError(
            "One of snp_gene_weight_adata_path or ldscore_h5ad_path must be provided."
        )
    if get("ldscore_h5ad_path") is not None and not get("additional_baseline_h5ad_path_list"):
        raise ValueError(
            "additional_baseline_h5ad_path_list must be provided when using ldscore_h5ad_path "
            "(it supplies all baseline annotations for the regression)."
        )


def resolve_w_ld_dir(config) -> Path:
    """The configured weights directory, or the one generated by ldscore-weight-matrix."""
    if getattr(config, "w_ld_dir", None) is not None:
        return Path(config.w_ld_dir)
    w_ld_dir = Path(config.ldscore_save_dir) / "w_ld"
    if not w_ld_dir.exists():
        raise ValueError(
            "No w_ld_dir provided and no weights directory found in generate_ldscore output. "
            "Either provide --w-ld-dir or run generate_ldscore first."
        )
    return w_ld_dir


def check_spatial_ldsc_done(config: SpatialLDSCConfig, trait_name: str) -> bool:
    """
    Check if spatial_ldsc step is done for a specific trait.
    """
    result_file = config.get_ldsc_result_file(trait_name)
    return result_file.exists()
