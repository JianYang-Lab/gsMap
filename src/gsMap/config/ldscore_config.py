"""
Configuration dataclasses for the general LD score framework.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from gsMap.config.base import BaseConfig, ConfigWithAutoPaths

logger = logging.getLogger(__name__)


@dataclass
class LDScoreConfig(BaseConfig):
    """LD Score Weights Configuration"""

    # Paths
    bfile_root: Annotated[
        str, typer.Option(help="Reference panel prefix template (e.g., 'data/1000G.{chr}')")
    ]

    hm3_snp_dir: Annotated[
        Path,
        typer.Option(
            help="Directory containing per-chromosome HM3 SNP lists (e.g., hm.{chr}.snp). "
            "Typically set to '<ldscore_data_dir>/hapmap3_snps'.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ]

    output_dir: Annotated[
        Path | None,
        typer.Option(
            help="Output directory. If None, uses {workdir}/{project_name}/generate_ldscore"
        ),
    ] = None

    output_filename: Annotated[str, typer.Option(help="Prefix for output files")] = (
        "ld_score_weights"
    )

    # Omics Input
    omics_h5ad_path: Annotated[
        Path | None,
        typer.Option(
            help="Path to omics H5AD file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    # Mapping Input (Strategy A/B)
    mapping_type: Annotated[str, typer.Option(help="Mapping type: 'bed' or 'dict'")] = "bed"

    mapping_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to mapping file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            resolve_path=True,
        ),
    ] = None

    # Annotation Input (Strategy C - Direct Annotation Matrix)
    annot_file: Annotated[
        str | None,
        typer.Option(help="Template for annotation files (e.g., 'baseline.{chr}.annot.gz')"),
    ] = None

    # Mapping Strategy parameters
    feature_window_size: Annotated[
        int, typer.Option(help="bp window for mapping (e.g. TSS window)")
    ] = 0

    strategy: Annotated[
        str, typer.Option(help="Strategy: 'score', 'tss', 'center', 'allow_repeat'")
    ] = "score"

    # LD Calculation parameters
    ld_wind: Annotated[float, typer.Option(help="LD window size")] = 1.0

    ld_unit: Annotated[str, typer.Option(help="LD unit: 'SNP', 'KB', 'CM'")] = "CM"

    snps_only: Annotated[
        bool,
        typer.Option(
            help=(
                "Restrict the broad reference set to biallelic single-nucleotide "
                "A/C/G/T variants before decoding the BED payload."
            )
        ),
    ] = True

    mac_min: Annotated[
        int,
        typer.Option(
            help=(
                "Minimum minor-allele count for the broad reference SNP set. "
                "Variants are retained when MAC is strictly greater than this value; "
                "the default therefore implements MAC > 5."
            ),
            min=0,
        ),
    ] = 5

    maf_min: Annotated[
        float | None,
        typer.Option(
            help=(
                "Optional minimum MAF filter for the broad reference SNP set. "
                "Disabled by default because MAC > 5 is the default QC rule."
            ),
            min=0.0,
            max=0.5,
        ),
    ] = None

    exclude_mhc_from_regression: Annotated[
        bool,
        typer.Option(
            help=(
                "Exclude extended-MHC SNPs from HM3 regression targets and the "
                "HM3-only weight-LD universe. MHC variants remain in the broad "
                "reference SNP set used in annotation LD-score sums."
            )
        ),
    ] = True

    mhc_start_bp: Annotated[
        int,
        typer.Option(help="GRCh38 extended-MHC start coordinate (inclusive)"),
    ] = 28_510_120

    mhc_end_bp: Annotated[
        int,
        typer.Option(help="GRCh38 extended-MHC end coordinate (inclusive)"),
    ] = 33_480_577

    # Computation
    chromosomes: Annotated[
        str,
        typer.Option(
            help="Chromosomes to process. 'all' uses 1-22 autosomes, or provide a comma-separated list of chromosomes (e.g., '1,2,3')"
        ),
    ] = "all"

    batch_size_hm3: Annotated[int, typer.Option(help="Batch size for HM3 SNPs")] = 50

    # w_ld Calculation
    calculate_w_ld: Annotated[bool, typer.Option(help="Whether to calculate w_ld")] = False

    w_ld_dir: Annotated[Path | None, typer.Option(help="Directory for w_ld outputs")] = None

    def __post_init__(self):
        """
        Post-initialization processing:
        1. Parse chromosome string to list.
        2. Fix PLINK file template.
        3. Validate file existence.
        """
        if self.mac_min < 0:
            raise ValueError("mac_min must be non-negative")
        if self.maf_min is not None and not 0 <= self.maf_min <= 0.5:
            raise ValueError("maf_min must be between 0 and 0.5 or None")
        if self.mhc_start_bp < 0 or self.mhc_end_bp <= self.mhc_start_bp:
            raise ValueError("MHC coordinates must satisfy 0 <= start < end")

        # Parse output_dir if provided
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
        else:
            raise ValueError("output_dir must be provided for LDScoreConfig.")

        # 3. Parse Chromosomes
        if self.chromosomes == "all":
            self.chromosomes = list(range(1, 23))
        elif isinstance(self.chromosomes, str):
            # Handle string input like "1,2,22"
            self.chromosomes = [int(x) for x in self.chromosomes.split(",")]
        # Else it's already a list or properly set

        # 4. Handle PLINK Prefix Template
        # Ensure bfile_root has {chr} placeholder
        if "{chr}" not in self.bfile_root:
            logger.warning(
                f"The 'bfile_root' ({self.bfile_root}) does not contain the '{{chr}}' placeholder. "
                f"Appending '.{{chr}}' to the prefix."
            )
            self.bfile_root = f"{self.bfile_root}.{{chr}}"

        # 5. Validate PLINK Files
        logger.info("Validating PLINK binary files based on template...")
        missing_paths = []

        for chrom in self.chromosomes:
            prefix = self.bfile_root.format(chr=chrom)
            bed_path = Path(f"{prefix}.bed")
            if not bed_path.exists():
                missing_paths.append(str(bed_path))

        if missing_paths:
            error_msg = (
                f"PLINK .bed files missing for {len(missing_paths)} chromosomes. "
                "The following files were not found:\n"
                + "\n".join(f"  - {p}" for p in missing_paths)
            )
            raise FileNotFoundError(error_msg)

        logger.info(f"Confirmed all PLINK files exist for {len(self.chromosomes)} chromosomes.")

        self.show_config(LDScoreConfig)


@dataclass
class GenerateLDScoreConfig(ConfigWithAutoPaths):
    """Generate LDScore Configuration"""

    # Required from parent
    workdir: Annotated[
        Path,
        typer.Option(
            help="Path to the working directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ]

    chrom: Annotated[str, typer.Option(help='Chromosome id (1-22) or "all"')]

    bfile_root: Annotated[
        str, typer.Option(help="Root path for genotype plink bfiles (.bim, .bed, .fam)")
    ]

    gtf_annotation_file: Annotated[
        Path,
        typer.Option(
            help="Path to GTF annotation file", exists=True, file_okay=True, dir_okay=False
        ),
    ]

    sample_name: Annotated[str | None, typer.Option(help="Name of the sample")] = None

    keep_snp_root: str | None = None  # Internal field

    gene_window_size: Annotated[
        int, typer.Option(help="Gene window size in base pairs", min=1000, max=1000000)
    ] = 50000

    enhancer_annotation_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to enhancer annotation file", exists=True, file_okay=True, dir_okay=False
        ),
    ] = None

    snp_multiple_enhancer_strategy: Annotated[
        str,
        typer.Option(
            help="Strategy for handling multiple enhancers per SNP", case_sensitive=False
        ),
    ] = "max_mkscore"

    gene_window_enhancer_priority: Annotated[
        str | None, typer.Option(help="Priority between gene window and enhancer annotations")
    ] = None

    additional_baseline_annotation: Annotated[
        str | None, typer.Option(help="Path of additional baseline annotations")
    ] = None

    spots_per_chunk: Annotated[
        int, typer.Option(help="Number of spots per chunk", min=100, max=10000)
    ] = 1000

    ld_wind: Annotated[int, typer.Option(help="LD window size", min=1, max=10)] = 1

    ld_unit: Annotated[str, typer.Option(help="Unit for LD window", case_sensitive=False)] = "CM"

    # Additional fields
    ldscore_save_format: str = "feather"
    save_pre_calculate_snp_gene_weight_matrix: bool = False
    baseline_annotation_dir: str | None = None
    SNP_gene_pair_dir: str | None = None

    def __post_init__(self):
        super().__post_init__()
        self.show_config(GenerateLDScoreConfig)
