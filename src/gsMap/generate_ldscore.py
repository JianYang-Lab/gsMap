"""
Module for generating LD scores for each spot in spatial transcriptomics data.

This module is responsible for assigning gene specificity scores to SNPs
and computing stratified LD scores that will be used for spatial LDSC analysis.
"""

import gc
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
from scipy.sparse import csr_matrix
from tqdm import trange

from gsMap.config import GenerateLDScoreConfig
from gsMap.utils.generate_r2_matrix import PlinkBEDFile

# Configure warning behavior more precisely
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
logger = logging.getLogger(__name__)


def load_gtf(
    gtf_file: str, mk_score: pd.DataFrame, window_size: int
) -> tuple[pr.PyRanges, pd.DataFrame]:
    """
    Load and process the gene annotation file (GTF).

    Parameters
    ----------
    gtf_file : str
        Path to the GTF file
    mk_score : pd.DataFrame
        DataFrame containing marker scores
    window_size : int
        Window size around gene bodies in base pairs

    Returns
    -------
    tuple
        A tuple containing (gtf_pr, mk_score) where:
        - gtf_pr is a PyRanges object with gene coordinates
        - mk_score is the filtered marker score DataFrame
    """
    logger.info("Loading GTF data from %s", gtf_file)

    # Load GTF file
    gtf = pr.read_gtf(gtf_file, as_df=True)

    # Filter for gene features
    gtf = gtf[gtf["Feature"] == "gene"]

    # Find common genes between GTF and marker scores
    # common_gene = np.intersect1d(mk_score.index, gtf.gene_name)
    common_gene = list(set(mk_score.index) & set(gtf.gene_name))
    logger.info(f"Found {len(common_gene)} common genes between GTF and marker scores")

    # Filter GTF and marker scores to common genes
    gtf = gtf[gtf.gene_name.isin(common_gene)]
    mk_score = mk_score[mk_score.index.isin(common_gene)]

    # Remove duplicated gene entries
    gtf = gtf.drop_duplicates(subset="gene_name", keep="first")

    # Process the GTF (open window around gene coordinates)
    gtf_bed = gtf[["Chromosome", "Start", "End", "gene_name", "Strand"]].copy()
    gtf_bed["Chromosome"] = gtf_bed["Chromosome"].apply(
        lambda x: f"chr{x}" if not str(x).startswith("chr") else x
    )
    gtf_bed.loc[:, "TSS"] = gtf_bed["Start"]
    gtf_bed.loc[:, "TED"] = gtf_bed["End"]

    # Create windows around genes
    gtf_bed.loc[:, "Start"] = gtf_bed["TSS"] - window_size
    gtf_bed.loc[:, "End"] = gtf_bed["TED"] + window_size
    gtf_bed.loc[gtf_bed["Start"] < 0, "Start"] = 0

    # Handle genes on negative strand (swap TSS and TED)
    tss_neg = gtf_bed.loc[gtf_bed["Strand"] == "-", "TSS"]
    ted_neg = gtf_bed.loc[gtf_bed["Strand"] == "-", "TED"]
    gtf_bed.loc[gtf_bed["Strand"] == "-", "TSS"] = ted_neg
    gtf_bed.loc[gtf_bed["Strand"] == "-", "TED"] = tss_neg
    gtf_bed = gtf_bed.drop("Strand", axis=1)

    # Convert to PyRanges
    gtf_pr = pr.PyRanges(gtf_bed)

    return gtf_pr, mk_score


def load_marker_score(mk_score_file: str) -> pd.DataFrame:
    """
    Load marker scores from a feather file.

    Parameters
    ----------
    mk_score_file : str
        Path to the marker score feather file

    Returns
    -------
    pd.DataFrame
        DataFrame with marker scores indexed by gene names
    """
    mk_score = pd.read_feather(mk_score_file).set_index("HUMAN_GENE_SYM").rename_axis("gene_name")
    mk_score = mk_score.astype(np.float32, copy=False)
    return mk_score


def overlaps_gtf_bim(gtf_pr: pr.PyRanges, bim_pr: pr.PyRanges) -> pd.DataFrame:
    """
    Find overlaps between GTF and BIM data, and select nearest gene for each SNP.

    Parameters
    ----------
    gtf_pr : pr.PyRanges
        PyRanges object with gene coordinates
    bim_pr : pr.PyRanges
        PyRanges object with SNP coordinates

    Returns
    -------
    pd.DataFrame
        DataFrame with SNP-gene pairs where each SNP is matched to its closest gene
    """
    # Join the PyRanges objects to find overlaps
    overlaps = gtf_pr.join(bim_pr)
    overlaps = overlaps.df

    # Calculate distance to TSS
    overlaps["Distance"] = np.abs(overlaps["Start_b"] - overlaps["TSS"])

    # For each SNP, select the closest gene
    nearest_genes = overlaps.loc[overlaps.groupby("SNP").Distance.idxmin()]

    return nearest_genes


class LDScoreCalculator:
    """
    Class for calculating LD scores from gene specificity scores.
    """

    def __init__(self, config: GenerateLDScoreConfig):
        """Initialize LDScoreCalculator."""
        self.config = config
        self.validate_config()

        # Load marker scores
        self.mk_score = load_marker_score(config.mkscore_feather_path)

        # Load GTF and get common markers
        self.gtf_pr, self.mk_score_common = load_gtf(
            config.gtf_annotation_file, self.mk_score, window_size=config.gene_window_size
        )

        # Initialize enhancer data if provided
        self.enhancer_pr = self._initialize_enhancer() if config.enhancer_annotation_file else None

    def validate_config(self):
        """Validate configuration parameters."""
        if not Path(self.config.mkscore_feather_path).exists():
            raise FileNotFoundError(
                f"Marker score file not found: {self.config.mkscore_feather_path}"
            )

        if not Path(self.config.gtf_annotation_file).exists():
            raise FileNotFoundError(
                f"GTF annotation file not found: {self.config.gtf_annotation_file}"
            )

        if (
            self.config.enhancer_annotation_file
            and not Path(self.config.enhancer_annotation_file).exists()
        ):
            raise FileNotFoundError(
                f"Enhancer annotation file not found: {self.config.enhancer_annotation_file}"
            )

    def _initialize_enhancer(self) -> pr.PyRanges:
        """
        Initialize enhancer data.

        Returns
        -------
        pr.PyRanges
            PyRanges object with enhancer data
        """
        # Load enhancer data
        enhancer_df = pr.read_bed(self.config.enhancer_annotation_file, as_df=True)
        enhancer_df.set_index("Name", inplace=True)
        enhancer_df.index.name = "gene_name"

        # Keep common genes and add marker score information
        avg_mkscore = pd.DataFrame(self.mk_score_common.mean(axis=1), columns=["avg_mkscore"])
        enhancer_df = enhancer_df.join(
            avg_mkscore,
            how="inner",
            on="gene_name",
        )

        # Add TSS information
        enhancer_df["TSS"] = self.gtf_pr.df.set_index("gene_name").reindex(enhancer_df.index)[
            "TSS"
        ]

        # Convert to PyRanges
        return pr.PyRanges(enhancer_df.reset_index())

    def process_chromosome(self, chrom: int):
        """
        Process a single chromosome to calculate LD scores.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        logger.info(f"Processing chromosome {chrom}")

        # Initialize PlinkBEDFile once for this chromosome
        plink_bed = PlinkBEDFile(f"{self.config.bfile_root}.{chrom}")

        # Get SNPs passing MAF filter using built-in method
        self.snp_pass_maf = plink_bed.get_snps_by_maf(0.05)

        # Get SNP-gene dummy pairs
        self.snp_gene_pair_dummy = self._get_snp_gene_dummy(chrom, plink_bed)

        # Apply SNP filter if provided
        self._apply_snp_filter(chrom)

        # Process additional baseline annotations if provided
        if self.config.additional_baseline_annotation:
            self._process_additional_baseline(chrom, plink_bed)
        else:
            # Calculate SNP-gene weight matrix using built-in methods
            ld_scores = plink_bed.get_ldscore(
                annot_matrix=self.snp_gene_pair_dummy.values,
                ld_wind=self.config.ld_wind,
                ld_unit=self.config.ld_unit,
            )

            self.snp_gene_weight_matrix = pd.DataFrame(
                ld_scores,
                index=self.snp_gene_pair_dummy.index,
                columns=self.snp_gene_pair_dummy.columns,
            )

            # Apply SNP filter if needed
            if self.keep_snp_mask is not None:
                self.snp_gene_weight_matrix = self.snp_gene_weight_matrix[self.keep_snp_mask]

        # Generate w_ld file if keep_snp_root is provided
        if self.config.keep_snp_root:
            self._generate_w_ld(chrom, plink_bed)

        # Save pre-calculated SNP-gene weight matrix if requested
        self._save_snp_gene_weight_matrix_if_needed(chrom)

        # Convert to sparse matrix for memory efficiency
        self.snp_gene_weight_matrix = csr_matrix(self.snp_gene_weight_matrix)
        logger.info(f"SNP-gene weight matrix shape: {self.snp_gene_weight_matrix.shape}")

        # Calculate baseline LD scores
        logger.info(f"Calculating baseline LD scores for chr{chrom}")
        self._calculate_baseline_ldscores(chrom, plink_bed)

        # Calculate LD scores for annotation
        logger.info(f"Calculating annotation LD scores for chr{chrom}")
        self._calculate_annotation_ldscores(chrom, plink_bed)

        # Clear memory
        self._clear_memory()

    def _generate_w_ld(self, chrom: int, plink_bed):
        """
        Generate w_ld file for the chromosome using filtered SNPs.

        Parameters
        ----------
        chrom : int
            Chromosome number
        plink_bed : PlinkBEDFile
            Initialized PlinkBEDFile object
        """
        if not self.config.keep_snp_root:
            logger.info(
                f"Skipping w_ld generation for chr{chrom} as keep_snp_root is not provided"
            )
            return

        logger.info(f"Generating w_ld for chr{chrom}")

        # Get the indices of SNPs to keep based on the keep_snp
        keep_snps_indices = plink_bed.bim_df[
            plink_bed.bim_df.SNP.isin(self.snp_name)
        ].index.tolist()

        # Create a simple unit annotation (all ones) for the filtered SNPs
        unit_annotation = np.ones((len(keep_snps_indices), 1), dtype="float32")

        # Calculate LD scores
        w_ld_scores = plink_bed.get_ldscore(
            annot_matrix=unit_annotation,
            ld_wind=self.config.ld_wind,
            ld_unit=self.config.ld_unit,
            keep_snps_index=keep_snps_indices,
        )

        # Create the w_ld DataFrame
        bim_subset = plink_bed.bim_df.loc[keep_snps_indices]
        w_ld_df = pd.DataFrame(
            {
                "SNP": bim_subset.SNP,
                "L2": w_ld_scores.flatten(),
                "CHR": bim_subset.CHR,
                "BP": bim_subset.BP,
                "CM": bim_subset.CM,
            }
        )

        # Reorder columns
        w_ld_df = w_ld_df[["CHR", "SNP", "BP", "CM", "L2"]]

        # Save to file
        w_ld_dir = Path(self.config.ldscore_save_dir) / "w_ld"
        w_ld_dir.mkdir(parents=True, exist_ok=True)
        w_ld_file = w_ld_dir / f"weights.{chrom}.l2.ldscore.gz"
        w_ld_df.to_csv(w_ld_file, sep="\t", index=False, compression="gzip")

        logger.info(f"Saved w_ld for chr{chrom} to {w_ld_file}")

    def _apply_snp_filter(self, chrom: int):
        """
        Apply SNP filter based on keep_snp_root.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        if self.config.keep_snp_root is not None:
            keep_snp_file = f"{self.config.keep_snp_root}.{chrom}.snp"
            keep_snp = pd.read_csv(keep_snp_file, header=None)[0].to_list()
            self.keep_snp_mask = self.snp_gene_pair_dummy.index.isin(keep_snp)
            self.snp_name = self.snp_gene_pair_dummy.index[self.keep_snp_mask].to_list()
            logger.info(f"Kept {len(self.snp_name)} SNPs after filtering with {keep_snp_file}")
            logger.info("These filtered SNPs will be used to calculate w_ld")
        else:
            self.keep_snp_mask = None
            self.snp_name = self.snp_gene_pair_dummy.index.to_list()
            logger.info(f"Using all {len(self.snp_name)} SNPs (no filter applied)")
            logger.warning("No keep_snp_root provided, all SNPs will be used to calculate w_ld.")

    def _process_additional_baseline(self, chrom: int, plink_bed):
        """
        Process additional baseline annotations.

        Parameters
        ----------
        chrom : int
            Chromosome number
        plink_bed : PlinkBEDFile
            Initialized PlinkBEDFile object
        """
        # Load additional baseline annotations
        additional_baseline_path = Path(self.config.additional_baseline_annotation)
        annot_file_path = additional_baseline_path / f"baseline.{chrom}.annot.gz"

        # Verify file existence
        if not annot_file_path.exists():
            raise FileNotFoundError(
                f"Additional baseline annotation file not found: {annot_file_path}"
            )

        # Load annotations
        additional_baseline_df = pd.read_csv(annot_file_path, sep="\t")
        additional_baseline_df.set_index("SNP", inplace=True)

        # Drop unnecessary columns
        for col in ["CHR", "BP", "CM"]:
            if col in additional_baseline_df.columns:
                additional_baseline_df.drop(col, axis=1, inplace=True)

        # Check for SNPs not in the additional baseline
        missing_snps = ~self.snp_gene_pair_dummy.index.isin(additional_baseline_df.index)
        missing_count = missing_snps.sum()

        if missing_count > 0:
            logger.warning(
                f"{missing_count} SNPs not found in additional baseline annotations. "
                "Setting their values to 0."
            )
        additional_baseline_df = additional_baseline_df.reindex(
            self.snp_gene_pair_dummy.index, fill_value=0
        )

        # Combine annotations into a single matrix
        combined_annotations = pd.concat(
            [self.snp_gene_pair_dummy, additional_baseline_df], axis=1
        )

        # Calculate LD scores
        ld_scores = plink_bed.get_ldscore(
            annot_matrix=combined_annotations.values.astype(np.float32, copy=False),
            ld_wind=self.config.ld_wind,
            ld_unit=self.config.ld_unit,
        )

        # Split results
        # total_cols = combined_annotations.shape[1]
        gene_cols = self.snp_gene_pair_dummy.shape[1]
        #         baseline_cols = additional_baseline_df.shape[1]

        # Create DataFrames with proper indices and columns
        self.snp_gene_weight_matrix = pd.DataFrame(
            ld_scores[:, :gene_cols],
            index=combined_annotations.index,
            columns=self.snp_gene_pair_dummy.columns,
        )

        additional_ldscore = pd.DataFrame(
            ld_scores[:, gene_cols:],
            index=combined_annotations.index,
            columns=additional_baseline_df.columns,
        )

        # Filter by keep_snp_mask if specified
        if self.keep_snp_mask is not None:
            additional_ldscore = additional_ldscore[self.keep_snp_mask]
            self.snp_gene_weight_matrix = self.snp_gene_weight_matrix[self.keep_snp_mask]

        # Save additional baseline LD scores
        ld_score_file = f"{self.config.ldscore_save_dir}/additional_baseline/baseline.{chrom}.l2.ldscore.feather"
        m_file_path = f"{self.config.ldscore_save_dir}/additional_baseline/baseline.{chrom}.l2.M"
        m_5_file_path = (
            f"{self.config.ldscore_save_dir}/additional_baseline/baseline.{chrom}.l2.M_5_50"
        )
        Path(m_file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save LD scores
        self._save_ldscore_to_feather(
            additional_ldscore.values,
            column_names=additional_ldscore.columns,
            save_file_name=ld_score_file,
        )

        # Calculate and save M values
        m_chr_chunk = additional_baseline_df.values.sum(axis=0, keepdims=True)
        m_5_chr_chunk = additional_baseline_df.loc[self.snp_pass_maf].values.sum(
            axis=0, keepdims=True
        )

        # Save M statistics
        np.savetxt(m_file_path, m_chr_chunk, delimiter="\t")
        np.savetxt(m_5_file_path, m_5_chr_chunk, delimiter="\t")

    def _save_snp_gene_weight_matrix_if_needed(self, chrom: int):
        """
        Save pre-calculated SNP-gene weight matrix if requested.

        Parameters
        ----------
        chrom : int
            Chromosome number
        """
        if self.config.save_pre_calculate_snp_gene_weight_matrix:
            save_dir = Path(self.config.ldscore_save_dir) / "snp_gene_weight_matrix"
            save_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Saving SNP-gene weight matrix for chr{chrom}")

            save_path = save_dir / f"{chrom}.snp_gene_weight_matrix.feather"
            self.snp_gene_weight_matrix.reset_index().to_feather(save_path)

    def _calculate_baseline_ldscores(self, chrom: int, plink_bed):
        """
        Calculate and save baseline LD scores.

        Parameters
        ----------
        chrom : int
            Chromosome number
        plink_bed : PlinkBEDFile
            Initialized PlinkBEDFile object
        """
        # Create baseline scores
        baseline_mk_score = np.ones((self.snp_gene_pair_dummy.shape[1], 2))
        baseline_mk_score[-1, 0] = 0  # all_gene column

        baseline_df = pd.DataFrame(
            baseline_mk_score, index=self.snp_gene_pair_dummy.columns, columns=["all_gene", "base"]
        )

        # Define file paths
        ld_score_file = (
            f"{self.config.ldscore_save_dir}/baseline/baseline.{chrom}.l2.ldscore.feather"
        )
        m_file = f"{self.config.ldscore_save_dir}/baseline/baseline.{chrom}.l2.M"
        m_5_file = f"{self.config.ldscore_save_dir}/baseline/baseline.{chrom}.l2.M_5_50"

        # Calculate LD scores
        ldscore_chunk = self._calculate_ldscore_from_weights(
            baseline_df, plink_bed, drop_dummy_na=False
        )

        # Save LD scores and M values
        self._save_ldscore_to_feather(
            ldscore_chunk,
            column_names=baseline_df.columns,
            save_file_name=ld_score_file,
        )

        self._calculate_and_save_m_values(
            baseline_df,
            m_file,
            m_5_file,
            drop_dummy_na=False,
        )

        # If keep_snp_root is not provided, use the first column of baseline ldscore as w_ld
        if not self.config.keep_snp_root:
            self._save_baseline_as_w_ld(chrom, ldscore_chunk, plink_bed)

    def _save_baseline_as_w_ld(self, chrom: int, ldscore_chunk: np.ndarray, plink_bed):
        """
        Save the first column of baseline ldscore as w_ld.

        Parameters
        ----------
        chrom : int
            Chromosome number
        ldscore_chunk : np.ndarray
            Array with baseline LD scores
        plink_bed : PlinkBEDFile
            Initialized PlinkBEDFile object
        """
        logger.info(f"Using first column of baseline ldscore as w_ld for chr{chrom}")

        # Create w_ld directory
        w_ld_dir = Path(self.config.ldscore_save_dir) / "w_ld"
        w_ld_dir.mkdir(parents=True, exist_ok=True)

        # Define file path
        w_ld_file = w_ld_dir / f"weights.{chrom}.l2.ldscore.gz"

        # Extract the first column
        w_ld_values = ldscore_chunk[:, 0]

        # Create a DataFrame with SNP information from the BIM file
        snp_indices = (
            plink_bed.kept_snps
            if hasattr(plink_bed, "kept_snps")
            else np.arange(len(self.snp_name))
        )
        bim_subset = plink_bed.bim_df.iloc[snp_indices]

        w_ld_df = pd.DataFrame(
            {
                "SNP": self.snp_name,
                "L2": w_ld_values,
                "CHR": bim_subset.CHR.values[: len(self.snp_name)],  # Ensure length matches
                "BP": bim_subset.BP.values[: len(self.snp_name)],
                "CM": bim_subset.CM.values[: len(self.snp_name)],
            }
        )

        # Reorder columns
        w_ld_df = w_ld_df[["CHR", "SNP", "BP", "CM", "L2"]]

        w_ld_df.to_csv(w_ld_file, sep="\t", index=False, compression="gzip")

        logger.info(f"Saved w_ld for chr{chrom} to {w_ld_file}")

    def _calculate_annotation_ldscores(self, chrom: int, plink_bed):
        """
        Calculate and save LD scores for spatial annotations.

        Parameters
        ----------
        chrom : int
            Chromosome number
        plink_bed : PlinkBEDFile
            Initialized PlinkBEDFile object
        """
        # Get marker scores for gene columns (excluding dummy NA column)
        mk_scores = self.mk_score_common.loc[self.snp_gene_pair_dummy.columns[:-1]]

        # Process in chunks
        chunk_index = 1
        for i in trange(
            0,
            mk_scores.shape[1],
            self.config.spots_per_chunk,
            desc=f"Calculating LD scores for chr{chrom}",
        ):
            # Get marker scores for current chunk
            mk_score_chunk = mk_scores.iloc[:, i : i + self.config.spots_per_chunk]

            # Define file paths
            sample_name = self.config.sample_name
            ld_score_file = f"{self.config.ldscore_save_dir}/{sample_name}_chunk{chunk_index}/{sample_name}.{chrom}.l2.ldscore.feather"
            m_file = f"{self.config.ldscore_save_dir}/{sample_name}_chunk{chunk_index}/{sample_name}.{chrom}.l2.M"
            m_5_file = f"{self.config.ldscore_save_dir}/{sample_name}_chunk{chunk_index}/{sample_name}.{chrom}.l2.M_5_50"

            # Calculate LD scores
            ldscore_chunk = self._calculate_ldscore_from_weights(mk_score_chunk, plink_bed)

            # Save LD scores based on format
            if self.config.ldscore_save_format == "feather":
                self._save_ldscore_to_feather(
                    ldscore_chunk,
                    column_names=mk_score_chunk.columns,
                    save_file_name=ld_score_file,
                )
            else:
                raise ValueError(f"Invalid ldscore_save_format: {self.config.ldscore_save_format}")

            # Calculate and save M values
            self._calculate_and_save_m_values(
                mk_score_chunk,
                m_file,
                m_5_file,
                drop_dummy_na=True,
            )

            chunk_index += 1

            # Clear memory
            del ldscore_chunk
            gc.collect()

    def _calculate_ldscore_from_weights(
        self, marker_scores: pd.DataFrame, plink_bed, drop_dummy_na: bool = True
    ) -> np.ndarray:
        """
        Calculate LD scores using SNP-gene weight matrix.

        Parameters
        ----------
        marker_scores : pd.DataFrame
            DataFrame with marker scores
        plink_bed : PlinkBEDFile
            Initialized PlinkBEDFile object
        drop_dummy_na : bool, optional
            Whether to drop the dummy NA column, by default True

        Returns
        -------
        np.ndarray
            Array with calculated LD scores
        """
        weight_matrix = self.snp_gene_weight_matrix

        if drop_dummy_na:
            # Use all columns except the last one (dummy NA)
            ldscore = weight_matrix[:, :-1] @ marker_scores
        else:
            ldscore = weight_matrix @ marker_scores

        return ldscore

    def _save_ldscore_to_feather(
        self, ldscore_data: np.ndarray, column_names: list[str], save_file_name: str
    ):
        """
        Save LD scores to a feather file.

        Parameters
        ----------
        ldscore_data : np.ndarray
            Array with LD scores
        column_names : list
            List of column names
        save_file_name : str
            Path to save the feather file
        """
        # Create directory if needed
        save_dir = Path(save_file_name).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # Convert to float16 for storage efficiency
        ldscore_data = ldscore_data.astype(np.float16, copy=False)

        # Handle numerical overflow
        ldscore_data[np.isinf(ldscore_data)] = np.finfo(np.float16).max

        # Create DataFrame and save
        df = pd.DataFrame(
            ldscore_data,
            index=self.snp_name,
            columns=column_names,
        )
        df.index.name = "SNP"
        df.reset_index().to_feather(save_file_name)

    def _calculate_and_save_m_values(
        self,
        marker_scores: pd.DataFrame,
        m_file_path: str,
        m_5_file_path: str,
        drop_dummy_na: bool = True,
    ):
        """
        Calculate and save M statistics.

        Parameters
        ----------
        marker_scores : pd.DataFrame
            DataFrame with marker scores
        m_file_path : str
            Path to save M values
        m_5_file_path : str
            Path to save M_5_50 values
        drop_dummy_na : bool, optional
            Whether to drop the dummy NA column, by default True
        """
        # Create directory if needed
        save_dir = Path(m_file_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)

        # Get sum of SNP-gene pairs
        snp_gene_sum = self.snp_gene_pair_dummy.values.sum(axis=0, keepdims=True)
        snp_gene_sum_maf = self.snp_gene_pair_dummy.loc[self.snp_pass_maf].values.sum(
            axis=0, keepdims=True
        )

        # Drop dummy NA column if requested
        if drop_dummy_na:
            snp_gene_sum = snp_gene_sum[:, :-1]
            snp_gene_sum_maf = snp_gene_sum_maf[:, :-1]

        # Calculate M values
        m_values = snp_gene_sum @ marker_scores
        m_5_values = snp_gene_sum_maf @ marker_scores

        # Save M values
        np.savetxt(m_file_path, m_values, delimiter="\t")
        np.savetxt(m_5_file_path, m_5_values, delimiter="\t")

    def _get_snp_gene_dummy(self, chrom: int, plink_bed) -> pd.DataFrame:
        """
        Get dummy matrix for SNP-gene pairs.

        Parameters
        ----------
        chrom : int
            Chromosome number
        plink_bed : PlinkBEDFile

        Returns
        -------
        pd.DataFrame
            DataFrame with dummy variables for SNP-gene pairs
        """
        logger.info(f"Creating SNP-gene mappings for chromosome {chrom}")

        # Load BIM file
        bim = plink_bed.bim_df
        bim_pr = plink_bed.convert_bim_to_pyrange(bim)

        # Determine mapping strategy
        if self.config.gene_window_enhancer_priority in ["gene_window_first", "enhancer_first"]:
            # Use both gene window and enhancer
            snp_gene_pair = self._combine_gtf_and_enhancer_mappings(bim, bim_pr)

        elif self.config.gene_window_enhancer_priority is None:
            # Use only gene window
            snp_gene_pair = self._get_snp_gene_pair_from_gtf(bim, bim_pr)

        elif self.config.gene_window_enhancer_priority == "enhancer_only":
            # Use only enhancer
            snp_gene_pair = self._get_snp_gene_pair_from_enhancer(bim, bim_pr)

        else:
            raise ValueError(
                f"Invalid gene_window_enhancer_priority: {self.config.gene_window_enhancer_priority}"
            )

        # Save SNP-gene pair mapping
        self._save_snp_gene_pair_mapping(snp_gene_pair, chrom)

        # Create dummy variables
        snp_gene_dummy = pd.get_dummies(snp_gene_pair["gene_name"], dummy_na=True)

        return snp_gene_dummy

    def _combine_gtf_and_enhancer_mappings(
        self, bim: pd.DataFrame, bim_pr: pr.PyRanges
    ) -> pd.DataFrame:
        """
        Combine gene window and enhancer mappings.

        Parameters
        ----------
        bim : pd.DataFrame
            BIM DataFrame
        bim_pr : pr.PyRanges
            BIM PyRanges object

        Returns
        -------
        pd.DataFrame
            Combined SNP-gene pair mapping
        """
        # Get mappings from both sources
        gtf_mapping = self._get_snp_gene_pair_from_gtf(bim, bim_pr)
        enhancer_mapping = self._get_snp_gene_pair_from_enhancer(bim, bim_pr)

        # Find SNPs with missing mappings in each source
        mask_of_nan_gtf = gtf_mapping.gene_name.isna()
        mask_of_nan_enhancer = enhancer_mapping.gene_name.isna()

        # Combine based on priority
        if self.config.gene_window_enhancer_priority == "gene_window_first":
            # Use gene window mappings first, fill missing with enhancer mappings
            combined_mapping = gtf_mapping.copy()
            combined_mapping.loc[mask_of_nan_gtf, "gene_name"] = enhancer_mapping.loc[
                mask_of_nan_gtf, "gene_name"
            ]
            logger.info(
                f"Filled {mask_of_nan_gtf.sum()} SNPs with no GTF mapping using enhancer mappings"
            )

        elif self.config.gene_window_enhancer_priority == "enhancer_first":
            # Use enhancer mappings first, fill missing with gene window mappings
            combined_mapping = enhancer_mapping.copy()
            combined_mapping.loc[mask_of_nan_enhancer, "gene_name"] = gtf_mapping.loc[
                mask_of_nan_enhancer, "gene_name"
            ]
            logger.info(
                f"Filled {mask_of_nan_enhancer.sum()} SNPs with no enhancer mapping using GTF mappings"
            )

        else:
            raise ValueError(
                f"Invalid gene_window_enhancer_priority for combining: {self.config.gene_window_enhancer_priority}"
            )

        return combined_mapping

    def _get_snp_gene_pair_from_gtf(self, bim: pd.DataFrame, bim_pr: pr.PyRanges) -> pd.DataFrame:
        """
        Get SNP-gene pairs based on GTF annotations.

        Parameters
        ----------
        bim : pd.DataFrame
            BIM DataFrame
        bim_pr : pr.PyRanges
            BIM PyRanges object

        Returns
        -------
        pd.DataFrame
            SNP-gene pairs based on GTF
        """
        logger.info(
            "Getting SNP-gene pairs from GTF. SNPs in multiple genes will be assigned to the nearest gene (by TSS)"
        )

        # Find overlaps between SNPs and gene windows
        overlaps = overlaps_gtf_bim(self.gtf_pr, bim_pr)

        # Get SNP information
        annot = bim[["CHR", "BP", "SNP", "CM"]]

        # Create SNP-gene pairs DataFrame
        snp_gene_pair = (
            overlaps[["SNP", "gene_name"]]
            .set_index("SNP")
            .join(annot.set_index("SNP"), how="right")
        )

        logger.info(f"Found {overlaps.shape[0]} SNP-gene pairs from GTF")

        return snp_gene_pair

    def _get_snp_gene_pair_from_enhancer(
        self, bim: pd.DataFrame, bim_pr: pr.PyRanges
    ) -> pd.DataFrame:
        """
        Get SNP-gene pairs based on enhancer annotations.

        Parameters
        ----------
        bim : pd.DataFrame
            BIM DataFrame
        bim_pr : pr.PyRanges
            BIM PyRanges object

        Returns
        -------
        pd.DataFrame
            SNP-gene pairs based on enhancer
        """
        if self.enhancer_pr is None:
            raise ValueError("Enhancer annotation file is required but not provided")

        # Find overlaps between SNPs and enhancers
        overlaps = self.enhancer_pr.join(bim_pr).df

        # Get SNP information
        annot = bim[["CHR", "BP", "SNP", "CM"]]

        if self.config.snp_multiple_enhancer_strategy == "max_mkscore":
            logger.info(
                "SNPs in multiple enhancers will be assigned to the gene with highest marker score"
            )
            overlaps = overlaps.loc[overlaps.groupby("SNP").avg_mkscore.idxmax()]

        elif self.config.snp_multiple_enhancer_strategy == "nearest_TSS":
            logger.info("SNPs in multiple enhancers will be assigned to the gene with nearest TSS")
            overlaps["Distance"] = np.abs(overlaps["Start_b"] - overlaps["TSS"])
            overlaps = overlaps.loc[overlaps.groupby("SNP").Distance.idxmin()]

        # Create SNP-gene pairs DataFrame
        snp_gene_pair = (
            overlaps[["SNP", "gene_name"]]
            .set_index("SNP")
            .join(annot.set_index("SNP"), how="right")
        )

        logger.info(f"Found {overlaps.shape[0]} SNP-gene pairs from enhancers")

        return snp_gene_pair

    def _save_snp_gene_pair_mapping(self, snp_gene_pair: pd.DataFrame, chrom: int):
        """
        Save SNP-gene pair mapping to a feather file.

        Parameters
        ----------
        snp_gene_pair : pd.DataFrame
            SNP-gene pair mapping
        chrom : int
            Chromosome number
        """
        save_path = (
            Path(self.config.ldscore_save_dir) / f"SNP_gene_pair/SNP_gene_pair_chr{chrom}.feather"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        snp_gene_pair.reset_index().to_feather(save_path)

    def _clear_memory(self):
        """Clear memory to prevent leaks."""
        gc.collect()


def run_generate_ldscore(config: GenerateLDScoreConfig):
    """
    Main function to run the LD score generation.

    Parameters
    ----------
    config : GenerateLDScoreConfig
        Configuration object
    """
    # Create output directory
    Path(config.ldscore_save_dir).mkdir(parents=True, exist_ok=True)

    if config.ldscore_save_format == "quick_mode":
        logger.info(
            "Running in quick_mode. Skip the process of generating ldscore. Using the pre-calculated ldscore."
        )
        ldscore_save_dir = Path(config.ldscore_save_dir)

        # Set up symbolic links
        baseline_dir = ldscore_save_dir / "baseline"
        baseline_dir.parent.mkdir(parents=True, exist_ok=True)
        if not baseline_dir.exists():
            baseline_dir.symlink_to(config.baseline_annotation_dir, target_is_directory=True)

        snp_gene_pair_dir = ldscore_save_dir / "SNP_gene_pair"
        snp_gene_pair_dir.parent.mkdir(parents=True, exist_ok=True)
        if not snp_gene_pair_dir.exists():
            snp_gene_pair_dir.symlink_to(config.SNP_gene_pair_dir, target_is_directory=True)

        # Create a done file to mark completion
        done_file = ldscore_save_dir / f"{config.sample_name}_generate_ldscore.done"
        done_file.touch()

        return

    # Initialize calculator
    calculator = LDScoreCalculator(config)

    # Process chromosomes
    if config.chrom == "all":
        # Process all chromosomes
        for chrom in range(1, 23):
            try:
                calculator.process_chromosome(chrom)
            except Exception as e:
                logger.error(f"Error processing chromosome {chrom}: {e}")
                raise
    else:
        # Process one chromosome
        try:
            chrom = int(config.chrom)
        except ValueError:
            logger.error(f"Invalid chromosome: {config.chrom}")
            raise ValueError(
                f"Invalid chromosome: {config.chrom}. Must be an integer between 1-22 or 'all'"
            ) from None
        else:
            calculator.process_chromosome(chrom)

    # Create a done file to mark completion
    done_file = Path(config.ldscore_save_dir) / f"{config.sample_name}_generate_ldscore.done"
    done_file.touch()

    logger.info(f"LD score generation completed for {config.sample_name}")
