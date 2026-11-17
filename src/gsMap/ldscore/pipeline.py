"""
Chromosome-wise pipeline for LD score calculation using NumPy/Scipy.

This module orchestrates the complete workflow:
1. Loop through chromosomes
2. Construct batches
3. Load genotypes once per chromosome
4. Process each batch to compute LD weights
5. Save results as AnnData
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyranges1 as pr
import scipy.sparse
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from gsMap.config.ldscore_config import LDScoreConfig
from gsMap.logging_config import stage_logging

from .batch_construction import construct_batches, resolve_ld_window
from .compute import (
    compute_batch_weights_and_ld_scores,
    compute_batch_weights_sparse,
)
from .io import PlinkBEDReader
from .mapping import create_snp_feature_map

logger = logging.getLogger(__name__)

MHC_CHROMOSOME = 6


def exclude_mhc_regression_snps(
    hm3_snps: list[str],
    bim_df: pd.DataFrame,
    chromosome: int,
    mhc_start_bp: int,
    mhc_end_bp: int,
) -> tuple[list[str], int, int]:
    """Remove extended-MHC SNPs from targets without changing reference SNPs.

    Returns the retained HM3 IDs, the number of matched HM3 target IDs removed,
    and the number of broad-reference variants retained inside the MHC interval.
    ``bim_df`` itself is never filtered, which preserves MHC variants in the
    annotation LD-score sum.
    """
    if chromosome != MHC_CHROMOSOME:
        return hm3_snps, 0, 0

    mhc_mask = bim_df["BP"].between(mhc_start_bp, mhc_end_bp, inclusive="both")
    mhc_reference_snps = set(bim_df.loc[mhc_mask, "SNP"].astype(str))
    retained = [snp for snp in hm3_snps if str(snp) not in mhc_reference_snps]
    return retained, len(hm3_snps) - len(retained), int(mhc_mask.sum())


@dataclass
class ChromosomeResult:
    """
    Results for a single chromosome.

    Attributes
    ----------
    hm3_snp_names : List[str]
        Names of HM3 SNPs processed
    weights : Union[scipy.sparse.csr_matrix, np.ndarray]
        Weight matrix for features (sparse or dense), shape (n_hm3_snps, n_features)
    feature_names : List[str]
        Names of mapped features
    """

    hm3_snp_names: list[str]
    hm3_snp_chr: list[int]
    hm3_snp_bp: list[int]
    weights: scipy.sparse.csr_matrix | np.ndarray
    feature_names: list[str]
    mapping_df: pd.DataFrame | None = None
    ld_scores: np.ndarray | None = None


class LDScorePipeline:
    """
    Pipeline for computing LD scores across chromosomes.
    """

    def __init__(self, config: LDScoreConfig):
        self.config = config
        self.hm3_dir = Path(config.hm3_snp_dir)

        logger.info("=" * 80)
        logger.info("LD Score Pipeline Configuration (NumPy/Scipy)")
        logger.info("=" * 80)
        logger.info(f"PLINK template: {config.bfile_root}")
        logger.info(f"HM3 directory: {config.hm3_snp_dir}")
        logger.info(f"Batch size (HM3): {config.batch_size_hm3}")
        logger.info(f"LD window: {config.ld_wind} {config.ld_unit}")
        logger.info(f"Broad-reference SNP-only filter: {config.snps_only}")
        logger.info(f"Broad-reference MAC filter: MAC > {config.mac_min}")
        logger.info(f"Optional broad-reference MAF filter: {config.maf_min}")
        logger.info(
            "Exclude MHC from HM3 regression/weight targets: %s (chr6:%d-%d)",
            config.exclude_mhc_from_regression,
            config.mhc_start_bp,
            config.mhc_end_bp,
        )
        logger.info(f"Chromosomes: {config.chromosomes}")
        logger.info(f"Output Directory: {config.output_dir}")
        logger.info(f"Output Filename: {config.output_filename}")
        logger.info("=" * 80)

    def run(self):
        """
        Main entry point. Dispatches to specific pipeline mode based on configuration.
        """
        with stage_logging(self.config, "ldscore_weight_matrix") as session:
            session.record_config(self.config, stage="ldscore_weight_matrix")
            if self.config.annot_file:
                logger.info(f"Mode: Annotation File ({self.config.annot_file})")
                self._run_with_annotation()
            elif self.config.mapping_file:
                logger.info(f"Mode: SNP-Feature Mapping ({self.config.mapping_type})")
                self._run_with_mapping()
            else:
                raise ValueError(
                    "Invalid Configuration: Neither 'annot_file' nor 'mapping_file' specified."
                )

    def _run_with_mapping(self):
        """Run the pipeline using SNP-Feature mapping (BED/Dictionary)."""
        # 1. Load Mapping Data
        mapping_data = self._load_mapping_data()

        results = {}
        for chrom in self.config.chromosomes:
            result = self._process_chromosome_from_mapping(chrom, mapping_data=mapping_data)
            if result is not None:
                results[str(chrom)] = result

        if not results:
            logger.warning("No results generated. Skipping save.")
            return

        self._save_aggregated_results(results, self.config.output_dir, self.config.output_filename)

    def _run_with_annotation(self):
        """Run the pipeline using external annotation matrices."""
        results = {}
        for chrom in self.config.chromosomes:
            result = self._process_chromosome_from_annotation(chrom)
            if result:
                results[str(chrom)] = result

        if not results:
            logger.warning("No results generated.")
            return

        self._save_aggregated_results(results, self.config.output_dir, self.config.output_filename)

    def _load_mapping_data(self) -> pd.DataFrame | dict[str, str]:
        """Helper to load mapping file based on config."""
        logger.info(f"Loading mapping data from: {self.config.mapping_file}")

        if self.config.mapping_type == "bed":
            # Use pyranges1 to read standard BED file
            try:
                bed_df = pr.read_bed(str(self.config.mapping_file)).copy()

                # Convert pyranges1 BED format to expected format
                # Standard BED columns: Chromosome, Start, End, Name, Score, Strand
                # Expected format: Feature, Chromosome, Start, End, [Score], [Strand]

                if "Name" not in bed_df.columns:
                    raise ValueError(
                        f"BED file {self.config.mapping_file} has no 'Name' column "
                        "(4th column); expected BED6 without a header line (chr, start, end, name, score, strand)."
                    )

                # Rename 'Name' to 'Feature' for internal use
                bed_df = bed_df.rename(columns={"Name": "Feature"})

                # Ensure required columns exist
                required_cols = ["Chromosome", "Start", "End", "Feature"]
                if not all(col in bed_df.columns for col in required_cols):
                    raise ValueError(
                        f"BED file {self.config.mapping_file} lacks columns {required_cols} "
                        "after parsing; expected BED6 without a header line (chr, start, end, name, score, strand)."
                    )

                logger.info(f"Successfully loaded BED file: {len(bed_df)} features")
                logger.info(f"Columns: {list(bed_df.columns)}")
                return bed_df

            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    f"Failed to read BED file {self.config.mapping_file} ({e}); "
                    "expected BED6 without a header line (chr, start, end, name, score, strand)."
                ) from e

        elif self.config.mapping_type == "dict":
            df_map = pd.read_csv(self.config.mapping_file, sep=None, engine="python")

            if "SNP" in df_map.columns and "Feature" in df_map.columns:
                return dict(zip(df_map["SNP"], df_map["Feature"], strict=False))
            elif len(df_map.columns) >= 2:
                logger.info("Assuming first column is SNP and second is Feature.")
                return dict(zip(df_map.iloc[:, 0], df_map.iloc[:, 1], strict=False))
            else:
                raise ValueError(
                    f"Dictionary mapping file {self.config.mapping_file} must have at least 2 columns."
                )
        else:
            raise ValueError(f"Unsupported mapping_type: {self.config.mapping_type}")

    def _load_hm3_snps(self, chromosome: int) -> list[str]:
        """Load HM3 SNP list for a chromosome."""
        possible_paths = [
            self.hm3_dir / f"hm.{chromosome}.snp",
            self.hm3_dir / f"hm3_snps.chr{chromosome}.txt",
            self.hm3_dir / f"hapmap3_snps.chr{chromosome}.txt",
            self.hm3_dir / f"chr{chromosome}.snplist",
            self.hm3_dir / f"w_hm3.snplist.chr{chromosome}",
        ]

        for path in possible_paths:
            if path.exists():
                try:
                    snps = pd.read_csv(path, header=None, names=["SNP"])["SNP"].tolist()
                    logger.info(f"Loaded {len(snps)} HM3 SNPs from {path}")
                    return snps
                except Exception:  # noqa: BLE001
                    continue

        tried = "\n".join(f"  - {p}" for p in possible_paths)
        raise FileNotFoundError(
            f"No HM3 SNP file found for chromosome {chromosome} in '{self.hm3_dir}'.\n"
            f"Tried the following paths:\n{tried}\n"
            f"Please ensure the HM3 SNP directory contains per-chromosome files "
            f"(e.g., hm.{chromosome}.snp)."
        )

    def _load_plink_reader(self, chromosome: int) -> PlinkBEDReader | None:
        """Helper to initialize the PLINK reader for a chromosome."""
        bfile_prefix = self.config.bfile_root.format(chr=chromosome)
        logger.info(f"Loading PLINK data from: {bfile_prefix}")

        try:
            reader = PlinkBEDReader(
                bfile_prefix,
                snps_only=self.config.snps_only,
                mac_min=self.config.mac_min,
                maf_min=self.config.maf_min,
                preload=True,
            )
            return reader
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"PLINK files not found for requested chromosome {chromosome}: {e}"
            ) from e

    def _process_chromosome_from_mapping(
        self,
        chromosome: int,
        mapping_data: pd.DataFrame | dict[str, str],
    ) -> ChromosomeResult | None:
        """Process a single chromosome using mapping rules (BED/Dict)."""
        logger.info("=" * 80)
        logger.info(f"Processing Chromosome {chromosome}")
        logger.info("=" * 80)

        target_hm3_snps = self._load_hm3_snps(chromosome)
        if not target_hm3_snps:
            return None

        reader = self._load_plink_reader(chromosome)
        if not reader:
            return None

        if self.config.exclude_mhc_from_regression:
            target_hm3_snps, n_removed, n_reference_mhc = exclude_mhc_regression_snps(
                hm3_snps=target_hm3_snps,
                bim_df=reader.bim,
                chromosome=chromosome,
                mhc_start_bp=self.config.mhc_start_bp,
                mhc_end_bp=self.config.mhc_end_bp,
            )
            if chromosome == MHC_CHROMOSOME:
                logger.info(
                    "MHC handling: retained %d broad-reference MHC variants; "
                    "excluded %d matched HM3 MHC regression targets",
                    n_reference_mhc,
                    n_removed,
                )

        logger.info(f"Creating SNP-feature mapping for chromosome {chromosome}...")
        # create_snp_feature_map now returns (matrix, names, df)
        mapping_matrix, feature_names, mapping_df = create_snp_feature_map(
            bim_df=reader.bim,
            mapping_type=self.config.mapping_type,
            mapping_data=mapping_data,
            feature_window_size=self.config.feature_window_size,
            strategy=self.config.strategy,
        )

        # Save Curated Mapping if it exists (BED type)
        if mapping_df is not None:
            logger.debug(
                f"  Captured SNP-Feature mapping ({len(mapping_df)} rows) for chromosome {chromosome}"
            )

        # Ensure feature names align with matrix width (handle Unmapped bin)
        if mapping_matrix.shape[1] == len(feature_names) + 1:
            feature_names_full = feature_names + ["Unmapped"]
        else:
            feature_names_full = feature_names

        logger.info(f"  Features: {len(feature_names_full)} | Matrix nnz: {mapping_matrix.nnz}")

        return self._compute_chromosome_weights(
            chromosome=chromosome,
            reader=reader,
            target_hm3_snps=target_hm3_snps,
            mapping_matrix=mapping_matrix,
            feature_names=feature_names_full,
            mapping_df=mapping_df,
            output_format="sparse",
        )

    def _process_chromosome_from_annotation(
        self,
        chromosome: int,
    ) -> ChromosomeResult | None:
        """Process a single chromosome using external annotation files."""
        logger.info("=" * 80)
        logger.info(f"Processing Chromosome {chromosome} (Annotation Mode)")
        logger.info("=" * 80)

        target_hm3_snps = self._load_hm3_snps(chromosome)
        if not target_hm3_snps:
            return None

        reader = self._load_plink_reader(chromosome)
        if not reader:
            return None

        if self.config.exclude_mhc_from_regression:
            target_hm3_snps, n_removed, n_reference_mhc = exclude_mhc_regression_snps(
                hm3_snps=target_hm3_snps,
                bim_df=reader.bim,
                chromosome=chromosome,
                mhc_start_bp=self.config.mhc_start_bp,
                mhc_end_bp=self.config.mhc_end_bp,
            )
            if chromosome == MHC_CHROMOSOME:
                logger.info(
                    "MHC handling: retained %d broad-reference MHC variants; "
                    "excluded %d matched HM3 MHC regression targets",
                    n_reference_mhc,
                    n_removed,
                )

        # Load Annotation File
        try:
            annot_file = self.config.annot_file.format(chr=chromosome)
        except KeyError:
            annot_file = self.config.annot_file.format(chromosome=chromosome)

        if not Path(annot_file).exists():
            raise FileNotFoundError(
                f"Annotation file not found for requested chromosome {chromosome}: {annot_file}"
            )

        logger.info(f"Loading annotation from: {annot_file}")
        try:
            df_annot = pd.read_csv(annot_file, sep=r"\s+")
        except Exception as e:
            raise ValueError(
                f"Failed to read annotation file for chromosome {chromosome}: {annot_file}: {e}"
            ) from e

        if "SNP" in df_annot.columns:
            # Align to filtered BIM
            df_annot = df_annot.set_index("SNP")
            df_annot = df_annot.reindex(reader.bim["SNP"], fill_value=0)
        else:
            logger.warning(
                "Annotation file missing 'SNP' column. Thin annotation format detected, assuming strict alignment."
            )

            if len(df_annot) == reader.m_original:
                # Filter df_annot according to filtered SNPs using snp_ids_original
                logger.info(
                    f"Filtering annotation from {len(df_annot)} to {reader.m} SNPs based on MAF/QC filters"
                )
                df_annot.index = reader.snp_ids_original
                df_annot = df_annot.reindex(reader.bim["SNP"], fill_value=0)
            else:
                raise ValueError(
                    f"Annotation rows for chromosome {chromosome} cannot be aligned: without a "
                    f"'SNP' column (thin format), the {len(df_annot)} annotation rows must match "
                    f"the {reader.m_original} SNPs in the PLINK panel."
                )

        # Drop non-feature columns
        feature_df = df_annot.drop(
            columns=[c for c in ["CHR", "BP", "CM"] if c in df_annot.columns]
        )
        feature_names = feature_df.columns.tolist()

        # QC
        for feature in feature_names:
            if np.count_nonzero(feature_df[feature].values) / len(feature_df) < 0.0001:
                logger.warning(f"Feature '{feature}' has < 0.01% non-zero entries.")

        logger.info("Converting annotation to sparse matrix...")
        annot_matrix = scipy.sparse.csr_matrix(feature_df.values, dtype=np.float32)

        return self._compute_chromosome_weights(
            chromosome=chromosome,
            reader=reader,
            target_hm3_snps=target_hm3_snps,
            mapping_matrix=annot_matrix,
            feature_names=feature_names,
            mapping_df=None,
            output_format="dense",
        )

    def _compute_chromosome_weights(
        self,
        chromosome: int,
        reader: PlinkBEDReader,
        target_hm3_snps: list[str],
        mapping_matrix: scipy.sparse.csr_matrix | np.ndarray,
        feature_names: list[str],
        mapping_df: pd.DataFrame | None = None,
        output_format: str = "sparse",
    ) -> ChromosomeResult | None:
        """
        Core logic: Batches -> Load Genotypes -> Slice Mapping -> Compute Weights.
        """
        logger.info("Constructing batches...")
        batch_infos = construct_batches(
            bim_df=reader.bim,
            hm3_snp_names=target_hm3_snps,
            batch_size_hm3=self.config.batch_size_hm3,
            ld_wind=self.config.ld_wind,
            ld_unit=self.config.ld_unit,
        )

        if len(batch_infos) == 0:
            logger.warning(f"No batches created for chromosome {chromosome}")
            return None

        total_snps = sum(len(b.hm3_indices) for b in batch_infos)
        logger.info(f"Actual HM3 SNPs found in BIM: {total_snps}")

        if reader.genotypes is None:
            raise RuntimeError("LD-score computation requires preloaded genotypes")

        batch_weights: list[scipy.sparse.csr_matrix | np.ndarray] = []
        batch_w_ld: list[np.ndarray] = []
        ordered_hm3_indices: list[np.ndarray] = []
        ld_coordinates, max_ld_distance = resolve_ld_window(
            reader.bim,
            self.config.ld_wind,
            self.config.ld_unit,
        )
        w_ld_reference_mask = None
        if self.config.calculate_w_ld:
            regression_snp_set = set(target_hm3_snps)
            w_ld_reference_mask = reader.bim["SNP"].isin(regression_snp_set).to_numpy()
            n_weight_snps = int(w_ld_reference_mask.sum())
            if n_weight_snps == 0:
                raise ValueError(
                    f"No regression SNPs are available for chromosome {chromosome} w_ld"
                )
            logger.info(
                "Fused w_ld: %d broad-reference columns belong to the HM3 no-MHC universe",
                n_weight_snps,
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task(f"[cyan]Chr {chromosome}", total=total_snps)

            for batch_info in batch_infos:
                X_hm3 = reader.genotypes[:, batch_info.hm3_indices]
                ref_slice = slice(batch_info.ref_start_idx, batch_info.ref_end_idx)
                X_ref_block = reader.genotypes[:, ref_slice]
                block_mapping = mapping_matrix[ref_slice, :]
                target_coordinates = ld_coordinates[batch_info.hm3_indices]
                reference_coordinates = ld_coordinates[ref_slice]
                ld_window_mask = (
                    np.abs(target_coordinates[:, None] - reference_coordinates[None, :])
                    <= max_ld_distance
                )
                if w_ld_reference_mask is None:
                    weights = compute_batch_weights_sparse(
                        X_hm3,
                        X_ref_block,
                        block_mapping,
                        ld_window_mask=ld_window_mask,
                    )
                else:
                    weights, w_ld_values = compute_batch_weights_and_ld_scores(
                        X_hm3,
                        X_ref_block,
                        block_mapping,
                        ld_window_mask=ld_window_mask,
                        ld_score_reference_mask=w_ld_reference_mask[ref_slice],
                    )
                    if w_ld_values is None:
                        raise RuntimeError("Fused w_ld kernel did not return LD scores")
                    batch_w_ld.append(w_ld_values)
                if output_format == "sparse":
                    batch_weights.append(scipy.sparse.csr_matrix(weights))
                else:
                    batch_weights.append(weights)
                ordered_hm3_indices.append(batch_info.hm3_indices)
                progress.update(task, advance=len(batch_info.hm3_indices))

        all_hm3_indices = np.concatenate(ordered_hm3_indices)
        all_hm3_bim = reader.bim.iloc[all_hm3_indices]
        if output_format == "sparse":
            weights_out = scipy.sparse.vstack(batch_weights, format="csr")
        else:
            weights_out = np.vstack(batch_weights)

        if self.config.calculate_w_ld:
            if not batch_w_ld:
                raise RuntimeError(f"No w_ld values were computed for chromosome {chromosome}")
            w_ld_values = np.concatenate(batch_w_ld)
            if len(w_ld_values) != len(all_hm3_bim):
                raise RuntimeError(
                    f"w_ld row count mismatch for chromosome {chromosome}: "
                    f"{len(w_ld_values)} != {len(all_hm3_bim)}"
                )
            self._write_w_ld(chromosome, all_hm3_bim, w_ld_values)

        return ChromosomeResult(
            hm3_snp_names=all_hm3_bim["SNP"].tolist(),
            hm3_snp_chr=all_hm3_bim["CHR"].tolist(),
            hm3_snp_bp=all_hm3_bim["BP"].tolist(),
            weights=weights_out,
            feature_names=feature_names,
            mapping_df=mapping_df,
        )

    def _write_w_ld(
        self,
        chromosome: int,
        hm3_bim: pd.DataFrame,
        w_ld_values: np.ndarray,
    ) -> None:
        """Write one chromosome of fused HM3-no-MHC regression weights."""
        required_columns = ["CHR", "SNP", "BP", "CM", "MAF"]
        missing = [column for column in required_columns if column not in hm3_bim.columns]
        if missing:
            raise ValueError(f"Cannot write w_ld; BIM metadata are missing {missing}")

        df_w_ld = hm3_bim[required_columns].copy()
        df_w_ld["L2"] = np.asarray(w_ld_values, dtype=np.float64)
        if not np.all(np.isfinite(df_w_ld["L2"])):
            raise ValueError(f"Non-finite w_ld values for chromosome {chromosome}")

        w_ld_base = (
            Path(self.config.w_ld_dir)
            if self.config.w_ld_dir
            else Path(self.config.output_dir) / "w_ld"
        )
        w_ld_base.mkdir(parents=True, exist_ok=True)
        out_file = w_ld_base / f"weights.{chromosome}.l2.ldscore.gz"
        temp_file = w_ld_base / f".weights.{chromosome}.l2.ldscore.tmp.gz"
        df_w_ld.to_csv(temp_file, sep="\t", index=False, compression="gzip")
        temp_file.replace(out_file)
        logger.info("Saved fused w_ld to: %s", out_file)

    def _save_aggregated_results(
        self, results: dict[str, ChromosomeResult], output_dir: str, output_filename: str
    ):
        """Helper to concatenate and save results."""
        logger.info("\n" + "=" * 80)
        logger.info("Processing Complete. Concatenating and Saving...")
        logger.info("=" * 80)

        all_snp_names = []
        all_snp_chr = []  # numeric CHR from BIM
        all_snp_bp = []  # numeric BP from BIM
        matrices = []

        # Check consistency
        first_res = next(iter(results.values()))
        feature_names = first_res.feature_names

        sorted_chroms = sorted(results.keys(), key=lambda x: int(x) if x.isdigit() else x)

        for chrom in sorted_chroms:
            res = results[chrom]
            if res.feature_names != feature_names:
                raise ValueError(
                    f"Chromosome {chrom} has different features than chromosome "
                    f"{sorted_chroms[0]}; the weight matrix needs one feature set."
                )

            all_snp_names.extend(res.hm3_snp_names)
            all_snp_chr.extend(res.hm3_snp_chr)
            all_snp_bp.extend(res.hm3_snp_bp)
            matrices.append(res.weights)

        # Concatenate
        if scipy.sparse.issparse(matrices[0]):
            X_full = scipy.sparse.vstack(matrices, format="csr")
        else:
            X_full = np.vstack(matrices)

        # AnnData
        obs = pd.DataFrame({"CHR": all_snp_chr, "BP": all_snp_bp}, index=all_snp_names)
        obs.index.name = "SNP"

        var = pd.DataFrame(index=feature_names)
        var.index.name = "Feature"

        logger.info(
            f"Creating AnnData object: {X_full.shape[0]} SNPs x {X_full.shape[1]} Features"
        )
        adata = ad.AnnData(X=X_full, obs=obs, var=var)

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"{output_filename}.h5ad"

        # Publish atomically: an interrupted write never leaves a truncated matrix.
        temporary = out_file.with_name(f".{out_file.name}.{os.getpid()}.partial")
        try:
            adata.write(temporary)
            os.replace(temporary, out_file)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info(f"Successfully saved AnnData to {out_file}")

        # Save Combined Mapping CSV if available
        all_mapping_dfs = [
            res.mapping_df for res in results.values() if res.mapping_df is not None
        ]
        if all_mapping_dfs:
            combined_mapping_df = pd.concat(all_mapping_dfs, ignore_index=True)
            mapping_out_file = out_path / f"{output_filename}.csv"
            combined_mapping_df.to_csv(mapping_out_file, index=False)
            logger.info(f"Successfully saved combined SNP-Feature mapping to {mapping_out_file}")
