"""
Module for reading and processing PLINK genotype data and calculating LD scores.

Note:
This code is adapted and modified from:
https://github.com/bulik/ldsc/blob/master/ldsc/ldscore.py
"""

import logging

import torch
import numba
import bitarray as ba
import numpy as np
import pandas as pd
import pyranges as pr
from tqdm import tqdm

# Configure logger
logger = logging.getLogger("gsMap.utils.plink_ldscore_tool")

def torch_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

@numba.njit
def getBlockLefts(coords: np.ndarray, max_dist: float):
    """
    Converts coordinates + max block length to a list of coordinates of the leftmost
    SNPs to be included in blocks.
    """
    M = len(coords)
    j = 0
    block_left = np.zeros(M)
    for i in range(M):
        while j < M and abs(coords[j] - coords[i]) > max_dist:
            j += 1

        block_left[i] = j
    return block_left


@numba.njit
def normalized_snps(X: np.ndarray, b: int, minorRef, freq, currentSNP):
    """
    Normalize the SNPs and impute the missing ones with the mean

    Parameters
    ----------
    fam_file : str
        Path to the FAM file

    Returns
    -------
    pd.DataFrame
        DataFrame containing FAM data
    """
    Y = np.zeros(X.shape, dtype="float32")

    for j in range(0, b):
        newsnp = X[:, j]
        ii = newsnp != 9
        avg = np.mean(newsnp[ii])
        newsnp[np.logical_not(ii)] = avg
        denom = np.std(newsnp)
        if denom == 0:
            denom = 1

        if minorRef is not None and freq[currentSNP + j] > 0.5:
            denom = denom * -1

        Y[:, j] = (newsnp - avg) / denom
    return Y


def l2_unbiased(x: torch.Tensor, n: int):
    """
    Calculate the unbiased estimate of L2.
    """
    denom = n - 2 if n > 2 else n  # allow n<2 for testing purposes
    sq = torch.square(x)
    return sq - (1 - sq) / denom


class PlinkBEDFile:
    """
    Interface for Plink .bed format for reading and processing genotype data.
    """

    def __init__(self, bfile_prefix):
        """
        Initialize the PlinkBEDFile from a PLINK file prefix.

        Parameters
        ----------
        bfile_prefix : str
            PLINK file prefix (without .bed/.bim/.fam extension)
        """
        # Initialize bitarray for bed code mapping
        self._bedcode = {
            2: ba.bitarray("11"),
            9: ba.bitarray("10"),
            1: ba.bitarray("01"),
            0: ba.bitarray("00"),
        }

        # Load BIM file
        self.bim_df = self.load_bim(f"{bfile_prefix}.bim")

        # Load FAM file
        self.fam_df = self.load_fam(f"{bfile_prefix}.fam")

        # Set up initial parameters
        self.m_original = len(self.bim_df)
        self.n_original = len(self.fam_df)

        # Read the bed file
        logger.info(f"Loading Plink genotype data from {bfile_prefix}.bed")
        (self.nru_original, self.geno_original) = self._read(
            f"{bfile_prefix}.bed", self.m_original, self.n_original
        )

        # Pre-calculate MAF for all SNPs
        logger.info("Calculating MAF and QC for all SNPs")
        self.all_snp_info = self._calculate_all_snp_info()

        # Filter out invalid SNPs
        valid_mask = self.all_snp_info["valid_snp"]
        if num_invalid := np.sum(~valid_mask):
            logger.warning(
                f"Filtering out {num_invalid} bad quality SNPs: {self.bim_df.loc[~valid_mask, 'SNP'].tolist()}"
            )
        else:
            logger.info("All SNPs passed the basic quality check")

        # Create new genotype data with only the valid SNPs
        new_geno = ba.bitarray()
        for j in np.arange(self.m_original)[valid_mask]:
            new_geno += self.geno_original[
                2 * self.nru_original * j : 2 * self.nru_original * (j + 1)
            ]

        # Update original data to only include valid SNPs
        self.geno_original = new_geno

        # Only keep valid SNPs
        self.bim_df = self.bim_df.loc[valid_mask].reset_index(drop=True)
        self.m_original = len(self.bim_df)
        self.kept_snps = np.arange(self.m_original)

        # Initialize current state variables
        self._currentSNP = 0
        self.m = self.m_original
        self.n = self.n_original
        self.nru = self.nru_original
        self.geno = self.geno_original.copy()

        # Update frequency info based on valid SNPs
        self.freq = self.all_snp_info["freq"][valid_mask]
        self.maf = np.minimum(self.freq, 1 - self.freq)
        self.sqrtpq = np.sqrt(self.freq * (1 - self.freq))

        # Add MAF to the BIM dataframe
        self.bim_df["MAF"] = self.maf

        logger.info(f"Loaded genotype data with {self.m} SNPs and {self.n} individuals")

    @staticmethod
    def load_bim(bim_file):
        """
        Load a BIM file into a pandas DataFrame.

        Parameters
        ----------
        bim_file : str
            Path to the BIM file

        Returns
        -------
        pd.DataFrame
            DataFrame containing BIM data
        """
        df = pd.read_csv(
            bim_file, sep="\t", header=None, names=["CHR", "SNP", "CM", "BP", "A1", "A2"]
        )
        return df

    @staticmethod
    def convert_bim_to_pyrange(bim_df) -> pr.PyRanges:
        bim_pr = bim_df.copy()
        bim_pr.drop(columns=["MAF"], inplace=True)
        bim_pr.columns = ["Chromosome", "SNP", "CM", "Start", "A1", "A2"]
        bim_pr.Chromosome = "chr" + bim_pr["Chromosome"].astype(str)

        # Adjust coordinates (BIM is 1-based, PyRanges uses 0-based)
        bim_pr["End"] = bim_pr["Start"].copy()
        bim_pr["Start"] = bim_pr["Start"] - 1

        bim_pr = pr.PyRanges(bim_pr)

        return bim_pr

    @staticmethod
    def load_fam(fam_file):
        """
        Load a FAM file into a pandas DataFrame.

        Parameters
        ----------
        fam_file : str
            Path to the FAM file

        Returns
        -------
        pd.DataFrame
            DataFrame containing FAM data
        """
        df = pd.read_csv(fam_file, sep=r"\s+", header=None, usecols=[1], names=["IID"])
        return df

    def _read(self, fname, m, n):
        """
        Read the bed file and return the genotype data.
        """
        if not fname.endswith(".bed"):
            raise ValueError(".bed filename must end in .bed")

        fh = open(fname, "rb")
        magicNumber = ba.bitarray(endian="little")
        magicNumber.fromfile(fh, 2)
        bedMode = ba.bitarray(endian="little")
        bedMode.fromfile(fh, 1)
        e = (4 - n % 4) if n % 4 != 0 else 0
        nru = n + e

        # Check magic number
        if magicNumber != ba.bitarray("0011011011011000"):
            raise OSError("Magic number from Plink .bed file not recognized")

        if bedMode != ba.bitarray("10000000"):
            raise OSError("Plink .bed file must be in default SNP-major mode")

        # Check file length
        geno = ba.bitarray(endian="little")
        geno.fromfile(fh)
        self._test_length(geno, m, nru)
        return (nru, geno)

    def _test_length(self, geno, m, nru):
        """
        Test if the genotype data has the expected length.
        """
        exp_len = 2 * m * nru
        real_len = len(geno)
        if real_len != exp_len:
            s = "Plink .bed file has {n1} bits, expected {n2}"
            raise OSError(s.format(n1=real_len, n2=exp_len))

    def _calculate_all_snp_info(self):
        """
        Pre-calculate MAF and other information for all SNPs.

        Returns
        -------
        dict
            Dictionary containing information for all SNPs
        """
        nru = self.nru_original
        n = self.n_original
        m = self.m_original
        geno = self.geno_original

        snp_info = {
            "freq": np.zeros(m),  # Allele frequencies
            "het_miss_count": np.zeros(m),  # Count of het or missing genotypes
            "valid_snp": np.zeros(m, dtype=bool),  # Whether SNP passes basic criteria
        }

        # For each SNP, calculate statistics
        for j in range(m):
            z = geno[2 * nru * j : 2 * nru * (j + 1)]
            A = z[0::2]
            a = A.count()
            B = z[1::2]
            b = B.count()
            c = (A & B).count()
            major_ct = b + c  # number of copies of the major allele
            n_nomiss = n - a + c  # number of individuals with nonmissing genotypes
            f = major_ct / (2 * n_nomiss) if n_nomiss > 0 else 0
            het_miss_ct = a + b - 2 * c  # count of SNPs that are het or missing

            snp_info["freq"][j] = f
            snp_info["het_miss_count"][j] = het_miss_ct
            snp_info["valid_snp"][j] = het_miss_ct < n  # Basic validity check

        return snp_info

    def apply_filters(self, keep_snps=None, keep_indivs=None, mafMin=None):
        """
        Apply filters to the genotype data without reloading the bed file.

        Parameters
        ----------
        keep_snps : array-like, optional
            Indices of SNPs to keep.
        keep_indivs : array-like, optional
            Indices of individuals to keep.
        mafMin : float, optional
            Minimum minor allele frequency.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        # Reset to original state first
        self.geno = self.geno_original.copy()
        self.m = self.m_original
        self.n = self.n_original
        self.nru = self.nru_original
        self._currentSNP = 0

        # Initialize with all SNPs
        kept_snps = np.arange(self.m_original)

        # Apply MAF filter using pre-calculated values
        if mafMin is not None and mafMin > 0:
            # Remove the redundant valid_snp check since all SNPs are already valid
            maf_mask = self.maf > mafMin
            kept_snps = kept_snps[maf_mask]
            logger.info(f"After MAF filtering (>{mafMin}), {len(kept_snps)} SNPs remain")

        # Apply SNP filter if specified
        if keep_snps is not None:
            keep_snps = np.array(keep_snps, dtype="int")
            if np.any(keep_snps > self.m_original):
                raise ValueError("keep_snps indices out of bounds")

            # Intersect with current kept_snps
            kept_snps = np.intersect1d(kept_snps, keep_snps)
            logger.info(f"After keep_snps filtering, {len(kept_snps)} SNPs remain")

        # Filter SNPs in the genotype data
        if len(kept_snps) < self.m_original:
            # Create new genotype data with only the kept SNPs
            new_geno = ba.bitarray()
            for j in kept_snps:
                new_geno += self.geno_original[2 * self.nru * j : 2 * self.nru * (j + 1)]
            self.geno = new_geno
            self.m = len(kept_snps)

        # Filter individuals if specified
        if keep_indivs is not None:
            keep_indivs = np.array(keep_indivs, dtype="int")
            if np.any(keep_indivs > self.n):
                raise ValueError("keep_indivs indices out of bounds")

            (self.geno, self.m, self.n) = self._filter_indivs(
                self.geno, keep_indivs, self.m, self.n
            )

            if self.n > 0:
                logger.info(f"After filtering, {self.n} individuals remain")
            else:
                raise ValueError("After filtering, no individuals remain")

        # Update kept_snps and other attributes
        self.kept_snps = kept_snps
        self.freq = self.all_snp_info["freq"][kept_snps]
        self.maf = np.minimum(self.freq, 1 - self.freq)
        self.sqrtpq = np.sqrt(self.freq * (1 - self.freq))

        return self

    def _filter_indivs(self, geno, keep_indivs, m, n):
        """
        Filter individuals based on the keep_indivs parameter.
        """
        n_new = len(keep_indivs)
        e = (4 - n_new % 4) if n_new % 4 != 0 else 0
        nru_new = n_new + e
        nru = self.nru
        z = ba.bitarray(m * 2 * nru_new, endian="little")
        z.setall(0)
        for e, i in enumerate(keep_indivs):
            z[2 * e :: 2 * nru_new] = geno[2 * i :: 2 * nru]
            z[2 * e + 1 :: 2 * nru_new] = geno[2 * i + 1 :: 2 * nru]
        self.nru = nru_new
        return (z, m, n_new)

    def get_snps_by_maf(self, mafMin):
        """
        Get the list of SNPs that pass the MAF threshold.

        Parameters
        ----------
        mafMin : float
            Minimum MAF threshold

        Returns
        -------
        list
            List of SNP IDs that pass the MAF threshold
        """
        maf_mask = self.maf > mafMin

        # Get SNP names from the BIM dataframe
        snp_pass_maf = self.bim_df.loc[maf_mask, "SNP"].tolist()

        logger.info(f"{len(snp_pass_maf)} SNPs with MAF > f{mafMin}")

        return snp_pass_maf

    def get_ldscore(self, annot_matrix=None, ld_wind=1.0, ld_unit="CM", keep_snps_index=None):
        """
        Calculate LD scores using an annotation matrix.

        Parameters
        ----------
        annot_matrix : np.ndarray, optional
            Annotation matrix. If None, uses a matrix of all ones.
        ld_wind : float, optional
            LD window size, by default 1.0
        ld_unit : str, optional
            Unit for the LD window, by default "CM"
        keep_snps_index : list[int], optional
            Indices of SNPs to keep, by default None

        Returns
        -------
        np.ndarray
            Array with calculated LD scores
        """
        # Apply filters if needed
        if keep_snps_index is not None:
            original_kept_snps = self.kept_snps.copy()
            self.apply_filters(keep_snps=keep_snps_index)

        # Configure LD window based on specified unit
        if ld_unit == "SNP":
            max_dist = ld_wind
            coords = np.array(range(self.m))
        elif ld_unit == "KB":
            max_dist = ld_wind * 1000
            coords = np.array(self.bim_df.loc[self.kept_snps, "BP"])
        elif ld_unit == "CM":
            max_dist = ld_wind
            coords = np.array(self.bim_df.loc[self.kept_snps, "CM"])
            # Check if the CM is all 0
            if np.all(coords == 0):
                logger.warning(
                    "All CM values are 0. Using 1MB window size for LD score calculation."
                )
                max_dist = 1_000_000
                coords = np.array(self.bim_df.loc[self.kept_snps, "BP"])
        else:
            raise ValueError(f"Invalid ld_wind_unit: {ld_unit}. Must be one of: SNP, KB, CM")

        # Calculate blocks for LD computation
        block_left = getBlockLefts(coords, max_dist)
        assert block_left.sum() > 0, "Invalid window size, please check the ld_wind parameter."

        # Calculate LD scores
        ld_scores = self.ldScoreVarBlocks(block_left, 100, annot=annot_matrix)

        # Restore original state if filters were applied
        if keep_snps_index is not None:
            self.apply_filters(keep_snps=original_kept_snps)

        return ld_scores

    def restart(self):
        """
        Reset the current SNP index to 0.
        """
        self._currentSNP = 0

    def nextSNPs(self, b, minorRef=None):
        """
        Unpacks the binary array of genotypes and returns an n x b matrix of floats of
        normalized genotypes for the next b SNPs.
        """
        try:
            b = int(b)
            if b <= 0:
                raise ValueError("b must be > 0")
        except TypeError as e:
            raise TypeError("b must be an integer") from e

        if self._currentSNP + b > self.m:
            s = "{b} SNPs requested, {k} SNPs remain"
            raise ValueError(s.format(b=b, k=(self.m - self._currentSNP)))

        c = self._currentSNP
        n = self.n
        nru = self.nru
        slice = self.geno[2 * c * nru : 2 * (c + b) * nru]
        X = np.array(slice.decode(self._bedcode), dtype="float32").reshape((b, nru)).T
        X = X[0:n, :]
        Y = normalized_snps(X, b, minorRef, self.freq, self._currentSNP)

        self._currentSNP += b
        return Y

    def ldScoreVarBlocks(self, block_left: np.ndarray, c, annot=None):
        """
        Computes an unbiased estimate of L2(j) for j=1,..,M.
        """

        def func(x):
            return l2_unbiased(x, self.n)

        snp_getter = self.nextSNPs
        return self._corSumVarBlocks(block_left, c, func, snp_getter, annot)

    def _corSumVarBlocks(self, block_left, c, func, snp_getter, annot=None):
        """
        Calculate the sum of correlation coefficients.
        """
        m, n = self.m, self.n
        block_sizes = np.array(np.arange(m) - block_left)
        block_sizes = np.ceil(block_sizes / c) * c
        if annot is None:
            annot = np.ones((m, 1), dtype="float32")
        else:
            # annot = annot.astype("float32")  # Ensure annot is float32
            annot_m = annot.shape[0]
            if annot_m != self.m:
                raise ValueError("Incorrect number of SNPs in annot")

        n_a = annot.shape[1]  # number of annotations
        cor_sum = np.zeros((m, n_a), dtype="float32")
        # b = index of first SNP for which SNP 0 is not included in LD Score
        b = np.nonzero(block_left > 0)
        if np.any(b):
            b = b[0][0]
        else:
            b = m
        b = int(np.ceil(b / c) * c)  # round up to a multiple of c
        if b > m:
            c = 1
            b = m

        l_A = 0  # l_A := index of leftmost SNP in matrix A

        device = torch_device()
        A = torch.from_numpy(snp_getter(b)).to(device)  # This now returns float32 data
        cor_sum = torch.from_numpy(cor_sum).to(device)
        annot = torch.from_numpy(annot).to(device)
        rfuncAB = torch.zeros((b, c), dtype=torch.float32, device=device)
        rfuncBB = torch.zeros((c, c), dtype=torch.float32, device=device)

        # chunk inside of block
        for l_B in np.arange(0, b, c):  # l_B := index of leftmost SNP in matrix B
            B = A[:, l_B : l_B + c]
            # ld matrix
            torch.mm(A.T, B / n, out=rfuncAB)
            # ld matrix square
            rfuncAB = func(rfuncAB)
            cor_sum[l_A : l_A + b, :] += torch.mm(rfuncAB, annot[l_B : l_B + c, :].float())

        # chunk to right of block
        b0 = b
        md = int(c * np.floor(m / c))
        end = md + 1 if md != m else md
        for l_B in tqdm(np.arange(b0, end, c), desc="Compute SNP Gene Weight"):
            # check if the annot matrix is all zeros for this block + chunk
            # this happens w/ sparse categories (i.e., pathways)
            # update the block
            old_b = b
            b = int(block_sizes[l_B])
            if l_B > b0 and b > 0:
                # block_size can't increase more than c
                # block_size can't be less than c unless it is zero
                # both of these things make sense
                A = torch.hstack((A[:, old_b - b + c : old_b], B))
                l_A += old_b - b + c
            elif l_B == b0 and b > 0:
                A = A[:, b0 - b : b0]
                l_A = b0 - b
            elif b == 0:  # no SNPs to left in window, e.g., after a sequence gap
                A = torch.zeros((n, 0), dtype=torch.float32, device=device)
                l_A = l_B
            if l_B == md:
                c = m - md
                rfuncAB = torch.zeros((b, c), dtype=torch.float32, device=device)
                rfuncBB = torch.zeros((c, c), dtype=torch.float32, device=device)
            if b != old_b:
                rfuncAB = torch.zeros((b, c), dtype=torch.float32, device=device)

            B = torch.from_numpy(snp_getter(c)).to(device)  # This now returns float32 data

            annot_l_A = annot[l_A : l_A + b, :].float()
            annot_l_B = annot[l_B : l_B + c, :].float()
            p1 = torch.all(annot_l_A == 0)
            p2 = torch.all(annot_l_B == 0)
            if p1 and p2:
                continue

            B_n = B / n

            rfuncAB = func(torch.mm(A.T, B_n))
            cor_sum[l_A : l_A + b, :] += torch.mm(rfuncAB, annot_l_B)
            cor_sum[l_B : l_B + c, :] += torch.mm(annot_l_A.T, rfuncAB).T

            rfuncBB = func(torch.mm(B.T, B_n))
            cor_sum[l_B : l_B + c, :] += torch.mm(rfuncBB, annot_l_B)

        torch.mps.synchronize()

        return cor_sum.cpu().numpy()
