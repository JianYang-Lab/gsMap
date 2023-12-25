from pathlib import Path
import bitarray as ba
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import save_npz, load_npz
from tqdm import trange, tqdm


# Define the log class
class Logger(object):
    # -
    def __init__(self, fh):
        self.log_fh = open(fh, 'w')

    # -
    def log(self, msg):
        '''
        Print to log file and stdout.
        '''
        print(msg, file=self.log_fh)
        print(msg)

    # -
    def close(self):
        self.log_fh.close()


# Compute ld-score using cellular annotations
def get_compression(fh):
    '''Which sort of compression should we use with read_csv?'''
    if fh.endswith('gz'):
        compression = 'gzip'
    elif fh.endswith('bz2'):
        compression = 'bz2'
    else:
        compression = None
    # -
    return compression


# Define the reading functions
def ID_List_Factory(colnames, keepcol, fname_end, header=None, usecols=None):
    # -
    class IDContainer(object):
        """
        A class to read data from a file, store it as a DataFrame, and provide a method for a left outer join operation.
        """

        def __init__(self, fname):
            """
            Initialize the IDContainer with the given filename and reading options.
            """
            self.usecols = usecols
            self.colnames = colnames
            self.keepcol = keepcol
            self.fname_end = fname_end
            self.header = header
            self.read(fname)
            self.n = len(self.df)

        # -
        def read(self, fname):
            """
            Read data from the given file and store it as a DataFrame.
            """
            end = self.fname_end
            if end and not fname.endswith(end):
                raise ValueError('{f} filename must end in {f}'.format(f=end))
            comp = get_compression(fname)
            self.df = pd.read_csv(fname, header=self.header, usecols=self.usecols,
                                  delim_whitespace=True, compression=comp)
            if self.colnames:
                self.df.columns = self.colnames
            if self.keepcol is not None:
                self.IDList = self.df.iloc[:, [self.keepcol]].astype('object')

        # -
        def loj(self, externalDf):
            """
            Perform a left outer join operation with the given external DataFrame.
            """
            r = externalDf.columns[0]
            l = self.IDList.columns[0]
            merge_df = externalDf.iloc[:, [0]]
            merge_df['keep'] = True
            z = pd.merge(self.IDList, merge_df, how='left', left_on=l, right_on=r,
                         sort=False)
            ii = z['keep'] == True
            return np.nonzero(ii)[0]

    # -
    return IDContainer


def getBlockLefts(coords, max_dist):
    '''
    Converts coordinates + max block length to the a list of coordinates of the leftmost
    SNPs to be included in blocks.
    Parameters
    ----------
    coords : array
        Array of coordinates. Must be sorted.
    max_dist : float
        Maximum distance between SNPs included in the same window.
    Returns
    -------
    block_left : 1D np.ndarray with same length as block_left
        block_left[j] :=  min{k | dist(j, k) < max_dist}.
    '''
    M = len(coords)
    j = 0
    block_left = np.zeros(M)
    for i in range(M):
        while j < M and abs(coords[j] - coords[i]) > max_dist:
            j += 1

        block_left[i] = j
    return block_left


def block_left_to_right(block_left):
    '''
    Converts block lefts to block rights.
    Parameters
    ----------
    block_left : array
        Array of block lefts.
    Returns
    -------
    block_right : 1D np.ndarray with same length as block_left
        block_right[j] := max {k | block_left[k] <= j}
    '''
    M = len(block_left)
    j = 0
    block_right = np.zeros(M)
    for i in range(M):
        while j < M and block_left[j] <= i:
            j += 1
        block_right[i] = j

    return block_right


class GenotypeArrayInMemory(object):
    '''
    Parent class for various classes containing interfaces for files with genotype
    matrices, e.g., plink .bed files, etc
    '''

    def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None, mafMin=None):
        self.m = len(snp_list.IDList)
        self.n = n
        self.keep_snps = keep_snps
        self.keep_indivs = keep_indivs
        self.df = np.array(snp_list.df[['CHR', 'SNP', 'BP', 'CM']])
        self.colnames = ['CHR', 'SNP', 'BP', 'CM']
        self.mafMin = mafMin if mafMin is not None else 0
        self._currentSNP = 0
        (self.nru, self.geno) = self.__read__(fname, self.m, n)
        # filter individuals
        if keep_indivs is not None:
            keep_indivs = np.array(keep_indivs, dtype='int')
            if np.any(keep_indivs > self.n):
                raise ValueError('keep_indivs indices out of bounds')
            # -
            (self.geno, self.m, self.n) = self.__filter_indivs__(self.geno, keep_indivs, self.m, self.n)
            # -
            if self.n > 0:
                print('After filtering, {n} individuals remain'.format(n=self.n))
            else:
                raise ValueError('After filtering, no individuals remain')
        # -
        # filter SNPs
        if keep_snps is not None:
            keep_snps = np.array(keep_snps, dtype='int')
            if np.any(keep_snps > self.m):  # if keep_snps is None, this returns False
                raise ValueError('keep_snps indices out of bounds')
        # -
        (self.geno, self.m, self.n, self.kept_snps, self.freq) = self.__filter_snps_maf__(
            self.geno, self.m, self.n, self.mafMin, keep_snps)
        # -
        if self.m > 0:
            print('After filtering, {m} SNPs remain'.format(m=self.m))
        else:
            raise ValueError('After filtering, no SNPs remain')
        # -
        self.df = self.df[self.kept_snps, :]
        self.maf = np.minimum(self.freq, np.ones(self.m) - self.freq)
        self.sqrtpq = np.sqrt(self.freq * (np.ones(self.m) - self.freq))
        self.df = np.c_[self.df, self.maf]
        self.colnames.append('MAF')

    # -
    def __read__(self, fname, m, n):
        raise NotImplementedError

    def __restart__(self):
        self._currentSNP = 0

    # -
    def __filter_indivs__(geno, keep_indivs, m, n):
        raise NotImplementedError

    # -
    def __filter_maf_(geno, m, n, maf):
        raise NotImplementedError

    # -
    def ldScoreVarBlocks(self, block_left, c, annot=None):
        '''Computes an unbiased estimate of L2(j) for j=1,..,M.'''
        func = lambda x: self.__l2_unbiased__(x, self.n)
        snp_getter = self.nextSNPs
        return self.__corSumVarBlocks__(block_left, c, func, snp_getter, annot)

    # -
    # In small samples, the observed r^2 tends to be higher than the true r^2 due to sampling variability.
    # The bias correction term (1-sq) / denom adjusts for this bias by subtracting a small value that depends on the sample size and the observed r^2.
    def __l2_unbiased__(self, x, n):
        denom = n - 2 if n > 2 else n  # allow n<2 for testing purposes
        sq = np.square(x)
        return sq - (1 - sq) / denom

    # -
    # Methods for calculating sums of Pearson correlation coefficients (i.e.,ld-score)
    # c stands for the chunk size (default = 50)
    def __corSumVarBlocks__(self, block_left, c, func, snp_getter, annot=None):
        '''
        Parameters
        ----------
        block_left : np.ndarray with shape (M, )
            block_left[i] = index of leftmost SNP included in LD Score of SNP i.
            if c > 1, then only entries that are multiples of c are examined, and it is
            assumed that block_left[a*c+i] = block_left[a*c], except at
            the beginning of the chromosome where the 0th SNP is included in the window.
        c : int
            Chunk size.
        func : function
            Function to be applied to the genotype correlation matrix. Before dotting with
            annot. Examples: for biased L2, np.square. For biased L4,
            lambda x: np.square(np.square(x)). For L1, lambda x: x.
        snp_getter : function(int)
            The method to be used to get the next SNPs
        annot: numpy array with shape (m,n_a)
            SNP annotations.
        Returns
        -------
        cor_sum : np.ndarray with shape (M, num_annots)
            Estimates.
        '''
        m, n = self.m, self.n
        block_sizes = np.array(np.arange(m) - block_left)
        block_sizes = np.ceil(block_sizes / c) * c
        if annot is None:
            annot = np.ones((m, 1))
        else:
            annot_m = annot.shape[0]
            if annot_m != self.m:
                raise ValueError('Incorrect number of SNPs in annot')
        # -
        n_a = annot.shape[1]  # number of annotations
        cor_sum = np.zeros((m, n_a))
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
        A = snp_getter(b)
        rfuncAB = np.zeros((b, c))
        rfuncBB = np.zeros((c, c))
        # chunk inside of block
        for l_B in np.arange(0, b, c):  # l_B := index of leftmost SNP in matrix B
            B = A[:, l_B:l_B + c]
            # ld matrix
            np.dot(A.T, B / n, out=rfuncAB)
            # ld matrix square
            rfuncAB = func(rfuncAB)
            cor_sum[l_A:l_A + b, :] += np.dot(rfuncAB, annot[l_B:l_B + c, :])

        # chunk to right of block
        b0 = b
        md = int(c * np.floor(m / c))
        end = md + 1 if md != m else md
        for l_B in np.arange(b0, end, c):
            # check if the annot matrix is all zeros for this block + chunk
            # this happens w/ sparse categories (i.e., pathways)
            # update the block
            old_b = b
            b = int(block_sizes[l_B])
            if l_B > b0 and b > 0:
                # block_size can't increase more than c
                # block_size can't be less than c unless it is zero
                # both of these things make sense
                A = np.hstack((A[:, old_b - b + c:old_b], B))
                l_A += old_b - b + c
            elif l_B == b0 and b > 0:
                A = A[:, b0 - b:b0]
                l_A = b0 - b
            elif b == 0:  # no SNPs to left in window, e.g., after a sequence gap
                A = np.array(()).reshape((n, 0))
                l_A = l_B
            if l_B == md:
                c = m - md
                rfuncAB = np.zeros((b, c))
                rfuncBB = np.zeros((c, c))
            if b != old_b:
                rfuncAB = np.zeros((b, c))
            # -
            B = snp_getter(c)
            p1 = np.all(annot[l_A:l_A + b, :] == 0)
            p2 = np.all(annot[l_B:l_B + c, :] == 0)
            if p1 and p2:
                continue
            # -
            np.dot(A.T, B / n, out=rfuncAB)
            rfuncAB = func(rfuncAB)
            cor_sum[l_A:l_A + b, :] += np.dot(rfuncAB, annot[l_B:l_B + c, :])
            cor_sum[l_B:l_B + c, :] += np.dot(annot[l_A:l_A + b, :].T, rfuncAB).T
            np.dot(B.T, B / n, out=rfuncBB)
            rfuncBB = func(rfuncBB)
            cor_sum[l_B:l_B + c, :] += np.dot(rfuncBB, annot[l_B:l_B + c, :])
        # -
        return cor_sum


class PlinkBEDFile(GenotypeArrayInMemory):
    '''
    Interface for Plink .bed format
    '''

    def __init__(self, fname, n, snp_list, keep_snps=None, keep_indivs=None, mafMin=None):
        self._bedcode = {
            2: ba.bitarray('11'),
            9: ba.bitarray('10'),
            1: ba.bitarray('01'),
            0: ba.bitarray('00')
        }
        # -
        GenotypeArrayInMemory.__init__(self, fname, n, snp_list, keep_snps=keep_snps, keep_indivs=keep_indivs,
                                       mafMin=mafMin)

    # -
    def __read__(self, fname, m, n):
        if not fname.endswith('.bed'):
            raise ValueError('.bed filename must end in .bed')
        # -
        fh = open(fname, 'rb')
        magicNumber = ba.bitarray(endian="little")
        magicNumber.fromfile(fh, 2)
        bedMode = ba.bitarray(endian="little")
        bedMode.fromfile(fh, 1)
        e = (4 - n % 4) if n % 4 != 0 else 0
        nru = n + e
        self.nru = nru
        # check magic number
        if magicNumber != ba.bitarray('0011011011011000'):
            raise IOError("Magic number from Plink .bed file not recognized")
        # -
        if bedMode != ba.bitarray('10000000'):
            raise IOError("Plink .bed file must be in default SNP-major mode")
        # check file length
        self.geno = ba.bitarray(endian="little")
        self.geno.fromfile(fh)
        self.__test_length__(self.geno, self.m, self.nru)
        return (self.nru, self.geno)

    # -
    def __test_length__(self, geno, m, nru):
        exp_len = 2 * m * nru
        real_len = len(geno)
        if real_len != exp_len:
            s = "Plink .bed file has {n1} bits, expected {n2}"
            raise IOError(s.format(n1=real_len, n2=exp_len))

    # -
    def __filter_indivs__(self, geno, keep_indivs, m, n):
        n_new = len(keep_indivs)
        e = (4 - n_new % 4) if n_new % 4 != 0 else 0
        nru_new = n_new + e
        nru = self.nru
        z = ba.bitarray(m * 2 * nru_new, endian="little")
        z.setall(0)
        for e, i in enumerate(keep_indivs):
            z[2 * e::2 * nru_new] = geno[2 * i::2 * nru]
            z[2 * e + 1::2 * nru_new] = geno[2 * i + 1::2 * nru]
        self.nru = nru_new
        return (z, m, n_new)

    # -
    def __filter_snps_maf__(self, geno, m, n, mafMin, keep_snps):
        '''
        Credit to Chris Chang and the Plink2 developers for this algorithm
        Modified from plink_filter.c
        https://github.com/chrchang/plink-ng/blob/master/plink_filter.c
        Genotypes are read forwards (since we are cheating and using endian="little")
        A := (genotype) & 1010...
        B := (genotype) & 0101...
        C := (A >> 1) & B
        Then
        a := A.count() = missing ct + hom major ct
        b := B.count() = het ct + hom major ct
        c := C.count() = hom major ct
        Which implies that
        missing ct = a - c
        # of indivs with nonmissing genotype = n - a + c
        major allele ct = b + c
        major allele frequency = (b+c)/(2*(n-a+c))
        het ct + missing ct = a + b - 2*c
        Why does bitarray not have >> ????
        '''
        nru = self.nru
        m_poly = 0
        y = ba.bitarray()
        if keep_snps is None:
            keep_snps = range(m)
        kept_snps = []
        freq = []
        for e, j in enumerate(keep_snps):
            z = geno[2 * nru * j:2 * nru * (j + 1)]
            A = z[0::2]
            a = A.count()
            B = z[1::2]
            b = B.count()
            c = (A & B).count()
            major_ct = b + c  # number of copies of the major allele
            n_nomiss = n - a + c  # number of individuals with nonmissing genotypes
            f = major_ct / (2 * n_nomiss) if n_nomiss > 0 else 0
            het_miss_ct = a + b - 2 * c  # remove SNPs that are only either het or missing
            if np.minimum(f, 1 - f) > mafMin and het_miss_ct < n:
                freq.append(f)
                y += z
                m_poly += 1
                kept_snps.append(j)
        # -
        return (y, m_poly, n, kept_snps, freq)

    # -
    def nextSNPs(self, b, minorRef=None):
        '''
        Unpacks the binary array of genotypes and returns an n x b matrix of floats of
        normalized genotypes for the next b SNPs, where n := number of samples.
        Parameters
        ----------
        b : int
            Number of SNPs to return.
        minorRef: bool, default None
            Should we flip reference alleles so that the minor allele is the reference?
            (This is useful for computing l1 w.r.t. minor allele).
        Returns
        -------
        X : np.array with dtype float64 with shape (n, b), where n := number of samples
            Matrix of genotypes normalized to mean zero and variance one. If minorRef is
            not None, then the minor allele will be the positive allele (i.e., two copies
            of the minor allele --> a positive number).
        '''
        # -
        try:
            b = int(b)
            if b <= 0:
                raise ValueError("b must be > 0")
        except TypeError:
            raise TypeError("b must be an integer")
        # -
        if self._currentSNP + b > self.m:
            s = '{b} SNPs requested, {k} SNPs remain'
            raise ValueError(s.format(b=b, k=(self.m - self._currentSNP)))
        # -
        c = self._currentSNP
        n = self.n
        nru = self.nru
        slice = self.geno[2 * c * nru:2 * (c + b) * nru]
        X = np.array(slice.decode(self._bedcode), dtype="float64").reshape((b, nru)).T
        X = X[0:n, :]
        Y = np.zeros(X.shape)
        # normalize the SNPs and impute the missing one with the mean
        for j in range(0, b):
            newsnp = X[:, j]
            ii = newsnp != 9
            avg = np.mean(newsnp[ii])
            newsnp[np.logical_not(ii)] = avg
            denom = np.std(newsnp)
            if denom == 0:
                denom = 1
            # -
            if minorRef is not None and self.freq[self._currentSNP + j] > 0.5:
                denom = denom * -1
            # -
            Y[:, j] = (newsnp - avg) / denom
        # -
        self._currentSNP += b
        return Y


class PlinkBEDFileWithR2Cache(PlinkBEDFile):
    def compute_r2_cache(self,
                         block_left,
                         output_cache_file_dir: Path,
                         chunk_size=500_000_000,
                         c=500,
                         r2_threshold=1e-4,
                         annot=None):

        func = np.square
        snp_getter = self.nextSNPs
        data, rows, cols = [], [], []

        def add_rfuncAB(rfuncAB, l_A, l_B):
            non_zero_indices = np.nonzero(rfuncAB > r2_threshold)
            data.extend(rfuncAB[non_zero_indices])
            rows.extend(l_A + non_zero_indices[0])
            cols.extend(l_B + non_zero_indices[1])

        # def add_rfuncAB(rfuncAB, l_A, l_B):
        #     # not need select non zero indices
        #     data.extend(rfuncAB.flatten())
        #     rows.extend(l_A + np.repeat(np.arange(rfuncAB.shape[0]), rfuncAB.shape[1]))
        #     cols.extend(l_B + np.tile(np.arange(rfuncAB.shape[1]), rfuncAB.shape[0]))

        # def add_rfuncBB(rfuncBB, l_B):
        #     non_zero_indices = np.nonzero(rfuncBB)
        #     data.extend(rfuncBB[non_zero_indices])
        #     rows.extend(l_B + non_zero_indices[0])
        #     cols.extend(l_B + non_zero_indices[1])

        def add_rfuncBB(rfuncBB, l_B):
            non_zero_indices = np.nonzero(rfuncBB > r2_threshold)
            data.extend(rfuncBB[non_zero_indices])
            rows.extend(l_B + non_zero_indices[0])
            cols.extend(l_B + non_zero_indices[1])
            if len(data) > chunk_size:
                # save the cache
                print(f'Start saving the cache file: {output_cache_file_dir / f"{l_B}.npz"}')
                r2_sparse_matrix = csr_matrix((data, (rows, cols)), shape=(self.m, self.m), dtype='float16')
                save_npz(output_cache_file_dir / f'{l_B}.npz', r2_sparse_matrix)
                # reset the data
                data.clear()
                rows.clear()
                cols.clear()

        m, n = self.m, self.n
        block_sizes = np.array(np.arange(m) - block_left)
        block_sizes = np.ceil(block_sizes / c) * c
        if annot is None:
            annot = np.ones((m, 1))
        else:
            annot_m = annot.shape[0]
            if annot_m != self.m:
                raise ValueError('Incorrect number of SNPs in annot')
        # -
        n_a = annot.shape[1]  # number of annotations
        # cor_sum = np.zeros((m, n_a))
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
        A = snp_getter(b)
        rfuncAB = np.zeros((b, c))
        rfuncBB = np.zeros((c, c))
        # chunk inside of block
        for l_B in np.arange(0, b, c):  # l_B := index of leftmost SNP in matrix B
            B = A[:, l_B:l_B + c]
            # ld matrix
            np.dot(A.T, B / n, out=rfuncAB)
            # ld matrix square
            rfuncAB = func(rfuncAB)
            add_rfuncAB(rfuncAB, l_A, l_B)
            # cor_sum[l_A:l_A + b, :] += np.dot(rfuncAB, annot[l_B:l_B + c, :])

        # chunk to right of block
        b0 = b
        md = int(c * np.floor(m / c))
        end = md + 1 if md != m else md
        for l_B in trange(b0, end, c, desc=f'Compute r2 cache for {output_cache_file_dir.name}'):
            # check if the annot matrix is all zeros for this block + chunk
            # this happens w/ sparse categories (i.e., pathways)
            # update the block
            old_b = b
            b = int(block_sizes[l_B])
            if l_B > b0 and b > 0:
                # block_size can't increase more than c
                # block_size can't be less than c unless it is zero
                # both of these things make sense
                A = np.hstack((A[:, old_b - b + c:old_b], B))
                l_A += old_b - b + c
            elif l_B == b0 and b > 0:
                A = A[:, b0 - b:b0]
                l_A = b0 - b
            elif b == 0:  # no SNPs to left in window, e.g., after a sequence gap
                A = np.array(()).reshape((n, 0))
                l_A = l_B
            if l_B == md:
                c = m - md
                rfuncAB = np.zeros((b, c))
                rfuncBB = np.zeros((c, c))
            if b != old_b:
                rfuncAB = np.zeros((b, c))
            # -
            B = snp_getter(c)
            p1 = np.all(annot[l_A:l_A + b, :] == 0)
            p2 = np.all(annot[l_B:l_B + c, :] == 0)
            if p1 and p2:
                continue
            # -
            np.dot(A.T, B / n, out=rfuncAB)
            rfuncAB = func(rfuncAB)
            # cor_sum[l_A:l_A + b, :] += np.dot(rfuncAB, annot[l_B:l_B + c, :])
            # cor_sum[l_B:l_B + c, :] += np.dot(annot[l_A:l_A + b, :].T, rfuncAB).T
            add_rfuncAB(rfuncAB, l_A, l_B)
            add_rfuncAB(rfuncAB.T, l_B, l_A)
            np.dot(B.T, B / n, out=rfuncBB)
            rfuncBB = func(rfuncBB)
            # cor_sum[l_B:l_B + c, :] += np.dot(rfuncBB, annot[l_B:l_B + c, :])
            add_rfuncBB(rfuncBB, l_B)
        if len(data) > 0:
            # save remaining data
            # save the cache
            print(f'Start saving the cache file: {output_cache_file_dir / f"{l_B}.npz"}')
            r2_sparse_matrix = csr_matrix((data, (rows, cols)), shape=(m, m), dtype='float16')
            save_npz(output_cache_file_dir / f'{l_B}.npz', r2_sparse_matrix)
        # combine the cache files
        print(f'Start combining the cache files in {output_cache_file_dir}')
        cached_r2_matrix_files = list(output_cache_file_dir.glob('*.npz'))
        combined_r2_matrix_files = self.load_r2_matrix_from_cache_files(output_cache_file_dir)
        # remove the cache files
        for cached_r2_matrix_file in cached_r2_matrix_files:
            cached_r2_matrix_file.unlink()
        # save the combined r2 matrix
        print(f'Start saving the combined r2 matrix in {output_cache_file_dir}')
        combined_r2_matrix_file = output_cache_file_dir / 'combined_r2_matrix.npz'
        save_npz(combined_r2_matrix_file, combined_r2_matrix_files)

    def get_ldscore_using_r2_cache(self, annot_matrix, cached_r2_matrix_dir):
        """
        Compute the r2 matrix multiplication with annot_matrix
        """
        # Compute the r2 matrix multiplication with annot_matrix
        cached_r2_matrix_dir = Path(cached_r2_matrix_dir)
        # iter the cached r2 matrix files
        result_matrix = np.zeros((self.m, annot_matrix.shape[1]))
        cached_r2_matrix_files = list(cached_r2_matrix_dir.glob('*.npz'))
        assert len(cached_r2_matrix_files) > 0, (f'No cached r2 matrix files in {cached_r2_matrix_dir}'
                                                 f'Please run the function compute_r2_cache first!')
        for r2_matrix_file in tqdm(cached_r2_matrix_files, desc=f'Compute ld score for {cached_r2_matrix_dir.name}'):
            print(f'Compute r2 matrix multiplication for {r2_matrix_file}')
            r2_matrix = load_npz(r2_matrix_file)
            result_matrix += r2_matrix.dot(annot_matrix)
        return result_matrix

    def load_r2_matrix_from_cache_files(self, cached_r2_matrix_dir):
        """
        Load the r2 matrix from cache
        """
        cached_r2_matrix_dir = Path(cached_r2_matrix_dir)
        # iter the cached r2 matrix files
        cached_r2_matrix_files = list(cached_r2_matrix_dir.glob('*.npz'))
        assert len(cached_r2_matrix_files) > 0, (f'No cached r2 matrix files in {cached_r2_matrix_dir}'
                                                 f'Please run the function compute_r2_cache first!')
        # load the r2 matrix
        r2_matrix = load_npz(cached_r2_matrix_files[0])
        for r2_matrix_file in tqdm(cached_r2_matrix_files[1:], desc=f'Load r2 matrix from {cached_r2_matrix_dir.name}'):
            print(f'Load r2 matrix from {r2_matrix_file}')
            r2_matrix += load_npz(r2_matrix_file)
        # to float16
        r2_matrix = r2_matrix.astype('float16')
        return r2_matrix
    def load_combined_r2_matrix(self, cached_r2_matrix_dir):
        """
        Load the combined r2 matrix
        """
        combined_r2_matrix_file = Path(cached_r2_matrix_dir) / 'combined_r2_matrix.npz'
        assert combined_r2_matrix_file.exists(), (f'No combined r2 matrix file in {cached_r2_matrix_dir}'
                                                  f'Should delete the cache files and run the function compute_r2_cache first!')
        # load the r2 matrix
        r2_matrix = load_npz(combined_r2_matrix_file)
        # to float16
        r2_matrix = r2_matrix.astype('float16')
        return r2_matrix

def load_bfile(bfile_chr_prefix):
    PlinkBIMFile = ID_List_Factory(['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'], 1, '.bim', usecols=[0, 1, 2, 3, 4, 5])
    PlinkFAMFile = ID_List_Factory(['IID'], 0, '.fam', usecols=[1])

    snp_file, snp_obj = bfile_chr_prefix + '.bim', PlinkBIMFile
    array_snps = snp_obj(snp_file)
    m = len(array_snps.IDList)
    print(f'Read list of {m} SNPs from {snp_file}')
    #
    # Load fam
    ind_file, ind_obj = bfile_chr_prefix + '.fam', PlinkFAMFile
    array_indivs = ind_obj(ind_file)
    n = len(array_indivs.IDList)
    print(f'Read list of {n} individuals from {ind_file}')

    # Load genotype array
    array_file, array_obj = bfile_chr_prefix + '.bed', PlinkBEDFileWithR2Cache
    geno_array = array_obj(array_file, n, array_snps, keep_snps=None, keep_indivs=None, mafMin=None)

    return array_snps, array_indivs, geno_array


def generate_r2_matrix_chr_cache(bfile_chr_prefix, ld_wind_cm, output_cache_file_dir):
    # Load genotype array
    array_snps, array_indivs, geno_array = load_bfile(bfile_chr_prefix)
    # Compute block lefts
    block_left = getBlockLefts(geno_array.df[:, 3], ld_wind_cm)
    # Compute LD score
    r2_matrix = geno_array.load_r2_matrix_from_cache(output_cache_file_dir)


def generate_r2_matrix_cache(bfile_prefix, chromosome_list, r2_cache_dir, ld_wind_cm=1):
    r2_cache_dir = Path(r2_cache_dir)

    for chr in chromosome_list:
        output_cache_file_prefix = r2_cache_dir / f'chr{chr}'
        output_cache_file_prefix.mkdir(parents=True, exist_ok=True)
        bfile_chr_prefix = bfile_prefix + '.' + str(chr)
        generate_r2_matrix_chr_cache(bfile_chr_prefix,
                                     ld_wind_cm=ld_wind_cm,
                                     output_cache_file_dir=output_cache_file_prefix)
        print(f'Compute r2 matrix for chr{chr} done!')


if __name__ == '__main__':
    bfile_prefix = '/storage/yangjianLab/sharedata/LDSC_resource/1000G_EUR_Phase3_plink/1000G.EUR.QC'
    chromosome_list = range(1, 22)
    r2_cache_dir = Path('/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/GPS_test/r2_matrix')
    ld_wind_cm = 1
    generate_r2_matrix_cache(bfile_prefix, chromosome_list, r2_cache_dir, ld_wind_cm)
