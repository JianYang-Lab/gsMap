#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:34:35 2023

@author: songliyang
"""
import logging
import sys
from pathlib import Path

import pyranges as pr
from progress.bar import IncrementalBar

sys.path.append('/storage/yangjianLab/songliyang/SpatialData/spatial_ldsc_v1')
from Build_LD_Score_old import *
from GPS.generate_r2_matrix import PlinkBEDFileWithR2Cache
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
	'[{asctime}] {levelname:8s} {filename} {message}', style='{'))
logger.addHandler(handler)

class Snp_Annotator:
    """
    1. Annotate SNPs based on score of genes. 
    2. Add baseline annotations. 
    """
    def __init__(self, mk_score_file, gtf_file, bfile_root,annot_root,annot_name, chr = None,base_root = None,
                 window_size = 50000, const_max_size = 100):
        #
        # marker score
        self.mk_score_file = mk_score_file
        self.mk_score = self.load_marker_score()
        #
        # chunk cells
        #self.const_max_size = const_max_size
        self.n_cells = len(self.mk_score.columns)
        self.max_chunk = const_max_size
        #self.max_chunk = floor(self.n_cells / self.const_max_size)
        #
        # gtf data
        self.gtf_file = gtf_file
        self.window_size = window_size
        self.gtf_pr = self.load_gtf(mk_score = self.mk_score)
        #
        self.bfile_root = bfile_root
        self.annot_root = annot_root
        self.base_root = base_root
        self.chr = chr
    
        self.data_name = annot_name
    
    #    
    def load_marker_score(self):
        """
        Load marker scores of each cell.
        """
        mk_score = pd.read_feather(self.mk_score_file).set_index('HUMAN_GENE_SYM').rename_axis('gene_name')
        mk_score.insert(0, 'all_gene', 1)
        return mk_score
    
    #
    def load_gtf(self,mk_score):
        """
        Load the gene annotation file (gtf). 
        """
        print("Loading gtf data")
        #
        # Load GTF file
        gtf = pr.read_gtf(self.gtf_file)
        gtf = gtf.df
        #
        # Select the common genes
        gtf = gtf[gtf['Feature'] == 'gene']
        common_gene = np.intersect1d(mk_score.index, gtf.gene_name)
        #
        gtf = gtf[gtf.gene_name.isin(common_gene)]
        mk_score = mk_score[mk_score.index.isin(common_gene)]
        #
        # Remove duplicated lines
        gtf = gtf.drop_duplicates(subset='gene_name', keep="first")
        #
        # Process the GTF (open 100-KB window: Tss - Ted)
        gtf_bed = gtf[['Chromosome', 'Start', 'End', 'gene_name']].copy()
        gtf_bed.loc[:,'TSS'] = gtf_bed['Start']
        gtf_bed.loc[:,'TED'] = gtf_bed['End']
        
        gtf_bed.loc[:,'Start'] = gtf_bed['TSS'] - self.window_size
        gtf_bed.loc[:,'End'] = gtf_bed['TED'] + self.window_size
        gtf_bed.loc[gtf_bed['Start'] < 0, 'Start'] = 0
        #
        # Transform the GTF to PyRanges
        gtf_pr = pr.PyRanges(gtf_bed)
        return gtf_pr
    
    #
    def load_baseline(self,chr):
        """
        Load baseline annotations.
        """
        baseline = pd.read_csv(f'{self.base_root}.{chr}.annot.gz',sep='\t')
        baseline.drop(['CHR','BP','CM'], axis=1, inplace=True)
        return baseline
    
    #-
    def Load_bim(self,chr):
        """
        Load bim files.
        """
        bim_file = f'{self.bfile_root}.{chr}.bim'
        bim = pd.read_csv(bim_file, sep='\t', header=None)
        bim.columns = ["CHR", "SNP", "CM", "BP", "A1", "A2"]
        #    
        # Transform bim to PyRanges
        bim_pr = bim.copy()
        bim_pr.columns = ["Chromosome", "SNP", "CM", "Start", "A1", "A2"]
        bim_pr['End'] = bim_pr['Start']
        bim_pr = pr.PyRanges(bim_pr)
        bim_pr.Chromosome = f'chr{chr}'
        return bim_pr, bim
   
    #-
    def Overlaps_gtf_bim(self,bim_pr):
        """
        Find overlaps between gtf and bim file. 
        """
        # Select the overlapped regions (SNPs in gene windows)
        overlaps = self.gtf_pr.join(bim_pr)
        overlaps = overlaps.df
        overlaps['Distance'] = np.abs(overlaps['Start_b'] - overlaps['TSS'])
        # 
        # For SNPs in multiple gene windows, assign them to the nearest genes (snp pos - gene tss)
        # TODO!!!: bug here, we need the SNP with the smallest distance to the TSS, but in the original code, it calculates the distance to the start of the gene in the gtf file, only when gene in plus strand, it is the same as the TSS, but when gene in minus strand, it is not the same as the TSS
        overlaps_small = overlaps.copy()
        overlaps_small = overlaps_small.loc[overlaps_small.groupby('SNP').Distance.idxmin()]
        return overlaps_small
    
    #-
    def map_baseline(self,snp_score,baseline,chr):
        """
        Generate the baseline annotations for SNPs.
        """
        
        header = snp_score.columns[0:6].to_list()
        
        if baseline is None:
            print(f'Baseline annotations of chr{chr} are not provided, using uniform annotations for genes and SNPs')
            baseline_score = snp_score[header + ['all_gene']].copy()
            baseline_score.loc[:,'base'] = 1
            
        else:    
            print(f'Mapping baseline annotations of chr{chr}')
            snp_score_baseline = pd.merge(snp_score,baseline,how='left', on='SNP').fillna(0).copy()

            baseline_score = snp_score_baseline[header + ['all_gene'] + baseline.columns.to_list()]
            baseline_score = baseline_score.loc[:,~baseline_score.columns.duplicated()].copy()

        # Create the folder (for baseline annotation)     
        file_base_root = f'{self.annot_root}/baseline'
        if not os.path.exists(file_base_root):
            os.makedirs(file_base_root, mode=0o777, exist_ok=True)    

        # Save baseline annotations (in parquet format)
        file_base = f'{file_base_root}/baseline.{chr}.feather'
        baseline_score.to_feather(file_base)
        
        return 0
    
    #-   
    def annotate_chr(self,chr):
        """
        Annotate SNPs of each chr. 
        """
        # Load the baseline file
        baseline = None
        if self.base_root is not None:
            baseline = self.load_baseline(chr)
        
        # Load the bim file    
        bim_pr, bim = self.Load_bim(chr)
        
        # Find overlapping
        overlaps_small = self.Overlaps_gtf_bim(bim_pr)
        
        # Do annotations
        all_chunks = int(np.ceil(self.n_cells / self.max_chunk))
        bar = IncrementalBar(f'Mapping the gene marker scores to SNPs in chr{chr}', max=all_chunks)
        bar.check_tty = False

        # Preprocess bim outside the loop as it doesn't change
        anno_template = bim[["CHR", "BP", "SNP", "CM"]]

        for chunk_index, left in enumerate(range(0, self.n_cells, self.max_chunk), start=1):
            right = min(left + self.max_chunk, self.n_cells)
            mk_score_current = self.mk_score.iloc[:, left:right]

            # Process marker scores for SNPs
            anno = anno_template.copy()
            merged_data = overlaps_small[['SNP', 'gene_name', 'TSS']].merge(mk_score_current, on='gene_name',
                                                                            how='left')
            snp_score = pd.merge(anno, merged_data, how='left', on='SNP').fillna(0)
            snp_score = snp_score.rename(columns={'gene_name': 'Gene'})
            snp_score.loc[snp_score.Gene == 0, 'Gene'] = 'None'

            # Process baseline annotations for the first chunk
            if chunk_index == 1:
                self.map_baseline(snp_score, baseline, chr)
                snp_score = snp_score.drop('all_gene', axis=1)

            # Create the folder and save SNP annotations
            file_root = f'{self.annot_root}/{self.data_name}_chunk{chunk_index}'
            os.makedirs(file_root, mode=0o777, exist_ok=True)
            file_anno = f'{file_root}/{self.data_name}.{chr}.feather'
            snp_score.to_feather(file_anno)

            bar.next()

        bar.finish()
        
        return all_chunks
            
    #
    def annotate(self):
        """
        Perform SNP annotations for each chromosome. 
        """
        if self.chr == None:
            for chr in range(1, 23):
                const_max_size = self.annotate_chr(chr=chr)
        else:
            const_max_size = self.annotate_chr(chr=self.chr)  
        
        return const_max_size 

    
    
    
    
class LDscore_Generator:
    def __init__(self, bfile_root, annot_root, const_max_size,annot_name,
                 chr=None,ld_wind_snps=None, ld_wind_kb=None, ld_wind_cm=1, keep_snp=None,
                 r2_cache_dir=None,
                 ):
        self.bfile_root = bfile_root
        self.annot_root = annot_root
        self.ld_wind_snps = ld_wind_snps
        self.ld_wind_kb = ld_wind_kb
        self.ld_wind_cm = ld_wind_cm
        self.keep_snp = keep_snp
        self.chr = chr
        
        self.data_name = annot_name
        self.const_max_size = const_max_size
        self.generate_r2_cache = False

        # Set the r2 cache
        if r2_cache_dir is None:
            logger.info('No r2 cache directory specified, will not use r2 cache')
            self.chr_r2_cache_dir = None
        else:
            assert chr is not None, 'Must specify chr when using r2 cache'
            chr_r2_cache_dir = os.path.join(r2_cache_dir, f'chr{chr}')
            self.chr_r2_cache_dir = chr_r2_cache_dir
            if not os.path.exists(chr_r2_cache_dir) or len(glob.glob(os.path.join(chr_r2_cache_dir, '*.npz'))) == 0:
                logger.warning(f'No r2 cache found for chr{chr}, will generate r2 cache for this chromosome, first time may take a while')
                os.makedirs(chr_r2_cache_dir, exist_ok=True, mode=0o777, )
                self.generate_r2_cache= True
            else:
                logger.info(f'Found r2 cache for chr{chr}, will use r2 cache for this chromosome')

    def compute_ldscore(self):
        """
        Compute LD scores.
        """
        if self.chr == None:
            for chr in range(1, 23):
                self.compute_ldscore_chr(chr=chr)
        else:
            self.compute_ldscore_chr(chr=self.chr) 
            
            
    def compute_ldscore_chunk(self,annot_file,ld_score_file,M_file,M_5_file,geno_array:PlinkBEDFileWithR2Cache, block_left, snp):
        """
        Compute and save LD scores for each chunk
        :param annot_file: Path to the annotation file
        :param ld_score_file: Path to the LD score file
        :param M_file: Path to the M file
        :param M_5_file: Path to the M_5_50 file
        :param geno_array: Genotype array
        :param block_left: Block left
        :param snp: SNP to be kept
        :return: None
        """
        annot_df = pd.read_feather(annot_file)
        n_annot, ma = len(annot_df.columns) - 6, len(annot_df)

        # print("Read {A} annotations for {M} SNPs from {f}".format(f=annot_file, A=n_annot, M=ma))
        annot_matrix = np.array(annot_df.iloc[:, 6:])
        annot_colnames = annot_df.columns[6:]

        # Reset the SNP point
        geno_array.__restart__()

        # Compute annotated LD score
        if self.chr_r2_cache_dir is None:
            lN_df = pd.DataFrame(geno_array.ldScoreVarBlocks(block_left, 50, annot=annot_matrix))
        else:
            lN_df = pd.DataFrame(self.get_ldscore_use_cache(annot_matrix))

        ldscore = pd.concat([annot_df.iloc[:, 0:6], lN_df], axis=1)
        ldscore.columns = annot_df.columns
        
        # Keep the targeted SNPs
        if not snp is None:
            ldscore = ldscore.loc[ldscore.SNP.isin(snp)]
        
        # Save the LD score annotations
        ldscore = ldscore.reset_index()
        ldscore.drop(columns=['index'], inplace=True)
        # ldscore.to_feather(ld_score_file)
        
        # Compute the .M (.M_5_50) file
        M = np.atleast_1d(np.squeeze(np.asarray(np.sum(annot_matrix, axis=0))))
        ii = geno_array.maf > 0.05
        M_5_50 = np.atleast_1d(np.squeeze(np.asarray(np.sum(annot_matrix[ii, :], axis=0))))
        
        # Save the sum of score annotations (all and maf > 0.05)
        np.savetxt(M_file, M, delimiter='\t')
        np.savetxt(M_5_file, M_5_50, delimiter='\t')

    def get_ldscore_use_cache(self, annot_matrix,):
        return self.r2_matrix.dot(annot_matrix)

    def compute_ldscore_chr(self, chr):
        bfile = f"{self.bfile_root}.{chr}"
        #
        # Load bim file
        snp_file, snp_obj = bfile + '.bim', PlinkBIMFile
        array_snps = snp_obj(snp_file)
        m = len(array_snps.IDList)
        print(f'Read list of {m} SNPs from {snp_file}')
        # 
        # Load fam
        ind_file, ind_obj = bfile + '.fam', PlinkFAMFile
        array_indivs = ind_obj(ind_file)
        n = len(array_indivs.IDList)
        print(f'Read list of {n} individuals from {ind_file}')
        #
        # Load genotype array
        array_file, array_obj = bfile + '.bed', PlinkBEDFileWithR2Cache
        geno_array = array_obj(array_file, n, array_snps, keep_snps=None, keep_indivs=None, mafMin=None)
        
        # Load the snp to be print
        if not self.keep_snp is None:
            snp = pd.read_csv(f'{self.keep_snp}.{chr}.snp',header=None)[0].to_list()
            num_snp = len(snp)
            print(f'Loading {num_snp} SNPs')
        else:
            snp = None

        # TODO : these three arguments are mutually exclusive, which should be processed in the initialization of the arguments parser
        #Determin LD blocks
        x = np.array((self.ld_wind_snps, self.ld_wind_kb, self.ld_wind_cm), dtype=bool)
        if np.sum(x) != 1:
            raise ValueError('Must specify exactly one ld-wind option')
        # 
        if self.ld_wind_snps:
            max_dist = self.ld_wind_snps
            coords = np.array(range(geno_array.m))
        elif self.ld_wind_kb:
            max_dist = self.ld_wind_kb * 1000
            coords = np.array(array_snps.df['BP'])[geno_array.kept_snps]
        elif self.ld_wind_cm:
            max_dist = self.ld_wind_cm
            coords = np.array(array_snps.df['CM'])[geno_array.kept_snps]
        block_left = getBlockLefts(coords, max_dist)
        if self.generate_r2_cache:
            logger.info(f'Generating r2 cache for chr{chr}, this may take a while')
            geno_array.compute_r2_cache(block_left,
                                        Path(self.chr_r2_cache_dir))
            logger.info(f'Finished generating r2 cache for chr{chr}')
        if self.chr_r2_cache_dir is not None:
            self.r2_matrix = geno_array.load_combined_r2_matrix(cached_r2_matrix_dir=self.chr_r2_cache_dir)
        
        # Set the baseline root   
        annot_file = f'{self.annot_root}/baseline/baseline.{chr}.feather'
        ld_score_file = f'{self.annot_root}/baseline/baseline.{chr}.l2.ldscore.feather'
        M_file = f'{self.annot_root}/baseline/baseline.{chr}.l2.M'
        M_5_file = f'{self.annot_root}/baseline/baseline.{chr}.l2.M_5_50'

        # Compute annotations of the baseline
        print(f"Computing LD score for baseline annotations of chr{chr}")
        self.compute_ldscore_chunk(annot_file,ld_score_file,M_file,M_5_file,geno_array, block_left, snp)

        # Load annotations of chunks
        bar = IncrementalBar(f"Computing LD scores for spatial data annotations of chr{chr}", max = self.const_max_size)
        bar.check_tty = False 
        for chunk_index in range(1,self.const_max_size+1):
            
            # Set the file root
            annot_file = f'{self.annot_root}/{self.data_name}_chunk{chunk_index}/{self.data_name}.{chr}.feather'
            ld_score_file = f'{self.annot_root}/{self.data_name}_chunk{chunk_index}/{self.data_name}.{chr}.l2.ldscore.feather'
            M_file = f'{self.annot_root}/{self.data_name}_chunk{chunk_index}/{self.data_name}.{chr}.l2.M'
            M_5_file = f'{self.annot_root}/{self.data_name}_chunk{chunk_index}/{self.data_name}.{chr}.l2.M_5_50'
            
            # Compute annotations of the current chunk
            self.compute_ldscore_chunk(annot_file,ld_score_file,M_file,M_5_file,geno_array, block_left, snp)
            
            bar.next()
            
        bar.finish()


parser = argparse.ArgumentParser()
parser.add_argument('--mk_score_file', default=None, type=str)
parser.add_argument('--gtf_file', default=None, type=str)
parser.add_argument('--bfile_root', default=None, type=str)
parser.add_argument('--annot_root', default=None, type=str)
parser.add_argument('--annot_name', default=None, type=str)
parser.add_argument('--base_root', default=None, type=str)
parser.add_argument('--keep_snp', default=None, type=str)

parser.add_argument('--chr', default=None, type=int)
parser.add_argument('--window_size', default=50000, type=int)
parser.add_argument('--const_max_size', default=100, type=int)
parser.add_argument('--ld_wind_snps', default=None, type=float)
parser.add_argument('--ld_wind_kb', default=None, type=float)
parser.add_argument('--ld_wind_cm', default=None, type=float)
parser.add_argument('--r2_cache_dir', default=None, type=str)


# Defin the Container for plink files   
PlinkBIMFile = ID_List_Factory(['CHR', 'SNP', 'CM', 'BP', 'A1', 'A2'], 1, '.bim', usecols=[0, 1, 2, 3, 4, 5])
PlinkFAMFile = ID_List_Factory(['IID'], 0, '.fam', usecols=[1])
FilterFile = ID_List_Factory(['ID'], 0, None, usecols=[0])


if __name__ == '__main__':

    # Store the Params
    TEST = True
    if TEST:
        name='Cortex_151507'
        TASK_ID = 15
        args = parser.parse_args([
            '--mk_score_file',
            f'/storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/annotation/{name}/gene_markers/{name}_rank.feather',
            '--gtf_file', '/storage/yangjianLab/songliyang/ReferenceGenome/GRCh37/gencode.v39lift37.annotation.gtf',
            '--bfile_root', '/storage/yangjianLab/sharedata/LDSC_resource/1000G_EUR_Phase3_plink/1000G.EUR.QC',
            '--annot_root',
            f'/storage/yangjianLab/songliyang/SpatialData/Data/Brain/Human/Nature_Neuroscience_2021/annotation/{name}/snp_annotation',
            '--keep_snp', '/storage/yangjianLab/sharedata/LDSC_resource/hapmap3_snps/hm',
            '--annot_name', f'{name}',
            '--const_max_size', '500',
            '--chr', f'{TASK_ID}',
            '--ld_wind_cm', '1',
            '--r2_cache_dir', '/storage/yangjianLab/chenwenhao/projects/202312_GPS/data/GPS_test/r2_matrix',
        ])
    else:
        args = parser.parse_args()

    # Mapping gene score to SNPs
    snp_annotate = Snp_Annotator(mk_score_file=args.mk_score_file, gtf_file=args.gtf_file,
                                 bfile_root=args.bfile_root, annot_root=args.annot_root, 
                                 base_root=args.base_root,annot_name=args.annot_name,
                                 window_size=args.window_size, chr=args.chr, const_max_size=args.const_max_size)
    const_max_size = snp_annotate.annotate()


    # Generate LD scores annotations
    ldscore_generate = LDscore_Generator(bfile_root=args.bfile_root, annot_root=args.annot_root, const_max_size=const_max_size,
                                         annot_name=args.annot_name, chr=args.chr, ld_wind_snps=args.ld_wind_snps, ld_wind_kb=args.ld_wind_kb, 
                                         ld_wind_cm=args.ld_wind_cm, keep_snp = args.keep_snp,
                                         r2_cache_dir=args.r2_cache_dir)
    ldscore_generate.compute_ldscore()

