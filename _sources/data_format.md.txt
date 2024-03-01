(data-format)=
# Data Format


## ST Data

The input ST data should be a h5ad file, which include, at least, gene expression matrix and spatial coordinates.
```python
import scanpy as sc
HDF5_PATH="example_data/ST/E16.5_E1S1.MOSTA.h5ad"
adata = sc.read_h5ad(f'{HDF5_PATH}')
```

####  gene expression matrix
The gene expression matrix should be stored in the `layers` attribute of the h5ad file.  
The parameter `Type` of the `find_latent_representations` specifies which data layer to use. If you set `Type` to `counts`, GPS will use the gene count matrix and automatically normalize it. If `Type` is set to another value, GPS will use the data provided in the corresponding layer without any normalization.

```python
>>> adata.layers['counts']

<121767x28204 sparse matrix of type '<class 'numpy.int64'>'
	with 211994186 stored elements in Compressed Sparse Row format>
```

####  spatial coordinates
The spatial coordinates should be stored in the `obsm` attribute of the h5ad file with the key `spatial`.

```python
>>> adata.obsm['spatial']

[[158.05955033 220.80121954]
 [159.04435808 220.97486772]
 [160.02916583 221.1485159 ]
 ...
 [112.83052085 730.69369339]
 [113.8153286  730.86734157]
 [114.80013636 731.04098975]]
```

#### spots (cells) annotation
Optionally, the h5ad file could contain annotations of the spots (cells) in the `obs` attribute.  
The parameter `annotation` of the `find_latent_representations` specifies the names of annotation columns in the `obs` attribute.
```python
>>> print(adata.obs['annotation'].value_counts().head())

Brain                17374
Liver                14167
Cavity               11287
Muscle                9853
Connective tissue     8803
Name: annotation, dtype: int64
```


## GWAS Data
The input GWAS data is in a gzip text file, which should, at least, include SNP (rs number), Z (Z-statistics), and N (sample size). Columns A1 and A2 represent the effect allele and non-effect allele, respectively, but they are not necessary. The column orders can be customized, but headers are keywords and will be used by GPS.
```shell
zcat example_data/GWAS/IQ_NG_2018.sumstats.gz | head -n 5

SNP		A1	A2	Z	N
rs12184267	T	C	0.916	225955
rs12184277	G	A	0.656	226215
rs12184279	A	C	1.050	226224
rs116801199	T	G	0.300	226626
```

####  Tips for formatting the GWAS data
You can convert the GWAS summary data into the required format using customized code. For ease of use, we've provided a command in GPS to do this. We'll demonstrate how to use this command with the human height GWAS data as an example.  

First, download the human height GWAS data and decompress it.
```bash
wegt https://portals.broadinstitute.org/collaboration/giant/images/4/4e/GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL.gz
gzip -d GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL.gz
```
The first 5 lines of the file should look like this:
```bash
head -n 5 GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL

SNPID	RSID	CHR	POS	EFFECT_ALLELE	OTHER_ALLELE	EFFECT_ALLELE_FREQ	BETA	SE	P	N
1:566875:C:T	rs2185539	1	566875	T	C	0.00413	0.00762959	0.0197	0.699009	607057
1:728951:C:T	rs11240767	1	728951	T	C	0.0438	0.0206708	0.00956	0.0306735	241218
1:734462:A:G	rs12564807	1	734462	A	G	0.896	0.00426816	0.0110	0.697777	117377
1:752721:A:G	rs3131972	1	752721	G	A	0.786	0.0000413581	0.00205	0.983898	1009515
```

To convert the summary statistics, use the commands:
```bash
GPS format_sumstats \
--sumstats GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL 
--out HEIGHT
```
you will get HEIGHT.sumstats.gz, which should look like this:
```bash
zcat HEIGHT.sumstats.gz | head -n 5

SNP		A1	A2	Z	N	
rs3131969	G	A	0.328	1494218.000	
rs3131967	C	T	0.386	1488150.000	
rs12562034	A	G	1.714	1554976.000	
rs4040617	G	A	-0.463	1602016.000	
```
In most cases, this command can automatically recognize different column names of GWAS data, but you can also specify the column names to ensure correctness.
```bash
GPS format_sumstats \
--sumstats GIANT_HEIGHT_YENGO_2022_GWAS_SUMMARY_STATS_ALL 
--out HEIGHT \
--a1 EFFECT_ALLELE \
--a2 OTHER_ALLELE \
--p P \
--frq EFFECT_ALLELE_FREQ \
--snp RSID \
--beta BETA \
--se SE \
--n N
```

For more usage options, please see:
```bash
GPS format_sumstats -h
```