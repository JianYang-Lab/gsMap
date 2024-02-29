(data-format)=
# Data Format


## 1. ST Data Format

The input ST data should be the h5ad format.

```python
import scanpy as sc
HDF5_PATH="example_data/ST/E16.5_E1S1.MOSTA.h5ad"
adata = sc.read_h5ad(f'{HDF5_PATH}')
```

## 1.1 Spatial Coordinates
The spatial coordinate information should be stored in the `obsm` attribute with the key `spatial`, which is a 2D numpy array with the shape of `(n_spots, 2)`. The first column is the x coordinate and the second column is the y coordinate.

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

## 1.2 DATA_TYPE

`DATA_TYPE` parameter used in the :ref:`find_latent_representations_mouse` is used to specify the data layer where the gene expression data is stored.
If you set the `DATA_TYPE` to `count`, the GPS will automatically normalize the data. Otherwise the GPS will use the data provided in the `DATA_TYPE` layer.

```python
>>> adata.layers['counts']

<121767x28204 sparse matrix of type '<class 'numpy.int64'>'
	with 211994186 stored elements in Compressed Sparse Row format>
```


## 1.3 Spots (cells) Annotation
Optional, the h5ad file could contain a annotation of the spots (cells) in the `obs` attribute. The annotation could be a region label or cell type label.

```python
>>> print(adata.obs['annotation'].value_counts().head())

Brain                17374
Liver                14167
Cavity               11287
Muscle                9853
Connective tissue     8803
Name: annotation, dtype: int64
```


## GWAS Data Format