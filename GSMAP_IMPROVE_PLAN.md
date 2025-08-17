1. Have a more modular design
    structure:

- models:
  - gnn
    - _model
    - _train
    - _inference
  - scRNA [will implement in future, keep it for now]
    - _model
    - _train
    - _inference
  - ATAC [will implement in future, keep it for now]

gsMap.py [gsMap main class]

2. Provide the Python API to run
  1. Provide a gsMap class to use
  2. Use the instance to run the workflow
3. More scalable design
  1. Find latent representation using subsampling
  2. Latent2Gene
    1. Single cell mode
      1. Find neighbor only by embedding
      2. Homogeneous spot finding, cross slices or not
    2. Spatial mode
      1. Single slice or mulitple slice
      2. Max pooling or not