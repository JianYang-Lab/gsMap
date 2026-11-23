3D mapping
==========

The human CS8 embryo contains **38,562 spots across 62 sections**, making it a
compact example for a quick first run. The mouse E11.5 embryo contains
**6,993,667 cells across 84 sections** and demonstrates a large-scale analysis
that requires substantially more computation time and memory.

Both follow the same sequence: get the data, run gsMap, view the report.
Each report also exports the 3D results for interactive viewing in
`SpatialVista <https://github.com/JianYang-Lab/spatial-vista-py>`_. If you use it,
please cite Wei, W. et al. "**SpatialVista as a unified ecosystem for
high-performance visualization and exploration of 3D spatial transcriptomics
data.**" *Nature Genetics* (2026).
`doi:10.1038/s41588-026-02696-7 <https://doi.org/10.1038/s41588-026-02696-7>`_


Tutorials
---------

.. toctree::
    :maxdepth: 1

    CS8 Human Embryo <human_gastrulation.md>
    Mouse E11.5 embryo <mouse_e11.md>
