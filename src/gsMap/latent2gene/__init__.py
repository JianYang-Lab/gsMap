"""
latent2gene subpackage for gsMap

This package contains all components for converting latent representations to gene marker scores:
- Rank calculation from latent representations
- Connectivity matrix building (spatial → anchor → homogeneous)
- Marker score calculation
- Zarr-backed storage utilities
"""

from .rank_calculator import RankCalculator
from .connectivity import ConnectivityMatrixBuilder
from .marker_scores import MarkerScoreCalculator
from .zarr_utils import ZarrBackedDense, ZarrBackedCSR
from .entry_point import run_latent_to_gene

__all__ = [
    'RankCalculator',
    'ConnectivityMatrixBuilder', 
    'MarkerScoreCalculator',
    'ZarrBackedDense',
    'ZarrBackedCSR',
    'run_latent_to_gene'
]