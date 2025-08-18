"""
gsMap configuration module.

This module provides:
- Configuration dataclasses for all gsMap commands
- Base classes with automatic path generation
- Decorators for CLI integration and resource tracking
"""

from collections import OrderedDict
from collections import namedtuple

# Base classes and utilities
from .base import ConfigWithAutoPaths, ensure_path_exists

# Decorators
from .decorators import dataclass_typer, track_resource_usage, show_banner

# Create a legacy registry for backward compatibility with main.py
cli_function_registry = OrderedDict()
subcommand = namedtuple("subcommand", ["name", "func", "add_args_function", "description"])

# Configuration dataclasses
from .dataclasses import (
    RunAllModeConfig,
    FindLatentRepresentationsConfig,
    LatentToGeneConfig,
    SpatialLDSCConfig,
    ReportConfig,
    MaxPoolingConfig,
    GenerateLDScoreConfig,
    CauchyCombinationConfig,
    CreateSliceMeanConfig,
    FormatSumstatsConfig,
    DiagnosisConfig,
    VisualizeConfig,
    ThreeDCombineConfig,
    RunLinkModeConfig,
)

__all__ = [
    # Base classes
    'ConfigWithAutoPaths',
    'ensure_path_exists',
    
    # Decorators
    'dataclass_typer',
    'track_resource_usage',
    'show_banner',
    
    # Legacy compatibility
    'cli_function_registry',
    
    # Configurations
    'RunAllModeConfig',
    'FindLatentRepresentationsConfig',
    'LatentToGeneConfig',
    'SpatialLDSCConfig',
    'ReportConfig',
    'MaxPoolingConfig',
    'GenerateLDScoreConfig',
    'CauchyCombinationConfig',
    'CreateSliceMeanConfig',
    'FormatSumstatsConfig',
    'DiagnosisConfig',
    'VisualizeConfig',
    'ThreeDCombineConfig',
    'RunLinkModeConfig',
]