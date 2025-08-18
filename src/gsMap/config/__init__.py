"""
gsMap configuration module.

This module provides:
- Configuration dataclasses for all gsMap commands
- Base classes with automatic path generation
- Decorators for CLI integration and resource tracking
"""

# Base classes and utilities
from .base import ConfigWithAutoPaths, ensure_path_exists

# Decorators
from .decorators import dataclass_typer, track_resource_usage, show_banner

# Configuration dataclasses
from .dataclasses import (
    RunAllModeConfig,
    FindLatentRepresentationsConfig,
    LatentToGeneConfig,
    SpatialLDSCConfig,
    ReportConfig,
)

__all__ = [
    # Base classes
    'ConfigWithAutoPaths',
    'ensure_path_exists',
    
    # Decorators
    'dataclass_typer',
    'track_resource_usage',
    'show_banner',
    
    # Configurations
    'RunAllModeConfig',
    'FindLatentRepresentationsConfig',
    'LatentToGeneConfig',
    'SpatialLDSCConfig',
    'ReportConfig',
]