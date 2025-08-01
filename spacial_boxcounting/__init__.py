"""
Spatial Boxcounting Package
============================

This package provides tools for spatial box counting and fractal analysis
with both CPU and GPU support.

Modules
-------
- api: High-level API functions
- core: Core algorithms (CPU and GPU)
- cli: Command-line interface
- batch: Batch processing functionality
- io: File I/O operations
- hilbert: Hilbert curve transformations
- visualize: Visualization utilities
- utils: Utility functions

"""
from . import api, core, cli, batch, io, hilbert, visualize, utils
from ._version import VERSION

__version__ = VERSION
