"""
Spatial Boxcounting Package
============================

This package provides tools for spatial box counting and fractal analysis
with both CPU and GPU support.

Modules
-------
- api: High-level API functions (supports ``backend="cpu"`` and ``backend="gpu"``)
- core: Core algorithms (CPU and GPU)
- cli: Command-line interface (supports ``--backend cpu|gpu``)
- batch: Batch processing functionality
- io: File I/O operations
- hilbert: Hilbert curve transformations
- visualize: Visualization utilities
- utils: Utility functions

"""
from . import api, batch, cli, core, hilbert, io, utils, visualize
from ._version import VERSION
from .core import CUPY_AVAILABLE, GPU_BACKEND, Z_boxcount_gpu, spacialBoxcount_gpu

__version__ = VERSION