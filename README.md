# spacial-boxcounting: Spatial Boxcount Algorithm & Fractal Analysis

An implementation of a spatial boxcount algorithm for fractal analysis, with both CPU and GPU support for accelerated computation.

[![PyPI version](https://badge.fury.io/py/spacial-boxcounting.svg)](https://badge.fury.io/py/spacial-boxcounting)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/spacial-boxcounting.svg)](https://pypi.org/project/spacial-boxcounting/)

## Abstract
This project implements a spatial boxcount algorithm that characterizes 2D arrays by topological complexity and spatial heterogeneity. With both CPU and GPU support, it enables spatial similarity search, edge detection, and statistical analysis of image datasets.

## Key Features
- **Spatial Box Counting**: Produces 2D maps of box count ratios and lacunarity
- **Fractal Dimension Analysis**: Multi-scale fractal dimension computation
- **Multiple Processing Modes**: Spatial maps or single-value results
- **CPU & GPU Support**: Numba JIT compilation and optional CuPy acceleration
- **Batch Processing**: Process entire directories of images
- **Multiple Input Formats**: JPEG, BMP, PNG, and NumPy arrays
- **Cross-Platform**: Works on Windows, Linux, and macOS (NVIDIA GPU support)

## Installation
Install via pip:

```bash
# Basic CPU-only installation
pip install spacial_boxcounting

# With GPU support (NVIDIA CUDA)
pip install spacial_boxcounting[gpu]

# Development installation
pip install -e .
```

Ensure dependencies are installed: numpy, numba, Pillow, matplotlib, hilbertcurve, pandas, and optionally cupy for GPU acceleration.

## Quick Start
### Processing a Single File

```python
from spacial_boxcounting.api import boxcount_from_file, fractal_dimension_from_file

# Get spatial box count map (2D maps of box count ratios and lacunarity)
result_spatial = boxcount_from_file('path/to/your/image.jpg', mode='spatial')
print('Spatial Box Count Map shape:', [r.shape for r in result_spatial])

# Get overall box count & lacunarity
result_single = boxcount_from_file('path/to/your/image.jpg', mode='single')
print('Box Count & Lacunarity:', result_single)

# Compute fractal dimension
fd = fractal_dimension_from_file('path/to/your/image.jpg')
print('Fractal Dimension:', fd)
```

### Processing from a Numpy Array

```python
import numpy as np
from spacial_boxcounting.api import boxcount_from_array, fractal_dimension_from_array

arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)

# Spatial processing
result_spatial = boxcount_from_array(arr, mode='spatial')
print('Spatial Result shape:', [r.shape for r in result_spatial])

# Single value processing
result_single = boxcount_from_array(arr, mode='single')
print('Single Result:', result_single)

# Fractal dimension
fd = fractal_dimension_from_array(arr)
print('Fractal Dimension:', fd)
```

## Command-Line Interface
Process images directly from the command line:

```bash
# Process a single file
spacial-boxcount single --file path/to/image.jpg --mode spatial

# Process all images in a directory
spacial-boxcount batch --folder path/to/images/ --mode single

# Process with specific pattern
spacial-boxcount batch --folder path/to/images/ --pattern "*.jpg"
```

## GPU Acceleration
If Cupy is installed with CUDA support, GPU accelerated functions will automatically be used:

```python
import numpy as np
from spacial_boxcounting.core import spacialBoxcount_gpu

arr = np.random.randint(0, 256, size=(512, 512)).astype(np.uint8)
# GPU processing for large images (significant speedup)
result_gpu = spacialBoxcount_gpu(arr, iteration=2, MaxValue=256)  # box size 8
print('GPU spatial result shape:', [r.shape for r in result_gpu])
```

## Batch Processing
Process multiple images with progress tracking:

```python
from spacial_boxcounting.batch import batch_boxcount

# Process all images in a directory
results = batch_boxcount('path/to/images/', mode='single')
for filename, result in results.items():
    print(f'{filename}: {result}')
```

## Performance
Performance varies by hardware and image size:
- **Small images (< 256x256)**: CPU often faster due to GPU overhead
- **Large images (> 512x512)**: GPU provides 2-10x speedup
- **Batch processing**: GPU provides 5-50x speedup for large batches
- **AMD users**: CPU optimization available (ROCm support experimental)

See [PERFORMANCE_ANALYSIS.md](PERFORMANCE_ANALYSIS.md) for detailed benchmarks.

## Documentation
- [Tutorial](tutorial.md) - Comprehensive usage guide
- [API Reference](docs/api/) - Detailed function documentation
- [Platform Compatibility](PLATFORM_COMPATIBILITY.md) - Installation for all systems
- [Development Roadmap](DEVELOPMENT_ROADMAP.md) - Future plans and progress

## Packaging & Distribution
This project is structured as a pip-installable package and is available on PyPI. Future releases will include additional features and performance improvements.

## Testing
Run unit tests with:

```bash
pytest
```

Or use the verification script:

```bash
python final_verification.py
```

## Academic Context
Originally derived from academic work in spatial analysis, this repository provides the tools for box counting and lacunarity computation. For a full exposition, please review the accompanying Jupyter Notebook:
[Spacial boxcount algorithm CPU and GPU.ipynb](https://colab.research.google.com/github/ollimacp/spacial-boxcounting-cpu-gpu/blob/main/Spacial%20boxcount%20algorithm%20CPU%20and%20GPU.ipynb)

## Contributing
Contributions are welcome! Please see the [Development Roadmap](DEVELOPMENT_ROADMAP.md) for planned features and improvements.

## License
See [LICENSE.txt](LICENSE.txt) for details.
