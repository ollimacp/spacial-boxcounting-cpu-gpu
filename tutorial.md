# spacial-boxcounting Tutorial

This tutorial demonstrates how to install and use the package for spatial box counting and fractal analysis.

## Installation

Clone the repository and install dependencies:

```bash
pip install .
```

Ensure that all dependencies (numpy, numba, Pillow, matplotlib, hilbertcurve, cupy (optional for GPU), pytest) are installed.

## Basic Usage

### Processing a Single File

```python
from spacial_boxcounting.api import boxcount_from_file, fractal_dimension

# For obtaining spatial box count maps:
result_spatial = boxcount_from_file('path/to/your/image.jpg', mode='spatial')
print('Spatial Box Count Map:', result_spatial)

# For a single overall box count and lacunarity:
result_single = boxcount_from_file('path/to/your/image.jpg', mode='single')
print('Box Count & Lacunarity:', result_single)

# Estimating fractal dimension:
fd = fractal_dimension('path/to/your/image.jpg')
print('Fractal Dimension:', fd)
```

### Processing Directly from a Numpy Array

```python
import numpy as np
from spacial_boxcounting.api import boxcount_from_array

# Create or load a 2D numpy array (example random array):
arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)
result = boxcount_from_array(arr, mode='spatial')
print('Spatial Result from Array:', result)
```

### Fractal Dimension from Arrays and Files

You can compute the fractal dimension either from a 2D numpy array or directly from an image file.

#### From a Numpy Array

```python
from spacial_boxcounting.api import fractal_dimension_from_array

# Create or load a 2D numpy array
arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)
fd = fractal_dimension_from_array(arr)
print('Fractal Dimension (Array):', fd)
```

#### From an Image File

```python
from spacial_boxcounting.api import fractal_dimension_from_file

fd = fractal_dimension_from_file('path/to/your/image.jpg')
print('Fractal Dimension (File):', fd)
```

Optional parameters include:

* `maxvalue`: maximum pixel value (default is 256)
* `box_sizes`: list of box sizes to use for box counting (default is powers of 2)
* `hilbert`: whether to apply a Hilbert curve transformation (for file-based function)

## Batch Processing

Use the provided CLI for processing all files in a directory:

```bash
python3 -m spacial_boxcounting.batch path/to/your/input_folder
```

## Using GPU Acceleration (if available)

The package can utilize Cupy for GPU-accelerated operations. If cupy is installed, GPU functions such as `spacialBoxcount_gpu` and `Z_boxcount_gpu` will execute on the GPU:

```python
import numpy as np
from spacial_boxcounting.core import spacialBoxcount_gpu, Z_boxcount_gpu

# Create a random array:
arr = np.random.randint(0, 256, size=(64, 64)).astype(np.uint8)
result_gpu = spacialBoxcount_gpu(arr, iteration=0, MaxValue=256)
print('GPU spatial result:', result_gpu)
```

## Testing

Run unit tests:

```bash
pytest
```
