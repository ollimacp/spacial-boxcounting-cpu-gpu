# spacial-boxcounting: Spatial Boxcount Algorithm & Fractal Analysis

An implementation of a spatial boxcount algorithm for fractal analysis, with an option to use a convolutional neural network to accelerate computation via GPU.

## Abstract
This project implements a spatial boxcount algorithm that characterizes 2D arrays by topological complexity and spatial heterogeneity. With both CPU and GPU support, it enables spatial similarity search, edge detection, and statistical analysis of image datasets. The algorithm has been translated into a convolutional neural network for faster processing on compatible hardware.

## Installation
Install via pip:

```bash
pip install .
```

Ensure dependencies are installed: numpy, numba, Pillow, matplotlib, hilbertcurve, cupy (optional), and pytest.

## Basic Usage
### Processing a Single File

```python
from spacial_boxcounting.api import boxcount_from_file, fractal_dimension

# Get spatial box count map
result_spatial = boxcount_from_file('path/to/your/image.jpg', mode='spatial')
print('Spatial Box Count Map:', result_spatial)

# Get overall box count & lacunarity
result_single = boxcount_from_file('path/to/your/image.jpg', mode='single')
print('Box Count & Lacunarity:', result_single)

# Compute fractal dimension
fd = fractal_dimension('path/to/your/image.jpg')
print('Fractal Dimension:', fd)
```

### Processing from a Numpy Array

```python
import numpy as np
from spacial_boxcounting.api import boxcount_from_array

arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)
result = boxcount_from_array(arr, mode='spatial')
print('Spatial Result from Array:', result)
```

### Fractal Dimension from Array or File

```python
from spacial_boxcounting.api import fractal_dimension_from_array, fractal_dimension_from_file

# From array
fd_array = fractal_dimension_from_array(arr)
print('Fractal Dimension (Array):', fd_array)

# From file
fd_file = fractal_dimension_from_file('path/to/your/image.jpg')
print('Fractal Dimension (File):', fd_file)
```

## Batch Processing
Run the CLI to process all files in a directory:

```bash
python3 -m spacial_boxcounting.batch path/to/your/input_folder
```

## GPU Acceleration
If Cupy is installed, GPU accelerated functions will execute:

```python
import numpy as np
from spacial_boxcounting.core import spacialBoxcount_gpu, Z_boxcount_gpu

arr = np.random.randint(0, 256, size=(64, 64)).astype(np.uint8)
result_gpu = spacialBoxcount_gpu(arr, iteration=0, MaxValue=256)
print('GPU spatial result:', result_gpu)
```

## Packaging & Distribution
This project is structured as a pip-installable package. Future releases may be distributed via PyPI. Contributions towards expanding its functionality are welcome.

## Testing
Run unit tests with:

```bash
pytest
```

## Academic Context
Originally derived from academic work in spatial analysis, this repository provides the tools for box counting and lacunarity computation as described in the accompanying Jupyter Notebook. For a full exposition, please review the notebook:
[Spacial boxcount algorithm CPU and GPU.ipynb](https://colab.research.google.com/github/ollimacp/spacial-boxcounting-cpu-gpu/blob/main/Spacial%20boxcount%20algorithm%20CPU%20and%20GPU.ipynb)

## License
See LICENSE.txt for details.
