import math
import numpy as np
from .io import load_file_as_ndarray
from .core import spacialBoxcount, Z_boxcount


def boxcount_from_file(filepath, mode='spatial', hilbert=False, **kwargs):
    """Compute box count from a file.

    Parameters:
        filepath (str): Path to the input file.
        mode (str): 'spatial' for 2D result, 'single' for overall count.
        hilbert (bool): If True, apply Hilbert curve transformation.
        **kwargs: Additional parameters for future extensions.

    Returns:
        np.ndarray or dict: Spatial box count array or dict with single box count and lacunarity.
    """
    arr = load_file_as_ndarray(filepath, mode='auto', hilbert=hilbert)
    maxvalue = 256  # assuming 8-bit data
    if mode == 'spatial':
        # For demonstration, use the first iteration (box size = 2)
        result = spacialBoxcount(arr, 0, maxvalue)
        return result
    elif mode == 'single':
        # Use an arbitrary box size, e.g., 8 (index 2 in list [2,4,8,...])
        counted, lacunarity = Z_boxcount(arr, 8, maxvalue)  
        return {'boxcount': counted, 'lacunarity': lacunarity}
    else:
        raise ValueError("Unsupported mode. Use 'spatial' or 'single'.")


def boxcount_from_array(arr, mode='spatial', hilbert=False, maxvalue=256):
    """Compute box count from a numpy array.

    Parameters:
        arr (np.ndarray): Input array
        mode (str): 'spatial' for spatial box count map, 'single' for overall count
        hilbert (bool): If True, apply Hilbert transform (if needed).
        maxvalue (int): Maximum value, defaults to 256 for 8-bit data

    Returns:
        np.ndarray or dict: Spatial box count result or dictionary with box count and lacunarity in single mode.
    """
    from .core import spacialBoxcount, Z_boxcount
    if mode == 'spatial':
        result = spacialBoxcount(arr, 0, maxvalue)
        return result
    elif mode == 'single':
        counted, lacunarity = Z_boxcount(arr, 8, maxvalue)
        return {'boxcount': counted, 'lacunarity': lacunarity}
    else:
        raise ValueError("Unsupported mode. Use 'spatial' or 'single'.")


def fractal_dimension(arr_or_path, **kwargs):
    """Compute fractal dimension from either an array or file path.
    
    Parameters:
        arr_or_path (np.ndarray or str): Input array or file path
        **kwargs: Additional parameters
        
    Returns:
        float: Fractal dimension estimate
    """
    if isinstance(arr_or_path, str):
        return fractal_dimension_from_file(arr_or_path, **kwargs)
    else:
        return fractal_dimension_from_array(arr_or_path, **kwargs)


def multi_scale_fractal_dimension_from_array(arr, scales=range(10), maxvalue=256, BoxSizes=None):
    """Compute fractal dimension from a numpy array using multi-scale box counting.
    
    Parameters:
        arr (np.ndarray): Input 2D array.
        scales (iterable): Indices of scales to use (default: 0-9).
        maxvalue (int): Maximum value, default is 256 (for 8-bit).
        BoxSizes (list): Box sizes, defaults to powers of 2

    Returns:
        float: Fractal dimension estimate.
    """
    # Set default box sizes if not provided
    if BoxSizes is None:
        BoxSizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    # Import required function
    from .core import Z_boxcount
    
    # Collect box counts for different scales
    box_counts = []
    box_sizes_used = []
    
    for iteration in scales:
        bs = BoxSizes[iteration]
        # Only process if box size is smaller than array dimensions
        if bs <= min(arr.shape):
            counted, _ = Z_boxcount(arr, bs, maxvalue)
            if counted > 0:  # Only include non-zero counts
                box_counts.append(counted)
                box_sizes_used.append(bs)
    
    # Compute fractal dimension from log-log regression
    if len(box_counts) >= 2:
        log_sizes = np.log(box_sizes_used)
        log_counts = np.log(box_counts)
        # Linear regression (slope = -D, so D = -slope)
        slope, _ = np.polyfit(log_sizes, log_counts, 1)
        return -slope
    else:
        return 0.0  # Not enough data points


def global_boxcount_from_array(arr, scales=range(10), maxvalue=256, BoxSizes = None):
    """Compute overall box counts for multiple scales from a numpy array.

    Parameters:
        arr (np.ndarray): Input 2D array.
        scales (iterable): Indices of scales to use (default: 0-9).
        maxvalue (int): Maximum value, default is 256 (for 8-bit).

    Returns:
        dict: Mapping from box size to overall box count.
    """
    if BoxSizes is None:
        BoxSizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    from .core import Z_boxcount
    result = {}
    for iteration in scales:
        bs = BoxSizes[iteration]
        counted, _ = Z_boxcount(arr, bs, maxvalue)
        result[bs] = counted
    return result



def fractal_dimension_from_array(arr, maxvalue=256, box_sizes=None):
    """
    Compute fractal dimension from a numpy array using multi-scale box counting.
    
    Parameters:
        arr (np.ndarray): Input 2D array
        maxvalue (int): Maximum pixel value (default 256)
        box_sizes (list): Box sizes to use (default powers of 2)
        
    Returns:
        float: Fractal dimension estimate
    """
    # Set default box sizes (powers of 2 up to array size)
    if box_sizes is None:
        max_size = min(arr.shape)
        box_sizes = [2**i for i in range(1, int(np.log2(max_size)) + 1)]
    
    counts = []
    for bs in box_sizes:
        count, _ = Z_boxcount(arr, bs, maxvalue)
        counts.append(count)
    
    # Filter zero counts and compute logs
    valid_idx = [i for i, c in enumerate(counts) if c > 0]
    log_sizes = np.log([box_sizes[i] for i in valid_idx])
    log_counts = np.log([counts[i] for i in valid_idx])
    
    # Linear regression (slope = -D)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return -slope

def fractal_dimension_from_file(filepath, maxvalue=256, box_sizes=None, hilbert=False):
    """
    Compute fractal dimension from file using multi-scale box counting.
    
    Parameters:
        filepath (str): Path to input file
        maxvalue (int): Maximum pixel value (default 256)
        box_sizes (list): Box sizes to use
        hilbert (bool): Apply Hilbert curve transformation
        
    Returns:
        float: Fractal dimension estimate
    """
    arr = load_file_as_ndarray(filepath, mode='auto', hilbert=hilbert)
    return fractal_dimension_from_array(arr, maxvalue, box_sizes)