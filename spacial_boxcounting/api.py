from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .core import (
    CUPY_AVAILABLE,
    GPU_BACKEND,
    Z_boxcount,
    Z_boxcount_gpu,
    spacialBoxcount,
    spacialBoxcount_gpu,
)
from .io import load_file_as_ndarray


def _resolve_backend(backend: str) -> str:
    """Normalise and validate the backend string.

    Returns ``"gpu"`` or ``"cpu"``.  If ``"gpu"`` is requested but CuPy is
    not installed an :class:`ImportError` is raised.
    """
    backend = backend.lower().strip()
    if backend not in ("cpu", "gpu"):
        raise ValueError(
            f"Unsupported backend '{backend}'. Use 'cpu' or 'gpu'."
        )
    if backend == "gpu" and not CUPY_AVAILABLE:
        raise ImportError(
            "GPU backend requested but CuPy is not installed. "
            "Install with: pip install spacial_boxcounting[gpu]"
        )
    return backend


def boxcount_from_file(
    filepath: str,
    mode: str = "spatial",
    hilbert: bool = False,
    backend: str = "cpu",
    **kwargs: Any,
) -> Union[List[np.ndarray], Dict[str, float]]:
    """Compute box count from a file.

    Parameters
    ----------
    filepath : str
        Path to the input file (image, .npy, or binary).
    mode : str
        ``"spatial"`` for 2D spatial box count map, ``"single"`` for
        overall box count and lacunarity. Default is ``"spatial"``.
    hilbert : bool
        If True, apply Hilbert curve transformation for binary files.
        Default is False.
    backend : str
        ``"cpu"`` (default) or ``"gpu"``.  When ``"gpu"`` is selected
        the CuPy-accelerated implementation is used.  Requires CuPy to
        be installed (``pip install spacial_boxcounting[gpu]``).
    **kwargs : Any
        Additional parameters for future extensions.

    Returns
    -------
    list of np.ndarray or dict
        In spatial mode: list of two 2D arrays
        (box-count-ratio map, spatial lacunarity map).
        In single mode: dict with keys ``"boxcount"`` and ``"lacunarity"``.
    """
    arr = load_file_as_ndarray(filepath, mode="auto", hilbert=hilbert)
    return boxcount_from_array(arr, mode=mode, backend=backend)


def boxcount_from_array(
    arr: np.ndarray,
    mode: str = "spatial",
    hilbert: bool = False,
    maxvalue: int = 256,
    backend: str = "cpu",
) -> Union[List[np.ndarray], Dict[str, float]]:
    """Compute box count from a numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array.
    mode : str
        ``"spatial"`` for spatial box count map, ``"single"`` for
        overall count. Default is ``"spatial"``.
    hilbert : bool
        If True, apply Hilbert transform. Default is False.
    maxvalue : int
        Maximum value for box quantization. Default is 256 (8-bit).
    backend : str
        ``"cpu"`` (default) or ``"gpu"``.  When ``"gpu"`` is selected
        the CuPy-accelerated implementation is used.  Requires CuPy to
        be installed (``pip install spacial_boxcounting[gpu]``).

    Returns
    -------
    list of np.ndarray or dict
        Spatial result or dictionary with ``"boxcount"`` and
        ``"lacunarity"``.
    """
    backend = _resolve_backend(backend)

    if mode == "spatial":
        if backend == "gpu":
            return spacialBoxcount_gpu(arr, 0, maxvalue)
        return spacialBoxcount(arr, 0, maxvalue)
    elif mode == "single":
        if backend == "gpu":
            counted, lacunarity = Z_boxcount_gpu(arr, 8, maxvalue)
        else:
            counted, lacunarity = Z_boxcount(arr, 8, maxvalue)
        return {"boxcount": counted, "lacunarity": lacunarity}
    else:
        raise ValueError("Unsupported mode. Use 'spatial' or 'single'.")


def fractal_dimension(
    arr_or_path: Union[np.ndarray, str],
    backend: str = "cpu",
    **kwargs: Any,
) -> float:
    """Compute fractal dimension from an array or file path.

    Parameters
    ----------
    arr_or_path : np.ndarray or str
        Input 2D array or path to a file.
    backend : str
        ``"cpu"`` (default) or ``"gpu"``.
    **kwargs : Any
        Additional parameters forwarded to the specific implementation.

    Returns
    -------
    float
        Fractal dimension estimate.
    """
    if isinstance(arr_or_path, str):
        return fractal_dimension_from_file(arr_or_path, backend=backend, **kwargs)
    return fractal_dimension_from_array(arr_or_path, backend=backend, **kwargs)


def fractal_dimension_from_array(
    arr: np.ndarray,
    maxvalue: int = 256,
    box_sizes: Optional[List[int]] = None,
    backend: str = "cpu",
) -> float:
    """Compute fractal dimension from a numpy array via multi-scale box counting.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array.
    maxvalue : int
        Maximum pixel value. Default is 256.
    box_sizes : list of int, optional
        Box sizes to use. Defaults to powers of 2 up to the array size.
    backend : str
        ``"cpu"`` (default) or ``"gpu"``.

    Returns
    -------
    float
        Fractal dimension estimate. Returns 0.0 if there is insufficient data.
    """
    backend = _resolve_backend(backend)

    if box_sizes is None:
        max_size = min(arr.shape)
        box_sizes = [2**i for i in range(1, int(np.log2(max_size)) + 1)]

    _zbox = Z_boxcount_gpu if backend == "gpu" else Z_boxcount

    counts = []
    for bs in box_sizes:
        count, _ = _zbox(arr, bs, maxvalue)
        counts.append(count)

    valid_idx = [i for i, c in enumerate(counts) if c > 0]
    if len(valid_idx) < 2:
        return 0.0

    log_sizes = np.log([box_sizes[i] for i in valid_idx])
    log_counts = np.log([counts[i] for i in valid_idx])
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return -slope


def fractal_dimension_from_file(
    filepath: str,
    maxvalue: int = 256,
    box_sizes: Optional[List[int]] = None,
    hilbert: bool = False,
    backend: str = "cpu",
) -> float:
    """Compute fractal dimension from a file via multi-scale box counting.

    Parameters
    ----------
    filepath : str
        Path to the input file.
    maxvalue : int
        Maximum pixel value. Default is 256.
    box_sizes : list of int, optional
        Box sizes to use.
    hilbert : bool
        Apply Hilbert curve transformation for binary files. Default is False.
    backend : str
        ``"cpu"`` (default) or ``"gpu"``.

    Returns
    -------
    float
        Fractal dimension estimate.
    """
    arr = load_file_as_ndarray(filepath, mode="auto", hilbert=hilbert)
    return fractal_dimension_from_array(arr, maxvalue, box_sizes, backend=backend)


def multi_scale_fractal_dimension_from_array(
    arr: np.ndarray,
    scales: range = range(10),
    maxvalue: int = 256,
    BoxSizes: Optional[List[int]] = None,
    backend: str = "cpu",
) -> float:
    """Compute fractal dimension using multi-scale box counting.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array.
    scales : iterable
        Indices of scales to use. Default is ``range(10)``.
    maxvalue : int
        Maximum value. Default is 256 (8-bit).
    BoxSizes : list of int, optional
        Box sizes. Defaults to powers of 2.
    backend : str
        ``"cpu"`` (default) or ``"gpu"``.

    Returns
    -------
    float
        Fractal dimension estimate. Returns 0.0 if there is insufficient data.
    """
    backend = _resolve_backend(backend)

    if BoxSizes is None:
        BoxSizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    _zbox = Z_boxcount_gpu if backend == "gpu" else Z_boxcount

    box_counts = []
    box_sizes_used = []

    for iteration in scales:
        bs = BoxSizes[iteration]
        if bs <= min(arr.shape):
            counted, _ = _zbox(arr, bs, maxvalue)
            if counted > 0:
                box_counts.append(counted)
                box_sizes_used.append(bs)

    if len(box_counts) < 2:
        return 0.0

    log_sizes = np.log(box_sizes_used)
    log_counts = np.log(box_counts)
    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return -slope


def global_boxcount_from_array(
    arr: np.ndarray,
    scales: range = range(10),
    maxvalue: int = 256,
    BoxSizes: Optional[List[int]] = None,
    backend: str = "cpu",
) -> Dict[int, int]:
    """Compute overall box counts for multiple scales from a numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D array.
    scales : iterable
        Indices of scales to use. Default is ``range(10)``.
    maxvalue : int
        Maximum value. Default is 256 (8-bit).
    BoxSizes : list of int, optional
        Box sizes. Defaults to powers of 2.
    backend : str
        ``"cpu"`` (default) or ``"gpu"``.

    Returns
    -------
    dict
        Mapping from box size to overall box count.
    """
    backend = _resolve_backend(backend)

    if BoxSizes is None:
        BoxSizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    _zbox = Z_boxcount_gpu if backend == "gpu" else Z_boxcount

    result: Dict[int, int] = {}
    for iteration in scales:
        bs = BoxSizes[iteration]
        counted, _ = _zbox(arr, bs, maxvalue)
        result[bs] = counted
    return result