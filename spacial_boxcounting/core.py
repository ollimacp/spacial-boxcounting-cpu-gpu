import numpy as np
from numba import jit


@jit(nopython=True)
def Z_boxcount(GlidingBox, boxsize, MaxValue):
    """Compute the box count and lacunarity for a given gliding box."""
    continualIndexes = GlidingBox / boxsize
    Boxindexes = np.floor(continualIndexes)
    unique_Boxes = np.unique(Boxindexes)
    counted_Boxes = len(unique_Boxes)

    # Build histogram: count pixels per box index, starting with first box
    first_box = unique_Boxes[0]
    first_count = np.sum(Boxindexes == first_box)
    # Use float for type consistency with EmptyBoxes (np.zeros) below
    SumPixInBox = np.array([float(first_count)])
    for unique_BoxIndex in unique_Boxes[1:]:
        ElementsCountedTRUTHTABLE = Boxindexes == unique_BoxIndex
        ElementsCounted = np.sum(ElementsCountedTRUTHTABLE)
        SumPixInBox = np.append(SumPixInBox, ElementsCounted)
    Max_Num_Boxes = int(MaxValue / boxsize)
    Num_empty_Boxes = Max_Num_Boxes - counted_Boxes
    if Num_empty_Boxes >= 1:
        EmptyBoxes = np.zeros(Num_empty_Boxes)
        SumPixInBox = np.append(SumPixInBox, EmptyBoxes)
    mean = np.mean(SumPixInBox)
    standardDeviation = np.std(SumPixInBox)
    Lacunarity = np.power(standardDeviation / mean, 2)
    return counted_Boxes, Lacunarity

@jit(nopython=True)
def spacialBoxcount(npOutputFile, iteration, MaxValue):
    """Compute the spatial box count ratio and lacunarity for an image array."""
    Boxsize = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    boxsize = Boxsize[iteration]
    BoxBoundriesX = np.array([0, boxsize])
    BoxBoundriesY = np.array([0, boxsize])
    YRange, XRange = npOutputFile.shape
    maxIndexY = int(YRange / boxsize) + 1
    maxIndexX = int(XRange / boxsize) + 1
    BoxCountR_map = np.zeros((maxIndexY, maxIndexX))
    spa_Lac_map = np.zeros((maxIndexY, maxIndexX))
    while BoxBoundriesY[1] <= YRange:
        while BoxBoundriesX[1] <= XRange:
            indexY = int(BoxBoundriesY[0] / boxsize)
            indexX = int(BoxBoundriesX[0] / boxsize)
            GlidingBox = npOutputFile[BoxBoundriesY[0]:BoxBoundriesY[1], BoxBoundriesX[0]:BoxBoundriesX[1]]
            counted_Boxes, Lacunarity = Z_boxcount(GlidingBox, boxsize, MaxValue)
            Max_Num_Boxes = int(MaxValue / boxsize)
            counted_Box_Ratio = counted_Boxes / Max_Num_Boxes
            BoxCountR_map[indexY, indexX] = counted_Box_Ratio
            spa_Lac_map[indexY, indexX] = Lacunarity
            BoxBoundriesX[0] += boxsize
            BoxBoundriesX[1] += boxsize
        BoxBoundriesX[0] = 0
        BoxBoundriesX[1] = boxsize
        BoxBoundriesY[0] += boxsize
        BoxBoundriesY[1] += boxsize
    return [BoxCountR_map, spa_Lac_map]

# GPU Acceleration — auto-detect backend
try:
    import cupy as cp

    CUPY_AVAILABLE = True

    # Detect whether CuPy was built for CUDA or ROCm
    try:
        _cupy_backend = cp.cuda.runtime.runtimeGetVersion()
        GPU_BACKEND = "cuda"
    except Exception:
        try:
            _cupy_backend = cp.cuda.runtime.getDeviceCount()
            GPU_BACKEND = "rocm"
        except Exception:
            GPU_BACKEND = "unknown"
except ImportError:
    CUPY_AVAILABLE = False
    GPU_BACKEND = None


def Z_boxcount_gpu(GlidingBox, boxsize, MaxValue):
    """Compute box count and lacunarity on GPU (fully vectorized, no Python loops).

    Parameters
    ----------
    GlidingBox : np.ndarray or cp.ndarray
        2D sliding window (will be transferred to GPU if needed).
    boxsize : int
        Size of the quantization box.
    MaxValue : int
        Maximum intensity value (typically 256 for 8-bit).

    Returns
    -------
    tuple[int, float]
        (counted_boxes, lacunarity)
    """
    GlidingBox_gpu = cp.asarray(GlidingBox)
    continualIndexes = GlidingBox_gpu / boxsize
    Boxindexes = cp.floor(continualIndexes).astype(cp.int32).ravel()

    # Count elements per box index via bincount (fully GPU-vectorized)
    counts = cp.bincount(Boxindexes)
    # Remove the zero bin (index 0 always exists in bincount)
    BoxCounts = counts[counts > 0]
    counted_Boxes: int = int(BoxCounts.size)

    Max_Num_Boxes = int(MaxValue / boxsize)
    Num_empty_Boxes = Max_Num_Boxes - counted_Boxes
    if Num_empty_Boxes >= 1:
        BoxCounts = cp.append(BoxCounts, cp.zeros(Num_empty_Boxes, dtype=cp.float64))

    mean = cp.mean(BoxCounts)
    standardDeviation = cp.std(BoxCounts)
    Lacunarity = cp.power(standardDeviation / mean, 2)
    return counted_Boxes, float(Lacunarity.get())


def spacialBoxcount_gpu(npOutputFile, iteration, MaxValue):
    """Compute spatial box count ratio and lacunarity on GPU.

    All sliding windows are processed in a single batched kernel launch
    via 4D-tensor reshaping and vectorised CuPy operations — no Python
    loops inside the window iteration.

    Parameters
    ----------
    npOutputFile : np.ndarray
        2D input array.
    iteration : int
        Index into ``[2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]``.
    MaxValue : int
        Maximum intensity value (typically 256 for 8-bit data).

    Returns
    -------
    list of np.ndarray
        ``[boxcount_ratio_map, spatial_lacunarity_map]`` — each a 2D array
        of shape ``(ny+1, nx+1)`` where the last row/column is zero-padded.
    """
    if not CUPY_AVAILABLE:
        raise ImportError("cupy is not installed")
    arr_gpu = cp.asarray(npOutputFile)
    Boxsize = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    boxsize = Boxsize[iteration]
    H, W = arr_gpu.shape
    Max_Num_Boxes = int(MaxValue / boxsize)
    n_bins = Max_Num_Boxes

    # Number of complete windows along each axis
    ny = H // boxsize
    nx = W // boxsize

    if ny == 0 or nx == 0:
        raise ValueError(
            f"Image too small for boxsize={boxsize}: "
            f"image is {H}×{W}, need at least {boxsize}×{boxsize}"
        )

    # Crop to exact multiple of boxsize (matches CPU while-loop behaviour)
    arr_cropped = arr_gpu[: ny * boxsize, : nx * boxsize]

    # Reshape into (ny * nx, boxsize * boxsize) — one row per window
    windows_4d = arr_cropped.reshape(ny, boxsize, nx, boxsize)
    windows_4d = windows_4d.transpose(0, 2, 1, 3)  # (ny, nx, boxsize, boxsize)
    N_windows = ny * nx
    windows = windows_4d.reshape(N_windows, boxsize * boxsize)

    # Step 1 — integer box indices via floor division
    indices = cp.floor(windows / boxsize).astype(cp.int32)  # (N_windows, bs²)

    # Step 2 — count unique box indices per window
    indices_sorted = cp.sort(indices, axis=1)
    diffs = cp.diff(indices_sorted, axis=1)
    counted_Boxes = cp.count_nonzero(diffs, axis=1) + 1  # (N_windows,)

    # Step 3 — per-window histogram via offset bincount
    offsets = cp.arange(N_windows, dtype=cp.int64) * n_bins
    ws = boxsize * boxsize
    flat_indices = (indices.ravel() + cp.repeat(offsets, ws)).astype(cp.int64)
    hist = cp.bincount(flat_indices, minlength=N_windows * n_bins).reshape(
        N_windows, n_bins
    )

    # Step 4 — box-count-ratio map
    BoxCountR_map = (counted_Boxes.astype(cp.float64) / Max_Num_Boxes).reshape(ny, nx)

    # Step 5 — lacunarity per window
    hist_f64 = hist.astype(cp.float64)
    means = cp.mean(hist_f64, axis=1)
    stds = cp.std(hist_f64, axis=1)
    Lacunarity = cp.power(stds / means, 2)
    spa_Lac_map = Lacunarity.reshape(ny, nx)

    # Pad with an extra row/column of zeros (matching legacy output shape)
    BoxCountR_map_full = cp.zeros((ny + 1, nx + 1), dtype=cp.float64)
    spa_Lac_map_full = cp.zeros((ny + 1, nx + 1), dtype=cp.float64)
    BoxCountR_map_full[:ny, :nx] = BoxCountR_map
    spa_Lac_map_full[:ny, :nx] = spa_Lac_map

    return [cp.asnumpy(BoxCountR_map_full), cp.asnumpy(spa_Lac_map_full)]
