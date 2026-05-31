import numpy as np
from numba import jit


@jit(nopython=True)
def Z_boxcount(GlidingBox, boxsize, MaxValue):
    """Compute the box count and lacunarity for a given gliding box."""
    continualIndexes = GlidingBox / boxsize
    Boxindexes = np.floor(continualIndexes)
    unique_Boxes = np.unique(Boxindexes)
    counted_Boxes = len(unique_Boxes)
    InitalEntry = [0.0]
    SumPixInBox = np.array(InitalEntry)
    for unique_BoxIndex in unique_Boxes:
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

# GPU Acceleration Functions
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


def Z_boxcount_gpu(GlidingBox, boxsize, MaxValue):
    """Compute the box count and lacunarity using GPU via cupy."""
    GlidingBox_gpu = cp.asarray(GlidingBox)
    continualIndexes = GlidingBox_gpu / boxsize
    Boxindexes = cp.floor(continualIndexes)
    unique_Boxes = cp.unique(Boxindexes)
    counted_Boxes: int = int(unique_Boxes.size)
    SumPixInBox = cp.array([0.0])
    for ub in unique_Boxes:
        mask = (Boxindexes == ub)
        ElementsCounted = cp.sum(mask)
        SumPixInBox = cp.append(SumPixInBox, ElementsCounted)
    Max_Num_Boxes = int(MaxValue / boxsize)
    Num_empty_Boxes = Max_Num_Boxes - int(counted_Boxes)
    if Num_empty_Boxes >= 1:
        EmptyBoxes = cp.zeros(Num_empty_Boxes)
        SumPixInBox = cp.append(SumPixInBox, EmptyBoxes)
    mean = cp.mean(SumPixInBox)
    standardDeviation = cp.std(SumPixInBox)
    Lacunarity = cp.power(standardDeviation / mean, 2)
    return counted_Boxes, float(Lacunarity.get())


def spacialBoxcount_gpu(npOutputFile, iteration, MaxValue):
    """Compute spatial box count ratio and lacunarity on GPU via cupy."""
    if not CUPY_AVAILABLE:
        raise ImportError("cupy is not installed")
    arr_gpu = cp.asarray(npOutputFile)
    Boxsize = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    boxsize = Boxsize[iteration]
    YRange, XRange = arr_gpu.shape
    maxIndexY = int(YRange / boxsize) + 1
    maxIndexX = int(XRange / boxsize) + 1
    BoxCountR_map = cp.zeros((maxIndexY, maxIndexX))
    spa_Lac_map = cp.zeros((maxIndexY, maxIndexX))
    y_idx = 0
    for i in range(0, int(YRange), boxsize):
        x_idx = 0
        for j in range(0, int(XRange), boxsize):
            GlidingBox = arr_gpu[i:i+boxsize, j:j+boxsize]
            # Use CPU function on the small block for simplicity
            counted_Boxes, Lacunarity = Z_boxcount_gpu(cp.asnumpy(GlidingBox), boxsize, MaxValue)
            Max_Num_Boxes = int(MaxValue / boxsize)
            counted_Box_Ratio = counted_Boxes / Max_Num_Boxes
            BoxCountR_map[y_idx, x_idx] = counted_Box_Ratio
            spa_Lac_map[y_idx, x_idx] = Lacunarity
            x_idx += 1
        y_idx += 1
    return [cp.asnumpy(BoxCountR_map), cp.asnumpy(spa_Lac_map)]
