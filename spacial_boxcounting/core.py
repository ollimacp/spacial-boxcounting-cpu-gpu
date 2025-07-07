import numpy as np
from numba import jit
import time
import linecache
import sys
import matplotlib.pyplot as plt
from PIL import Image


def PrintException():
    """Prints details of the current exception."""
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))


def show_np_array_as_image(np2ddArray, title, colormap):
    """Display a numpy 2D array as an image."""
    plt.figure()
    plt.imshow(np2ddArray, interpolation='none', cmap=colormap)
    plt.title(title)
    plt.show(block=False)

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
    counted_Boxes = unique_Boxes.size
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
    return int(counted_Boxes.get()), float(Lacunarity.get())


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



def MultithreadBoxcount(npOutputFile):
    """Compute spatial box count in parallel over multiple scales."""
    BoxsizeDict = {"2": 0, "4": 1, "8": 2, "16": 3, "32": 4, "64": 5, "128": 6, "256": 7, "512": 8, "1024": 9}
    Height, width = npOutputFile.shape
    Height, width = int(Height), int(width)
    BaseITERMinVal = min(16, Height, width)
    BaseIteration = BoxsizeDict[str(int(BaseITERMinVal))]
    maxiteration = BaseIteration + 1

    from threading import Thread
    class ThreadWithReturnValue(Thread):
        def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
            Thread.__init__(self, group, target, name, args, kwargs)
            self._return = None
        def run(self):
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)
        def join(self, *args):
            Thread.join(self, *args)
            return self._return

    def BoxcountBoxsizeWorker(npOutputFile, iteration):
        maxvalue = 256  # using 8-bit grayscale max value
        return spacialBoxcount(npOutputFile, iteration, maxvalue)

    threads = [None] * maxiteration
    start = time.time()
    for i in range(len(threads)):
        threads[i] = ThreadWithReturnValue(target=BoxcountBoxsizeWorker, args=(npOutputFile, i))
        threads[i].start()
    BoxCountR_SpacialLac_map_Dict = {"iteration": np.array(["BoxcountRatio", "spacialLacunarity"]) }
    for i in range(len(threads)):
        BoxCountR_SpacialLac_map = np.array(threads[i].join())
        BoxCountR_SpacialLac_map_Dict[i] = BoxCountR_SpacialLac_map
    end = time.time()
    print(round(end - start, 3), "seconds for spacial boxcounting with", i+1, "iterations/scalings")
    return BoxCountR_SpacialLac_map_Dict




def boxcount_to_be_proven(Z, k):
    """Count number of non-empty/non-full boxes of size k x k in image Z."""
    S = np.add.reduceat(
        np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
        np.arange(0, Z.shape[1], k), axis=1)
    # count non-empty and non-full boxes
    return len(np.where((S > 0) & (S < k*k))[0])


def fractal_dimension_to_be_proven(Z):
    """Estimate fractal dimension of a 2D array using box-counting method."""
    # Only for 2D array
    assert(len(Z.shape) == 2)
    # Transform Z into binary image
    Z = (Z > 0).astype(np.uint8)

    # Minimal dimension of image
    p = min(Z.shape)
    # Greatest power of 2 less than or equal to p
    n = 2**int(np.floor(np.log2(p)))
    Z = Z[:n, :n]

    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = []
    for size in sizes:
        counts.append(boxcount_to_be_proven(Z, size))

    # Fit to the line: log(count) = -D log(size) + constant
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
