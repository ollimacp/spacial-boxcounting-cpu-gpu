import os
import numpy as np
from PIL import Image

# Installation erforderlich:
# pip install hilbertcurve
from hilbertcurve.hilbertcurve import HilbertCurve

def load_file_as_ndarray(filepath, mode='auto', hilbert=False):
    # wie gehabt …
    if mode == 'auto':
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            mode = 'image'
        elif ext in ['.npy']:
            mode = 'npy'
        else:
            mode = 'binary'

    if mode == 'image':
        img = Image.open(filepath).convert('L')
        arr = np.array(img)
    elif mode == 'npy':
        arr = np.load(filepath)
    elif mode == 'binary':
        with open(filepath, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        if hilbert:
            arr = _map_bytes_to_hilbert(data)
        else:
            # fallback: Quadrat mit Zeilenweise Reshape
            side = int(np.floor(np.sqrt(data.size)))
            arr = data[:side*side].reshape(side, side)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return arr

def _map_bytes_to_hilbert(data: np.ndarray) -> np.ndarray:
    """
    Mapping eines 1D Byte-Arrays auf ein 2D-Array via Hilbert-Kurve.
    Das Quadrat muss 2^p × 2^p groß sein.
    """
    length = data.size
    # Bestimme minimal p: 2^(2p) >= length
    p = 0
    while (2 ** (2 * p)) < length:
        p += 1
    side = 2 ** p
    total = side * side
    # Padd das Array auf volle Größe
    if total > length:
        data = np.pad(data, (0, total - length), mode='constant', constant_values=0)

    # Hilbert-Kurve initialisieren: n=2 Dim, p Iterationen
    hc = HilbertCurve(p, 2)
    # generiere alle distanzbasierten Koordinaten
    distances = np.arange(total)
    coords = np.array(hc.points_from_distances(distances))  # shape (total, 2)

    # Mappe Byte-Werte in 2D Raster
    arr2d = np.zeros((side, side), dtype=data.dtype)
    arr2d[coords[:,0], coords[:,1]] = data
    return arr2d
