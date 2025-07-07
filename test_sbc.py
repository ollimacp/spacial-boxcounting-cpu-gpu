import os
import numpy as np
import traceback

def debug_print(header, content):
    print(f"\n=== {header} ===")
    print(content)

def test_from_file(image_path):
    try:
        from spacial_boxcounting.api import boxcount_from_file, fractal_dimension_from_file

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        debug_print("Testing Spatial Mode from File", image_path)
        result_spatial = boxcount_from_file(image_path, mode='spatial')
        debug_print("Spatial Box Count Map", result_spatial)

        debug_print("Testing Single Mode from File", image_path)
        result_single = boxcount_from_file(image_path, mode='single')
        debug_print("Single Box Count & Lacunarity", result_single)

        debug_print("Testing Fractal Dimension from File", image_path)
        fd = fractal_dimension_from_file(image_path)
        debug_print("Fractal Dimension", fd)

    except Exception as e:
        print("\n[ERROR] Failed processing image file.")
        traceback.print_exc()

def test_from_array():
    try:
        from spacial_boxcounting.api import boxcount_from_array, fractal_dimension_from_array

        debug_print("Creating Random Test Array", "Shape: (256, 256)")
        arr = np.random.randint(0, 256, size=(256, 256)).astype(np.uint8)

        debug_print("Testing Spatial Mode from Array", "")
        result = boxcount_from_array(arr, mode='spatial')
        debug_print("Spatial Result from Array", result)

        debug_print("Testing Fractal Dimension from Array", "")
        fd = fractal_dimension_from_array(arr)
        debug_print("Fractal Dimension from Array", fd)

    except Exception as e:
        print("\n[ERROR] Failed processing numpy array.")
        traceback.print_exc()

def test_gpu():
    try:
        import cupy
        from spacial_boxcounting.core import spacialBoxcount_gpu

        debug_print("Creating Random Test Array for GPU", "Shape: (64, 64)")
        arr = np.random.randint(0, 256, size=(64, 64)).astype(np.uint8)

        debug_print("Testing GPU Boxcount", "")
        result_gpu = spacialBoxcount_gpu(arr, iteration=0, MaxValue=256)
        debug_print("GPU Spatial Result", result_gpu)

    except ImportError:
        print("[INFO] cupy not installed, skipping GPU test.")
    except Exception as e:
        print("\n[ERROR] Failed during GPU boxcount.")
        traceback.print_exc()

if __name__ == '__main__':
    print("=== spacial-boxcounting Test Runner ===")

    # Optional: change this path to an existing image to test from file
    test_image_path = "/home/raghat/projects/spacial-boxcounting-cpu-gpu/0Data/Images/12_3_3700x.bmp"  # <-- Replace with actual image path
    # /home/raghat/projects/spacial-boxcounting-cpu-gpu/0Data/Images/
    test_from_array()
    test_from_file(test_image_path)
    #test_gpu()
