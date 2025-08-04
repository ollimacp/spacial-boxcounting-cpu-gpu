"""
Benchmark script to compare CPU vs GPU performance for spatial boxcounting.
"""
import time
import numpy as np
import os
from spacial_boxcounting.api import boxcount_from_array, fractal_dimension_from_array
from spacial_boxcounting.core import spacialBoxcount, Z_boxcount

# Try to import GPU functions
try:
    from spacial_boxcounting.core import spacialBoxcount_gpu, Z_boxcount_gpu
    GPU_AVAILABLE = True
    print("GPU support detected")
except ImportError:
    GPU_AVAILABLE = False
    print("GPU support not available")

def benchmark_single_image_cpu(arr):
    """Benchmark spatial boxcounting on CPU for a single image."""
    start_time = time.time()
    result = boxcount_from_array(arr, mode='spatial')
    end_time = time.time()
    return end_time - start_time, result

def benchmark_single_image_gpu(arr):
    """Benchmark spatial boxcounting on GPU for a single image."""
    if not GPU_AVAILABLE:
        return None, None
    
    start_time = time.time()
    try:
        # Using iteration=0 for smallest box size (2x2)
        result = spacialBoxcount_gpu(arr, iteration=0, MaxValue=256)
        end_time = time.time()
        return end_time - start_time, result
    except Exception as e:
        print(f"GPU benchmark failed: {e}")
        return None, None

def benchmark_fractal_dimension_cpu(arr):
    """Benchmark fractal dimension computation on CPU."""
    start_time = time.time()
    fd = fractal_dimension_from_array(arr)
    end_time = time.time()
    return end_time - start_time, fd

def benchmark_fractal_dimension_gpu(arr):
    """Benchmark fractal dimension computation on GPU."""
    # For this version, fractal dimension runs on CPU regardless
    return benchmark_fractal_dimension_cpu(arr)

def run_benchmarks():
    """Run comprehensive benchmarks across different image sizes."""
    print("Running spatial boxcounting benchmarks...")
    
    # Test with different image sizes
    sizes = [64, 128, 256, 512, 1024]
    
    results = {
        'size': [],
        'cpu_time': [],
        'gpu_time': [],
        'speedup': [],
        'fractal_cpu_time': [],
        'fractal_gpu_time': [],
        'fractal_speedup': []
    }
    
    for size in sizes:
        print(f"\nTesting image size: {size}x{size}")
        # Create test image
        arr = np.random.randint(0, 256, size=(size, size)).astype(np.uint8)
        
        # Benchmark spatial boxcounting CPU
        cpu_time, cpu_result = benchmark_single_image_cpu(arr)
        print(f"  CPU spatial boxcount time: {cpu_time:.4f}s")
        
        # Benchmark spatial boxcounting GPU
        gpu_time, gpu_result = benchmark_single_image_gpu(arr)
        if gpu_time:
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"  GPU spatial boxcount time: {gpu_time:.4f}s (speedup: {speedup:.2f}x)")
        else:
            speedup = 0
            print("  GPU spatial boxcount: Skipped (no GPU support)")
        
        # Benchmark fractal dimension CPU
        fractal_cpu_time, fractal_cpu_result = benchmark_fractal_dimension_cpu(arr)
        print(f"  CPU fractal dimension time: {fractal_cpu_time:.4f}s")
        
        # Benchmark fractal dimension GPU (same as CPU for now)
        fractal_gpu_time, fractal_gpu_result = benchmark_fractal_dimension_gpu(arr)
        fractal_speedup = fractal_cpu_time / fractal_gpu_time if fractal_gpu_time > 0 else 0
        print(f"  GPU fractal dimension time: {fractal_gpu_time:.4f}s (speedup: {fractal_speedup:.2f}x)")
        
        # Store results
        results['size'].append(size)
        results['cpu_time'].append(cpu_time)
        results['gpu_time'].append(gpu_time if gpu_time else 0)
        results['speedup'].append(speedup)
        results['fractal_cpu_time'].append(fractal_cpu_time)
        results['fractal_gpu_time'].append(fractal_gpu_time)
        results['fractal_speedup'].append(fractal_speedup)
    
    return results

def benchmark_batch_processing():
    """Benchmark batch processing performance."""
    print("\n\nRunning batch processing benchmarks...")
    
    # Test with different batch sizes
    batch_sizes = [1, 5, 10, 20]
    image_size = 256
    
    batch_results = {
        'batch_size': [],
        'cpu_times': [],
        'gpu_times': []
    }
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create batch of images
        images = [np.random.randint(0, 256, size=(image_size, image_size)).astype(np.uint8) 
                  for _ in range(batch_size)]
        
        # Benchmark CPU batch processing
        start_time = time.time()
        cpu_results = []
        for img in images:
            result = boxcount_from_array(img, mode='single')
            cpu_results.append(result)
        cpu_time = time.time() - start_time
        print(f"  CPU batch processing time: {cpu_time:.4f}s")
        
        # Benchmark GPU batch processing (sequential for now)
        if GPU_AVAILABLE:
            start_time = time.time()
            gpu_results = []
            for img in images:
                try:
                    # Process each image individually on GPU
                    result = spacialBoxcount_gpu(img, iteration=0, MaxValue=256)
                    # Convert to similar format as CPU result for comparison
                    gpu_results.append({'boxcount': result[0].mean(), 'lacunarity': result[1].mean()})
                except Exception as e:
                    print(f"GPU processing failed for image: {e}")
                    gpu_results.append(None)
            gpu_time = time.time() - start_time
            print(f"  GPU batch processing time: {gpu_time:.4f}s")
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"  Batch processing speedup: {speedup:.2f}x")
        else:
            gpu_time = 0
            print("  GPU batch processing: Skipped (no GPU support)")
        
        # Store results
        batch_results['batch_size'].append(batch_size)
        batch_results['cpu_times'].append(cpu_time)
        batch_results['gpu_times'].append(gpu_time)
    
    return batch_results

if __name__ == "__main__":
    # Run benchmarks
    results = run_benchmarks()
    batch_results = benchmark_batch_processing()
    
    # Print summary
    print("\n\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print("\nSingle Image Spatial Boxcounting:")
    print("Size\tCPU Time\tGPU Time\tSpeedup")
    print("-" * 50)
    for i in range(len(results['size'])):
        size = results['size'][i]
        cpu_time = results['cpu_time'][i]
        gpu_time = results['gpu_time'][i]
        speedup = results['speedup'][i]
        if gpu_time > 0:
            print(f"{size}\t{cpu_time:.4f}s\t{gpu_time:.4f}s\t{speedup:.2f}x")
        else:
            print(f"{size}\t{cpu_time:.4f}s\tN/A\t\tN/A")
    
    print("\nBatch Processing:")
    print("Batch Size\tCPU Time\tGPU Time\tSpeedup")
    print("-" * 50)
    for i in range(len(batch_results['batch_size'])):
        batch_size = batch_results['batch_size'][i]
        cpu_time = batch_results['cpu_times'][i]
        gpu_time = batch_results['gpu_times'][i]
        if gpu_time > 0 and cpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"{batch_size}\t\t{cpu_time:.4f}s\t{gpu_time:.4f}s\t{speedup:.2f}x")
        else:
            print(f"{batch_size}\t\t{cpu_time:.4f}s\tN/A\t\tN/A")
