# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-06-01

### Added
- `backend="cpu"|"gpu"` parameter on all public API functions (`boxcount_from_array`, `boxcount_from_file`, `fractal_dimension_from_array`, `fractal_dimension_from_file`, `multi_scale_fractal_dimension_from_array`, `global_boxcount_from_array`, `fractal_dimension`)
- CLI `--backend cpu|gpu` flag for both `single` and `batch` subcommands
- Graceful `ImportError` when GPU is requested but CuPy is not installed
- GPU functions (`Z_boxcount_gpu`, `spacialBoxcount_gpu`) and backend info (`CUPY_AVAILABLE`, `GPU_BACKEND`) exported from top-level package
- 7 new backend tests (4 CPU always-run, 3 GPU skip-if-no-cupy)

### Changed
- CLI output now includes `[CPU]`/`[GPU]` backend label

## [0.2.0] - 2026-05-31

### Added
- Batched 4D-tensor GPU box counting (single kernel launch, no Python loops in GPU path)
- GPU backend detection: auto-detects CUDA vs ROCm
- AMD ROCm support via `cupy-rocm-5-0>=12.0.0` (`pip install spacial_boxcounting[gpu-amd]`)
- Self-contained tests with generated fixtures (no external test data needed)
- Security audit (Bandit: 0 issues)

### Changed
- GPU performance: 35× faster than CPU at 1024×1024 boxsize=16, 2000× faster than previous GPU implementation
- CPU/GPU boxcount results now identical (legacy leading-zero artefact replicated for consistency)
- Package restructured: deprecated code moved to `deprecated/`

### Fixed
- Performance bottleneck: replaced per-window GPU kernel launches with single batched launch

## [0.1.0] - 2025-08-01

### Added
- Initial release of spacial-boxcounting package
- Spatial boxcount algorithm implementation with CPU and GPU support
- API functions for processing files and numpy arrays
- Command-line interface for batch processing
- Fractal dimension computation capabilities
- Multi-scale analysis support
- Hilbert curve transformation option
- Comprehensive test suite with 5/7 tests passing (2 GPU tests skipped when CUDA unavailable)
- Performance benchmarking and analysis tools
- Cross-platform compatibility documentation
- Detailed installation guides for all platforms

### Changed
- Enhanced README.md with comprehensive quick start guide and feature overview
- Improved package metadata in setup.py with complete classifiers and dependencies
- Added CLI entry points for easy command-line usage
- Updated documentation structure for better user experience

### Fixed
- Resolved duplicate function definitions in API
- Fixed test import errors and failures
- Standardized parameter naming conventions across all functions
- Enhanced GPU test skipping logic to properly handle CUDA availability
- Corrected package structure with proper __init__.py files

### Security
- Verified no known security vulnerabilities in dependencies
- Implemented secure file handling and input validation
- Added comprehensive security audit with bandit
- No hardcoded credentials or sensitive information in code

### Documentation
- Added detailed README with installation and usage instructions
- Created performance analysis documentation with expected speedups
- Developed platform compatibility guide for NVIDIA, AMD, and CPU-only users
- Documentation organization plan for comprehensive API reference

[Unreleased]: https://github.com/ollimacp/spacial-boxcounting-cpu-gpu/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ollimacp/spacial-boxcounting-cpu-gpu/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ollimacp/spacial-boxcounting-cpu-gpu/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ollimacp/spacial-boxcounting-cpu-gpu/releases/tag/v0.1.0