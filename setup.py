from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='spacial_boxcounting',
    version='0.1.0',
    description='Convenient package for spatial box counting and fractal analysis across data types with CPU and GPU support',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/spacial-boxcounting-cpu-gpu',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'numba>=0.53.0',
        'Pillow>=8.0.0',
        'matplotlib>=3.3.0',
        'hilbertcurve>=1.0.0',
        'pandas>=1.3.0'
    ],
    extras_require={
        'gpu': ['cupy-cuda12x>=9.0.0'],
        'dev': [
            'pytest>=6.0.0',
            'sphinx>=4.0.0'
        ]
    },
    entry_points={
        'console_scripts': [
            'spacial-boxcount=spacial_boxcounting.cli:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering :: Image Processing',
        'Topic :: Scientific/Engineering :: Mathematics',
    ],
    python_requires='>=3.8',
    keywords='fractal-analysis, box-counting, image-processing, spatial-analysis, gpu-acceleration',
)
