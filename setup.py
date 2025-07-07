from setuptools import setup, find_packages

setup(
    name='spacial_boxcounting',
    version='0.1.0',
    description='Convenient package for spatial box counting and fractal analysis across data types',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'numba',
        'Pillow',
        'matplotlib',
        'hilbertcurve'
    ],
    extras_require={
        'dev': [
            'pytest',
            'sphinx'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
