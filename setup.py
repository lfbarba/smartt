"""
Setup script for the smartt library.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "smartt" / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")
else:
    long_description = "Smart Tensor Tomography Library"

setup(
    name="smartt",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Smart Tensor Tomography Library for ML dataset generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/smartTT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "h5py>=3.0.0",
        "mumott",  # Ensure mumott is installed separately if not on PyPI
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "smartt-generate=smartt.data_processing:main",
            "mumott-al-synthetic-dataset=mumott_al.synthetic_data_processing:main",
        ],
    },
)
