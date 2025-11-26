# packages/grw-smoothing/setup.py
from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).with_name("README.md").read_text(encoding="utf-8")

setup(
    name="grw-smoothing",
    version="0.1.0",
    author="Gil Goldman",
    author_email="gilgoldm@gmail.com",
    description="Functional implementation of GRW-smoothing loss",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/gilgoldm/grw-smoothing",

    # Use src/ layout
    packages=find_packages(where="src"),
    package_dir={"": "src"},

    # Minimal runtime deps only
    install_requires=[
        "numpy",
        "torch>=2.3.0",
    ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
