from setuptools import setup, find_packages

setup(
    name="grw-smoothing",
    version="0.1.0",
    author="Gil Goldman",
    author_email="gilgoldm@gmail.com",
    description="Functional implementation of GRW-smoothing loss",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/gilgoldm/grw-smoothing",

    # Finds the 'grw_smoothing' package directory automatically
    packages=find_packages(exclude=("configs", "notebooks", "scripts", "tests")),

    # --- Minimal Core Requirements ---
    # Only the dependencies needed to import and use the grw_smoothing functional
    install_requires=[
        "numpy",
        "torch>=2.3.0",
    ],
    # --- End of Requirements ---

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)