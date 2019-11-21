from setuptools import setup, find_packages

setup(
    name="whynot",
    version="0.10.14",
    author="John Miller",
    author_email="miller_john@berkeley.edu",
    description="A framework for benchmarking causal inference.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "autograd",
        "causality",
        "dataclasses; python_version<'3.7'",
        "gym",
        "mesa",
        "numpy",
        "networkx",
        "pandas",
        "pyomo",
        "py_mini_racer",
        "scipy",
        "sklearn",
        "statsmodels",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
    ],
    python_requires=">=3.6",
)
