from setuptools import setup, find_packages

setup(
    name="whynot",
    version="0.12.0",
    author="John Miller",
    author_email="miller_john@berkeley.edu",
    description="A framework for benchmarking causal inference.",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "autograd",
        "dataclasses; python_version<'3.7'",
        "gym==0.23.0",
        "mesa<=0.8.7",
        "numpy",
        "networkx",
        "pandas",
        "py_mini_racer",
        "pyomo",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
    ],
    python_requires=">=3.8.10, <=3.10.11",
)
