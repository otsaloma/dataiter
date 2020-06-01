#!/usr/bin/env python3

from setuptools import find_packages
from setuptools import setup

setup(
    name="dataiter",
    version="0.11",
    author="Osmo Salomaa",
    author_email="otsaloma@iki.fi",
    description="Classes for data manipulation",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/otsaloma/dataiter",
    license="MIT",
    packages=find_packages(exclude=["*.test"]),
    python_requires=">=3.6.0",
    install_requires=["attd>=0.3", "numpy>=1.7", "pandas>=1.0"],
)
