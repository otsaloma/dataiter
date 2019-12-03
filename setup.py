#!/usr/bin/env python3

from setuptools import setup

setup(
    name="dataiter",
    version="0.6",
    author="Osmo Salomaa",
    author_email="otsaloma@iki.fi",
    description="Classes for data manipulation",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/otsaloma/dataiter",
    license="MIT",
    py_modules=["dataiter"],
    python_requires=">=3.1.0",
    install_requires=["attd>=0.3"],
)
