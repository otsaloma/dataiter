Python Classes for Data Manipulation
====================================

[![Build Status](https://travis-ci.org/otsaloma/dataiter.svg)](https://travis-ci.org/otsaloma/dataiter)
[![Documentation Status](https://readthedocs.org/projects/dataiter/badge/?version=latest)](https://dataiter.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/dataiter.svg)](https://pypi.org/project/dataiter/)

dataiter currently includes classes `DataFrame` and `ListOfDicts`.

**`DataFrame`** is a class for tabular data similar to R's `data.frame`
or `pandas.DataFrame`. It is under the hood a dictionary of NumPy arrays
and thus capable of fast vectorized operations. You can consider this to
be a very experimental, very light-weight alternative to Pandas with a
simple and consistent API. Performance-wise dataiter relies on NumPy and
is likely to be at best comparable to Pandas.

**`ListOfDicts`** is a class useful for manipulating data from JSON
APIs. It provides functionality similar to libraries such as
Underscore.js, with manipulation functions that iterate over the data
and return a shallow modified copy of the original. `attd.AttributeDict`
is used to provide convenient access to dictionary keys.

## Installation

```bash
pip install dataiter
```

## Documentation

https://dataiter.readthedocs.io/
