Python Classes for Data Manipulation
====================================

[![Test](https://github.com/otsaloma/dataiter/workflows/Test/badge.svg)](https://github.com/otsaloma/dataiter/actions)
[![Documentation Status](https://readthedocs.org/projects/dataiter/badge/?version=latest)](https://dataiter.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/dataiter.svg)](https://pypi.org/project/dataiter/)

dataiter currently includes the following classes.

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

**`GeoJSON`** is a simple wrapper class that allows reading a GeoJSON
file into a `DataFrame` and writing a data frame to a GeoJSON file. Any
operations on the data are thus done with methods provided by the data
frame class. Geometry is read as-is into the "geometry" column, but no
special geometric operations are currently supported.

## Installation

```bash
# Latest stable version
pip install -U dataiter

# Latest development version
pip install -U git+https://github.com/otsaloma/dataiter#egg=dataiter
```

## Documentation

https://dataiter.readthedocs.io/
