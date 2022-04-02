Python Classes for Data Manipulation
====================================

[![Test](https://github.com/otsaloma/dataiter/workflows/Test/badge.svg)](https://github.com/otsaloma/dataiter/actions)
[![Documentation Status](https://readthedocs.org/projects/dataiter/badge/?version=latest)](https://dataiter.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/dataiter.svg)](https://pypi.org/project/dataiter/)
[![Downloads](https://pepy.tech/badge/dataiter/month)](https://pepy.tech/project/dataiter)

Dataiter currently includes the following classes.

**`DataFrame`** is a class for tabular data similar to R's `data.frame`
or `pandas.DataFrame`. It is under the hood a dictionary of NumPy arrays
and thus capable of fast vectorized operations. You can consider this to
be a light-weight alternative to Pandas with a simple and consistent
API. Performance-wise Dataiter relies on NumPy and Numba and is likely
to be at best comparable to Pandas.

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
pip install -U git+https://github.com/otsaloma/dataiter

# Numba (optional)
pip install -U numba
```

Dataiter optionally uses **Numba** to speed up certain operations. If
you have Numba installed and importing it succeeds, Dataiter will use it
automatically. It's currently not a hard dependency, so you need to
install it separately.

## Documentation

https://dataiter.readthedocs.io/

If you're familiar with either dplyr (R) or Pandas (Python), the
comparison table in the documentation will give you a quick overview of
the differences and similarities.

https://dataiter.readthedocs.io/en/latest/comparison.html
