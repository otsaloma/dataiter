# -*- coding: utf-8 -*-

# Copyright (c) 2020 Osmo Salomaa
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import contextlib
import functools
import numpy as np

from dataiter import aggregate
from dataiter import util

# API
from dataiter.vector import Vector # noqa
from dataiter.data_frame import DataFrame # noqa
from dataiter.data_frame import DataFrameColumn # noqa
from dataiter.geojson import GeoJSON # noqa
from dataiter.list_of_dicts import ListOfDicts # noqa

try:
    import numba # noqa
    USE_NUMBA = True
except Exception:
    USE_NUMBA = False

with contextlib.suppress(LookupError):
    USE_NUMBA = util.parse_env_boolean("DATAITER_USE_NUMBA")

__version__ = "0.28"

DEFAULT_PEEK_ELEMENTS = 10
DEFAULT_PEEK_ITEMS = 3
DEFAULT_PEEK_ROWS = 10
PRINT_FLOAT_PRECISION = 6
PRINT_MAX_ELEMENTS = 100
PRINT_MAX_ITEMS = 10
PRINT_MAX_ROWS = 100

# Only used as a fallback, see util.get_print_width.
PRINT_MAX_WIDTH = 80


def _ensure_x_type(function):
    @functools.wraps(function)
    def wrapper(x, *args, **kwargs):
        if not isinstance(x, (Vector, str)):
            raise TypeError(
                f"Bad type for x: {type(x)}, "
                f"expected dataiter.Vector or str")
        return function(x, *args, **kwargs)
    return wrapper

@_ensure_x_type
def all(x):
    """
    Return whether all elements of `x` evaluate to ``True``.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    Uses ``numpy.all``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.all.html

    >>> di.all(di.Vector([True, False]))
    >>> di.all(di.Vector([True, True]))
    >>> di.all("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return all(data[x])
        if USE_NUMBA:
            f = aggregate.generic(x, np.all, dropna=False, default=True)
            f.fallback = fallback
            return f
        return fallback
    if len(x) < 1:
        return True
    return np.all(x).item()

@_ensure_x_type
def any(x):
    """
    Return whether any element of `x` evaluates to ``True``.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    Uses ``numpy.any``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.any.html

    >>> di.any(di.Vector([False, False]))
    >>> di.any(di.Vector([True, False]))
    >>> di.any("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return any(data[x])
        if USE_NUMBA:
            f = aggregate.generic(x, np.any, dropna=False, default=False)
            f.fallback = fallback
            return f
        return fallback
    if len(x) < 1:
        return False
    return np.any(x).item()

# x type check skipped on purpose due to allowing calls with no x given.
def count(x="", dropna=False):
    """
    Return the amount of elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`. Since
    all columns in a data frame should have the same amount of elements (i.e.
    rows), you can just leave the x argument at its default blank string, which
    will give you that row count.

    >>> di.count(di.Vector([1, 2, 3]))
    >>> di.count()
    """
    if isinstance(x, str):
        def fallback(data):
            if not x:
                return data.nrow
            return count(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.generic(x or "_group_", len, dropna=dropna, default=0, nrequired=0)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    return len(x)

@_ensure_x_type
def count_unique(x, dropna=False):
    """
    Return the amount of unique elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.count_unique(di.Vector([1, 2, 2, 3, 3, 3]))
    >>> di.count_unique("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return count_unique(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.count_unique(x, dropna=dropna)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    return len(np.unique(x))

@_ensure_x_type
def first(x):
    """
    Return the first element of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.first(di.Vector([1, 2, 3]))
    >>> di.first("x")
    """
    return nth(x, 0)

@_ensure_x_type
def last(x):
    """
    Return the last element of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.last(di.Vector([1, 2, 3]))
    >>> di.last("x")
    """
    return nth(x, -1)

@_ensure_x_type
def max(x, dropna=True):
    """
    Return the maximum of elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.max(di.Vector([4, 5, 6]))
    >>> di.max("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return max(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.generic(x, np.amax, dropna=dropna, default=np.nan)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 1:
        return np.nan
    return np.amax(x).item()

@_ensure_x_type
def mean(x, dropna=True):
    """
    Return the arithmetic mean of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    Uses ``numpy.mean``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.mean.html

    >>> di.mean(di.Vector([1, 2, 10]))
    >>> di.mean("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return mean(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.generic(x, np.mean, dropna=dropna, default=np.nan)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 1:
        return np.nan
    return np.mean(x).item()

@_ensure_x_type
def median(x, dropna=True):
    """
    Return the median of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    Uses ``numpy.median``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.median.html

    >>> di.median(di.Vector([5, 1, 2]))
    >>> di.median("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return median(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.generic(x, np.median, dropna=dropna, default=np.nan)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 1:
        return np.nan
    return np.median(x).item()

@_ensure_x_type
def min(x, dropna=True):
    """
    Return the minimum of elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.min(di.Vector([4, 5, 6]))
    >>> di.min("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return min(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.generic(x, np.amin, dropna=dropna, default=np.nan)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 1:
        return np.nan
    return np.amin(x).item()

@_ensure_x_type
def mode(x, dropna=True):
    """
    Return the most common value in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.mode(di.Vector([1, 2, 2, 3, 3, 3]))
    >>> di.mode("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return mode(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.mode(x, dropna=dropna)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 1:
        return x.missing_value
    values, counts = np.unique(x, return_counts=True)
    return values[counts.argmax()]

def nrow(data):
    """
    Return the amount of rows in `data`.

    This is a useful shorthand for `data.nrow` in contexts where you don't have
    direct access to the data frame in question, e.g. in group-by-aggregate

    .. warning:: Deprecated, please use :func:`count` instead.

    >>> data = di.read_csv("data/listings.csv")
    >>> data.group_by("hood").aggregate(n=di.nrow)
    """
    if not getattr(nrow, "warning_shown", False):
        print('Warning: nrow is deprecated, please use count instead')
        print('e.g. data.group_by("hood").aggregate(n=di.count())')
        nrow.warning_shown = True
    return data.nrow

@_ensure_x_type
def nth(x, index):
    """
    Return the element of `x` at `index`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.nth(di.Vector([1, 2, 3]), 1)
    >>> di.nth("x", 1)
    """
    if isinstance(x, str):
        def fallback(data):
            return nth(data[x], index)
        if USE_NUMBA:
            f = aggregate.nth(x, index)
            f.fallback = fallback
            return f
        return fallback
    try:
        return x[index].item()
    except IndexError:
        return x.missing_value

@_ensure_x_type
def quantile(x, q, dropna=True):
    """
    Return the `qth` quantile of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    Uses ``numpy.quantile``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.quantile.html

    >>> di.quantile(di.Vector([1, 5, 6]), 0.5)
    >>> di.quantile("x", 0.5)
    """
    if isinstance(x, str):
        def fallback(data):
            return quantile(data[x], q, dropna=dropna)
        if USE_NUMBA:
            f = aggregate.quantile(x, q, dropna=dropna)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 1:
        return np.nan
    return np.quantile(x, q).item()

def read_csv(path, encoding="utf-8", sep=",", header=True, columns=[], dtypes={}):
    return DataFrame.read_csv(path,
                              encoding=encoding,
                              sep=sep,
                              header=header,
                              columns=columns,
                              dtypes=dtypes)

def read_geojson(path, encoding="utf-8", columns=[], dtypes={}, **kwargs):
    return GeoJSON.read(path,
                        encoding=encoding,
                        columns=columns,
                        dtypes=dtypes,
                        **kwargs)

def read_json(path, encoding="utf-8", keys=[], types={}, **kwargs):
    return ListOfDicts.read_json(path,
                                 encoding=encoding,
                                 keys=keys,
                                 types=types,
                                 **kwargs)

def read_npz(path, allow_pickle=True):
    return DataFrame.read_npz(path, allow_pickle=allow_pickle)

read_csv.__doc__ = util.format_alias_doc(read_csv, DataFrame.read_csv)
read_geojson.__doc__ = util.format_alias_doc(read_geojson, GeoJSON.read)
read_json.__doc__ = util.format_alias_doc(read_json, ListOfDicts.read_json)
read_npz.__doc__ = util.format_alias_doc(read_npz, DataFrame.read_npz)

@_ensure_x_type
def std(x, dropna=True):
    """
    Return the standard deviation of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    Uses ``numpy.std``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.std.html

    >>> di.std(di.Vector([3, 6, 7]))
    >>> di.std("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return std(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.generic(x, np.std, dropna=dropna, default=np.nan, nrequired=2)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return np.std(x).item()

@_ensure_x_type
def sum(x, dropna=True):
    """
    Return the sum of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.sum(di.Vector([1, 2, 3]))
    >>> di.sum("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return sum(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.generic(x, np.sum, dropna=dropna, default=0, nrequired=0)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    return np.sum(x).item()

@_ensure_x_type
def var(x, dropna=True):
    """
    Return the variance of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    Uses ``numpy.var``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.var.html

    >>> di.var(di.Vector([3, 6, 7]))
    >>> di.var("x")
    """
    if isinstance(x, str):
        def fallback(data):
            return var(data[x], dropna=dropna)
        if USE_NUMBA:
            f = aggregate.generic(x, np.var, dropna=dropna, default=np.nan, nrequired=2)
            f.fallback = fallback
            return f
        return fallback
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return np.var(x).item()
