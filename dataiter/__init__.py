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

import numpy as np
import os

from dataiter.vector import Vector # noqa
from dataiter.data_frame import DataFrame # noqa
from dataiter.data_frame import DataFrameColumn # noqa
from dataiter.geojson import GeoJSON # noqa
from dataiter.list_of_dicts import ListOfDicts # noqa
from dataiter import aggregate

try:
    import numba # noqa
    USE_NUMBA = True
except Exception:
    USE_NUMBA = False

if os.getenv("DATAITER_DONT_USE_NUMBA", ""):
    USE_NUMBA = False

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


def all(x):
    """
    Return whether all elements of `x` evaluate to ``True``.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.all(di.Vector(range(10)))
    >>> di.all("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.generic(x, np.all, False, True)
        return lambda data: all(data[x])
    if len(x) == 0:
        return True
    return np.all(x).item()

def any(x):
    """
    Return whether any element of `x` evaluates to ``True``.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.any(di.Vector(range(10)))
    >>> di.any("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.generic(x, np.any, False, False)
        return lambda data: any(data[x])
    if len(x) == 0:
        return False
    return np.any(x).item()

def first(x):
    """
    Return the first element of `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.first(di.Vector(range(10)))
    >>> di.first("x")
    """
    return nth(x, 0)

def last(x):
    """
    Return the last element of `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.last(di.Vector(range(10)))
    >>> di.last("x")
    """
    return nth(x, -1)

def max(x, dropna=True):
    """
    Return the maximum of `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.max(di.Vector(range(10)))
    >>> di.max("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.generic(x, np.amax, dropna, np.nan)
        return lambda data: max(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.amax(x).item()

def mean(x, dropna=True):
    """
    Return the arithmetic mean of `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.mean(di.Vector(range(10)))
    >>> di.mean("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.generic(x, np.mean, dropna, np.nan)
        return lambda data: mean(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.mean(x).item()

def median(x, dropna=True):
    """
    Return the median of `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.median(di.Vector(range(10)))
    >>> di.median("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.generic(x, np.median, dropna, np.nan)
        return lambda data: median(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.median(x).item()

def min(x, dropna=True):
    """
    Return the minimum of `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.min(di.Vector(range(10)))
    >>> di.min("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.generic(x, np.amin, dropna, np.nan)
        return lambda data: min(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.amin(x).item()

def mode(x, dropna=True):
    """
    Return the most common value in `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.mode(di.Vector(range(10)))
    >>> di.mode("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.mode(x, dropna)
        return lambda data: mode(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) == 0:
        if not isinstance(x, Vector):
            raise TypeError("Not a dataiter.Vector")
        return x.missing_value
    values, counts = np.unique(x, return_counts=True)
    return values[counts.argmax()]

def n(x="", dropna=False):
    """
    Return the amount of elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.n(di.Vector(range(10)))
    >>> di.n("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.count(x, dropna)
        return lambda data: n(data[x or data.colnames[0]], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    return len(x)

def ncol(data):
    """
    Return the amount of columns in `data`.
    """
    return data.ncol

def nrow(data):
    """
    Return the amount of rows in `data`.

    This is a useful shorthand for `data.nrow` in contexts where you don't have
    direct access to the data frame in question, e.g. in group-by-aggregate

    >>> data = di.DataFrame.read_csv("data/listings.csv")
    >>> data.group_by("hood").aggregate(n=di.nrow)
    """
    return data.nrow

def nth(x, index):
    """
    Return the nth element of `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.nth(di.Vector(range(10)), 2)
    >>> di.nth("x", 2)
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.nth(x, index)
        return lambda data: nth(data[x], index)
    try:
        return x[index].item()
    except IndexError:
        if not isinstance(x, Vector):
            raise TypeError("Not a dataiter.Vector")
        return x.missing_value

def nunique(x, dropna=False):
    """
    Return the amount of elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.nunique(di.Vector(range(10)))
    >>> di.nunique("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.count(x, dropna, True)
        return lambda data: nunique(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    return len(x)

def quantile(x, q, dropna=True):
    """
    Return the `q`th quantile of `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.quantile(di.Vector(range(10)), 0.5)
    >>> di.quantile("x", 0.5)
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.quantile(x, q, dropna)
        return lambda data: quantile(data[x], q, dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.quantile(x, q).item()

def std(x, dropna=True):
    """
    Return the standard deviation of `x`.

    Uses ``numpy.std``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.std.html

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.std(di.Vector(range(10)))
    >>> di.std("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.generic(x, np.std, dropna, np.nan, nrequired=2)
        return lambda data: std(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return np.std(x).item()

def sum(x, dropna=True):
    """
    Return the sum of `x`.

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.sum(di.Vector(range(10)))
    >>> di.sum("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.generic(x, np.sum, dropna, np.nan)
        return lambda data: sum(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.sum(x).item()

def var(x, dropna=True):
    """
    Return the variance of `x`.

    Uses ``numpy.var``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.var.html

    If `x` is a string, return a function usable with
    :meth:`DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.var(di.Vector(range(10)))
    >>> di.var("x")
    """
    if isinstance(x, str):
        if USE_NUMBA:
            return aggregate.generic(x, np.var, dropna, np.nan, nrequired=2)
        return lambda data: var(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return np.var(x).item()
