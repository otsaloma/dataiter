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
            return aggregate.ff(x, np.amax, dropna)
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
            return aggregate.ff(x, np.mean, dropna)
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
            return aggregate.ff(x, np.median, dropna)
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
            return aggregate.ff(x, np.amin, dropna)
        return lambda data: min(data[x], dropna=dropna)
    if dropna:
        x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.amin(x).item()

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
