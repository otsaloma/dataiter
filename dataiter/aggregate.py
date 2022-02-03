# -*- coding: utf-8 -*-

# Copyright (c) 2022 Osmo Salomaa
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

import dataiter
import functools
import numpy as np
import statistics

from collections import Counter
from dataiter import Vector

try:
    import numba
except Exception:
    class numba:
        @classmethod
        def jit(cls, *args, **kwargs):
            def outer_wrapper(function):
                def inner_wrapper(*args, **kwargs):
                    print("Using dummy jit, this shouldn't happen")
                    return function(*args, **kwargs)
                return inner_wrapper
            return outer_wrapper
        generated_jit = jit
        njit = jit


def composite(function):
    @functools.wraps(function)
    def wrapper(x, *args, **kwargs):
        if not isinstance(x, (Vector, str)):
            raise TypeError("Expected Vector or str")
        return function(x, *args, **kwargs)
    return wrapper

@composite
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
        def aggregate(data):
            f = (generic, generic_numba)
            f = select(f, data, x)(np.all)
            aggregate.default = True
            return f(data[x].as_boolean(),
                     data._group_,
                     drop_missing=False,
                     default=True,
                     nrequired=0)

        aggregate.numba = True
        return aggregate
    x = x.as_boolean()
    return np.all(x).item()

@composite
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
        def aggregate(data):
            f = (generic, generic_numba)
            f = select(f, data, x)(np.any)
            aggregate.default = False
            return f(data[x].as_boolean(),
                     data._group_,
                     drop_missing=False,
                     default=False,
                     nrequired=0)

        aggregate.numba = True
        return aggregate
    x = x.as_boolean()
    return np.any(x).item()

# @composite skipped on purpose due to allowing calls with no x given.
def count(x="", drop_missing=False):
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
        def aggregate(data):
            f = (generic, generic_numba)
            f = select(f, data, x or "_group_")(len)
            aggregate.default = 0
            return f(data[x or "_group_"],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         x and data[x].is_missing().any()),
                     default=0,
                     nrequired=0)

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    return len(x)

@composite
def count_unique(x, drop_missing=False):
    """
    Return the amount of unique elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.count_unique(di.Vector([1, 2, 2, 3, 3, 3]))
    >>> di.count_unique("x")
    """
    if isinstance(x, str):
        def aggregate(data):
            f = (count_unique_apply, count_unique_apply_numba)
            f = select(f, data, x)
            aggregate.default = 0
            return f(data[x],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()))

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    return len(set(x))

def count_unique_apply(x, group, drop_missing):
    out = []
    for xij in yield_groups(x, group, drop_missing):
        out.append(len(set(xij)))
    return out

@numba.njit(cache=True)
def count_unique_apply_numba(x, group, drop_missing):
    out = []
    for xij in yield_groups_numba(x, group, drop_missing):
        out.append(len(np.unique(xij)))
    return out

def first(x, drop_missing=False):
    """
    Return the first element of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.first(di.Vector([1, 2, 3]))
    >>> di.first("x")
    """
    return nth(x, 0, drop_missing=drop_missing)

@functools.lru_cache(256)
def generic(function, **kwargs):
    def aggregate(x, group, drop_missing, default, nrequired):
        out = []
        for xij in yield_groups(x, group, drop_missing):
            out.append(function(xij, **kwargs) if len(xij) >= nrequired else default)
        return out
    return aggregate

@functools.lru_cache(256)
def generic_numba(function):
    @numba.njit(cache=True)
    def aggregate(x, group, drop_missing, default, nrequired):
        out = []
        for xij in yield_groups_numba(x, group, drop_missing):
            out.append(function(xij) if len(xij) >= nrequired else default)
        return out
    return aggregate

@numba.generated_jit(nopython=True, cache=True)
def is_missing_item_numba(x):
    # Flexible specializations with @generated_jit
    # https://numba.pydata.org/numba-doc/dev/user/generated-jit.html
    if isinstance(x, numba.types.Float):
        return lambda x: np.isnan(x)
    if isinstance(x, numba.types.NPDatetime):
        return lambda x: np.isnat(x)
    if isinstance(x, numba.types.UnicodeType):
        return lambda x: x == ""
    return lambda x: False

@numba.njit(cache=True)
def is_missing_numba(x):
    missing = np.full(len(x), False)
    for i in range(len(x)):
        missing[i] = is_missing_item_numba(x[i])
    return missing

@composite
def last(x, drop_missing=False):
    """
    Return the last element of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.last(di.Vector([1, 2, 3]))
    >>> di.last("x")
    """
    return nth(x, -1, drop_missing=drop_missing)

@composite
def max(x, drop_missing=True):
    """
    Return the maximum of elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.max(di.Vector([4, 5, 6]))
    >>> di.max("x")
    """
    if isinstance(x, str):
        def aggregate(data):
            f = (generic, generic_numba)
            f = select(f, data, x)(np.amax)
            aggregate.default = data[x].missing_value
            return f(data[x],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()),
                     default=None,
                     nrequired=1)

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    if len(x) < 1:
        return x.missing_value
    return np.amax(x).item()

@composite
def mean(x, drop_missing=True):
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
        def aggregate(data):
            f = (generic, generic_numba)
            f = select(f, data, x)(np.mean)
            aggregate.default = np.nan
            return f(data[x],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()),
                     default=np.nan,
                     nrequired=1)

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    if len(x) < 1:
        return np.nan
    return np.mean(x).item()

@composite
def median(x, drop_missing=True):
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
        def aggregate(data):
            f = (generic, generic_numba)
            f = select(f, data, x)(np.median)
            aggregate.default = np.nan
            return f(data[x],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()),
                     default=np.nan,
                     nrequired=1)

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    if len(x) < 1:
        return np.nan
    return np.median(x).item()

@composite
def min(x, drop_missing=True):
    """
    Return the minimum of elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.min(di.Vector([4, 5, 6]))
    >>> di.min("x")
    """
    if isinstance(x, str):
        def aggregate(data):
            f = (generic, generic_numba)
            f = select(f, data, x)(np.amin)
            aggregate.default = data[x].missing_value
            return f(data[x],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()),
                     default=None,
                     nrequired=1)

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    if len(x) < 1:
        return x.missing_value
    return np.amin(x).item()

@composite
def mode(x, drop_missing=True):
    """
    Return the most common value in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.mode(di.Vector([1, 2, 2, 3, 3, 3]))
    >>> di.mode("x")
    """
    if isinstance(x, str):
        def aggregate(data):
            f = (mode_apply, mode_apply_numba)
            f = select(f, data, x)
            aggregate.default = data[x].missing_value
            return f(data[x],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()))

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    if len(x) < 1:
        return x.missing_value
    return mode1(x)

def mode1(x):
    try:
        return statistics.mode(x)
    except statistics.StatisticsError:
        # Python < 3.8 with several elements tied for mode,
        # will return one of the tied elements at random.
        return Counter(x).most_common(1)[0][0]

def mode_apply(x, group, drop_missing):
    out = []
    for xij in yield_groups(x, group, drop_missing):
        out.append(mode1(xij) if len(xij) > 0 else None)
    return out

@numba.njit(cache=True)
def mode_apply_numba(x, group, drop_missing):
    out = []
    for xij in yield_groups_numba(x, group, drop_missing):
        if len(xij) > 0:
            nij = np.repeat(1, len(xij))
            for k in range(len(xij)):
                for l in range(len(xij)):
                    if xij[l] == xij[k]:
                        nij[k] += 1
            out.append(xij[np.argmax(nij)])
        else:
            out.append(None)
    return out

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

@composite
def nth(x, index, drop_missing=False):
    """
    Return the element of `x` at `index`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.nth(di.Vector([1, 2, 3]), 1)
    >>> di.nth("x", 1)
    """
    if isinstance(x, str):
        def aggregate(data):
            f = (nth_apply, nth_apply_numba)
            f = select(f, data, x)
            aggregate.default = data[x].missing_value
            return f(data[x],
                     data._group_,
                     index,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()))

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    try:
        return x[index].item()
    except IndexError:
        return x.missing_value

def nth_apply(x, group, index, drop_missing):
    out = []
    for xij in yield_groups(x, group, drop_missing):
        try:
            out.append(xij[index])
        except IndexError:
            out.append(None)
    return out

@numba.njit(cache=True)
def nth_apply_numba(x, group, index, drop_missing):
    out = []
    for xij in yield_groups_numba(x, group, drop_missing):
        if 0 <= index < len(xij) or -len(xij) <= index < 0:
            out.append(xij[index])
        else:
            out.append(None)
    return out

@composite
def quantile(x, q, drop_missing=True):
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
        def aggregate(data):
            f = (quantile_apply, quantile_apply_numba)
            f = select(f, data, x)
            aggregate.default = np.nan
            return f(data[x].as_float(),
                     data._group_,
                     q,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()))

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    if len(x) < 1:
        return np.nan
    return np.quantile(x.as_float(), q).item()

def quantile_apply(x, group, q, drop_missing):
    out = []
    for xij in yield_groups(x, group, drop_missing):
        out.append(np.quantile(xij, q) if len(xij) > 0 else np.nan)
    return out

@numba.njit(cache=True)
def quantile_apply_numba(x, group, q, drop_missing):
    out = []
    for xij in yield_groups_numba(x, group, drop_missing):
        out.append(np.quantile(xij, q) if len(xij) > 0 else np.nan)
    return out

def select(functions, data, name):
    return functions[use_numba(data[name])]

@composite
def std(x, ddof=0, drop_missing=True):
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
        def aggregate(data):
            if ddof == 0:
                # Numba doesn't support the ddof argument,
                # so can only handle the default ddof=0.
                f = (generic, generic_numba)
                f = select(f, data, x)(np.std)
            else:
                f = generic(np.std, ddof=ddof)
            aggregate.default = np.nan
            return f(data[x],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()),
                     default=np.nan,
                     nrequired=2)

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    if len(x) < 2:
        return np.nan
    return np.std(x, ddof=ddof).item()

@composite
def sum(x, drop_missing=True):
    """
    Return the sum of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.sum(di.Vector([1, 2, 3]))
    >>> di.sum("x")
    """
    if isinstance(x, str):
        def aggregate(data):
            f = (generic, generic_numba)
            f = select(f, data, x)(np.sum)
            aggregate.default = 0
            return f(data[x],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()),
                     default=0,
                     nrequired=0)

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    return np.sum(x).item()

def use_numba(x):
    # Numba can't handle all dtypes, use conditionally.
    return dataiter.USE_NUMBA and (
        np.issubdtype(x.dtype, np.bool_) or
        np.issubdtype(x.dtype, np.datetime64) or
        np.issubdtype(x.dtype, np.floating) or
        np.issubdtype(x.dtype, np.integer) or
        np.issubdtype(x.dtype, np.unicode_))

@composite
def var(x, ddof=0, drop_missing=True):
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
        def aggregate(data):
            if ddof == 0:
                # Numba doesn't support the ddof argument,
                # so can only handle the default ddof=0.
                f = (generic, generic_numba)
                f = select(f, data, x)(np.var)
            else:
                f = generic(np.var, ddof=ddof)
            aggregate.default = np.nan
            return f(data[x],
                     data._group_,
                     drop_missing=(
                         drop_missing and
                         data[x].is_missing().any()),
                     default=np.nan,
                     nrequired=2)

        aggregate.numba = True
        return aggregate
    if drop_missing:
        x = x[~x.is_missing()]
    if len(x) < 2:
        return np.nan
    return np.var(x, ddof=ddof).item()

def yield_groups(x, group, drop_missing):
    # Groups must be contiguous for this to work!
    i = 0
    n = len(x)
    for j in range(1, n + 1):
        if j < n and group[j] == group[i]: continue
        xij = x[i:j]
        if drop_missing:
            xij = xij[~xij.is_missing()]
        yield xij
        i = j

@numba.njit(cache=True)
def yield_groups_numba(x, group, drop_missing):
    # Groups must be contiguous for this to work!
    i = 0
    n = len(x)
    for j in range(1, n + 1):
        if j < n and group[j] == group[i]: continue
        xij = x[i:j]
        if drop_missing:
            xij = xij[~is_missing_numba(xij)]
        yield xij
        i = j
