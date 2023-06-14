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
from dataiter import deco
from dataiter import Vector

try:
    assert dataiter.USE_NUMBA
    from numba import njit
    from numba import types
    from numba.extending import overload
except Exception:
    # We need the decorators used to exist to avoid import time errors,
    # but actual calls to the below shouldn't be made (see 'select').
    def dummy_jit(*args, **kwargs):
        def outer_wrapper(function):
            def inner_wrapper(*args, **kwargs):
                print("Using dummy jit, this shouldn't happen")
                return function(*args, **kwargs)
            return inner_wrapper
        return outer_wrapper
    njit = overload = dummy_jit


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
                     drop_na=False,
                     default=True,
                     nrequired=0)

        aggregate.group_aware = True
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
                     drop_na=False,
                     default=False,
                     nrequired=0)

        aggregate.group_aware = True
        return aggregate
    x = x.as_boolean()
    return np.any(x).item()

# @composite skipped on purpose due to allowing calls with no x given.
def count(x="", *, drop_na=False):
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
                     drop_na=(
                         drop_na and
                         x and data[x].is_na().any()),
                     default=0,
                     nrequired=0)

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return len(x)

@composite
def count_unique(x, *, drop_na=False):
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
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()))

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return len(set(x))

@deco.listify
def count_unique_apply(x, group, drop_na):
    for xg in yield_groups(x, group, drop_na):
        yield len(set(xg))

@njit(cache=dataiter.USE_NUMBA_CACHE)
def count_unique_apply_numba(x, group, drop_na):
    out = []
    for xg in yield_groups_numba(x, group, drop_na):
        out.append(len(np.unique(xg)))
    return out

def first(x, *, drop_na=False):
    """
    Return the first element of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.first(di.Vector([1, 2, 3]))
    >>> di.first("x")
    """
    return nth(x, 0, drop_na=drop_na)

@functools.lru_cache(256)
def generic(function, **kwargs):
    @deco.listify
    def aggregate(x, group, drop_na, default, nrequired):
        for xg in yield_groups(x, group, drop_na):
            yield function(xg, **kwargs) if len(xg) >= nrequired else default
    return aggregate

@functools.lru_cache(256)
def generic_numba(function):
    @njit(cache=dataiter.USE_NUMBA_CACHE)
    def aggregate(x, group, drop_na, default, nrequired):
        out = []
        for xg in yield_groups_numba(x, group, drop_na):
            out.append(function(xg) if len(xg) >= nrequired else default)
        return out
    return aggregate

def handle_na(x, drop_na):
    return x[~x.is_na()] if drop_na else x

def is_na_item_numba(x):
    raise NotImplementedError

@overload(is_na_item_numba)
def is_na_item_numba_overload(x):
    # "Called at compile-time with the types of the function's runtime arguments."
    # https://numba.readthedocs.io/en/stable/reference/deprecation.html#deprecation-of-generated-jit
    # https://numba.readthedocs.io/en/stable/extending/high-level.html#implementing-functions
    if isinstance(x, types.Float):
        return lambda x: np.isnan(x)
    if isinstance(x, types.NPDatetime):
        return lambda x: np.isnat(x)
    if isinstance(x, types.UnicodeType):
        return lambda x: x == ""
    return lambda x: False

@njit(cache=dataiter.USE_NUMBA_CACHE)
def is_na_numba(x):
    na = np.full(len(x), False)
    for i in range(len(x)):
        na[i] = is_na_item_numba(x[i])
    return na

@composite
def last(x, *, drop_na=False):
    """
    Return the last element of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.last(di.Vector([1, 2, 3]))
    >>> di.last("x")
    """
    return nth(x, -1, drop_na=drop_na)

@composite
def max(x, *, drop_na=True):
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
            aggregate.default = data[x].na_value
            return f(data[x],
                     data._group_,
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()),
                     default=None,
                     nrequired=1)

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return np.amax(x).item() if len(x) >= 1 else x.na_value

@composite
def mean(x, *, drop_na=True):
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
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()),
                     default=np.nan,
                     nrequired=1)

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return np.mean(x).item() if len(x) >= 1 else np.nan

@composite
def median(x, *, drop_na=True):
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
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()),
                     default=np.nan,
                     nrequired=1)

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return np.median(x).item() if len(x) >= 1 else np.nan

@composite
def min(x, *, drop_na=True):
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
            aggregate.default = data[x].na_value
            return f(data[x],
                     data._group_,
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()),
                     default=None,
                     nrequired=1)

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return np.amin(x).item() if len(x) >= 1 else x.na_value

@composite
def mode(x, *, drop_na=True):
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
            aggregate.default = data[x].na_value
            return f(data[x],
                     data._group_,
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()))

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return mode1(x) if len(x) >= 1 else x.na_value

@deco.listify
def mode_apply(x, group, drop_na):
    for xg in yield_groups(x, group, drop_na):
        yield mode1(xg) if len(xg) >= 1 else None

@njit(cache=dataiter.USE_NUMBA_CACHE)
def mode_apply_numba(x, group, drop_na):
    out = []
    for xg in yield_groups_numba(x, group, drop_na):
        if len(xg) > 0:
            ng = np.full(len(xg), 0)
            for i in range(len(xg)):
                for j in range(len(xg)):
                    if xg[j] == xg[i]:
                        ng[i] += 1
            out.append(xg[np.argmax(ng)])
        else:
            out.append(None)
    return out

def mode1(x):
    try:
        return statistics.mode(x)
    except statistics.StatisticsError:
        # Python < 3.8 with several elements tied for mode.
        # Return the first encountered of the tied elements.
        return Counter(x).most_common(1)[0][0]

@composite
def nth(x, index, *, drop_na=False):
    """
    Return the element of `x` at `index` (zero-based).

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.nth(di.Vector([1, 2, 3]), 1)
    >>> di.nth("x", 1)
    """
    if isinstance(x, str):
        def aggregate(data):
            f = (nth_apply, nth_apply_numba)
            f = select(f, data, x)
            aggregate.default = data[x].na_value
            return f(data[x],
                     data._group_,
                     index,
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()))

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    try:
        return x[index].item()
    except IndexError:
        return x.na_value

@deco.listify
def nth_apply(x, group, index, drop_na):
    for xg in yield_groups(x, group, drop_na):
        try:
            yield xg[index]
        except IndexError:
            yield None

@njit(cache=dataiter.USE_NUMBA_CACHE)
def nth_apply_numba(x, group, index, drop_na):
    out = []
    for xg in yield_groups_numba(x, group, drop_na):
        if 0 <= index < len(xg) or -len(xg) <= index < 0:
            out.append(xg[index])
        else:
            out.append(None)
    return out

@composite
def quantile(x, q, *, drop_na=True):
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
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()))

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return np.quantile(x.as_float(), q).item() if len(x) >= 1 else np.nan

@deco.listify
def quantile_apply(x, group, q, drop_na):
    for xg in yield_groups(x, group, drop_na):
        yield np.quantile(xg, q) if len(xg) >= 1 else np.nan

@njit(cache=dataiter.USE_NUMBA_CACHE)
def quantile_apply_numba(x, group, q, drop_na):
    out = []
    for xg in yield_groups_numba(x, group, drop_na):
        out.append(np.quantile(xg, q) if len(xg) >= 1 else np.nan)
    return out

def select(functions, data, name):
    return functions[use_numba(data[name])]

@composite
def std(x, *, ddof=0, drop_na=True):
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
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()),
                     default=np.nan,
                     nrequired=2)

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return np.std(x, ddof=ddof).item() if len(x) >= 2 else np.nan

@composite
def sum(x, *, drop_na=True):
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
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()),
                     default=0,
                     nrequired=0)

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return np.sum(x).item()

def use_numba(x):
    # Numba can't handle all dtypes, use conditionally.
    # Strings are supported, but performance is bad.
    # https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#str
    return dataiter.USE_NUMBA and (
        np.issubdtype(x.dtype, np.bool_) or
        np.issubdtype(x.dtype, np.datetime64) or
        np.issubdtype(x.dtype, np.floating) or
        np.issubdtype(x.dtype, np.integer))

@composite
def var(x, *, ddof=0, drop_na=True):
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
                     drop_na=(
                         drop_na and
                         data[x].is_na().any()),
                     default=np.nan,
                     nrequired=2)

        aggregate.group_aware = True
        return aggregate
    x = handle_na(x, drop_na)
    return np.var(x, ddof=ddof).item() if len(x) >= 2 else np.nan

def yield_groups(x, group, drop_na):
    # Groups must be contiguous for this to work!
    i = 0
    n = len(x)
    for j in range(1, n + 1):
        if j < n and group[j] == group[i]: continue
        xij = x[i:j]
        if drop_na:
            xij = xij[~xij.is_na()]
        yield xij
        i = j

@njit(cache=dataiter.USE_NUMBA_CACHE)
def yield_groups_numba(x, group, drop_na):
    # Groups must be contiguous for this to work!
    i = 0
    n = len(x)
    out = []
    for j in range(1, n + 1):
        if j < n and group[j] == group[i]: continue
        xij = x[i:j]
        if drop_na:
            xij = xij[~is_na_numba(xij)]
        out.append(xij)
        i = j
    return out
