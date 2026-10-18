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
def all(x, *, drop_na=True):
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
    return all_any(x, np.all, True, drop_na)

def all_any(x, function, empty, drop_na):
    # Replacing missing values with the result for empty input
    # gives the same result as dropping missing values.
    if isinstance(x, str):
        def aggregate(data):
            column = data[x]
            na = column.is_na()
            xb = Vector.fast(np.full(len(column), empty), bool)
            xb[~na] = column[~na].as_boolean()
            f = (generic, generic_numba)[use_numba(xb)]
            aggregate.default = empty
            out = f(function)(xb, data._group_, drop_na=False, default=empty, nrequired=0)
            if drop_na or not na.any(): return out
            has_na = f(np.any)(na, data._group_, drop_na=False, default=False, nrequired=0)
            # Prevent replacing the below missing values with the default.
            aggregate.default = None
            return [None if y == empty and z else bool(y)
                    for y, z in zip(out, has_na)]

        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    na = x.is_na()
    out = function(x[~na].as_boolean()).item()
    return None if not drop_na and out == empty and na.any() else out

@composite
def any(x, *, drop_na=True):
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
    return all_any(x, np.any, False, drop_na)

def check_not_datetime(x, name):
    if x.is_datetime():
        raise TypeError(f"Can't calculate {name} of datetimes")

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
        aggregate.colnames = [x] if x else []
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
        aggregate.colnames = [x]
        return aggregate
    x = handle_na(x, drop_na)
    return len(x.unique())

@deco.listify
def count_unique_apply(x, group, drop_na):
    for xg in yield_groups(x, group, drop_na):
        yield len(xg.unique())

@njit(cache=dataiter.USE_NUMBA_CACHE)
def count_unique_apply_numba(x, group, drop_na):
    out = []
    for xg in yield_groups_numba(x, group, drop_na):
        out.append(len(np.unique(xg)))
    return out

def datetime_as_float(x):
    if not (x.is_datetime() or x.is_timedelta()): return x
    # NaT would otherwise become the smallest int64 as float.
    out = x.as_float()
    out[x.is_na()] = np.nan
    return out

def datetime_from_float(value, x):
    if not (x.is_datetime() or x.is_timedelta()): return value
    # Float conversion loses precision for datetime64[ns],
    # results are exact to only about a microsecond.
    return np.round(np.asarray(value, float)).astype(x.dtype)

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
            # A conditional expression here (x if y else z) would be
            # miscompiled for datetimes, returning wrong values and breaking
            # also other functions compiled later that return lists of
            # datetimes. This is likely a bug in Numba (observed on version
            # 0.67 with Python 3.14), but easy to work around.
            if len(xg) >= nrequired:
                out.append(function(xg))
            else:
                out.append(default)
        return out
    return aggregate

def group_starts(group):
    # Return starts of groups, group being contiguous group indices per row.
    change = np.ones(len(group), bool)
    change[1:] = group[1:] != group[:-1]
    return np.flatnonzero(change)

def handle_na(x, drop_na):
    return x[~x.is_na()] if drop_na else x

def handle_na_groups(x, starts, drop_na):
    # Return x and starts of its groups, dropping missing values if drop_na.
    # Groups can be left empty, i.e. their start is the next group's start.
    if not drop_na: return x, starts
    valid = ~x.is_na()
    if valid.all(): return x, starts
    nvalid_before = np.cumsum(valid) - valid
    return x[valid], nvalid_before[starts]

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

def item(x):
    # NumPy returns strings as str, other scalars as NumPy generics.
    return x.item() if isinstance(x, np.generic) else x

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
        aggregate.colnames = [x]
        return aggregate
    x = handle_na(x, drop_na)
    return item(np.amax(x)) if len(x) >= 1 else x.na_value

@composite
def mean(x, *, drop_na=True):
    """
    Return the arithmetic mean of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    Equivalent to ``numpy.mean``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.mean.html

    >>> di.mean(di.Vector([1, 2, 10]))
    >>> di.mean("x")
    """
    def mean_groups(x, starts):
        y, starts = handle_na_groups(datetime_as_float(x), starts, drop_na)
        n = np.diff(starts, append=len(y))
        out = np.full(len(starts), np.nan)
        if n.any():
            # reduceat gives the next element for empty groups, skip them.
            out[n > 0] = np.add.reduceat(y, starts[n > 0], dtype=float) / n[n > 0]
        return datetime_from_float(out, x)
    if isinstance(x, str):
        def aggregate(data):
            return mean_groups(data[x], group_starts(data._group_))
        aggregate.default = np.nan
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return mean_groups(x, np.array([0])).item(0)

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
            out = f(datetime_as_float(data[x]),
                    data._group_,
                    drop_na=(
                        drop_na and
                        data[x].is_na().any()),
                    default=np.nan,
                    nrequired=1)
            return datetime_from_float(out, data[x])

        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    x = handle_na(x, drop_na)
    if x.is_datetime():
        return datetime_from_float(median(datetime_as_float(x)), x).item()
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
        aggregate.colnames = [x]
        return aggregate
    x = handle_na(x, drop_na)
    return item(np.amin(x)) if len(x) >= 1 else x.na_value

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
        aggregate.colnames = [x]
        return aggregate
    x = handle_na(x, drop_na)
    return item(statistics.mode(x)) if len(x) >= 1 else x.na_value

@deco.listify
def mode_apply(x, group, drop_na):
    for xg in yield_groups(x, group, drop_na):
        yield statistics.mode(xg) if len(xg) >= 1 else None

@njit(cache=dataiter.USE_NUMBA_CACHE)
def mode_apply_numba(x, group, drop_na):
    out = []
    for xg in yield_groups_numba(x, group, drop_na):
        if len(xg) > 0:
            # Count runs of equal values in sorted order, breaking ties
            # by first appearance in data, as statistics.mode does.
            order = np.argsort(xg)
            xs = xg[order]
            best_count = 0
            best_first = 0
            i = 0
            while i < len(xs):
                j = i + 1
                first = order[i]
                while j < len(xs) and xs[j] == xs[i]:
                    if order[j] < first:
                        first = order[j]
                    j += 1
                if j - i > best_count or (j - i == best_count and first < best_first):
                    best_count = j - i
                    best_first = first
                i = j
            out.append(xg[best_first])
        else:
            out.append(None)
    return out

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
        aggregate.colnames = [x]
        return aggregate
    x = handle_na(x, drop_na)
    try:
        return item(x[index])
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
            out = f(datetime_as_float(data[x]).as_float(),
                    data._group_,
                    q,
                    drop_na=(
                        drop_na and
                        data[x].is_na().any()))
            return datetime_from_float(out, data[x])

        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    x = handle_na(x, drop_na)
    if x.is_datetime() or x.is_timedelta():
        return datetime_from_float(quantile(datetime_as_float(x), q), x).item()
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

    Equivalent to ``numpy.std``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.std.html

    >>> di.std(di.Vector([3, 6, 7]))
    >>> di.std("x")
    """
    if isinstance(x, str):
        f = var(x, ddof=ddof, drop_na=drop_na)
        def aggregate(data):
            check_not_datetime(data[x], "std")
            return np.sqrt(f(data))
        aggregate.default = np.nan
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    check_not_datetime(x, "std")
    return np.sqrt(var(x, ddof=ddof, drop_na=drop_na)).item()

@composite
def sum(x, *, drop_na=True):
    """
    Return the sum of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.sum(di.Vector([1, 2, 3]))
    >>> di.sum("x")
    """
    def sum_groups(x, starts):
        check_not_datetime(x, "sum")
        y, starts = handle_na_groups(x, starts, drop_na)
        n = np.diff(starts, append=len(y))
        # Like np.sum, promote small integers and booleans to int64,
        # but unlike it, sum also float32 in float64 for accuracy.
        dtype = float if y.dtype.kind == "f" else np.sum(y[:0]).dtype
        out = np.zeros(len(starts), dtype)
        if n.any():
            # reduceat gives the next element for empty groups, skip them.
            out[n > 0] = np.add.reduceat(y.astype(dtype, copy=False), starts[n > 0])
        return out
    if isinstance(x, str):
        def aggregate(data):
            return sum_groups(data[x], group_starts(data._group_))
        aggregate.default = 0
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return sum_groups(x, np.array([0])).item(0)

def use_numba(x):
    # Numba can't handle all dtypes, use conditionally.
    # https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#str
    # NumPy considers timedelta a subtype of integer, but our Numba code
    # doesn't handle missing timedeltas or mixing them with numbers.
    if np.issubdtype(x.dtype, np.timedelta64):
        return False
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

    Equivalent to ``numpy.var``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.var.html

    >>> di.var(di.Vector([3, 6, 7]))
    >>> di.var("x")
    """
    def var_groups(x, starts):
        check_not_datetime(x, "var")
        y, starts = handle_na_groups(x, starts, drop_na)
        n = np.diff(starts, append=len(y))
        out = np.full(len(starts), np.nan)
        if n.any():
            # reduceat gives the next element for empty groups, skip them.
            s, k = starts[n > 0], n[n > 0]
            mean = np.add.reduceat(y, s, dtype=float) / k
            with np.errstate(divide="ignore", invalid="ignore"):
                out[n > 0] = np.add.reduceat((y - np.repeat(mean, k))**2, s) / (k - ddof)
            out[n <= ddof] = np.nan
        return out
    if isinstance(x, str):
        def aggregate(data):
            return var_groups(data[x], group_starts(data._group_))
        aggregate.default = np.nan
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return var_groups(x, np.array([0])).item(0)

def yield_groups(x, group, drop_na):
    # Groups must be contiguous for this to work!
    if len(x) == 0: return
    if drop_na:
        valid = ~x.is_na()
    bounds = (np.flatnonzero(group[1:] != group[:-1]) + 1).tolist()
    for i, j in zip([0] + bounds, bounds + [len(x)]):
        xij = x[i:j]
        if drop_na:
            xij = xij[valid[i:j]]
        yield xij

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
