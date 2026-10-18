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
import itertools
import numpy as np

from dataiter import deco
from dataiter import util
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
    return all_any(x, np.logical_and, True, drop_na)

def all_any(x, function, empty, drop_na):
    # function is np.logical_and or np.logical_or. Replacing missing values
    # with the result for empty input gives the same result as dropping them.
    def all_any_groups(x, starts):
        na = x.is_na()
        y = np.full(len(x), empty)
        y[~na] = x[~na].as_boolean()
        n = np.diff(starts, append=len(x))
        out = np.full(len(starts), empty)
        if n.any():
            out[n > 0] = function.reduceat(y, starts[n > 0])
        if drop_na or not na.any(): return out
        # Without a decisive value, missing values make the result unknown.
        has_na = np.zeros(len(starts), bool)
        has_na[n > 0] = np.logical_or.reduceat(na, starts[n > 0])
        out = out.astype(object)
        out[(out == empty) & has_na] = None
        return out
    if isinstance(x, str):
        def aggregate(data):
            return all_any_groups(data[x], group_starts(data._group_))
        aggregate.default = None
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return item(all_any_groups(x, np.array([0]))[0])

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
    return all_any(x, np.logical_or, False, drop_na)

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
    def count_groups(x, starts):
        y, starts = handle_na_groups(x, starts, drop_na)
        return np.diff(starts, append=len(y))
    if isinstance(x, str):
        def aggregate(data):
            return count_groups(data[x or "_group_"], group_starts(data._group_))
        aggregate.default = 0
        aggregate.group_aware = True
        aggregate.colnames = [x] if x else []
        return aggregate
    return count_groups(x, np.array([0])).item(0)

@composite
def count_unique(x, *, drop_na=False):
    """
    Return the amount of unique elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.count_unique(di.Vector([1, 2, 2, 3, 3, 3]))
    >>> di.count_unique("x")
    """
    def count_unique_groups(x, starts):
        y, starts = handle_na_groups(x, starts, drop_na)
        n = np.diff(starts, append=len(y))
        group = np.repeat(np.arange(len(starts)), n)
        codes = value_codes(y)
        m = codes.max(initial=0) + 1
        # Count unique pairs of group and value.
        keys = util.unique_codes(group * m + codes)[0]
        return np.bincount(keys // m, minlength=len(starts))
    if isinstance(x, str):
        def aggregate(data):
            return count_unique_groups(data[x], group_starts(data._group_))
        aggregate.default = 0
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return count_unique_groups(x, np.array([0])).item(0)

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
    return min_max(x, np.maximum, drop_na)

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

    Equivalent to ``numpy.median``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.median.html

    >>> di.median(di.Vector([5, 1, 2]))
    >>> di.median("x")
    """
    def median_groups(x, starts):
        y, starts = handle_na_groups(datetime_as_float(x).as_float(), starts, drop_na)
        y = np.asarray(y)
        n = np.diff(starts, append=len(y))
        out = np.full(len(starts), np.nan)
        if len(starts) < 40_000:
            # Looping is faster than sorting for few groups.
            for i, (a, k) in enumerate(zip(starts.tolist(), n.tolist())):
                if k > 0: out[i] = np.median(y[a:a+k])
            return datetime_from_float(out, x)
        ys = sort_groups(y, starts)
        ok = n > 0
        s, k = starts[ok], n[ok]
        # Average the middle values as np.median does.
        values = (ys[s + (k - 1) // 2] + ys[s + k // 2]) / 2
        # Missing values are sorted last, making the result missing.
        values[np.isnan(ys[s + k - 1])] = np.nan
        out[ok] = values
        return datetime_from_float(out, x)
    if isinstance(x, str):
        def aggregate(data):
            return median_groups(data[x], group_starts(data._group_))
        aggregate.default = np.nan
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return median_groups(x, np.array([0])).item(0)

@composite
def min(x, *, drop_na=True):
    """
    Return the minimum of elements in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.min(di.Vector([4, 5, 6]))
    >>> di.min("x")
    """
    return min_max(x, np.minimum, drop_na)

def min_max(x, function, drop_na):
    # function is np.minimum or np.maximum.
    def min_max_groups(x, starts):
        y, starts = handle_na_groups(x, starts, drop_na)
        n = np.diff(starts, append=len(y))
        nonempty = n > 0
        if not nonempty.any():
            values = y[:0]
        elif y.is_string():
            # reduceat doesn't support strings, reduce their sorted codes
            # and map back to strings via a row index for each code.
            codes = util.sorted_codes(y)
            rows = np.empty(codes.max() + 1, int)
            rows[codes] = np.arange(len(y))
            values = y[rows[function.reduceat(codes, starts[nonempty])]]
        else:
            values = function.reduceat(y, starts[nonempty])
        if nonempty.all(): return values
        # Upcast to have missing values for empty groups, e.g. int to float.
        out = np.full(len(starts), x.na_value, x.na_dtype)
        out[nonempty] = values
        return out
    if isinstance(x, str):
        def aggregate(data):
            return min_max_groups(data[x], group_starts(data._group_))
        aggregate.default = None
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return item(min_max_groups(x, np.array([0]))[0])

@composite
def mode(x, *, drop_na=True):
    """
    Return the most common value in `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.mode(di.Vector([1, 2, 2, 3, 3, 3]))
    >>> di.mode("x")
    """
    def mode_groups(x, starts):
        y, starts = handle_na_groups(x, starts, drop_na)
        n = np.diff(starts, append=len(y))
        found = n > 0
        if found.any():
            group = np.repeat(np.arange(len(starts)), n)
            codes = value_codes(y)
            m = codes.max() + 1
            keys, first, pairs = util.unique_codes(group * m + codes)
            # Unique pairs of group and value are in order of first appearance
            # and thus by group. Pick the most common pair of each group,
            # breaking ties by first appearance, as statistics.mode does.
            count = np.bincount(pairs)
            pair_group = keys // m
            pair_starts = group_starts(pair_group)
            most = np.maximum.reduceat(count, pair_starts)
            most = np.repeat(most, np.diff(pair_starts, append=len(keys)))
            candidates = np.flatnonzero(count == most)
            chosen = candidates[group_starts(pair_group[candidates])]
            values = y[first[chosen]]
        else:
            values = y[:0]
        if found.all(): return values
        # Upcast to have missing values for empty groups, e.g. int to float.
        out = np.full(len(starts), x.na_value, x.na_dtype)
        out[found] = values
        return out
    if isinstance(x, str):
        def aggregate(data):
            return mode_groups(data[x], group_starts(data._group_))
        aggregate.default = None
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return item(mode_groups(x, np.array([0]))[0])

@composite
def nth(x, index, *, drop_na=False):
    """
    Return the element of `x` at `index` (zero-based).

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    >>> di.nth(di.Vector([1, 2, 3]), 1)
    >>> di.nth("x", 1)
    """
    def nth_groups(x, starts):
        y, starts = handle_na_groups(x, starts, drop_na)
        n = np.diff(starts, append=len(y))
        i = n + index if index < 0 else np.full(len(n), index)
        found = (i >= 0) & (i < n)
        values = y[(starts + i)[found]]
        if found.all(): return values
        # Upcast to have missing values for groups without index, e.g. int to float.
        out = np.full(len(starts), x.na_value, x.na_dtype)
        out[found] = values
        return out
    if isinstance(x, str):
        def aggregate(data):
            return nth_groups(data[x], group_starts(data._group_))
        aggregate.default = None
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return item(nth_groups(x, np.array([0]))[0])

@composite
def quantile(x, q, *, drop_na=True):
    """
    Return the `qth` quantile of `x`.

    If `x` is a string, return a function usable with
    :meth:`.DataFrame.aggregate` that operates group-wise on column `x`.

    Equivalent to ``numpy.quantile``, see the NumPy documentation for details:
    https://numpy.org/doc/stable/reference/generated/numpy.quantile.html

    >>> di.quantile(di.Vector([1, 5, 6]), 0.5)
    >>> di.quantile("x", 0.5)
    """
    def quantile_groups(x, starts):
        y, starts = handle_na_groups(datetime_as_float(x).as_float(), starts, drop_na)
        y = np.asarray(y)
        n = np.diff(starts, append=len(y))
        out = np.full(len(starts), np.nan)
        if len(starts) < 20_000:
            # Looping is faster than sorting for few groups.
            for i, (a, k) in enumerate(zip(starts.tolist(), n.tolist())):
                if k > 0: out[i] = np.quantile(y[a:a+k], q)
            return datetime_from_float(out, x)
        ys = sort_groups(y, starts)
        ok = n > 0
        s, k = starts[ok], n[ok]
        virtual = (k - 1) * q
        if isinstance(q, (int, np.integer)):
            # np.quantile takes elements without interpolating for integer q,
            # which differs from interpolating with infinities.
            values = ys[s + virtual]
        else:
            # Interpolate linearly as np.quantile does, including its
            # previous index of -1 and thus gamma at and above the top.
            top = virtual >= k - 1
            previous = np.where(top, -1, np.floor(virtual))
            gamma = virtual - previous
            lo = np.where(top, k - 1, previous).astype(int)
            hi = np.where(top, k - 1, previous + 1).astype(int)
            a, b = ys[s + lo], ys[s + hi]
            diff = b - a
            values = np.where(gamma >= 0.5, b - diff * (1 - gamma), a + diff * gamma)
        # Missing values are sorted last, making the result missing.
        values[np.isnan(ys[s + k - 1])] = np.nan
        out[ok] = values
        return datetime_from_float(out, x)
    if isinstance(x, str):
        def aggregate(data):
            return quantile_groups(data[x], group_starts(data._group_))
        aggregate.default = np.nan
        aggregate.group_aware = True
        aggregate.colnames = [x]
        return aggregate
    return quantile_groups(x, np.array([0])).item(0)

def select(functions, data, name):
    return functions[use_numba(data[name])]

def sort_groups(x, starts):
    # Return float x sorted within groups, NaN last.
    import pyarrow as pa
    import pyarrow.compute as pc
    group = np.repeat(np.arange(len(starts)), np.diff(starts, append=len(x)))
    table = pa.table([group, pa.array(x, mask=np.isnan(x))], names=["group", "x"])
    keys = [("group", "ascending", "at_end"), ("x", "ascending", "at_end")]
    return x[pc.sort_indices(table, sort_keys=keys).to_numpy()]

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

def value_codes(x):
    # Return integer codes of x, equal for equal values and for all missing
    # values. Floats are normalized first, -0.0 to 0.0 and all NaNs to one,
    # as hashing their bits would otherwise treat those as different.
    if x.is_object():
        found = {}
        return np.fromiter(map(found.setdefault, x.tolist(), itertools.count()), np.int64, len(x))
    if x.is_float():
        x = np.where(np.isnan(x), np.nan, x.astype(float) + 0.0).view(np.int64)
    elif x.is_datetime() or x.is_timedelta():
        x = x.view(np.int64)
    elif x.is_boolean():
        x = x.astype(np.int64)
    return util.unique_codes(x)[2]

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
