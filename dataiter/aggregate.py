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

import functools
import numpy as np

try:
    import numba
except Exception:
    class numba:
        @classmethod
        def njit(cls, function):
            print("Using dummy njit, this shouldn't happen")
            return function

# The below functions are designed solely to be used in conjunction with
# DataFrame.aggregate, which temporarily adds a '_group_' column and uses
# the 'numba' attribute of functions to figure out how to apply them. All
# '*_numba' aggregation functions require that the 'group' argument has
# contiguous groups and will not work right if that doesn't hold.

# There's a confusing amount of nested functions below. This is mostly due to
# two reasons: (1) Numba doesn't understand what a dataiter.DataFrame is, so
# the accelerated functions need to operate only on NumPy arrays and (2) Numba
# doesn't directly accept functions as arguments, but needs a wrapper instead.


def mark_numba(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        aggregate = function(*args, **kwargs)
        aggregate.numba = True
        return aggregate
    return wrapper

@mark_numba
def count(dropna):
    f = generic_numba(len)
    return lambda data: f(
        data.columns[0],
        data._group_,
        dropna=dropna,
        default=0,
        nrequired=0) if data.columns else 0

@mark_numba
def count_unique(name, dropna):
    f = generic_numba(count_unique1)
    return lambda data: f(
        data[name],
        data._group_,
        dropna=dropna,
        default=0,
        nrequired=0)

@numba.njit
def count_unique1(x):
    return len(np.unique(x))

@mark_numba
def generic(name, function, dropna, default, nrequired=1):
    f = generic_numba(function)
    return lambda data: f(
        data[name],
        data._group_,
        dropna=dropna,
        default=default,
        nrequired=nrequired)

@functools.lru_cache(256)
def generic_numba(function):
    import numba
    @numba.njit
    def aggregate(x, group, dropna, default, nrequired):
        dropna = dropna and np.isnan(x).any()
        out = np.repeat(default, len(np.unique(group)))
        g = i = 0
        n = len(x)
        for j in range(1, n + 1):
            if j < n and group[j] == group[i]: continue
            xij = x[i:j]
            if dropna:
                xij = xij[~np.isnan(xij)]
            if len(xij) >= nrequired:
                out[g] = function(xij)
            g += 1
            i = j
        return out
    return aggregate

@mark_numba
def mode(name, dropna):
    f = generic_numba(mode1)
    return lambda data: f(
        data[name],
        data._group_,
        dropna=dropna,
        default=data[name].missing_value,
        nrequired=1)

@numba.njit
def mode1(x):
    # Numba doesn't support all np.unique's arguments,
    # so we can't do the usual below, but want to match it.
    # > values, counts = np.unique(x, return_counts=True)
    # > return values[counts.argmax()]
    max_value = x[0]
    max_count = 1
    for i in range(1, len(x)):
        count = np.nansum(x == x[i])
        if count > max_count:
            max_value = x[i]
    return max_value

@mark_numba
def nth(name, index):
    return lambda data: nth_numba(
        data[name],
        data._group_,
        index=index,
        default=data[name].missing_value)

@numba.njit
def nth_numba(x, group, index, default):
    out = np.repeat(default, len(np.unique(group)))
    g = i = 0
    n = len(x)
    for j in range(1, n + 1):
        if j < n and group[j] == group[i]: continue
        xij = x[i:j]
        if (0 <= index < len(xij) or
            -len(xij) <= index < 0):
            out[g] = xij[index]
        g += 1
        i = j
    return out

@mark_numba
def quantile(name, q, dropna):
    return lambda data: quantile_numba(
        data[name],
        data._group_,
        q=q,
        dropna=dropna)

@numba.njit
def quantile_numba(x, group, q, dropna):
    dropna = dropna and np.isnan(x).any()
    out = np.repeat(np.nan, len(np.unique(group)))
    g = i = 0
    n = len(x)
    for j in range(1, n + 1):
        if j < n and group[j] == group[i]: continue
        xij = x[i:j]
        if dropna:
            xij = xij[~np.isnan(xij)]
        if len(xij) > 0:
            out[g] = np.quantile(xij, q)
        g += 1
        i = j
    return out
