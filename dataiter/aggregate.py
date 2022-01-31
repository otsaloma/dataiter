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
        def njit(cls, cache=False):
            def outer_wrapper(function):
                def inner_wrapper(*args, **kwargs):
                    print("Using dummy njit, this shouldn't happen")
                    return function(*args, **kwargs)
                return inner_wrapper
            return outer_wrapper

# The below functions are designed solely to be used in conjunction with
# DataFrame.aggregate, which temporarily adds a '_group_' column and uses the
# 'numba' attribute of functions to figure out how to apply them. All '*_numba'
# aggregation functions require that the 'group' argument has contiguous groups
# and will not work right if that doesn't hold.

# There's a confusing amount of nested functions below. This is mostly due to
# two reasons: (1) Numba doesn't understand what a dataiter.DataFrame is, so
# the accelerated functions need to operate only on NumPy arrays and (2) Numba
# doesn't directly accept functions as arguments, but needs a wrapper instead.


def count_unique(name, dropna):
    def aggregate(data):
        if not use_numba(data[name]):
            raise NotImplementedError
        return count_unique_numba(
            data[name],
            data._group_,
            dropna=dropna)
    aggregate.numba = True
    return aggregate

@numba.njit(cache=True)
def count_unique_numba(x, group, dropna):
    dropna = dropna and np.isnan(x).any()
    out = np.repeat(0, len(np.unique(group)))
    g = i = 0
    n = len(x)
    for j in range(1, n + 1):
        if j < n and group[j] == group[i]: continue
        xij = x[i:j]
        if dropna:
            xij = xij[~np.isnan(xij)]
        out[g] = len(np.unique(xij))
        g += 1
        i = j
    return out

def generic(name, function, dropna, default=None, nrequired=1):
    f = generic_numba(function)
    def aggregate(data):
        if not use_numba(data[name]):
            raise NotImplementedError
        return f(
            data[name],
            data._group_,
            dropna=dropna,
            default=(
                data[name].missing_value
                if default is None else default),
            nrequired=nrequired)
    aggregate.numba = True
    return aggregate

@functools.lru_cache(256)
def generic_numba(function):
    @numba.njit(cache=True)
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

def mode(name, dropna):
    def aggregate(data):
        if not use_numba(data[name]):
            raise NotImplementedError
        return mode_numba(
            data[name],
            data._group_,
            dropna=dropna,
            default=data[name].missing_value)
    aggregate.numba = True
    return aggregate

@numba.njit(cache=True)
def mode_numba(x, group, dropna, default):
    dropna = dropna and np.isnan(x).any()
    out = np.repeat(default, len(np.unique(group)))
    g = i = 0
    n = len(x)
    for j in range(1, n + 1):
        if j < n and group[j] == group[i]: continue
        xij = x[i:j]
        if dropna:
            xij = xij[~np.isnan(xij)]
        if len(xij) > 0:
            nij = np.repeat(1, len(xij))
            for k in range(len(xij)):
                nij[k] = np.nansum(xij == xij[k])
            out[g] = xij[np.argmax(nij)]
        g += 1
        i = j
    return out

def nth(name, index, dropna):
    def aggregate(data):
        if not use_numba(data[name]):
            raise NotImplementedError
        return nth_numba(
            data[name],
            data._group_,
            index=index,
            dropna=dropna,
            default=data[name].missing_value)
    aggregate.numba = True
    return aggregate

@numba.njit(cache=True)
def nth_numba(x, group, index, dropna, default):
    dropna = dropna and np.isnan(x).any()
    out = np.repeat(default, len(np.unique(group)))
    g = i = 0
    n = len(x)
    for j in range(1, n + 1):
        if j < n and group[j] == group[i]: continue
        xij = x[i:j]
        if dropna:
            xij = xij[~np.isnan(xij)]
        if (0 <= index < len(xij) or
            -len(xij) <= index < 0):
            out[g] = xij[index]
        g += 1
        i = j
    return out

def quantile(name, q, dropna):
    def aggregate(data):
        if not use_numba(data[name]):
            raise NotImplementedError
        return quantile_numba(
            data[name],
            data._group_,
            q=q,
            dropna=dropna)
    aggregate.numba = True
    return aggregate

@numba.njit(cache=True)
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

def use_numba(x):
    # Numba can't handle all dtypes, use conditionally.
    # XXX: Strings should be supported, but not all functions seem to work.
    # https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#built-in-types
    return (np.issubdtype(x.dtype, np.bool_) or
            np.issubdtype(x.dtype, np.datetime64) or
            np.issubdtype(x.dtype, np.floating) or
            np.issubdtype(x.dtype, np.integer))
