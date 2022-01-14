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
# DataFrame.aggregate, which temporarily adds a '_group_' column and
# uses the 'numba' attribute of functions to figure out how to apply them.


def generic(name, function, dropna, default, nrequired=1):
    # Since Numba doesn't understand what a dataiter.DataFrame is,
    # we need the accelerated function to be separate,
    # operating only on NumPy arrays.
    aggregate_numba = generic_numba(function)
    def aggregate(data):
        return aggregate_numba(data[name],
                               data._group_,
                               dropna,
                               default,
                               nrequired)

    aggregate.numba = True
    return aggregate

@functools.lru_cache(256)
def generic_numba(function):
    import numba
    @numba.njit
    def aggregate(x, group, dropna, default, nrequired):
        # Calculate function group-wise for x.
        # Groups are expected to be contiguous.
        dropna = dropna and np.isnan(x).any()
        out = np.repeat(default, len(np.unique(group)))
        g = 0
        i = 0
        n = len(x)
        for j in range(1, n + 1):
            if j == n or group[j] != group[i]:
                xij = x[i:j]
                if dropna:
                    xij = xij[~np.isnan(xij)]
                if len(xij) >= nrequired:
                    out[g] = function(xij)
                g += 1
                i = j
        return out
    return aggregate

def nth(name, index):
    # Since Numba doesn't understand what a dataiter.DataFrame is,
    # we need the accelerated function to be separate,
    # operating only on NumPy arrays.
    def aggregate(data):
        return nth_numba(data[name],
                         data._group_,
                         index,
                         data[name].missing_value)

    aggregate.numba = True
    return aggregate

@numba.njit
def nth_numba(x, group, index, default):
    # Calculate nth group-wise for x.
    # Groups are expected to be contiguous.
    out = np.repeat(default, len(np.unique(group)))
    g = 0
    i = 0
    n = len(x)
    for j in range(1, n + 1):
        if j == n or group[j] != group[i]:
            try:
                out[g] = x[i:j][index]
            except Exception:
                pass
            g += 1
            i = j
    return out
