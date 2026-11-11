# -*- coding: utf-8 -*-

# Copyright (c) 2026 Osmo Salomaa
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

from dataiter.aggregate import check_kind
from dataiter.aggregate import composite

def accumulate(ufunc, x, starts):
    # Return ufunc accumulated within groups. Looping ufunc.accumulate per
    # group is slow for many small groups, for which instead scan combining
    # elements at doubling distances. Unlike subtracting group offsets from
    # a cumulative sum of all, this keeps float error relative to values
    # within group.
    if len(starts) < len(x) / 20:
        out = np.empty_like(x)
        for a, b in zip(starts.tolist(), np.append(starts[1:], len(x)).tolist()):
            ufunc.accumulate(x[a:b], out=out[a:b])
        return out
    pos = positions(starts, len(x))
    out = x.copy()
    i = np.arange(len(x))
    k = 1
    while k <= pos.max(initial=0):
        i = i[pos[i] >= k]
        out[i] = ufunc(out[i], out[i - k])
        k *= 2
    return out

@composite
def cummax(x):
    """
    Return cumulative maximum of `x`.

    If `x` is a string, return a function usable with :meth:`.DataFrame.modify`
    that operates group-wise on column `x`. Missing values propagate, i.e. the
    result is missing from the first missing value on.

    >>> di.cummax(di.Vector([1, 3, 2]))
    >>> di.cummax("x")
    """
    def cummax_groups(x, starts):
        check_kind(x, "cumulative maximum", "biufmM")
        return accumulate(np.maximum, x, starts)
    return dispatch(cummax_groups, x)

@composite
def cummin(x):
    """
    Return cumulative minimum of `x`.

    If `x` is a string, return a function usable with :meth:`.DataFrame.modify`
    that operates group-wise on column `x`. Missing values propagate, i.e. the
    result is missing from the first missing value on.

    >>> di.cummin(di.Vector([3, 1, 2]))
    >>> di.cummin("x")
    """
    def cummin_groups(x, starts):
        check_kind(x, "cumulative minimum", "biufmM")
        return accumulate(np.minimum, x, starts)
    return dispatch(cummin_groups, x)

@composite
def cumsum(x):
    """
    Return cumulative sum of `x`.

    If `x` is a string, return a function usable with :meth:`.DataFrame.modify`
    that operates group-wise on column `x`. Missing values propagate, i.e. the
    result is missing from the first missing value on.

    >>> di.cumsum(di.Vector([1, 2, 3]))
    >>> di.cumsum("x")
    """
    def cumsum_groups(x, starts):
        check_kind(x, "cumulative sum", "biufcm")
        # Upcast as np.cumsum does, as the sum can overflow smaller types.
        if x.dtype.kind in "bi":
            x = x.astype(np.int64, copy=False)
        elif x.dtype.kind == "u":
            x = x.astype(np.uint64, copy=False)
        if x.dtype.kind in "iu":
            # Subtract the total before each group from a cumulative sum of
            # all, which is much faster than accumulate for many small groups.
            # Integer sums are exact, also if they overflow, as wraparound
            # cancels out in the subtraction.
            out = np.cumsum(x)
            before = np.where(starts > 0, out[starts - 1], 0)
            out -= np.repeat(before, np.diff(starts, append=len(x)))
            return out
        return accumulate(np.add, x, starts)
    return dispatch(cumsum_groups, x)

def dispatch(function, x):
    # function takes a vector and starts of its groups and returns a value
    # per element. Return a group-aware function for use with
    # DataFrame.modify if x is a column name, else the value for x.
    if isinstance(x, str):
        def window(data, starts):
            return function(data[x], starts)
        window.group_aware = True
        return window
    return function(x, np.arange(min(1, len(x))))

def positions(starts, length):
    # Return zero-based positions of elements within their groups.
    return np.arange(length) - np.repeat(starts, np.diff(starts, append=length))

@composite
def rank(x, *, method="min", dir=1):
    """
    Return the order of elements of `x` in sorted order.

    If `x` is a string, return a function usable with :meth:`.DataFrame.modify`
    that operates group-wise on column `x`. See :meth:`.Vector.rank` for
    `method`, `dir` and handling of missing values.

    >>> di.rank(di.Vector([3, 1, 1, 2]))
    >>> di.rank("x", method="ordinal", dir=-1)
    """
    def rank_groups(x, starts):
        if len(starts) <= 1:
            return x.rank(method=method, dir=dir)
        codes = x.rank(method="dense", dir=dir)
        lengths = np.diff(starts, append=len(x))
        group = np.repeat(np.arange(len(starts)), lengths)
        ranks = (group * (codes.max() + 1) + codes).rank(method=method)
        # Rank over all groups and subtract what precedes each group,
        # all rows or, for dense, all distinct values of previous groups.
        offsets = np.minimum.reduceat(ranks, starts) - 1 if method == "dense" else starts
        return ranks - np.repeat(offsets, lengths)
    return dispatch(rank_groups, x)

def row_number(start=1):
    """
    Return row numbers within groups.

    Return a function usable with :meth:`.DataFrame.modify` that operates
    group-wise. Rows are numbered in the order they're in,
    :meth:`.DataFrame.sort` first if needed. See :func:`rank` to number by the
    values of a column instead.

    >>> di.row_number()
    """
    def window(data, starts):
        return positions(starts, data.nrow) + start
    window.group_aware = True
    return window

@composite
def shift(x, n=1):
    """
    Return elements of `x` shifted by `n` positions.

    If `x` is a string, return a function usable with :meth:`.DataFrame.modify`
    that operates group-wise on column `x`. See :meth:`.Vector.shift` for the
    direction of `n`. Positions left without an element will be missing.

    >>> di.shift(di.Vector([1, 2, 3, 4]))
    >>> di.shift("x", -1)
    """
    def shift_groups(x, starts):
        out = x.shift(n)
        if len(starts) <= 1 or n == 0:
            return out
        pos = positions(starts, len(x))
        lengths = np.diff(starts, append=len(x))
        out[pos < n if n > 0 else pos >= np.repeat(lengths, lengths) + n] = x.na_value
        return out
    return dispatch(shift_groups, x)
