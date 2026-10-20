# -*- coding: utf-8 -*-

# Copyright (c) 2025 Osmo Salomaa
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
import re

from dataiter import dtypes
from dataiter import util
from dataiter import Vector
from numpy.dtypes import StringDType

def _apply(function, string, dtype=object, na_value=None, dedup=True):
    if util.is_scalar(string):
        return function(string)
    if not (isinstance(string, np.ndarray) and isinstance(string.dtype, StringDType)):
        label = string.view(Vector).dtype_label if isinstance(string, np.ndarray) else type(string).__name__
        raise TypeError(f"Expected string, got {label}")
    def apply(x):
        # np.fromiter keeps e.g. lists from findall as elements,
        # where assigning a list could make a multidimensional array.
        values = x.tolist()
        return np.fromiter(map(function, values), dtype, count=len(values))
    out = np.full_like(string, na_value, dtype)
    na = string == dtypes.string.na_object
    string = string[~na]
    if dedup and util.is_repetitive(string):
        # Call function only once per unique value. Not for mutable results,
        # e.g. lists, which equal rows would then share.
        unique, _, codes = util.unique_codes(string)
        out[~na] = apply(unique)[codes]
        return out.view(Vector)
    out[~na] = apply(string)
    return out.view(Vector)

def extract(pattern, string, group=0, flags=0):
    r"""
    Return `group` of the first match of `pattern` in `string`.

    `group` is a group number or name as in ``re.Match.group``, ``0`` for the
    whole match. Return a missing value if there is no match.

    >>> x = di.Vector(["Toy Story (1995)", "Jumanji (1995)", "No year"])
    >>> regex.extract(r"\((\d{4})\)", x, group=1)
    """
    p = re.compile(pattern, flags)
    def f(x):
        m = p.search(x)
        return (m.group(group) or "") if m else ""
    return _apply(f, string, dtypes.string, dtypes.string.na_object)

def findall(pattern, string, flags=0):
    """
    Return a list of matches of `pattern` in `string`.

    https://docs.python.org/3/library/re.html#re.findall

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.findall(r"[a-z]", x)
    """
    f = re.compile(pattern, flags).findall
    return _apply(f, string, dedup=False)

def fullmatch(pattern, string, flags=0):
    """
    Return a ``re.Match`` object or ``None``.

    https://docs.python.org/3/library/re.html#re.fullmatch

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.fullmatch(r"[a-z]+", x)
    """
    f = re.compile(pattern, flags).fullmatch
    return _apply(f, string)

def match(pattern, string, flags=0):
    """
    Return a ``re.Match`` object or ``None``.

    https://docs.python.org/3/library/re.html#re.match

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.match(r"[a-z]", x)
    """
    f = re.compile(pattern, flags).match
    return _apply(f, string)

def search(pattern, string, flags=0):
    """
    Return a ``re.Match`` object or ``None``.

    https://docs.python.org/3/library/re.html#re.search

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.search(r"[a-z]", x)
    """
    f = re.compile(pattern, flags).search
    return _apply(f, string)

def split(pattern, string, maxsplit=0, flags=0):
    """
    Return a list of `string` split by `pattern`.

    https://docs.python.org/3/library/re.html#re.split

    >>> x = di.Vector(["one two three", "four"])
    >>> regex.split(r" +", x)
    """
    p = re.compile(pattern, flags)
    f = lambda x: p.split(x, maxsplit=maxsplit)
    return _apply(f, string, dedup=False)

def sub(pattern, repl, string, count=0, flags=0):
    """
    Return `string` with instances of `pattern` replaced with `repl`.

    https://docs.python.org/3/library/re.html#re.sub

    >>> x = di.Vector(["great", "fantastic"])
    >>> regex.sub(r"$", r"!", x)
    """
    p = re.compile(pattern, flags)
    f = lambda x: p.sub(repl, x, count=count)
    return _apply(f, string, dtypes.string, dtypes.string.na_object)

def subn(pattern, repl, string, count=0, flags=0):
    """
    Return `string`, count of instances of `pattern` replaced with `repl`.

    https://docs.python.org/3/library/re.html#re.subn

    >>> x = di.Vector(["great", "fantastic"])
    >>> regex.subn(r"$", r"!", x)
    """
    p = re.compile(pattern, flags)
    f = lambda x: p.subn(repl, x, count=count)
    return _apply(f, string)
