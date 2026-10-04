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

def _apply(function, string, dtype=object, na_value=None):
    if util.is_scalar(string):
        return function(string)
    assert isinstance(string, np.ndarray)
    assert isinstance(string.dtype, StringDType)
    out = np.full_like(string, na_value, dtype)
    na = string == dtypes.string.na_object
    for i in np.flatnonzero(~na):
        out[i] = function(string[i])
    return Vector.fast(out, dtype)

def findall(pattern, string, flags=0):
    """
    Return a list of matches of `pattern` in `string`.

    https://docs.python.org/3/library/re.html#re.findall

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.findall(r"[a-z]", x)
    """
    f = lambda x: re.findall(pattern, x, flags=flags)
    return _apply(f, string)

def fullmatch(pattern, string, flags=0):
    """
    Return a ``re.Match`` object or ``None``.

    https://docs.python.org/3/library/re.html#re.fullmatch

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.fullmatch(r"[a-z]+", x)
    """
    f = lambda x: re.fullmatch(pattern, x, flags=flags)
    return _apply(f, string)

def match(pattern, string, flags=0):
    """
    Return a ``re.Match`` object or ``None``.

    https://docs.python.org/3/library/re.html#re.match

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.match(r"[a-z]", x)
    """
    f = lambda x: re.match(pattern, x, flags=flags)
    return _apply(f, string)

def search(pattern, string, flags=0):
    """
    Return a ``re.Match`` object or ``None``.

    https://docs.python.org/3/library/re.html#re.search

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.search(r"[a-z]", x)
    """
    f = lambda x: re.search(pattern, x, flags=flags)
    return _apply(f, string)

def split(pattern, string, maxsplit=0, flags=0):
    """
    Return a list of `string` split by `pattern`.

    https://docs.python.org/3/library/re.html#re.split

    >>> x = di.Vector(["one two three", "four"])
    >>> regex.split(r" +", x)
    """
    f = lambda x: re.split(pattern, x, maxsplit=maxsplit, flags=flags)
    return _apply(f, string)

def sub(pattern, repl, string, count=0, flags=0):
    """
    Return `string` with instances of `pattern` replaced with `repl`.

    https://docs.python.org/3/library/re.html#re.sub

    >>> x = di.Vector(["great", "fantastic"])
    >>> regex.sub(r"$", r"!", x)
    """
    f = lambda x: re.sub(pattern, repl, x, count=count, flags=flags)
    return _apply(f, string, dtypes.string, dtypes.string.na_object)

def subn(pattern, repl, string, count=0, flags=0):
    """
    Return `string`, count of instances of `pattern` replaced with `repl`.

    https://docs.python.org/3/library/re.html#re.subn

    >>> x = di.Vector(["great", "fantastic"])
    >>> regex.subn(r"$", r"!", x)
    """
    f = lambda x: re.subn(pattern, repl, x, count=count, flags=flags)
    return _apply(f, string)
