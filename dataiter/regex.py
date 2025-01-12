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

def _prep(string, dtype, default):
    assert isinstance(string, np.ndarray)
    assert isinstance(string.dtype, StringDType)
    out = np.full_like(string, default, dtype)
    na = string == dtypes.string.na_object
    return out, na

def findall(pattern, string, flags=0):
    """
    Return a list of matches of `pattern` in `string`.

    https://docs.python.org/3/library/re.html#re.findall

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.findall(r"[a-z]", x)
    """
    if util.is_scalar(string):
        return re.findall(pattern, string, flags=flags)
    out, na = _prep(string, object, None)
    for i in np.flatnonzero(~na):
        out[i] = re.findall(pattern, string[i], flags=flags)
    return Vector.fast(out, object)

def fullmatch(pattern, string, flags=0):
    """
    Return a ``re.Match`` object or ``None``.

    https://docs.python.org/3/library/re.html#re.fullmatch

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.fullmatch(r"[a-z]+", x)
    """
    if util.is_scalar(string):
        return re.fullmatch(pattern, string, flags=flags)
    out, na = _prep(string, object, None)
    for i in np.flatnonzero(~na):
        out[i] = re.fullmatch(pattern, string[i], flags=flags)
    return Vector.fast(out, object)

def match(pattern, string, flags=0):
    """
    Return a ``re.Match`` object or ``None``.

    https://docs.python.org/3/library/re.html#re.match

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.match(r"[a-z]", x)
    """
    if util.is_scalar(string):
        return re.match(pattern, string, flags=flags)
    out, na = _prep(string, object, None)
    for i in np.flatnonzero(~na):
        out[i] = re.match(pattern, string[i], flags=flags)
    return Vector.fast(out, object)

def search(pattern, string, flags=0):
    """
    Return a ``re.Match`` object or ``None``.

    https://docs.python.org/3/library/re.html#re.search

    >>> x = di.Vector(["asdf", "1234"])
    >>> regex.search(r"[a-z]", x)
    """
    if util.is_scalar(string):
        return re.search(pattern, string, flags=flags)
    out, na = _prep(string, object, None)
    for i in np.flatnonzero(~na):
        out[i] = re.search(pattern, string[i], flags=flags)
    return Vector.fast(out, object)

def split(pattern, string, maxsplit=0, flags=0):
    """
    Return a list of `string` split by `pattern`.

    https://docs.python.org/3/library/re.html#re.split

    >>> x = di.Vector(["one two three", "four"])
    >>> regex.split(r" +", x)
    """
    if util.is_scalar(string):
        return re.split(pattern, string, maxsplit=maxsplit, flags=flags)
    out, na = _prep(string, object, None)
    for i in np.flatnonzero(~na):
        out[i] = re.split(pattern, string[i], maxsplit=maxsplit, flags=flags)
    return Vector.fast(out, object)

def sub(pattern, repl, string, count=0, flags=0):
    """
    Return `string` with instances of `pattern` replaced with `repl`.

    https://docs.python.org/3/library/re.html#re.sub

    >>> x = di.Vector(["great", "fantastic"])
    >>> regex.sub(r"$", r"!", x)
    """
    if util.is_scalar(string):
        return re.sub(pattern, repl, string, count=count, flags=flags)
    out, na = _prep(string, dtypes.string, dtypes.string.na_object)
    for i in np.flatnonzero(~na):
        out[i] = re.sub(pattern, repl, string[i], count=count, flags=flags)
    return Vector.fast(out, dtypes.string)

def subn(pattern, repl, string, count=0, flags=0):
    """
    Return `string`, count of instances of `pattern` replaced with `repl`.

    https://docs.python.org/3/library/re.html#re.subn

    >>> x = di.Vector(["great", "fantastic"])
    >>> regex.subn(r"$", r"!", x)
    """
    if util.is_scalar(string):
        return re.subn(pattern, repl, string, count=count, flags=flags)
    out, na = _prep(string, object, None)
    for i in np.flatnonzero(~na):
        out[i] = re.subn(pattern, repl, string[i], count=count, flags=flags)
    return Vector.fast(out, object)
