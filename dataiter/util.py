# -*- coding: utf-8 -*-

# Copyright (c) 2020 Osmo Salomaa
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

import bz2
import dataiter
import datetime
import gzip
import itertools
import lzma
import math
import numpy as np
import os
import shutil
import string

from dataiter import deco
from pathlib import Path


def count_digits(value):
    if np.isnan(value): return 0, 0
    if math.isinf(value): return 0, 0
    parts = np.format_float_positional(value).split(".")
    n = len(parts[0].lstrip("0"))
    m = len(parts[1].rstrip("0"))
    return n, m

def format_alias_doc(alias, target):
    indent = " " * 8
    note = (
        ".. note:: :func:`{}` is a convenience alias for :meth:`{}`."
        .format(alias.__name__, target.__qualname__))
    return target.__doc__ + "\n\n" + indent + note

def format_floats(seq, ksep=None):
    precision = dataiter.PRINT_FLOAT_PRECISION
    if any(0 < abs(x) < 1/10**precision or abs(x) > 10**16 - 1 for x in seq):
        # Format tiny and huge numbers in scientific notation.
        f = np.format_float_scientific
        return [f(x, precision=precision, trim="-") for x in seq]
    if ksep is None:
        ksep = dataiter.PRINT_THOUSAND_SEPARATOR
    # Format like largest by significant digits.
    digits = [count_digits(x) for x in seq]
    n = max(x[0] for x in digits)
    m = max(x[1] for x in digits)
    precision = min(m, max(0, precision - n))
    return [f"{{:,.{precision}f}}".format(x).replace(",", ksep)
            for x in seq]

def generate_colnames(n):
    return list(itertools.islice(yield_colnames(), n))

def get_print_width():
    return shutil.get_terminal_size((dataiter.PRINT_MAX_WIDTH, 24))[0] - 1

def is_scalar(value):
    # np.isscalar doesn't cover all needed cases.
    return (np.isscalar(value) or
            value is None or
            isinstance(value, (bytes,
                               bool,
                               float,
                               int,
                               str,
                               datetime.date,
                               datetime.datetime,
                               datetime.timedelta)))

def length(value):
    return 1 if is_scalar(value) else len(value)

def makedirs_for_file(path):
    return Path(path).parent.mkdir(parents=True, exist_ok=True)

@deco.listify
def pad(strings, *, align="right"):
    width = max(ulen(x) for x in strings)
    for value in strings:
        padding = " " * (width - ulen(value))
        yield (padding + value
               if align == "right"
               else value + padding)

def parse_env_boolean(name):
    return {
        "1":     True,
        "t":     True,
        "true":  True,
        "y":     True,
        "yes":   True,
        "0":     False,
        "f":     False,
        "false": False,
        "n":     False,
        "no":    False,
    }[os.environ[name].strip().lower()]

def quote(value):
    return '"{}"'.format(str(value).replace('"', r'\"'))

def sequencify(value):
    if isinstance(value, (np.ndarray, list, tuple)):
        return value
    if is_scalar(value):
        return [value]
    if hasattr(value, "__iter__"):
        # Evaluate generator or iterator.
        return list(value)
    raise ValueError(f"Unexpected type: {type(value)}")

def unique_keys(keys):
    return list(dict.fromkeys(keys))

def ulen(string):
    # Return the display length of string accounting for
    # Unicode characters that have a display width of zero.
    return len(string
               .replace("\u200b", "") # zero width space
               .replace("\u200c", "") # zero width non-joiner
               .replace("\u200d", "") # zero width joiner
               .replace("\u2060", "") # word joiner
               )

def unique_types(seq):
    return set(x.__class__ for x in seq if
               x is not None and
               not (isinstance(x, float) and np.isnan(x)))

def xopen(path, mode="r", **kwargs):
    if "b" not in mode:
        kwargs.setdefault("encoding", "utf-8")
    if str(path).endswith(".bz2"):
        kwargs.setdefault("compresslevel", 6)
        return bz2.open(path, mode, **kwargs)
    if str(path).endswith(".gz"):
        kwargs.setdefault("compresslevel", 6)
        return gzip.open(path, mode, **kwargs)
    if str(path).endswith(".xz"):
        return lzma.open(path, mode)
    return open(path, mode, **kwargs)

def yield_colnames():
    # Like Excel: a, b, c, ..., aa, bb, cc, ...
    for batch in range(1, 1000):
        for letter in string.ascii_lowercase:
            yield letter * batch
