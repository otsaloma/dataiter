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

import itertools
import numpy as np
import string

from dataiter import deco


def count_decimals(value):
    value = str(float(value)).rstrip("0")
    if "." not in value: return 0
    return len(value.split(".")[-1])

def generate_colnames(n):
    return list(itertools.islice(yield_colnames(), n))

def length(value):
    return 1 if np.isscalar(value) else len(value)

@deco.listify
def pad(strings, align="right"):
    width = max(len(x) for x in strings)
    for value in strings:
        padding = " " * (width - len(value))
        yield (padding + value
               if align == "right"
               else value + padding)

def quote(value):
    return '"{}"'.format(str(value).replace('"', r'\"'))

def unique_keys(keys):
    return list(dict.fromkeys(keys))

def unique_types(seq):
    return set(x.__class__ for x in set(seq) if not x in set((np.nan, None)))

def yield_colnames():
    # Like Excel: a, b, c, ..., aa, bb, cc, ...
    for batch in range(1, 1000):
        for letter in string.ascii_lowercase:
            yield letter * batch
