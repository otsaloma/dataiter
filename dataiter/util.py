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

import dataiter
import itertools
import numpy as np
import string


def generate_colnames(n):
    return list(itertools.islice(yield_colnames(), n))

def is_missing(value):
    if value in [None, np.nan]:
        return True
    if (isinstance(value, float) and
        np.isnan(value)):
        return True
    if (isinstance(value, str) and
        not value):
        return True
    if (hasattr(value, "dtype") and
        np.issubdtype(value.dtype, np.datetime64) and
        np.isnat(value)):
        return True
    return False

def length(value):
    return 1 if np.isscalar(value) else len(value)

def np_to_string(value):
    if np.isscalar(value):
        value = np.array(value)
    return np.array2string(
        value,
        max_line_width=dataiter.PRINT_MAX_WIDTH,
        precision=dataiter.PRINT_FLOAT_PRECISION,
        formatter={
            "datetime": str,
            "numpystr": str,
            "str": str,
        },
    )

def unique_keys(keys):
    return list(dict.fromkeys(keys))

def unique_types(seq):
    return set(type(x) for x in seq if not is_missing(x))

def yield_colnames():
    # Like Excel: a, b, c, ..., aa, bb, cc, ...
    for batch in range(1, 1000):
        for letter in string.ascii_lowercase:
            yield letter * batch
