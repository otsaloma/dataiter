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
import datetime
import numpy as np

from dataiter import util

TYPE_CONVERSIONS = {
    datetime.date: "datetime64[D]",
    datetime.datetime: "datetime64[us]",
}


class Vector(np.ndarray):

    def __new__(cls, object, dtype=None):
        # If given a NumPy array, we can do a fast initialization.
        if isinstance(object, np.ndarray):
            dtype = dtype or object.dtype
            return np.array(object, dtype).view(cls)
        # If given a Python list, or something else generic, we need
        # to convert certain types and special values. This is really
        # slow, see Vector.fast for faster initialization.
        object = [object] if np.isscalar(object) else object
        return cls._std_to_np(object, dtype).view(cls)

    def __init__(self, object, dtype=None):
        self._check_dimensions()

    def __array_wrap__(self, array, context=None):
        # Avoid returning 0-dimensional arrays.
        # https://github.com/numpy/numpy/issues/7403
        return array[()] if array.shape == () else array

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    def as_boolean(self):
        return self.astype(bool)

    def as_bytes(self):
        if self.is_string:
            array = np.char.encode(self, "UTF-8")
            return array.view(self.__class__)
        return self.astype(bytes)

    def as_date(self):
        return self.astype(np.dtype("datetime64[D]"))

    def as_datetime(self):
        return self.astype(np.dtype("datetime64[us]"))

    def as_float(self):
        return self.astype(float)

    def as_integer(self):
        return self.astype(int)

    def as_string(self):
        return self.astype(str)

    def _check_dimensions(self):
        if self.ndim == 1: return
        raise ValueError(f"Bad dimensions: {self.ndim!r}")

    def equal(self, other):
        if not (isinstance(other, Vector) and
                self.length == other.length and
                str(self.missing_value) == str(other.missing_value)):
            return False
        ii = self.is_missing()
        jj = other.is_missing()
        return (np.all(ii == jj) and
                np.all(self[~ii] == other[~jj]))

    @classmethod
    def fast(cls, object, dtype=None):
        return np.array(object, dtype).view(cls)

    def head(self, n=None):
        if n is None:
            n = dataiter.DEFAULT_PEEK_ELEMENTS
        n = min(self.length, n)
        return self[np.arange(n)].copy()

    @property
    def is_boolean(self):
        return np.issubdtype(self.dtype, np.bool_)

    @property
    def is_bytes(self):
        return np.issubdtype(self.dtype, np.bytes_)

    @property
    def is_datetime(self):
        return np.issubdtype(self.dtype, np.datetime64)

    @property
    def is_float(self):
        return np.issubdtype(self.dtype, np.floating)

    @property
    def is_integer(self):
        return np.issubdtype(self.dtype, np.integer)

    def is_missing(self):
        if self.is_datetime:
            return np.isnat(self)
        if self.is_float:
            return np.isnan(self)
        if self.is_string:
            return self == ""
        return np.isin(self, [None])

    @property
    def is_number(self):
        return np.issubdtype(self.dtype, np.number)

    @property
    def is_object(self):
        return np.issubdtype(self.dtype, np.object_)

    @property
    def is_string(self):
        return np.issubdtype(self.dtype, np.unicode_)

    @property
    def length(self):
        self._check_dimensions()
        return self.size

    @property
    def missing_dtype(self):
        # Return corresponding dtype that can handle missing data.
        # Needed for upcasting when missing data is first introduced.
        if self.is_datetime:
            return self.dtype
        if self.is_float:
            return self.dtype
        if self.is_integer:
            return float
        if self.is_string:
            return self.dtype
        return object

    @property
    def missing_value(self):
        # Return value to use to represent missing values.
        if self.is_datetime:
            return np.datetime64("NaT")
        if self.is_float:
            return np.nan
        if self.is_integer:
            return np.nan
        if self.is_string:
            return ""
        # Note that using None, e.g. for a boolean vector,
        # might not work directly as it requires upcasting to object.
        return None

    def range(self):
        rng = [np.nanmin(self), np.nanmax(self)]
        return self.__class__(rng, self.dtype)

    def rank(self):
        rank = np.unique(self, return_inverse=True)[1]
        return rank.view(self.__class__)

    def sample(self, n=None):
        if n is None:
            n = dataiter.DEFAULT_PEEK_ELEMENTS
        n = min(self.length, n)
        indices = np.random.choice(self.length, n, replace=False)
        return self[np.sort(indices)].copy()

    def sort(self, dir=1):
        vector = self.copy()
        np.ndarray.sort(vector)
        if dir < 0:
            # Flip order, but keep missing last.
            na = vector.is_missing()
            ok = np.nonzero(~na)
            np.put(vector, ok, vector[ok][::-1])
        return vector

    @classmethod
    def _std_to_np(cls, seq, dtype=None):
        # Convert missing values in seq to NumPy equivalents.
        types = util.unique_types(seq)
        missing = cls._std_to_np_missing_value(types)
        seq = [missing if x is None else x for x in seq]
        if dtype is not None:
            return np.array(seq, dtype)
        # NaT values bring in np.datetime64 to types.
        types.discard(np.datetime64)
        for fm, to in TYPE_CONVERSIONS.items():
            if types and all(x == fm for x in types):
                return np.array(seq, to)
        # Let NumPy guess the appropriate dtype.
        return np.array(seq, dtype)

    @classmethod
    def _std_to_np_missing_value(cls, types):
        if str in types:
            return ""
        if all(x in [float, int] for x in types):
            return np.nan
        datetimes = [datetime.date, datetime.datetime, np.datetime64]
        if all(x in datetimes for x in types):
            return np.datetime64("NaT")
        # Usually causes dtype to be object!
        return None

    def tail(self, n=None):
        if n is None:
            n = dataiter.DEFAULT_PEEK_ELEMENTS
        n = min(self.length, n)
        return self[np.arange(self.length - n, self.length)].copy()

    def to_string(self, max_elements=None):
        def add_string_element(string, rows):
            if len(rows[-1]) <= 1:
                return rows[-1].append(string)
            row = " ".join(rows[-1] + [string])
            if len(row) < dataiter.PRINT_MAX_WIDTH:
                return rows[-1].append(string)
            # Start a new row with padding and string.
            return rows.append([" ", string])
        if max_elements is None:
            max_elements = dataiter.PRINT_MAX_ELEMENTS
        rows = [["["]]
        for string in self[:max_elements].to_strings(pad=True):
            add_string_element(string, rows)
        if max_elements < self.length:
            add_string_element("...", rows)
        add_string_element("]", rows)
        return "\n".join(" ".join(x) for x in rows)

    def to_strings(self, quote=True, pad=False):
        if self.length == 0:
            return self.__class__.fast([], str)
        identity = lambda x, *args, **kwargs: x
        quote = util.quote if quote else identity
        pad = util.pad if pad else identity
        if self.is_float:
            strings = util.format_floats(self)
            return self.__class__.fast(pad(strings), str)
        if self.is_integer:
            strings = ["{:d}".format(x) for x in self]
            return self.__class__.fast(pad(strings), str)
        if self.is_string:
            strings = [quote(x) for x in self]
            return self.__class__.fast(pad(strings), str)
        strings = [str(x) for x in self]
        return self.__class__.fast(pad(strings), str)

    def tolist(self):
        return np.where(self.is_missing(), None, self).tolist()

    def unique(self):
        u, indices = np.unique(self, return_index=True)
        return self[indices.sort()].copy()
