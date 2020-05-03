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

import datetime
import numpy as np

from dataiter import util

TYPE_CONVERSIONS = {
    datetime.date: "datetime64[D]",
    datetime.datetime: "datetime64[us]",
}


class Array(np.ndarray):

    def __new__(cls, object, dtype=None):
        # If given a NumPy array, we can do a fast initialization.
        if isinstance(object, np.ndarray):
            dtype = dtype or object.dtype
            return np.array(object, dtype).view(cls)
        # If given a Python list, or something else generic, we need
        # to convert certain types and special values. This is really
        # slow, see Array.fast for faster initialization.
        object = [object] if np.isscalar(object) else object
        return cls._std_to_np(object, dtype).view(cls)

    def __array_wrap__(self, array, context=None):
        # Avoid returning 0-dimensional arrays.
        # https://github.com/numpy/numpy/issues/7403
        return array[()] if array.shape == () else array

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return util.np_to_string(self)

    def as_boolean(self):
        return self.astype(bool)

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

    def equal(self, other):
        if self.is_datetime and other.is_datetime:
            return self._equal_missing(other, np.isnat)
        if self.is_float and other.is_float:
            return self._equal_missing(other, np.isnan)
        return np.array_equal(self, other)

    def _equal_missing(self, other, isna):
        ii, jj = map(isna, (self, other))
        return (np.all(ii == jj) and
                np.all(self[~ii] == other[~jj]))

    @classmethod
    def fast(cls, object, dtype=None):
        return np.array(object, dtype).view(cls)

    @property
    def is_boolean(self):
        return np.issubdtype(self.dtype, np.bool_)

    @property
    def is_datetime(self):
        return np.issubdtype(self.dtype, np.datetime64)

    @property
    def is_float(self):
        return np.issubdtype(self.dtype, np.floating)

    @property
    def is_integer(self):
        return np.issubdtype(self.dtype, np.integer)

    @property
    def is_number(self):
        return np.issubdtype(self.dtype, np.number)

    @property
    def is_object(self):
        return np.issubdtype(self.dtype, np.object_)

    @property
    def is_string(self):
        return np.issubdtype(self.dtype, np.character)

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
        # Note that using None, e.g. for a boolean array,
        # might not work directly as it requires upcasting to object.
        return None

    def rank(self):
        rank = np.unique(self, return_inverse=True)[1]
        return rank.view(self.__class__)

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

    def tolist(self):
        def replace(isna):
            return np.where(isna(self), None, self).tolist()
        if self.is_datetime:
            return replace(np.isnat)
        if self.is_float:
            return replace(np.isnan)
        if self.is_string:
            return replace(lambda x: x == "")
        return super().tolist()
