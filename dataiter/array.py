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

from dataiter import deco
from dataiter import util

TYPE_CONVERSIONS = {
    datetime.date: "datetime64[D]",
    datetime.datetime: "datetime64[us]",
}


class Array(np.ndarray):

    def __new__(cls, object, dtype=None):
        object = [object] if np.isscalar(object) else object
        if not isinstance(object, np.ndarray):
            object = list(cls._std_to_np(object))
        if dtype is None:
            types = util.unique_types(object)
            for fm, to in TYPE_CONVERSIONS.items():
                if types and all(x in [fm] for x in types):
                    dtype = to
        return np.array(object, dtype).view(cls)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return util.np_to_string(self)

    def equal(self, other):
        # XXX: Not really sure what we want to consider equal.
        if self.is_float and other.is_float:
            return np.allclose(self, other, equal_nan=True)
        if self.is_datetime and other.is_datetime:
            na1 = np.isnat(self)
            na2 = np.isnat(other)
            return np.all(na1 == na2) and np.all(self[~na1] == other[~na2])
        return np.array_equal(self, other)

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
        return np.issubdtype(self.dtype, np.dtype("object"))

    @property
    def is_string(self):
        return np.issubdtype(self.dtype, np.character)

    @property
    def missing_value(self):
        if self.is_datetime:
            return np.datetime64("NaT")
        if self.is_float:
            return np.nan
        if self.is_integer:
            return np.nan
        if self.is_string:
            return ""
        return None

    @classmethod
    def _std_to_np(cls, seq):
        # Convert special values in seq to NumPy equivalents.
        types = set(type(x) for x in seq if not util.is_missing(x))
        if str in types:
            missing = ""
        elif all(x in [float, int] for x in types):
            missing = np.nan
        elif all(x in [datetime.date, datetime.datetime] for x in types):
            missing = np.datetime64("NaT")
        else:
            missing = None
        for item in seq:
            if util.is_missing(item):
                item = missing
            yield item

    def rank(self):
        rank = np.unique(self, return_inverse=True)[1]
        return rank.view(self.__class__)

    @deco.listify
    def tolist(self):
        for item in super().tolist():
            if util.is_missing(item):
                item = None
            yield item
