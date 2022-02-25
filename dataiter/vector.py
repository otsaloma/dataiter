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
import sys

from dataiter import util
from math import inf

TYPE_CONVERSIONS = {
    datetime.date: "datetime64[D]",
    datetime.datetime: "datetime64[us]",
}


class Vector(np.ndarray):

    """
    A one-dimensional array.

    Vector is a subclass of NumPy ``ndarray``. Note that not all ``ndarray``
    methods have been overridden and thus by careless use of baseclass in-place
    methods you might manage to twist the data into multi-dimensional or other
    non-vector form, causing unexpected results.

    https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html
    """

    def __new__(cls, object, dtype=None):
        # If given a NumPy array, we can do a fast initialization.
        if isinstance(object, np.ndarray):
            dtype = dtype or object.dtype
            return np.array(object, dtype).view(cls)
        # If given a Python list, or something else generic, we need
        # to convert certain types and special values. This is really
        # slow, see Vector.fast for faster initialization.
        if (hasattr(object, "__iter__") and
            not isinstance(object, (list, tuple))):
            # Evaluate generator/iterator.
            object = list(object)
        return cls._std_to_np(object, dtype).view(cls)

    def __init__(self, object, dtype=None):
        """
        Return a new vector.

        `object` can be any one-dimensional sequence, such as a NumPy array,
        Python list or tuple. Creating a vector from a NumPy array will be
        fast, from other types slower as data types and special values will
        need to be converted.

        `dtype` is the NumPy-compatible data type for the vector. Providing
        `dtype` will make creating the vector faster, otherwise the appropriate
        data type will be guessed by introspecting the elements of `object`,
        which is potentially slow, especially for large objects.

        >>> di.Vector([1, 2, 3], int)
        """
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
        """
        Return vector converted to boolean data type.

        >>> vector = di.Vector([0, 1])
        >>> vector.as_boolean()
        """
        if self.is_string():
            # NumPy does bool(int(str)), which is weird.
            # https://github.com/numpy/numpy/issues/20898
            # https://github.com/numpy/numpy/pull/21024
            return self.map(bool)
        return self.astype(bool)

    def as_bytes(self):
        """
        Return vector converted to bytes data type.

        >>> vector = di.Vector(["a", "b"])
        >>> vector.as_bytes()
        """
        if self.is_string():
            array = np.char.encode(self, "utf-8")
            return array.view(self.__class__)
        return self.astype(bytes)

    def as_date(self):
        """
        Return vector converted to date data type.

        >>> vector = di.Vector(["2020-01-01"])
        >>> vector.as_date()
        """
        return self.astype(np.dtype("datetime64[D]"))

    def as_datetime(self):
        """
        Return vector converted to datetime data type.

        >>> vector = di.Vector(["2020-01-01T12:00:00"])
        >>> vector.as_datetime()
        """
        return self.astype(np.dtype("datetime64[us]"))

    def as_float(self):
        """
        Return vector converted to float data type.

        >>> vector = di.Vector([1, 2, 3])
        >>> vector.as_float()
        """
        return self.astype(float)

    def as_integer(self):
        """
        Return vector converted to integer data type.

        >>> vector = di.Vector([1.0, 2.0, 3.0])
        >>> vector.as_integer()
        """
        return self.astype(int)

    def as_object(self):
        """
        Return vector converted to object data type.

        >>> vector = di.Vector([1, 2, 3])
        >>> vector.as_object()
        """
        return self.__class__(self.tolist(), object)

    def as_string(self, length=None):
        """
        Return vector converted to string data type.

        >>> vector = di.Vector([1, 2, 3])
        >>> vector.as_string()
        >>> vector.as_string(64)
        """
        return self.astype(f"U{length}" if length else str)

    def _check_dimensions(self):
        if self.ndim == 1: return
        raise ValueError(f"Bad dimensions: {self.ndim!r}")

    def drop_na(self):
        """
        Return vector without missing values.

        >>> vector = di.Vector([1, 2, 3, None])
        >>> vector.drop_na()
        """
        return self[~self.is_na()].copy()

    def equal(self, other):
        """
        Return whether vectors are equal.

        Equality is tested with ``==``. As an exception, corresponding missing
        values are considered equal as well.

        >>> a = di.Vector([1, 2, 3, None])
        >>> b = di.Vector([1, 2, 3, None])
        >>> a
        >>> b
        >>> a.equal(b)
        """
        if not (isinstance(other, Vector) and
                self.length == other.length and
                str(self.na_value) == str(other.na_value)):
            return False
        ii = self.is_na()
        jj = other.is_na()
        return (np.all(ii == jj) and
                np.all(self[~ii] == other[~jj]))

    @classmethod
    def fast(cls, object, dtype=None):
        """
        Return a new vector.

        Unlike :meth:`__init__`, this will **not** convert special values in
        `object`. Use this only if you know `object` doesn't contain special
        values or if you know they are already of the correct type.
        """
        if (hasattr(object, "__iter__") and
            not isinstance(object, (np.ndarray, list, tuple))):
            # Evaluate generator/iterator.
            object = list(object)
        return np.array(object, dtype).view(cls)

    def get_memory_use(self):
        """
        Return memory use in bytes.

        >>> vector = di.Vector(range(100))
        >>> vector.get_memory_use()
        """
        if self.is_object():
            return sum(sys.getsizeof(x) for x in self)
        return self.nbytes

    def head(self, n=None):
        """
        Return the first `n` elements.

        >>> vector = di.Vector(range(100))
        >>> vector.head(10)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ELEMENTS
        n = min(self.length, n)
        return self[np.arange(n)].copy()

    def is_boolean(self):
        """
        Return whether vector data type is boolean.
        """
        return np.issubdtype(self.dtype, np.bool_)

    def is_bytes(self):
        """
        Return whether vector data type is bytes.
        """
        return np.issubdtype(self.dtype, np.bytes_)

    def is_datetime(self):
        """
        Return whether vector data type is datetime.

        Dates are considered datetimes as well.
        """
        return np.issubdtype(self.dtype, np.datetime64)

    def is_float(self):
        """
        Return whether vector data type is float.
        """
        return np.issubdtype(self.dtype, np.floating)

    def is_integer(self):
        """
        Return whether vector data type is integer.
        """
        return np.issubdtype(self.dtype, np.integer)

    def is_na(self):
        """
        Return a boolean vector indicating missing data elements.

        >>> vector = di.Vector([1, 2, 3, None])
        >>> vector
        >>> vector.is_na()
        """
        if self.is_datetime():
            return np.isnat(self)
        if self.is_float():
            return np.isnan(self)
        if self.is_string():
            return self == ""
        return np.isin(self, [None])

    def is_number(self):
        """
        Return whether vector data type is number.
        """
        return np.issubdtype(self.dtype, np.number)

    def is_object(self):
        """
        Return whether vector data type is object.
        """
        return np.issubdtype(self.dtype, np.object_)

    def is_string(self):
        """
        Return whether vector data type is string.
        """
        return np.issubdtype(self.dtype, np.unicode_)

    @property
    def length(self):
        """
        Return the amount of elements.

        >>> vector = di.Vector(range(100))
        >>> vector.length
        """
        self._check_dimensions()
        return self.size

    def map(self, function, *args, **kwargs):
        """
        Apply `function` element-wise and return a new vector.

        >>> import math
        >>> vector = di.Vector(range(10))
        >>> vector.map(math.pow, 2)
        """
        return self.__class__(function(x, *args, **kwargs) for x in self)

    @property
    def na_dtype(self):
        """
        Return the corresponding data type that can handle missing data.

        You might need this for upcasting when missing data is first introduced.

        >>> vector = di.Vector([1, 2, 3])
        >>> vector
        >>> vector.put([2], vector.na_value)
        >>> vector = vector.astype(vector.na_dtype)
        >>> vector
        >>> vector.put([2], vector.na_value)
        >>> vector
        """
        if self.is_datetime():
            return self.dtype
        if self.is_float():
            return self.dtype
        if self.is_integer():
            return float
        if self.is_string():
            return self.dtype
        return object

    @property
    def na_value(self):
        """
        Return the corresponding value to use to represent missing data.

        Dataiter is built on top of NumPy. NumPy doesn't support a proper
        missing value ("NA"), only two data type specific values: ``np.nan``
        and ``np.datetime64("NaT")``. Dataiter recommends the following values
        be used and internally supports them to an extent.

        ======== ========================
        datetime ``np.datetime64("NaT")``
        float    ``np.nan``
        integer  ``np.nan``
        string   ``""``
        other    ``None``
        ======== ========================

        Note that actually using these might require upcasting the vector.
        Integer will need to be upcast to float to contain ``np.nan``. Other,
        such as boolean, will need to be upcast to object to contain ``None``.

        If you need to avoid object columns, you can also consider converting
        booleans to float using :meth:`as_float`, which will give you 0.0 for
        false and 1.0 for true. Depending on how you use the data, that might
        work as well as an object vector of ``True``, ``False`` and ``None``.
        """
        if self.is_datetime():
            return np.datetime64("NaT")
        if self.is_float():
            return np.nan
        if self.is_integer():
            return np.nan
        if self.is_string():
            return ""
        # Note that using None, e.g. for a boolean vector,
        # might not work directly as it requires upcasting to object.
        return None

    def range(self):
        """
        Return the minimum and maximum values as a two-element vector.

        >>> vector = di.Vector(range(100))
        >>> vector.range()
        """
        rng = [np.nanmin(self), np.nanmax(self)]
        return self.__class__(rng, self.dtype)

    def rank(self, *, method="average"):
        """
        Return the order of elements in a sorted vector.

        `method` determines how ties are resolved. **'min'** assigns each of
        equal values the same rank, the minimum of the set (also called
        "competition ranking"). **'max'** is the same, but assigning the
        maximum of the set. **'average'** is the mean of 'min' and 'max'.
        **'ordinal'** gives each element a distinct rank with equal values
        ranked by their order in input.

        Ranks begin at 1. Missing values are ranked last.

        **References**

        * https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
        * https://www.rdocumentation.org/packages/base/topics/rank

        >>> vector = di.Vector([3, 1, 1, 1, 2, 2])
        >>> vector.rank(method="min")
        >>> vector.rank(method="max")
        >>> vector.rank(method="average")
        >>> vector.rank(method="ordinal")
        """
        if method not in ["min", "max", "average", "ordinal"]:
            raise ValueError(f"Unexpected method: {method!r}")
        na = self.is_na()
        if method == "average":
            rank_min = self.rank(method="min")
            rank_max = self.rank(method="max")
            rank = np.mean([rank_min, rank_max], axis=0)
            return self.__class__(rank)
        if method == "min":
            # https://stackoverflow.com/a/14672797/16369038
            inv = np.unique(self[~na], return_inverse=True)[1]
            arank = np.concatenate(([0], np.bincount(inv))).cumsum()[inv]
            zrank = arank.max() + 1
        if method == "max":
            # https://stackoverflow.com/a/14672797/16369038
            inv = np.unique(self[~na], return_inverse=True)[1]
            arank = np.bincount(inv).cumsum()[inv] - 1
            zrank = len(self) - 1
        if method == "ordinal":
            # https://stackoverflow.com/a/5284703/16369038
            indices = self[~na].argsort()
            arank = np.empty_like(indices)
            arank[indices] = np.arange(len(indices))
            zrank = arank.max() + 1 + np.arange(na.sum())
        out = np.zeros_like(self, int)
        out[~na] = arank + 1
        out[na] = zrank + 1
        return self.__class__(out)

    def replace_na(self, value):
        """
        Return vector with missing values replaced with `value`.

        >>> vector = di.Vector([1, 2, 3, None])
        >>> vector.replace_na(0)
        """
        vector = self.copy()
        vector[vector.is_na()] = value
        return vector

    def sample(self, n=None):
        """
        Return randomly chosen `n` elements.

        >>> vector = di.Vector(range(100))
        >>> vector.sample(10)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ELEMENTS
        n = min(self.length, n)
        indices = np.random.choice(self.length, n, replace=False)
        return self[np.sort(indices)].copy()

    def sort(self, *, dir=1):
        """
        Return elements in sorted order.

        `dir` is ``1`` for ascending sort, ``-1`` for descending.

        Missing values are sorted last, regardless of `dir`.

        >>> vector = di.Vector([1, 2, 3, None])
        >>> vector.sort(dir=1)
        >>> vector.sort(dir=-1)
        """
        na = self.is_na()
        a = self[~na].copy()
        z = self[na].copy()
        np.ndarray.sort(a)
        if dir < 0:
            a = a[::-1]
        vector = np.concatenate((a, z))
        return self.__class__(vector)

    @classmethod
    def _std_to_np(cls, seq, dtype=None):
        # Convert missing values in seq to NumPy equivalents.
        # Can be empty if all of seq are missing values.
        types = util.unique_types(seq)
        if dtype is not None:
            na = Vector.fast([], dtype).na_value
        elif len(types) == 1 and types.copy().pop().__module__ == "numpy":
            # If we have a regular Python list of NumPy scalars,
            # infer type. This should be rare, but can happen.
            dtype = types.copy().pop()().dtype
            na = Vector.fast([], dtype).na_value
        else:
            # Guess the missing value based on types in seq.
            na = cls._std_to_np_na_value(types)
        seq = [na if
               x is None or
               (isinstance(x, float) and np.isnan(x))
               else x for x in seq]
        if dtype is not None:
            if np.issubdtype(dtype, np.integer) and np.nan in seq:
                # Upcast from integer to float as required.
                dtype = float
            return np.array(seq, dtype)
        # NaT values bring in np.datetime64 to types.
        types.discard(np.datetime64)
        for fm, to in TYPE_CONVERSIONS.items():
            if types and all(x == fm for x in types):
                return np.array(seq, to)
        # Let NumPy guess the appropriate dtype.
        return np.array(seq, dtype)

    @classmethod
    def _std_to_np_na_value(cls, types):
        if not types:
            return None
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
        """
        Return the last `n` elements.

        >>> vector = di.Vector(range(100))
        >>> vector.tail(10)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ELEMENTS
        n = min(self.length, n)
        return self[np.arange(self.length - n, self.length)].copy()

    def to_string(self, *, max_elements=None):
        """
        Return vector as a string formatted for display.

        >>> vector = di.Vector([1/2, 1/3, 1/4])
        >>> vector.to_string()
        """
        print_width = util.get_print_width()
        def add_string_element(string, rows):
            if len(rows[-1]) <= 1:
                return rows[-1].append(string)
            row = " ".join(rows[-1] + [string])
            if len(row) < print_width:
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
        if len(rows) == 1:
            # Drop padding for single-line output.
            rows[0] = [x.strip() for x in rows[0]]
        rows.append([f"dtype: {self.dtype}"])
        return "\n".join(" ".join(x) for x in rows)

    def to_strings(self, *, quote=True, pad=False, truncate_width=inf):
        """
        Return vector as strings formatted for display.

        >>> vector = di.Vector([1/2, 1/3, 1/4])
        >>> vector.to_strings()
        """
        if self.length == 0:
            return self.__class__.fast([], str)
        identity = lambda x, *args, **kwargs: x
        quote = util.quote if quote else identity
        pad = util.pad if pad else identity
        if self.is_float():
            strings = util.format_floats(self)
            return self.__class__.fast(pad(strings), str)
        if self.is_integer():
            strings = ["{:d}".format(x) for x in self]
            return self.__class__.fast(pad(strings), str)
        if self.is_object():
            strings = [str(x) for x in self]
            for i in range(len(strings)):
                if len(strings[i]) > truncate_width:
                    strings[i] = strings[i][:(truncate_width-1)] + "…"
            return self.__class__.fast(pad(strings), str)
        if self.is_string():
            strings = [quote(x) for x in self]
            for i in range(len(strings)):
                if len(strings[i]) > truncate_width:
                    strings[i] = strings[i][:(truncate_width-1)] + "…"
            return self.__class__.fast(pad(strings), str)
        strings = [str(x) for x in self]
        return self.__class__.fast(pad(strings), str)

    def tolist(self):
        """
        Return vector as a list with elements of matching Python builtin type.

        Missing values are replaced with ``None``.
        """
        return np.where(self.is_na(), None, self).tolist()

    def unique(self):
        """
        Return unique elements.

        >>> vector = di.Vector([1, 1, 1, 2, 2, 3])
        >>> vector.unique()
        """
        u, indices = np.unique(self, return_index=True)
        return self[indices.sort()].copy()
