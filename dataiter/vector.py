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
import functools
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import random
import sys
import wcwidth

from dataiter import dtypes
from dataiter import util
from math import inf
from numpy.dtypes import StringDType

TYPE_CONVERSIONS = {
    datetime.date: "datetime64[D]",
    datetime.datetime: "datetime64[us]",
}

def as_vector(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        array = function(*args, **kwargs)
        return array.view(Vector)
    return wrapper

def not_implemented(*args, **kwargs):
    raise NotImplementedError

def of_type(*types):
    """
    Return a function that checks if a vector's data type is one of `types`.

    `types` are matched by kind as in the ``Vector.is_*`` methods, e.g.
    ``float`` matches all floats and ``np.datetime64`` both dates and
    datetimes. Use ``np.number`` to match all numbers.

    >>> is_float = di.of_type(float)
    >>> is_float(di.Vector([1.5, 2.5]))
    >>> is_float(di.Vector(["a", "b"]))
    """
    kinds = {
        bool: np.bool_,
        bytes: np.bytes_,
        float: np.floating,
        int: np.integer,
        object: np.object_,
    }
    types = [kinds.get(x, x) for x in types]
    def is_type(x):
        return any(isinstance(x.dtype, StringDType) if t is str
                   else np.issubdtype(x.dtype, t)
                   for t in types)

    return is_type

def str_partition(vector, function, sep):
    # np.strings.partition has no loop for StringDType
    # with a str separator, it needs to be StringDType too.
    sep = np.asarray(sep, vector.dtype)
    return tuple(x.view(Vector) for x in function(vector, sep))

def str_slice(vector, *args):
    # np.strings.slice raises for missing values, i.e. empty strings
    # with our na_object, so slice only the others.
    na = vector.is_na()
    if not na.any():
        return np.strings.slice(vector, *args)
    out = np.array(vector, copy=True)
    args = [x[~na] if np.ndim(x) > 0 else x for x in args]
    out[~na] = np.strings.slice(out[~na], *args)
    return out

class DtProxy:
    def __init__(self, vector):
        from dataiter import dt
        wrap = lambda f: functools.partial(f, vector)
        self.day = wrap(dt.day)
        self.floor = wrap(dt.floor)
        self.from_string = wrap(dt.from_string)
        self.hour = wrap(dt.hour)
        self.isoformat = wrap(dt.isoformat)
        self.isoweek = wrap(dt.isoweek)
        self.isoweekday = wrap(dt.isoweekday)
        self.microsecond = wrap(dt.microsecond)
        self.minute = wrap(dt.minute)
        self.month = wrap(dt.month)
        self.new = wrap(dt.new)
        self.quarter = wrap(dt.quarter)
        self.replace = wrap(dt.replace)
        self.second = wrap(dt.second)
        self.to_string = wrap(dt.to_string)
        self.total_seconds = wrap(dt.total_seconds)
        self.weekday = wrap(dt.weekday)
        self.year = wrap(dt.year)

class ReProxy:
    def __init__(self, vector):
        from dataiter import regex
        wrap = lambda f: functools.partial(f, string=vector)
        self.extract = wrap(regex.extract)
        self.findall = wrap(regex.findall)
        self.fullmatch = wrap(regex.fullmatch)
        self.match = wrap(regex.match)
        self.search = wrap(regex.search)
        self.split = wrap(regex.split)
        self.sub = wrap(regex.sub)
        self.subn = wrap(regex.subn)

class StrProxy:
    def __init__(self, vector):
        wrap = lambda name: as_vector(functools.partial(
            getattr(np.strings, name, not_implemented),
            vector))

        # https://numpy.org/doc/stable/reference/routines.strings.html
        self.add = wrap("add")
        self.capitalize = wrap("capitalize")
        self.center = wrap("center")
        self.contains = lambda x: self.find(x) > -1
        self.count = wrap("count")
        self.decode = wrap("decode")
        self.encode = wrap("encode")
        self.endswith = wrap("endswith")
        self.equal = wrap("equal")
        self.expandtabs = wrap("expandtabs")
        self.find = wrap("find")
        self.greater = wrap("greater")
        self.greater_equal = wrap("greater_equal")
        self.index = wrap("index")
        self.isalnum = wrap("isalnum")
        self.isalpha = wrap("isalpha")
        self.isdecimal = wrap("isdecimal")
        self.isdigit = wrap("isdigit")
        self.islower = wrap("islower")
        self.isnumeric = wrap("isnumeric")
        self.isspace = wrap("isspace")
        self.istitle = wrap("istitle")
        self.isupper = wrap("isupper")
        self.less = wrap("less")
        self.less_equal = wrap("less_equal")
        self.ljust = wrap("ljust")
        self.lower = wrap("lower")
        self.lstrip = wrap("lstrip")
        self.mod = wrap("mod")
        self.multiply = wrap("multiply")
        self.not_equal = wrap("not_equal")
        self.partition = functools.partial(str_partition, vector, np.strings.partition)
        self.replace = wrap("replace")
        self.rfind = wrap("rfind")
        self.rindex = wrap("rindex")
        self.rjust = wrap("rjust")
        self.rpartition = functools.partial(str_partition, vector, np.strings.rpartition)
        self.rstrip = wrap("rstrip")
        self.slice = as_vector(functools.partial(str_slice, vector))
        self.startswith = wrap("startswith")
        self.str_len = wrap("str_len")
        self.strip = wrap("strip")
        self.swapcase = wrap("swapcase")
        self.title = wrap("title")
        self.translate = wrap("translate")
        self.upper = wrap("upper")
        self.zfill = wrap("zfill")

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
        dtype = cls._map_input_dtype(dtype)
        # If given a NumPy array, we can do a fast initialization.
        if isinstance(object, np.ndarray):
            return cls._np_array(object, dtype).view(cls)
        # If given a Python list, or something else generic, we need
        # to convert certain types and special values. This is really
        # slow, see Vector.fast for faster initialization.
        object = util.sequencify(object)
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
        # Note that when we call array.view(Vector) for a numpy.ndarray,
        # this will not be called and no attributes added here will exist!
        self._check_dimensions()

    def __array_wrap__(self, array, context=None, return_scalar=False):
        # Avoid returning 0-dimensional arrays.
        # https://github.com/numpy/numpy/issues/7403
        if not array.shape or return_scalar:
            return array.dtype.type(array)
        return array.view(self.__class__)

    def __eq__(self, other):
        return super().__eq__(self._coerce_datetime(other))

    def __ge__(self, other):
        return super().__ge__(self._coerce_datetime(other))

    def __gt__(self, other):
        return super().__gt__(self._coerce_datetime(other))

    def __iter__(self):
        # NumPy iterates subclasses ~2x slower than base arrays.
        return iter(self.view(np.ndarray))

    def __le__(self, other):
        return super().__le__(self._coerce_datetime(other))

    def __lt__(self, other):
        return super().__lt__(self._coerce_datetime(other))

    def __ne__(self, other):
        return super().__ne__(self._coerce_datetime(other))

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    def as_boolean(self):
        """
        Return vector converted to boolean data type.

        Raise ``ValueError`` if the vector contains missing values, since
        booleans can't represent them. Use :meth:`replace_na` or
        :meth:`drop_na` first to handle them.

        >>> vector = di.Vector([0, 1])
        >>> vector.as_boolean()
        """
        if self.is_na().any():
            raise ValueError("Can't convert missing values to boolean, use replace_na or drop_na first")
        return self.astype(bool)

    def as_bytes(self):
        """
        Return vector converted to bytes data type.

        >>> vector = di.Vector(["a", "b"])
        >>> vector.as_bytes()
        """
        # Missing strings encode to b"", the missing value of bytes.
        return self.as_string().str.encode("utf-8")

    def as_date(self):
        """
        Return vector converted to date data type.

        >>> vector = di.Vector(["2020-01-01"])
        >>> vector.as_date()
        """
        return self.astype(np.dtype("datetime64[D]"))

    def as_datetime(self, precision="us"):
        """
        Return vector converted to datetime data type.

        >>> vector = di.Vector(["2020-01-01T12:00:00"])
        >>> vector.as_datetime()
        """
        return self.astype(np.dtype(f"datetime64[{precision}]"))

    def as_float(self, *, errors="strict"):
        """
        Return vector converted to float data type.

        `errors` determines how to handle values that can't be converted,
        such as non-numeric strings: ``"strict"`` to raise ``ValueError``,
        ``"replace"`` to replace them with missing values.

        >>> vector = di.Vector([1, 2, 3])
        >>> vector.as_float()
        >>> vector = di.Vector(["1", "2.5", "???"])
        >>> vector.as_float(errors="replace")
        """
        if errors not in ["strict", "replace"]:
            raise ValueError(f"Bad {errors=}")
        na = self.is_na()
        result = np.full(self.length, np.nan).view(self.__class__)
        try:
            # NumPy can't convert missing strings to float
            # and converts NaT to its integer sentinel value.
            result[~na] = self[~na].astype(float)
        except (TypeError, ValueError):
            if errors == "strict": raise
            def parse(x):
                try:
                    return float(x)
                except (TypeError, ValueError):
                    return np.nan
            result[~na] = [parse(x) for x in self[~na].tolist()]
        return result

    def as_integer(self):
        """
        Return vector converted to integer data type.

        Raise ``ValueError`` if the vector contains missing values, since
        integers can't represent them. Use :meth:`replace_na` or
        :meth:`drop_na` first to handle them.

        >>> vector = di.Vector([1.0, 2.0, 3.0])
        >>> vector.as_integer()
        """
        if self.is_na().any():
            raise ValueError("Can't convert missing values to integer, use replace_na or drop_na first")
        return self.astype(int)

    def as_object(self):
        """
        Return vector converted to object data type.

        >>> vector = di.Vector([1, 2, 3])
        >>> vector.as_object()
        """
        return self.__class__(self.tolist(), object)

    def as_string(self):
        """
        Return vector converted to string data type.

        >>> vector = di.Vector([1, 2, 3])
        >>> vector.as_string()
        """
        na = self.is_na()
        if na.any():
            # NumPy converts e.g. NaN and None to "nan" and "None".
            result = np.full(self.length, dtypes.string.na_object, dtypes.string).view(self.__class__)
            result[~na] = self[~na].astype(dtypes.string)
            return result
        return self.astype(dtypes.string)

    def _check_dimensions(self):
        if self.ndim == 1: return
        raise ValueError(f"Bad dimensions: {self.ndim!r}")

    def _coerce_datetime(self, other):
        # NumPy considers datetimes and strings unequal or raises
        # for ordering, convert strings to allow e.g. x >= "2025-01-01".
        if isinstance(other, str) and self.is_datetime():
            return np.datetime64(other)
        return other

    def concat(self, *others):
        """
        Return vector with elements from `others` appended.

        >>> a = di.Vector([1, 2, 3])
        >>> b = di.Vector([4, 5, 6])
        >>> c = di.Vector([7, 8, 9])
        >>> a.concat(b, c)
        """
        vectors = [self] + list(others)
        new = np.concatenate(vectors)
        return self.__class__(new)

    def drop_na(self):
        """
        Return vector without missing values.

        >>> vector = di.Vector([1, 2, 3, None])
        >>> vector.drop_na()
        """
        return self[~self.is_na()].copy()

    @property
    def dt(self) -> DtProxy:
        """
        Proxy object for calling :mod:`dataiter.dt` functions.

        >>> x = di.Vector(["2025-01-11"], np.datetime64)
        >>> x.dt.year()
        >>> x.dt.month()
        >>> x.dt.day()
        """
        if not hasattr(self, "_dt"):
            self._dt = DtProxy(self)
        return self._dt

    @property
    def dtype_label(self):
        """
        Return a human-readable label of vector data type.

        >>> vector = di.Vector(["abc", "def"])
        >>> vector.dtype
        >>> vector.dtype_label
        """
        if self.is_string():
            return "string"
        return str(self.dtype)

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
        dtype = cls._map_input_dtype(dtype)
        object = util.sequencify(object)
        return cls._np_array(object, dtype).view(cls)

    def get_memory_use(self):
        """
        Return memory use in bytes.

        >>> vector = di.Vector(range(100))
        >>> vector.get_memory_use()
        """
        if self.is_object():
            return sum(sys.getsizeof(x) for x in self)
        if self.is_string():
            # StringDType stores longer strings outside the array, not counted
            # in nbytes. Adding string lengths gives a rough estimate.
            return self.nbytes + int(self.str.str_len().sum())
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

    def is_in(self, values):
        """
        Return a boolean vector indicating elements found in `values`.

        `values` can be any kind of a sequence or iterable, such as a list, a
        set or a vector.

        >>> vector = di.Vector(["a", "b", "c", "d"])
        >>> vector.is_in(["a", "c"])
        """
        if util.is_scalar(values):
            raise TypeError(f"Expected a sequence of values, got {type(values).__name__}, use == for a single value")
        if not isinstance(values, np.ndarray):
            # np.isin treats sets and other non-sequences as a single element.
            values = list(values)
            if self.is_datetime():
                values = [self._coerce_datetime(x) for x in values]
        if self.is_string() and len(values) >= 10:
            # StringDType has hasobject set, so np.isin never sorts,
            # but compares one value at a time, slow for many values.
            # Hash values instead, with a set for small inputs, as it beats
            # Arrow's per-call overhead, e.g. many groups. Non-strings can't
            # match, so drop them for Arrow.
            if self.length < 1000:
                values = set(values)
                return np.fromiter((x in values for x in self), bool, self.length).view(self.__class__)
            values = pa.array([x for x in values if isinstance(x, str)], pa.large_string())
            found = pc.is_in(pa.array(np.asarray(self), pa.large_string()), value_set=values)
            return found.to_numpy(zero_copy_only=False).view(self.__class__)
        return np.isin(self, values).view(self.__class__)

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
        if self.is_timedelta():
            return np.isnat(self)
        if self.is_float():
            return np.isnan(self)
        if self.is_string() or self._is_string_fixed():
            return self == dtypes.string.na_object
        if self.is_bytes():
            return self == b""
        if self.is_integer() or self.is_boolean():
            return np.zeros(len(self), bool).view(self.__class__)
        # Can't use np.isin here since elements can be arrays.
        return self.fast([x is None for x in self], bool)

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
        return isinstance(self.dtype, StringDType)

    def _is_string_fixed(self):
        # Old-style fixed-width string type
        return np.issubdtype(self.dtype, np.str_)

    def is_timedelta(self):
        """
        Return whether vector data type is timedelta.
        """
        return np.issubdtype(self.dtype, np.timedelta64)

    @property
    def length(self):
        """
        Return the amount of elements.

        >>> vector = di.Vector(range(100))
        >>> vector.length
        """
        self._check_dimensions()
        return self.size

    def map(self, function, *args, dtype=None, **kwargs):
        """
        Apply `function` element-wise and return a new vector.

        `args` and `kwargs` are passed to `function`. To recode values with
        a mapping, use its ``get`` method, optionally with a default.

        >>> import math
        >>> vector = di.Vector(range(10))
        >>> vector.map(math.pow, 2)
        >>> # Recode using a lookup dict
        >>> vector = di.Vector(["a", "b", "c"])
        >>> vector.map({"a": 1, "b": 2}.get)
        >>> vector.map({"a": 1, "b": 2}.get, 0)
        """
        dtype = self._map_input_dtype(dtype)
        return self.__class__((function(x, *args, **kwargs) for x in self), dtype)

    @classmethod
    def _map_input_dtype(cls, dtype):
        if dtype is str:
            return dtypes.string
        return dtype

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
        if self.is_timedelta():
            return self.dtype
        if self.is_float():
            return self.dtype
        if self.is_integer():
            return float
        if self.is_string() or self._is_string_fixed():
            return self.dtype
        if self.is_bytes():
            return self.dtype
        return object

    @property
    def na_value(self):
        """
        Return the corresponding value to use to represent missing data.

        Dataiter is built on top of NumPy. NumPy doesn't support a proper
        missing value ("NA"), only data type specific values: ``np.nan``,
        ``np.datetime64("NaT")`` and ``np.timedelta64("NaT")``. Dataiter
        recommends the following values be used and internally supports them to
        an extent.

        ========= ========================
        bytes     ``b""``
        datetime  ``np.datetime64("NaT")``
        float     ``np.nan``
        integer   ``np.nan``
        string    ``""``
        timedelta ``np.timedelta64("NaT")``
        other     ``None``
        ========= ========================

        Note that actually using these might require upcasting the vector.
        Integer will need to be upcast to float to contain ``np.nan``. Other,
        such as boolean, will need to be upcast to object to contain ``None``.

        If you need to avoid object columns, you can also consider converting
        booleans to float using :meth:`as_float`, which will give you 0.0 for
        false and 1.0 for true. Depending on how you use the data, that might
        work as well as an object vector of ``True``, ``False`` and ``None``.
        """
        if self.is_datetime() or self.is_timedelta():
            unit = np.datetime_data(self.dtype)
            # Use a coarse unit for generic dtype so that
            # a finer unit from actual data wins in upcasting.
            if unit[0] == "generic":
                unit = "D"
            return self.dtype.type("NaT", unit)
        if self.is_float():
            return np.nan
        if self.is_integer():
            return np.nan
        if self.is_string() or self._is_string_fixed():
            return dtypes.string.na_object
        if self.is_bytes():
            return b""
        # Note that using None, e.g. for a boolean vector,
        # might not work directly as it requires upcasting to object.
        return None

    @classmethod
    def _np_array(cls, object, dtype=None):
        # NumPy still defaults to fixed width strings.
        # In some cases we can only fix the dtype ex-post.
        if dtype is None and not isinstance(object, np.ndarray):
            if object and isinstance(object[0], str):
                dtype = dtypes.string
        if (dtype is not None and
            np.dtype(dtype) == np.object_ and
            not isinstance(object, np.ndarray)):
            # np.array would make a 2D array of equal-length sequences.
            return np.fromiter(object, np.object_, len(object))
        array = np.array(object, dtype)
        if dtype is None:
            if np.issubdtype(array.dtype, np.str_):
                array = array.astype(dtypes.string)
        return array

    def range(self):
        """
        Return the minimum and maximum values as a two-element vector.

        >>> vector = di.Vector(range(100))
        >>> vector.range()
        """
        x = self.drop_na()
        if x.length == 0:
            return self.fast([self.na_value, self.na_value], self.na_dtype)
        return self.fast([x.min(), x.max()], self.dtype)

    def rank(self, *, method="min", dir=1):
        """
        Return the order of elements in a sorted vector.

        `method` determines how ties are resolved. **'min'** assigns each of
        equal values the same rank, the minimum of the set (also called
        "competition ranking"). **'max'** is the same, but assigning the
        maximum of the set. **'ordinal'** gives each element a distinct rank
        with equal values ranked by their order in input. Ranks begin at 1.
        `dir` is ``1`` for ascending order, ``-1`` for descending. Missing
        values are ranked last, regardless of `dir`.

        **References**

        * https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
        * https://www.rdocumentation.org/packages/base/topics/rank

        >>> vector = di.Vector([3, 1, 1, 1, 2, 2])
        >>> vector.rank(method="min")
        >>> vector.rank(method="max")
        >>> vector.rank(method="ordinal")
        """
        if self.length == 0:
            return self.fast([], int)
        na = self.is_na()
        if na.all():
            # Avoid trying to evaluate min/max/mean of all NA.
            return np.repeat(1, self.length).view(self.__class__).rank(method=method, dir=dir)
        out = np.zeros_like(self, int)
        def codes():
            if self.is_string():
                # Hashing and sorting only unique values is faster
                # than np.unique, which sorts all strings.
                inv = util.sorted_codes(self[~na])
            else:
                # return_index=True makes np.unique use a stable sort, the default
                # quicksort can segfault for StringDType (NumPy 2.5.4).
                # https://github.com/numpy/numpy/issues/33089
                inv = np.unique(self[~na], return_index=True, return_inverse=True)[2]
            return inv if dir > 0 else inv.max() - inv
        if method == "min":
            # https://stackoverflow.com/a/14672797/16369038
            inv = codes()
            out[~na] = np.concatenate(([0], np.bincount(inv))).cumsum()[inv] + 1
            out[na] = (~na).sum() + 1
            return out.view(self.__class__)
        if method == "max":
            # https://stackoverflow.com/a/14672797/16369038
            inv = codes()
            out[~na] = np.bincount(inv).cumsum()[inv]
            out[na] = len(self)
            return out.view(self.__class__)
        if method == "ordinal":
            # https://stackoverflow.com/a/5284703/16369038
            # Sorting values directly is faster than via codes,
            # except for strings with values that repeat.
            direct = dir > 0 and not (self.is_string() and util.is_repetitive(self))
            keys = self[~na] if direct else codes()
            indices = keys.argsort(stable=True)
            rank = np.zeros_like(indices)
            rank[indices] = np.arange(len(indices)) + 1
            out[~na] = rank
            out[na] = rank.max() + np.arange(na.sum()) + 1
            return out.view(self.__class__)
        raise ValueError(f"Unexpected method: {method!r}")

    @property
    def re(self) -> ReProxy:
        """
        Proxy object for calling :mod:`dataiter.regex` functions.

        >>> x = di.Vector(["great", "fantastic"])
        >>> x.re.sub(r"$", r"!")
        """
        if not hasattr(self, "_re"):
            self._re = ReProxy(self)
        return self._re

    def replace_na(self, value):
        """
        Return vector with missing values replaced with `value`.

        >>> vector = di.Vector([1, 2, 3, None])
        >>> vector.replace_na(0)
        """
        vector = self.copy()
        na = vector.is_na()
        if na.any():
            vector[na] = value
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
        indices = sorted(random.sample(range(self.length), n))
        return self[indices].copy()

    def shift(self, n=1):
        """
        Return elements shifted by `n` positions.

        Positive `n` shifts later, negative earlier, i.e. ``shift(1)`` gives at
        each position the previous element and ``shift(-1)`` the next one.
        Positions left without an element will be missing.

        >>> vector = di.Vector([1, 2, 3, 4])
        >>> vector.shift(1)
        >>> vector.shift(-1)
        """
        k = min(abs(n), self.length)
        if k == 0:
            return self.copy()
        result = self.astype(self.na_dtype)
        if n > 0:
            result[k:] = self[:self.length - k]
            result[:k] = self.na_value
        else:
            result[:self.length - k] = self[k:]
            result[self.length - k:] = self.na_value
        return result

    def sort(self, *, dir=1, **kwargs):
        """
        sort(*, dir=1)

        Return elements in sorted order.

        `dir` is ``1`` for ascending sort, ``-1`` for descending. Missing
        values are sorted last, regardless of `dir`.

        >>> vector = di.Vector([1, 2, 3, None])
        >>> vector.sort(dir=1)
        >>> vector.sort(dir=-1)
        """
        if kwargs:
            # np.sort calls ndarray.sort with these arguments.
            raise TypeError(f"Unexpected arguments {list(kwargs)}, "
                            "use vector.sort() instead of np.sort(vector)")
        if self.is_object():
            # Let's compare objects via str(x).
            lst = sorted(self, key=str, reverse=dir<0)
            new = self.fast(lst, object)
            na = new.is_na()
            return new[~na].concat(new[na])
        if self.is_string() and util.is_repetitive(self):
            # Sorting only unique values and repeating each by its count
            # is faster than sorting all strings if values repeat.
            unique, _, codes = util.unique_codes(self)
            order = np.argsort(unique, stable=True)
            counts = np.bincount(codes, minlength=len(unique))
            new = np.repeat(unique[order], counts[order]).view(self.__class__)
        else:
            new = self.copy()
            np.ndarray.sort(new, stable=True)
        if dir < 0:
            new = new[::-1]
        na = new.is_na()
        return new[~na].concat(new[na])

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
            dtype = np.dtype(types.copy().pop())
            na = Vector.fast([], dtype).na_value
        elif not types and any(isinstance(x, float) for x in seq):
            # All missing, but NaN is a float, unlike None.
            na = np.nan
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
            return cls._np_array(seq, dtype)
        # NaT values bring in np.datetime64 to types.
        types.discard(np.datetime64)
        for fm, to in TYPE_CONVERSIONS.items():
            if types and all(x == fm for x in types):
                return cls._np_array(seq, to)
        if any(issubclass(x, (list, tuple)) for x in types):
            return cls._np_array(seq, object)
        # Let NumPy guess the appropriate dtype.
        return cls._np_array(seq, dtype)

    @classmethod
    def _std_to_np_na_value(cls, types):
        if not types:
            return None
        if str in types:
            return dtypes.string.na_object
        if bytes in types:
            return b""
        if all(x in [float, int] or
               np.issubdtype(x, np.floating) or
               np.issubdtype(x, np.integer)
               for x in types):
            return np.nan
        datetimes = [datetime.date, datetime.datetime, np.datetime64]
        if all(x in datetimes for x in types):
            return np.datetime64("NaT", "D")
        # Usually causes dtype to be object!
        return None

    @property
    def str(self) -> StrProxy:
        """
        Proxy object for calling ``numpy.strings`` functions.

        https://numpy.org/doc/stable/reference/routines.strings.html

        Additionally, ``contains`` is available to check whether strings
        contain a substring.

        Note that, unlike :attr:`re`, NumPy functions and operators treat
        missing values, i.e. blank strings, as regular strings, e.g. ``x +
        "-"`` gives ``"-"`` and ``x.str.str_len()`` gives ``0`` for missing
        values. If you need to keep missing values missing, use e.g.
        ``np.where(x.is_na(), "", x + "-")``.

        >>> x = di.Vector(["asdf", "1234"])
        >>> x.str.contains("sd")
        >>> x.str.isalpha()
        >>> x.str.str_len()
        >>> x.str.upper()
        """
        if not hasattr(self, "_str"):
            self._str = StrProxy(self)
        return self._str

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
            if wcwidth.width(row) < print_width:
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
        add_string_element(f"] {self.dtype_label}", rows)
        if len(rows) == 1:
            # Drop padding for single-line output.
            rows[0] = [x.strip() for x in rows[0]]
        return "\n".join(" ".join(x) for x in rows)

    def to_strings(self, *, ksep=None, quote=True, pad=False, truncate_width=inf):
        """
        Return vector as strings formatted for display.

        >>> vector = di.Vector([1/2, 1/3, 1/4])
        >>> vector.to_strings()
        """
        if self.length == 0:
            return self.__class__.fast([], str)
        if ksep is None:
            ksep = dataiter.PRINT_THOUSAND_SEPARATOR
        if self.is_float():
            strings = util.format_floats(self, ksep=ksep)
        elif self.is_integer() and not self.is_timedelta():
            strings = [f"{x:,d}".replace(",", ksep) for x in self]
        elif self.is_object() or self.is_string():
            strings = [util.quote(x) if quote and self.is_string() else str(x) for x in self]
            for i in range(len(strings)):
                lines = strings[i].splitlines()
                if (wcwidth.width(strings[i]) > truncate_width or
                    (len(lines) > 1 and truncate_width < inf)):
                    strings[i] = wcwidth.clip(lines[0], 0, truncate_width-1) + "…"
        elif self.is_datetime():
            # Use the coarsest unit that shows all values exactly,
            # e.g. omit microseconds if all are zero.
            units = ["s", "ms", "us", "ns"]
            unit = np.datetime_data(self.dtype)[0]
            if unit in units:
                values = self[~self.is_na()]
                unit = next(x for x in units if np.all(values.astype(f"datetime64[{x}]") == values))
                strings = np.datetime_as_string(self, unit=unit).tolist()
            else:
                strings = np.datetime_as_string(self).tolist()
        # Years and months have no fixed length and finer units than
        # nanoseconds would be truncated, keep NumPy's format for those.
        elif self.is_timedelta() and np.datetime_data(self.dtype)[0] in ["W", "D", "h", "m", "s", "ms", "us", "ns"]:
            strings = util.format_timedeltas(self)
        else:
            strings = [str(x) for x in self]
        return self.__class__.fast(util.upad(strings) if pad else strings, str)

    def tolist(self):
        """
        Return vector as a list with elements of matching Python builtin type.

        Missing values are replaced with ``None``.
        """
        return np.where(self.is_na(), None, self).tolist()

    def unique(self):
        """
        Return unique elements in order of first appearance.

        >>> vector = di.Vector([3, 1, 1, 2, 3])
        >>> vector.unique()
        """
        if self.is_string() and self.length < 1000:
            # A dict beats Arrow's per-call overhead
            # for small inputs, e.g. many groups.
            return self.fast(list(dict.fromkeys(np.ndarray.tolist(self))), self.dtype)
        if self.dtype.kind in "biufmMT":
            # Hashing is faster than np.unique, which sorts.
            return util.unique_codes(self)[0]
        u, indices = np.unique(np.asarray(self), return_index=True)
        return self[np.sort(indices)]
