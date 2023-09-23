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
import math
import numpy as np

from dataiter import Vector

NaN = np.nan
NaT = np.datetime64("NaT")
DATE = datetime.date.today()
DATETIME = datetime.datetime.now()

class TestVector:

    def test___new___iterator(self):
        a = Vector(list(range(10)))
        b = Vector(range(10))
        c = Vector(x for x in range(10))
        d = Vector(map(lambda x: x, range(10)))
        assert a.equal(b)
        assert a.equal(c)
        assert a.equal(d)

    def test___new___list_of_numpy(self):
        assert Vector([np.bool_(True)]).is_boolean()
        assert Vector([np.datetime64(DATE)]).is_datetime()
        assert Vector([np.datetime64(DATETIME)]).is_datetime()
        assert Vector([np.float_(0.5)]).is_float()
        assert Vector([np.int_(1)]).is_integer()
        assert Vector([np.object_(np)]).is_object()
        assert Vector([np.str_("")]).is_string()

    def test___new___na(self):
        Vector([None])
        Vector([NaN])
        Vector([NaT])
        Vector([""])

    def test___new___na_boolean(self):
        # Should be upcast to object.
        # Missing values should be None.
        a = Vector([True, False, NaN, None])
        b = Vector([True, False, None, None])
        assert a.is_object()
        assert a.equal(b)

    def test___new___na_date(self):
        # Should be converted to np.datetime64.
        # Missing values should be NaT.
        a = Vector([DATE, NaT, NaN, None])
        b = Vector([DATE, NaT, NaT, NaT])
        assert a.is_datetime()
        assert a.equal(b)

    def test___new___na_datetime(self):
        # Should be converted to np.datetime64.
        # Missing values should be NaT.
        a = Vector([DATETIME, NaN, NaT, None])
        b = Vector([DATETIME, NaT, NaT, NaT])
        assert a.is_datetime()
        assert a.equal(b)

    def test___new___na_float(self):
        # Missing values should be NaN.
        a = Vector([1.1, 2.2, NaN, None])
        b = Vector([1.1, 2.2, NaN, NaN])
        assert a.is_float()
        assert a.equal(b)

    def test___new___na_float_mixed_nan(self):
        # Missing values should be NaN.
        # Mixing Python and NumPy floats should be fine.
        a = Vector([1.1, np.float64(2.2), NaN])
        b = Vector([1.1, 2.2, NaN])
        assert a.is_float()
        assert a.equal(b)

    def test___new___na_float_mixed_none(self):
        # Missing values should be NaN.
        # Mixing Python and NumPy floats should be fine.
        a = Vector([1.1, np.float64(2.2), None])
        b = Vector([1.1, 2.2, NaN])
        assert a.is_float()
        assert a.equal(b)

    def test___new___na_integer(self):
        # Should be upcast to float.
        # Missing values should be NaN.
        a = Vector([1, 2, NaN, None])
        b = Vector([1, 2, NaN, NaN])
        assert a.is_float()
        assert a.equal(b)

    def test___new___na_integer_mixed_nan(self):
        # Should be upcast to float.
        # Missing values should be NaN.
        # Mixing Python and NumPy integers should be fine.
        a = Vector([1, np.int64(2), NaN])
        b = Vector([1, 2, NaN])
        assert a.is_float()
        assert a.equal(b)

    def test___new___na_integer_mixed_none(self):
        # Should be upcast to float.
        # Missing values should be NaN.
        # Mixing Python and NumPy integers should be fine.
        a = Vector([1, np.int64(2), None])
        b = Vector([1, 2, NaN])
        assert a.is_float()
        assert a.equal(b)

    def test___new___na_object(self):
        # Missing values should be None.
        a = Vector(["a", "b", "", NaN, None], object)
        assert a.is_object()
        assert a[0] == "a"
        assert a[1] == "b"
        assert a[2] == ""
        assert a[3] is None
        assert a[4] is None

    def test___new___na_object_single(self):
        # Missing values should be None.
        a = Vector([None], object)
        assert a.is_object()
        assert a[0] is None

    def test___new___na_string(self):
        # Missing values should be blank strings.
        a = Vector(["a", "b", "", NaN, None])
        b = Vector(["a", "b", "", "", ""])
        assert a.is_string()
        assert a.equal(b)

    def test__array_wrap___expect_vector(self):
        result = Vector([1, 2, 3]) * 2
        assert isinstance(result, Vector)
        assert result.tolist() == [2, 4, 6]

    def test__array_wrap___expect_scalar(self):
        result = np.mean(Vector([1, 2, 3]))
        assert np.isscalar(result)
        assert result == 2

    def test___len__(self):
        assert len(Vector([1, 2, 3])) == 3

    def test_as_boolean(self):
        a = Vector([1, 0]).as_boolean()
        assert a.is_boolean()
        assert np.all(a == [True, False])

    def test_as_bytes(self):
        a = Vector([0, 1]).as_bytes()
        assert a.is_bytes()
        assert np.all(a == [b"0", b"1"])

    def test_as_bytes_string(self):
        a = Vector(["a", "ö"]).as_bytes()
        assert a.is_bytes()
        assert np.all(a == [b"a", b"\xc3\xb6"])

    def test_as_date(self):
        a = Vector([DATETIME]).as_date()
        assert a.is_datetime()
        assert np.all(a == [np.datetime64(DATE, "D")])

    def test_as_datetime(self):
        a = Vector([DATETIME]).as_datetime()
        assert a.is_datetime()
        assert np.all(a == [np.datetime64(DATETIME, "us")])

    def test_as_datetime_precision(self):
        a = Vector([DATETIME]).as_datetime("s")
        assert a.is_datetime()
        assert np.all(a == [np.datetime64(DATETIME, "s")])

    def test_as_float(self):
        a = Vector([1, 2]).as_float()
        assert a.is_float()
        assert np.all(a == [1.0, 2.0])

    def test_as_integer(self):
        a = Vector([1.1, 2.2]).as_integer()
        assert a.is_integer()
        assert np.all(a == [1, 2])

    def test_as_object(self):
        a = Vector([1, 2]).as_object()
        assert a.is_object()
        assert np.all(a == [1, 2])

    def test_as_string(self):
        a = Vector([1, 2]).as_string()
        assert a.is_string()
        assert np.all(a == ["1", "2"])

    def test_as_string_length(self):
        a = Vector([""]).as_string()
        a[0] = "hello"
        assert a[0] == "h"
        a = Vector([""]).as_string(5)
        a[0] = "hello"
        assert a[0] == "hello"

    def test_concat(self):
        a = Vector([1, 2, 3])
        b = Vector([4, 5, 6])
        c = Vector([7, 8, 9])
        assert a.concat(b, c).tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_drop_na(self):
        a = Vector([1, 2, 3, None])
        b = a.drop_na()
        assert b.tolist() == [1, 2, 3]

    def test_equal_boolean(self):
        a = Vector([True, False])
        b = Vector([True, False])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_date(self):
        a = Vector([DATE, NaT])
        b = Vector([DATE, NaT])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_datetime(self):
        a = Vector([DATETIME, NaT])
        b = Vector([DATETIME, NaT])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_float(self):
        a = Vector([1.1, 2.2, NaN])
        b = Vector([1.1, 2.2, NaN])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_integer(self):
        a = Vector([1, 2, 3])
        b = Vector([1, 2, 3])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_string(self):
        a = Vector(["a", "b", ""])
        b = Vector(["a", "b", ""])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_fast(self):
        a = Vector.fast([1, 2, 3], int)
        b = Vector([1, 2, 3], int)
        assert a.is_integer()
        assert a.equal(b)

    def test_fast_iterator(self):
        a = Vector.fast(list(range(10)))
        b = Vector.fast(range(10))
        c = Vector.fast(x for x in range(10))
        d = Vector.fast(map(lambda x: x, range(10)))
        assert a.equal(b)
        assert a.equal(c)
        assert a.equal(d)

    def test_get_memory_use(self):
        a = Vector(range(100))
        assert a.get_memory_use() > 100

    def test_get_memory_use_object(self):
        a = Vector(["1", "12", "123"], object)
        assert a.get_memory_use() > 3

    def test_head(self):
        a = Vector([1, 2, 3, 4, 5])
        assert a.head(3).tolist() == [1, 2, 3]

    def test_is_boolean(self):
        assert not Vector([b"1"]).is_boolean()
        assert Vector([True]).is_boolean()
        assert not Vector([1]).is_boolean()
        assert not Vector([1.1]).is_boolean()
        assert not Vector(["a"]).is_boolean()
        assert not Vector([DATE]).is_boolean()
        assert not Vector([DATETIME]).is_boolean()
        assert not Vector([self]).is_boolean()

    def test_is_bytes(self):
        assert Vector([b"1"]).is_bytes()
        assert not Vector([True]).is_bytes()
        assert not Vector([1]).is_bytes()
        assert not Vector([1.1]).is_bytes()
        assert not Vector(["a"]).is_bytes()
        assert not Vector([DATE]).is_bytes()
        assert not Vector([DATETIME]).is_bytes()
        assert not Vector([self]).is_bytes()

    def test_is_datetime(self):
        assert not Vector([b"1"]).is_datetime()
        assert not Vector([True]).is_datetime()
        assert not Vector([1]).is_datetime()
        assert not Vector([1.1]).is_datetime()
        assert not Vector(["a"]).is_datetime()
        assert Vector([DATE]).is_datetime()
        assert Vector([DATETIME]).is_datetime()
        assert not Vector([self]).is_datetime()

    def test_is_float(self):
        assert not Vector([b"1"]).is_float()
        assert not Vector([True]).is_float()
        assert not Vector([1]).is_float()
        assert Vector([1.1]).is_float()
        assert not Vector(["a"]).is_float()
        assert not Vector([DATE]).is_float()
        assert not Vector([DATETIME]).is_float()
        assert not Vector([self]).is_float()

    def test_is_integer(self):
        assert not Vector([b"1"]).is_integer()
        assert not Vector([True]).is_integer()
        assert Vector([1]).is_integer()
        assert not Vector([1.1]).is_integer()
        assert not Vector(["a"]).is_integer()
        assert not Vector([DATE]).is_integer()
        assert not Vector([DATETIME]).is_integer()
        assert not Vector([self]).is_integer()

    def test_is_na_date(self):
        a = Vector([DATE, DATE, NaT])
        assert a.is_na().tolist() == [False, False, True]

    def test_is_na_datetime(self):
        a = Vector([DATETIME, DATETIME, NaT])
        assert a.is_na().tolist() == [False, False, True]

    def test_is_na_float(self):
        a = Vector([1, 2, None])
        assert a.is_na().tolist() == [False, False, True]

    def test_is_na_object(self):
        a = Vector([self, self, None])
        assert a.is_na().tolist() == [False, False, True]

    def test_is_na_string(self):
        a = Vector(["a", "b", ""])
        assert a.is_na().tolist() == [False, False, True]

    def test_is_na_timedelta_date(self):
        a = Vector([DATE, DATE, NaT]) - Vector([DATE])
        assert a.is_na().tolist() == [False, False, True]

    def test_is_na_timedelta_datetime(self):
        a = Vector([DATETIME, DATETIME, NaT]) - Vector([DATETIME])
        assert a.is_na().tolist() == [False, False, True]

    def test_is_number(self):
        assert not Vector([b"1"]).is_number()
        assert not Vector([True]).is_number()
        assert Vector([1]).is_number()
        assert Vector([1.1]).is_number()
        assert not Vector(["a"]).is_number()
        assert not Vector([DATE]).is_number()
        assert not Vector([DATETIME]).is_number()
        assert not Vector([self]).is_number()

    def test_is_object(self):
        assert not Vector([b"1"]).is_object()
        assert not Vector([True]).is_object()
        assert not Vector([1]).is_object()
        assert not Vector([1.1]).is_object()
        assert not Vector(["a"]).is_object()
        assert not Vector([DATE]).is_object()
        assert not Vector([DATETIME]).is_object()
        assert Vector([self]).is_object()

    def test_is_string(self):
        assert not Vector([b"1"]).is_string()
        assert not Vector([True]).is_string()
        assert not Vector([1]).is_string()
        assert not Vector([1.1]).is_string()
        assert Vector(["a"]).is_string()
        assert not Vector([DATE]).is_string()
        assert not Vector([DATETIME]).is_string()
        assert not Vector([self]).is_string()

    def test_is_timedelta_date(self):
        date = Vector([DATE])
        assert (date - date).is_timedelta()

    def test_is_timedelta_datetime(self):
        datetime = Vector([DATETIME])
        assert (datetime - datetime).is_timedelta()

    def test_map(self):
        a = Vector([1, 2, 3, 4, 5])
        assert a.map(math.pow, 2).tolist() == [1, 4, 9, 16, 25]

    def test_map_dtype(self):
        a = Vector(["a", "ab", "abc"], object)
        b = a.map(str.replace, "a", "x", dtype=object)
        assert b.tolist() == ["x", "xb", "xbc"]
        assert b.is_object()

    def test_range(self):
        a = Vector([1, 2, 3, 4, 5, None])
        assert a.range().tolist() == [1, 5]

    def test_rank_average(self):
        a = Vector([3, 1, 1, 1, 2, 2])
        b = a.rank(method="average")
        assert b.tolist() == [6, 2, 2, 2, 4.5, 4.5]

    def test_rank_average_blank(self):
        a = Vector([], float)
        b = a.rank(method="average")
        assert b.tolist() == []

    def test_rank_average_na(self):
        a = Vector([NaN, 1, 2, 3, NaN])
        b = a.rank(method="average")
        assert b.tolist() == [4.5, 1, 2, 3, 4.5]

    def test_rank_average_na_all(self):
        a = Vector([NaN, NaN, NaN], float)
        b = a.rank(method="average")
        assert b.tolist() == [2, 2, 2]

    def test_rank_max(self):
        a = Vector([1, 2, 1, 2, 3])
        b = a.rank(method="max")
        assert b.tolist() == [2, 4, 2, 4, 5]

    def test_rank_max_na(self):
        a = Vector([NaN, 1, 2, 3, NaN])
        b = a.rank(method="max")
        assert b.tolist() == [5, 1, 2, 3, 5]

    def test_rank_min(self):
        a = Vector([1, 2, 1, 2, 3])
        b = a.rank(method="min")
        assert b.tolist() == [1, 3, 1, 3, 5]

    def test_rank_min_na(self):
        a = Vector([NaN, 1, 2, 3, NaN])
        b = a.rank(method="min")
        assert b.tolist() == [4, 1, 2, 3, 4]

    def test_rank_ordinal(self):
        a = Vector([1, 2, 1, 2, 3])
        b = a.rank(method="ordinal")
        assert b.tolist() == [1, 3, 2, 4, 5]

    def test_rank_ordinal_na(self):
        a = Vector([NaN, 1, 2, 3, NaN])
        b = a.rank(method="ordinal")
        assert b.tolist() == [4, 1, 2, 3, 5]

    def test_rank_without_ties(self):
        # Without ties, all methods should give the same result.
        a = Vector([NaN, 3, 2, 4, 5, 1])
        assert a.rank(method="min").equal(a.rank(method="average"))
        assert a.rank(method="min").equal(a.rank(method="max"))
        assert a.rank(method="min").equal(a.rank(method="min"))
        assert a.rank(method="min").equal(a.rank(method="ordinal"))

    def test_replace_na(self):
        a = Vector([1, 2, 3, None])
        assert a.replace_na(0).tolist() == [1, 2, 3, 0]

    def test_sample(self):
        a = Vector([1, 2, 3, 4, 5])
        assert np.all(np.isin(a.sample(3), a))
        assert a.sample(5).tolist() == [1, 2, 3, 4, 5]

    def test_sort(self):
        a = Vector([3, 4, 2, 1, 5])
        assert a.sort(dir=1).tolist() == [1, 2, 3, 4, 5]
        assert a.sort(dir=-1).tolist() == [5, 4, 3, 2, 1]

    def test_sort_object(self):
        a = Vector([1, None, True, None, "Hello"], object)
        assert a.sort(dir=1).tolist() == [1, "Hello", True, None, None]
        assert a.sort(dir=-1).tolist() == [True, "Hello", 1, None, None]

    def test_sort_na(self):
        a = Vector([None, 1, 2, 3, 4, 5, None])
        assert a.sort(dir=1).tolist() == [1, 2, 3, 4, 5, None, None]
        assert a.sort(dir=-1).tolist() == [5, 4, 3, 2, 1, None, None]

    def test_tail(self):
        a = Vector([1, 2, 3, 4, 5])
        assert a.tail(3).tolist() == [3, 4, 5]

    def test_to_string(self):
        assert Vector(np.arange(1)).to_string()
        assert Vector(np.arange(1000)).to_string()
        assert Vector(np.arange(1000000)).to_string()

    def test_to_strings_boolean(self):
        a = [True, False]
        b = ["True", "False"]
        assert Vector(a).to_strings().tolist() == b

    def test_to_strings_date(self):
        a = [DATE, DATE]
        b = [DATE.isoformat(), DATE.isoformat()]
        assert Vector(a).to_strings().tolist() == b

    def test_to_strings_datetime(self):
        a = [DATETIME, DATETIME]
        b = [DATETIME.isoformat(), DATETIME.isoformat()]
        assert Vector(a).to_strings().tolist() == b

    def test_to_strings_float(self):
        a = [0.1, 1/3]
        b = ["0.100000", "0.333333"]
        assert Vector(a).to_strings().tolist() == b

    def test_to_strings_integer(self):
        a = [1, 2]
        b = ["1", "2"]
        assert Vector(a).to_strings().tolist() == b

    def test_to_strings_string(self):
        a = ["a", "b"]
        b = ['"a"', '"b"']
        assert Vector(a).to_strings().tolist() == b

    def test_to_strings_string_newline(self):
        a = ["a\nb", "c\nd"]
        b = ['"a\nb"', '"c\nd"']
        assert Vector(a).to_strings().tolist() == b

    def test_to_strings_string_newline_truncate(self):
        # Used especially by DataFrame.to_strings to avoid ruining
        # tabular display with a multiline mess.
        a = ["a\nb", "c\nd"]
        b = ["a…", "c…"]
        assert Vector(a).to_strings(quote=False, truncate_width=36).tolist() == b

    def test_to_strings_timedelta_date(self):
        a = Vector([DATE, DATE])
        a = (a + np.timedelta64(1, "D")) - a
        b = ["1 days", "1 days"]
        assert a.to_strings().tolist() == b

    def test_to_strings_timedelta_datetime(self):
        a = Vector([DATETIME, DATETIME])
        a = (a + np.timedelta64(1, "us")) - a
        b = ["1 microseconds", "1 microseconds"]
        assert a.to_strings().tolist() == b

    def test_tolist_boolean(self):
        a = [True, False]
        b = [True, False]
        assert Vector(a).tolist() == b

    def test_tolist_date(self):
        a = [DATE, NaT]
        b = [DATE, None]
        assert Vector(a).tolist() == b

    def test_tolist_datetime(self):
        a = [DATETIME, NaT]
        b = [DATETIME, None]
        assert Vector(a).tolist() == b

    def test_tolist_float(self):
        a = [1.1, 2.2, NaN]
        b = [1.1, 2.2, None]
        assert Vector(a).tolist() == b

    def test_tolist_integer(self):
        a = [1, 2]
        b = [1, 2]
        assert Vector(a).tolist() == b

    def test_tolist_object(self):
        a = [True, False, None]
        b = [True, False, None]
        assert Vector(a).tolist() == b

    def test_tolist_string(self):
        a = ["a", "b", ""]
        b = ["a", "b", None]
        assert Vector(a).tolist() == b

    def test_tolist_timedelta_date(self):
        a = Vector([DATE, DATE])
        a = (a + np.timedelta64(1, "D")) - a
        b = [datetime.timedelta(days=1), datetime.timedelta(days=1)]
        assert Vector(a).tolist() == b

    def test_tolist_timedelta_datetime(self):
        a = Vector([DATETIME, DATETIME])
        a = (a + np.timedelta64(1, "us")) - a
        b = [datetime.timedelta(microseconds=1), datetime.timedelta(microseconds=1)]
        assert Vector(a).tolist() == b

    def test_unique(self):
        a = Vector([1, 2, None, 1, 2, 3])
        assert a.unique().tolist() == [1, 2, None, 3]
