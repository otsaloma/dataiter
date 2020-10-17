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

from dataiter import Vector

NaT = np.datetime64("NaT")
DATE = datetime.date.today()
DATETIME = datetime.datetime.now()


class TestVector:

    def test___new___missing(self):
        Vector([None])
        Vector([np.nan])
        Vector([NaT])
        Vector([""])

    def test___new___missing_boolean(self):
        # Should be upcast to object.
        # Missing values should be None.
        a = Vector([True, False, None])
        b = Vector([True, False, None])
        assert a.is_object
        assert a.equal(b)

    def test___new___missing_date(self):
        # Should be converted to np.datetime64.
        # Missing values should be NaT.
        a = Vector([DATE, NaT, None])
        b = Vector([DATE, NaT, NaT])
        assert a.is_datetime
        assert a.equal(b)

    def test___new___missing_datetime(self):
        # Should be converted to np.datetime64.
        # Missing values should be NaT.
        a = Vector([DATETIME, NaT, None])
        b = Vector([DATETIME, NaT, NaT])
        assert a.is_datetime
        assert a.equal(b)

    def test___new___missing_float(self):
        # Missing values should be NaN.
        a = Vector([1.1, 2.2, np.nan, None])
        b = Vector([1.1, 2.2, np.nan, np.nan])
        assert a.is_float
        assert a.equal(b)

    def test___new___missing_integer(self):
        # Should be upcast to float.
        # Missing values should be NaN.
        a = Vector([1, 2, np.nan, None])
        b = Vector([1, 2, np.nan, np.nan])
        assert a.is_float
        assert a.equal(b)

    def test___new___missing_string(self):
        # Missing values should be blank strings.
        a = Vector(["a", "b", "", None])
        b = Vector(["a", "b", "", ""])
        assert a.is_string
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
        assert a.is_boolean
        assert np.all(a == [True, False])

    def test_as_bytes(self):
        a = Vector([0, 1]).as_bytes()
        assert a.is_bytes
        assert np.all(a == [b"0", b"1"])

    def test_as_bytes_string(self):
        a = Vector(["a", "ö"]).as_bytes()
        assert a.is_bytes
        assert np.all(a == [b"a", b"\xc3\xb6"])

    def test_as_date(self):
        a = Vector([DATETIME]).as_date()
        assert a.is_datetime
        assert np.all(a == [np.datetime64(DATE, "D")])

    def test_as_datetime(self):
        a = Vector([DATE]).as_datetime()
        assert a.is_datetime
        assert np.all(a == [np.datetime64(DATE, "us")])

    def test_as_float(self):
        a = Vector([1, 2]).as_float()
        assert a.is_float
        assert np.all(a == [1.0, 2.0])

    def test_as_integer(self):
        a = Vector([1.1, 2.2]).as_integer()
        assert a.is_integer
        assert np.all(a == [1, 2])

    def test_as_object(self):
        a = Vector([1, 2]).as_object()
        assert a.is_object
        assert np.all(a == [1, 2])

    def test_as_string(self):
        a = Vector([1, 2]).as_string()
        assert a.is_string
        assert np.all(a == ["1", "2"])

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
        a = Vector([1.1, 2.2, np.nan])
        b = Vector([1.1, 2.2, np.nan])
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
        assert a.is_integer
        assert a.equal(b)

    def test_head(self):
        a = Vector([1, 2, 3, 4, 5])
        assert a.head(3).tolist() == [1, 2, 3]

    def test_is_boolean(self):
        assert not Vector([b"1"]).is_boolean
        assert Vector([True]).is_boolean
        assert not Vector([1]).is_boolean
        assert not Vector([1.1]).is_boolean
        assert not Vector(["a"]).is_boolean
        assert not Vector([DATE]).is_boolean
        assert not Vector([DATETIME]).is_boolean
        assert not Vector([self]).is_boolean

    def test_is_bytes(self):
        assert Vector([b"1"]).is_bytes
        assert not Vector([True]).is_bytes
        assert not Vector([1]).is_bytes
        assert not Vector([1.1]).is_bytes
        assert not Vector(["a"]).is_bytes
        assert not Vector([DATE]).is_bytes
        assert not Vector([DATETIME]).is_bytes
        assert not Vector([self]).is_bytes

    def test_is_datetime(self):
        assert not Vector([b"1"]).is_datetime
        assert not Vector([True]).is_datetime
        assert not Vector([1]).is_datetime
        assert not Vector([1.1]).is_datetime
        assert not Vector(["a"]).is_datetime
        assert Vector([DATE]).is_datetime
        assert Vector([DATETIME]).is_datetime
        assert not Vector([self]).is_datetime

    def test_is_float(self):
        assert not Vector([b"1"]).is_float
        assert not Vector([True]).is_float
        assert not Vector([1]).is_float
        assert Vector([1.1]).is_float
        assert not Vector(["a"]).is_float
        assert not Vector([DATE]).is_float
        assert not Vector([DATETIME]).is_float
        assert not Vector([self]).is_float

    def test_is_integer(self):
        assert not Vector([b"1"]).is_integer
        assert not Vector([True]).is_integer
        assert Vector([1]).is_integer
        assert not Vector([1.1]).is_integer
        assert not Vector(["a"]).is_integer
        assert not Vector([DATE]).is_integer
        assert not Vector([DATETIME]).is_integer
        assert not Vector([self]).is_integer

    def test_is_missing_date(self):
        a = Vector([DATE, DATE, NaT])
        assert a.is_missing().tolist() == [False, False, True]

    def test_is_missing_datetime(self):
        a = Vector([DATETIME, DATETIME, NaT])
        assert a.is_missing().tolist() == [False, False, True]

    def test_is_missing_float(self):
        a = Vector([1, 2, None])
        assert a.is_missing().tolist() == [False, False, True]

    def test_is_missing_object(self):
        a = Vector([self, self, None])
        assert a.is_missing().tolist() == [False, False, True]

    def test_is_missing_string(self):
        a = Vector(["a", "b", ""])
        assert a.is_missing().tolist() == [False, False, True]

    def test_is_number(self):
        assert not Vector([b"1"]).is_number
        assert not Vector([True]).is_number
        assert Vector([1]).is_number
        assert Vector([1.1]).is_number
        assert not Vector(["a"]).is_number
        assert not Vector([DATE]).is_number
        assert not Vector([DATETIME]).is_number
        assert not Vector([self]).is_number

    def test_is_object(self):
        assert not Vector([b"1"]).is_object
        assert not Vector([True]).is_object
        assert not Vector([1]).is_object
        assert not Vector([1.1]).is_object
        assert not Vector(["a"]).is_object
        assert not Vector([DATE]).is_object
        assert not Vector([DATETIME]).is_object
        assert Vector([self]).is_object

    def test_is_string(self):
        assert not Vector([b"1"]).is_string
        assert not Vector([True]).is_string
        assert not Vector([1]).is_string
        assert not Vector([1.1]).is_string
        assert Vector(["a"]).is_string
        assert not Vector([DATE]).is_string
        assert not Vector([DATETIME]).is_string
        assert not Vector([self]).is_string

    def test_range(self):
        a = Vector([1, 2, 3, 4, 5, None])
        assert a.range().tolist() == [1, 5]

    def test_rank(self):
        a = Vector([1, 2, 1, 2, 3])
        assert a.rank().tolist() == [0, 1, 0, 1, 2]

    def test_rank_missing(self):
        a = Vector([np.nan, 1, 2, 3, np.nan])
        assert a.rank().tolist() == [3, 0, 1, 2, 4]

    def test_sample(self):
        a = Vector([1, 2, 3, 4, 5])
        assert np.all(np.isin(a.sample(3), a))
        assert a.sample(5).tolist() == [1, 2, 3, 4, 5]

    def test_sort(self):
        a = Vector([1, 2, 3, 4, 5, None])
        assert a.sort(dir=1).tolist() == [1, 2, 3, 4, 5, None]
        assert a.sort(dir=-1).tolist() == [5, 4, 3, 2, 1, None]

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
        a = [1.1, 2.2, np.nan]
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

    def test_unique(self):
        a = Vector([1, 2, None, 1, 2, 3])
        assert a.unique().tolist() == [1, 2, None, 3]
