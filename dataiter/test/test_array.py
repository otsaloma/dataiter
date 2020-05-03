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

from dataiter import Array

NaT = np.datetime64("NaT")
DATE = datetime.date.today()
DATETIME = datetime.datetime.now()


class TestArray:

    def test___new___missing(self):
        Array([None])
        Array([np.nan])
        Array([NaT])
        Array([""])

    def test___new___missing_boolean(self):
        # Should be upcast to object.
        # Missing values should be None.
        a = Array([True, False, None])
        b = Array([True, False, None])
        assert a.is_object
        assert a.equal(b)

    def test___new___missing_date(self):
        # Should be converted to np.datetime64.
        # Missing values should be NaT.
        a = Array([DATE, NaT, None])
        b = Array([DATE, NaT, NaT])
        assert a.is_datetime
        assert a.equal(b)

    def test___new___missing_datetime(self):
        # Should be converted to np.datetime64.
        # Missing values should be NaT.
        a = Array([DATETIME, NaT, None])
        b = Array([DATETIME, NaT, NaT])
        assert a.is_datetime
        assert a.equal(b)

    def test___new___missing_float(self):
        # Missing values should be NaN.
        a = Array([1.1, 2.2, np.nan, None])
        b = Array([1.1, 2.2, np.nan, np.nan])
        assert a.is_float
        assert a.equal(b)

    def test___new___missing_integer(self):
        # Should be upcast to float.
        # Missing values should be NaN.
        a = Array([1, 2, np.nan, None])
        b = Array([1, 2, np.nan, np.nan])
        assert a.is_float
        assert a.equal(b)

    def test___new___missing_string(self):
        # Missing values should be blank strings.
        a = Array(["a", "b", "", None])
        b = Array(["a", "b", "", ""])
        assert a.is_string
        assert a.equal(b)

    def test__array_wrap___expect_array(self):
        result = Array([1, 2, 3]) * 2
        assert isinstance(result, Array)
        assert result.tolist() == [2, 4, 6]

    def test__array_wrap___expect_scalar(self):
        result = np.mean(Array([1, 2, 3]))
        assert np.isscalar(result)
        assert result == 2

    def test___len__(self):
        assert len(Array([1, 2, 3])) == 3

    def test_as_boolean(self):
        a = Array([1, 0]).as_boolean()
        assert a.is_boolean
        assert np.all(a == [True, False])

    def test_as_date(self):
        a = Array([DATETIME]).as_date()
        assert a.is_datetime
        assert np.all(a == [np.datetime64(DATE, "D")])

    def test_as_datetime(self):
        a = Array([DATE]).as_datetime()
        assert a.is_datetime
        assert np.all(a == [np.datetime64(DATE, "us")])

    def test_as_float(self):
        a = Array([1, 2]).as_float()
        assert a.is_float
        assert np.all(a == [1.0, 2.0])

    def test_as_integer(self):
        a = Array([1.1, 2.2]).as_integer()
        assert a.is_integer
        assert np.all(a == [1, 2])

    def test_as_string(self):
        a = Array([1, 2]).as_string()
        assert a.is_string
        assert np.all(a == ["1", "2"])

    def test_equal_boolean(self):
        a = Array([True, False])
        b = Array([True, False])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_date(self):
        a = Array([DATE, NaT])
        b = Array([DATE, NaT])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_datetime(self):
        a = Array([DATETIME, NaT])
        b = Array([DATETIME, NaT])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_float(self):
        a = Array([1.1, 2.2, np.nan])
        b = Array([1.1, 2.2, np.nan])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_integer(self):
        a = Array([1, 2, 3])
        b = Array([1, 2, 3])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_string(self):
        a = Array(["a", "b", ""])
        b = Array(["a", "b", ""])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_fast(self):
        a = Array.fast([1, 2, 3], int)
        b = Array([1, 2, 3], int)
        assert a.is_integer
        assert a.equal(b)

    def test_is_boolean(self):
        assert Array([True]).is_boolean
        assert not Array([1]).is_boolean
        assert not Array([1.1]).is_boolean
        assert not Array(["a"]).is_boolean
        assert not Array([DATE]).is_boolean
        assert not Array([DATETIME]).is_boolean
        assert not Array([self]).is_boolean

    def test_is_datetime(self):
        assert not Array([True]).is_datetime
        assert not Array([1]).is_datetime
        assert not Array([1.1]).is_datetime
        assert not Array(["a"]).is_datetime
        assert Array([DATE]).is_datetime
        assert Array([DATETIME]).is_datetime
        assert not Array([self]).is_datetime

    def test_is_float(self):
        assert not Array([True]).is_float
        assert not Array([1]).is_float
        assert Array([1.1]).is_float
        assert not Array(["a"]).is_float
        assert not Array([DATE]).is_float
        assert not Array([DATETIME]).is_float
        assert not Array([self]).is_float

    def test_is_integer(self):
        assert not Array([True]).is_integer
        assert Array([1]).is_integer
        assert not Array([1.1]).is_integer
        assert not Array(["a"]).is_integer
        assert not Array([DATE]).is_integer
        assert not Array([DATETIME]).is_integer
        assert not Array([self]).is_integer

    def test_is_number(self):
        assert not Array([True]).is_number
        assert Array([1]).is_number
        assert Array([1.1]).is_number
        assert not Array(["a"]).is_number
        assert not Array([DATE]).is_number
        assert not Array([DATETIME]).is_number
        assert not Array([self]).is_number

    def test_is_object(self):
        assert not Array([True]).is_object
        assert not Array([1]).is_object
        assert not Array([1.1]).is_object
        assert not Array(["a"]).is_object
        assert not Array([DATE]).is_object
        assert not Array([DATETIME]).is_object
        assert Array([self]).is_object

    def test_is_string(self):
        assert not Array([True]).is_string
        assert not Array([1]).is_string
        assert not Array([1.1]).is_string
        assert Array(["a"]).is_string
        assert not Array([DATE]).is_string
        assert not Array([DATETIME]).is_string
        assert not Array([self]).is_string

    def test_rank(self):
        a = Array([1, 2, 1, 2, 3])
        assert a.rank().tolist() == [0, 1, 0, 1, 2]

    def test_rank_missing(self):
        a = Array([np.nan, 1, 2, 3, np.nan])
        assert a.rank().tolist() == [3, 0, 1, 2, 4]

    def test_tolist_boolean(self):
        a = [True, False]
        b = [True, False]
        assert Array(a).tolist() == b

    def test_tolist_date(self):
        a = [DATE, NaT]
        b = [DATE, None]
        assert Array(a).tolist() == b

    def test_tolist_datetime(self):
        a = [DATETIME, NaT]
        b = [DATETIME, None]
        assert Array(a).tolist() == b

    def test_tolist_float(self):
        a = [1.1, 2.2, np.nan]
        b = [1.1, 2.2, None]
        assert Array(a).tolist() == b

    def test_tolist_integer(self):
        a = [1, 2]
        b = [1, 2]
        assert Array(a).tolist() == b

    def test_tolist_object(self):
        a = [True, False, None]
        b = [True, False, None]
        assert Array(a).tolist() == b

    def test_tolist_string(self):
        a = ["a", "b", ""]
        b = ["a", "b", None]
        assert Array(a).tolist() == b
