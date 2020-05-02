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

NAT = np.datetime64("NaT")
NOW = datetime.datetime.now()
TODAY = datetime.date.today()


class TestArray:

    def test___new___missing(self):
        Array([np.nan])
        Array([NAT])
        Array([None])

    def test___new___missing_boolean(self):
        # Should be upcast to handle missing values.
        x = Array([True, False, np.nan, None])
        y = Array([True, False, None, None])
        assert x.is_object
        assert x.equal(y)

    def test___new___missing_date(self):
        # Missing values should be NaT.
        x = Array([TODAY, np.nan, NAT, None])
        y = Array([TODAY, NAT, NAT, NAT])
        assert x.is_datetime
        assert x.equal(y)

    def test___new___missing_datetime(self):
        # Missing values should be NaT.
        x = Array([NOW, np.nan, NAT, None])
        y = Array([NOW, NAT, NAT, NAT])
        assert x.is_datetime
        assert x.equal(y)

    def test___new___missing_float(self):
        # Missing values should be NaN.
        x = Array([1.1, 2.2, np.nan, None])
        y = Array([1.1, 2.2, np.nan, np.nan])
        assert x.is_float
        assert x.equal(y)

    def test___new___missing_integer(self):
        # Should be upcast to handle missing values.
        x = Array([1, 2, np.nan, None])
        y = Array([1, 2, np.nan, np.nan])
        assert x.is_float
        assert x.equal(y)

    def test___new___missing_string(self):
        # Should use special string values.
        x = Array(["a", "b", np.nan, None])
        y = Array(["a", "b", "", ""])
        assert x.is_string
        assert x.equal(y)

    def test_equal_boolean(self):
        a = Array([True, False, np.nan])
        b = Array([True, False, np.nan])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_date(self):
        a = Array([TODAY, np.datetime64("NaT")])
        b = Array([TODAY, np.datetime64("NaT")])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_datetime(self):
        a = Array([NOW, np.datetime64("NaT")])
        b = Array([NOW, np.datetime64("NaT")])
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
        a = Array(["a", "b", "c", np.nan])
        b = Array(["a", "b", "c", np.nan])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_is_boolean(self):
        assert Array([True]).is_boolean
        assert not Array([1]).is_boolean
        assert not Array([1.1]).is_boolean
        assert not Array(["a"]).is_boolean
        assert not Array([TODAY]).is_boolean
        assert not Array([NOW]).is_boolean

    def test_is_datetime(self):
        assert not Array([True]).is_datetime
        assert not Array([1]).is_datetime
        assert not Array([1.1]).is_datetime
        assert not Array(["a"]).is_datetime
        assert Array([TODAY]).is_datetime
        assert Array([NOW]).is_datetime

    def test_is_float(self):
        assert not Array([True]).is_float
        assert not Array([1]).is_float
        assert Array([1.1]).is_float
        assert not Array(["a"]).is_float
        assert not Array([TODAY]).is_float
        assert not Array([NOW]).is_float

    def test_is_integer(self):
        assert not Array([True]).is_integer
        assert Array([1]).is_integer
        assert not Array([1.1]).is_integer
        assert not Array(["a"]).is_integer
        assert not Array([TODAY]).is_integer
        assert not Array([NOW]).is_integer

    def test_is_number(self):
        assert not Array([True]).is_number
        assert Array([1]).is_number
        assert Array([1.1]).is_number
        assert not Array(["a"]).is_number
        assert not Array([TODAY]).is_number
        assert not Array([NOW]).is_number

    def test_is_object(self):
        assert not Array([True]).is_object
        assert not Array([1]).is_object
        assert not Array([1.1]).is_object
        assert not Array(["a"]).is_object
        assert not Array([TODAY]).is_object
        assert not Array([NOW]).is_object

    def test_is_string(self):
        assert not Array([True]).is_string
        assert not Array([1]).is_string
        assert not Array([1.1]).is_string
        assert Array(["a"]).is_string
        assert not Array([TODAY]).is_string
        assert not Array([NOW]).is_string

    def test_rank(self):
        assert isinstance(Array([1, 2, 3]).rank(), Array)
        assert Array([1, 2, 1, 2, 3]).rank().tolist() == [0, 1, 0, 1, 2]
        assert Array([np.nan, 1, 2, 3, np.nan]).rank().tolist() == [3, 0, 1, 2, 4]

    def test_tolist_boolean(self):
        x = [True, False, None]
        y = [True, False, None]
        assert Array(x).tolist() == y

    def test_tolist_date(self):
        x = [TODAY, NAT]
        y = [TODAY, None]
        assert Array(x).tolist() == y

    def test_tolist_datetime(self):
        x = [NOW, NAT]
        y = [NOW, None]
        assert Array(x).tolist() == y

    def test_tolist_float(self):
        x = [1.1, 2.2, np.nan]
        y = [1.1, 2.2, None]
        assert Array(x).tolist() == y

    def test_tolist_integer(self):
        x = [1, 2, np.nan]
        y = [1.0, 2.0, None]
        assert Array(x).tolist() == y

    def test_tolist_string(self):
        x = ["a", "b", None]
        y = ["a", "b", None]
        assert Array(x).tolist() == y
