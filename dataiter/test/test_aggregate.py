# -*- coding: utf-8 -*-

# Copyright (c) 2022 Osmo Salomaa
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
import pytest

from dataiter import aggregate
from dataiter import USE_NUMBA

try:
    import numba # noqa
except Exception:
    pass

skipif = pytest.mark.skipif
xfail = pytest.mark.xfail


class TestAggregate:

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_count_unique1_bool(self):
        assert aggregate.count_unique1(np.array([True, False])) == 2
        assert aggregate.count_unique1(np.array([True, True]))  == 1

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_count_unique1_date(self):
        date1 = datetime.date.today()
        date2 = datetime.date.today() + datetime.timedelta(days=1)
        dtype = np.dtype("datetime64[D]")
        assert aggregate.count_unique1(np.array([date1, date2], dtype)) == 2
        assert aggregate.count_unique1(np.array([date1, date1], dtype)) == 1

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_count_unique1_datetime(self):
        datetime1 = datetime.datetime.now()
        datetime2 = datetime.datetime.now() + datetime.timedelta(days=1)
        dtype = np.dtype("datetime64[us]")
        assert aggregate.count_unique1(np.array([datetime1, datetime2], dtype)) == 2
        assert aggregate.count_unique1(np.array([datetime1, datetime1], dtype)) == 1

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_count_unique1_float(self):
        assert aggregate.count_unique1(np.array([0.1, 0.2, 0.3])) == 3
        assert aggregate.count_unique1(np.array([0.1, 0.2, 0.2])) == 2

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_count_unique1_int(self):
        assert aggregate.count_unique1(np.array([1, 2, 3])) == 3
        assert aggregate.count_unique1(np.array([1, 2, 2])) == 2

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_count_unique1_str(self):
        assert aggregate.count_unique1(np.array(["a", "b", "c"])) == 3
        assert aggregate.count_unique1(np.array(["a", "b", "b"])) == 2

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_mode1_bool(self):
        assert aggregate.mode1(np.array([True, True, False]))  == True
        assert aggregate.mode1(np.array([True, False, False])) == False

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_mode1_date(self):
        date1 = datetime.date.today()
        date2 = datetime.date.today() + datetime.timedelta(days=1)
        dtype = np.dtype("datetime64[D]")
        assert aggregate.mode1(np.array([date1, date1, date2], dtype)) == date1
        assert aggregate.mode1(np.array([date1, date2, date2], dtype)) == date2

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_mode1_datetime(self):
        datetime1 = datetime.datetime.now()
        datetime2 = datetime.datetime.now() + datetime.timedelta(days=1)
        dtype = np.dtype("datetime64[us]")
        assert aggregate.mode1(np.array([datetime1, datetime1, datetime2], dtype)) == datetime1
        assert aggregate.mode1(np.array([datetime1, datetime2, datetime2], dtype)) == datetime2

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_mode1_float(self):
        assert aggregate.mode1(np.array([0.1, 0.2, 0.1])) == 0.1
        assert aggregate.mode1(np.array([0.2, 0.2, 0.1])) == 0.2

    @skipif(not USE_NUMBA, reason="No Numba")
    def test_mode1_int(self):
        assert aggregate.mode1(np.array([1, 2, 1])) == 1
        assert aggregate.mode1(np.array([2, 2, 1])) == 2

    @xfail
    @skipif(not USE_NUMBA, reason="No Numba")
    def test_mode1_str(self):
        assert aggregate.mode1(np.array(["b", "a", "a"])) == "a"
        assert aggregate.mode1(np.array(["b", "a", "b"])) == "b"
