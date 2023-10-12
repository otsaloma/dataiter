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

import numpy as np

from dataiter import dt
from dataiter import Vector

NaT = np.datetime64("NaT")

class TestDT:

    def test_day(self):
        a = dt.new(["2022-10-15", NaT])
        assert dt.day(a).tolist() == [15, None]

    def test_day_nat(self):
        assert np.isnan(dt.day(NaT))

    def test_day_numpy(self):
        a = np.array(["2022-10-15", NaT], np.datetime64)
        assert dt.day(a).tolist() == [15, None]

    def test_day_scalar(self):
        x = np.datetime64("2022-10-15")
        assert dt.day(x) == 15

    def test_from_string_date(self):
        a = Vector(["14.11.2022", ""], str)
        b = dt.from_string(a, "%d.%m.%Y")
        assert b.is_datetime()
        assert b[0] == np.datetime64("2022-11-14")
        assert np.isnat(b[1])

    def test_from_string_datetime(self):
        a = Vector(["14.11.2022 22:49", ""], str)
        b = dt.from_string(a, "%d.%m.%Y %H:%M")
        assert b.is_datetime()
        assert b[0] == np.datetime64("2022-11-14T22:49:00")
        assert np.isnat(b[1])

    def test_hour(self):
        a = dt.new(["2022-10-15T12:34:56", NaT])
        assert dt.hour(a).tolist() == [12, None]

    def test_isoweek(self):
        a = dt.new(["2022-10-15", NaT])
        assert dt.isoweek(a).tolist() == [41, None]

    def test_isoweekday(self):
        a = dt.new(["2022-10-15", NaT])
        assert dt.isoweekday(a).tolist() == [6, None]

    def test_microsecond(self):
        a = dt.new(["2022-10-15T12:34:56.789", NaT])
        assert dt.microsecond(a).tolist() == [789_000, None]

    def test_minute(self):
        a = dt.new(["2022-10-15T12:34:56", NaT])
        assert dt.minute(a).tolist() == [34, None]

    def test_month(self):
        a = dt.new([NaT, NaT])
        assert dt.month(a).tolist() == [None, None]
        a = dt.new(["2022-10-15", NaT])
        assert dt.month(a).tolist() == [10, None]
        a = dt.new(["2022-10-15", "2022-11-15"])
        assert dt.month(a).tolist() == [10, 11]

    def test_new_date(self):
        a = dt.new(["2022-10-15"])
        b = Vector(["2022-10-15"]).as_date()
        assert a.equal(b)

    def test_new_datetime(self):
        a = dt.new(["2022-10-15T12:00:00"])
        b = Vector(["2022-10-15T12:00:00"]).as_datetime()
        assert a.equal(b)

    def test_new_scalar(self):
        a = dt.new("2022-10-15")
        b = np.datetime64("2022-10-15")
        assert a == b

    def test_now(self):
        assert isinstance(dt.now(), np.datetime64)

    def test_quarter(self):
        a = dt.new("2022-10-15")
        assert dt.quarter(a) == 4
        assert np.isnan(dt.quarter(NaT))
        a = dt.new(["2022-10-15"])
        assert dt.quarter(a).tolist() == [4]
        a = dt.new(["2022-10-15", NaT])
        assert dt.quarter(a).tolist() == [4, None]

    def test_replace(self):
        a = dt.new(["2022-10-15", NaT])
        b = dt.new(["2022-01-01", NaT])
        assert dt.replace(a, month=1, day=1).equal(b)

    def test_replace_vector(self):
        a = dt.new(["2023-08-09", "2023-08-10", "2023-08-11"])
        b = dt.new(["2023-07-01", "2023-07-02", "2023-07-03"])
        assert dt.replace(a, month=7, day=[1, 2, 3]).equal(b)

    def test_replace_vector_1m(self):
        a = np.repeat(dt.new("2023-08-09"), 1_000_000)
        month = np.repeat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 100_000)
        assert len(month) == len(a) == 1_000_000
        b = dt.replace(a, month=month, day=1)
        assert len(b) == len(a)

    def test_second(self):
        a = dt.new(["2022-10-15T12:34:56", NaT])
        assert dt.second(a).tolist() == [56, None]

    def test_to_string_date(self):
        a = dt.new(["2022-11-14", NaT])
        b = dt.to_string(a, "%d.%m.%Y")
        assert b.is_string()
        assert b.tolist() == ["14.11.2022", None]

    def test_to_string_datetime(self):
        a = dt.new(["2022-11-14T22:49:00", NaT])
        b = dt.to_string(a, "%Y%m%d-%H%M%S")
        assert b.is_string()
        assert b.tolist() == ["20221114-224900", None]

    def test_today(self):
        assert isinstance(dt.today(), np.datetime64)

    def test_weekday(self):
        a = dt.new(["2022-10-15", NaT])
        assert dt.weekday(a).tolist() == [5, None]

    def test_year(self):
        a = dt.new(["2022-10-15", NaT])
        assert dt.year(a).tolist() == [2022, None]
