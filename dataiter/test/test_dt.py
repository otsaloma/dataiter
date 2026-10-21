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

from dataiter import dt
from dataiter import Vector

NaT = np.datetime64("NaT", "D")

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

    def test_floor_date(self):
        a = dt.new(["2022-10-15", "2022-11-20", "1969-12-31", NaT])
        assert dt.floor(a, "year").equal(dt.new(["2022-01-01", "2022-01-01", "1969-01-01", NaT]))
        assert dt.floor(a, "quarter").equal(dt.new(["2022-10-01", "2022-10-01", "1969-10-01", NaT]))
        assert dt.floor(a, "month").equal(dt.new(["2022-10-01", "2022-11-01", "1969-12-01", NaT]))
        assert dt.floor(a, "week").equal(dt.new(["2022-10-10", "2022-11-14", "1969-12-29", NaT]))
        assert dt.floor(a, "day").equal(a)

    def test_floor_datetime(self):
        a = dt.new(["2022-10-15T12:34:56.789", NaT])
        assert dt.floor(a, "week").equal(dt.new(["2022-10-10T00:00:00.000", NaT]))
        assert dt.floor(a, "hour").equal(dt.new(["2022-10-15T12:00:00.000", NaT]))
        assert dt.floor(a, "second").equal(dt.new(["2022-10-15T12:34:56.000", NaT]))

    def test_floor_scalar(self):
        assert dt.floor(dt.new("2022-10-15"), "week") == dt.new("2022-10-10")

    def test_floor_unit(self):
        a = dt.new(["2022-10-15T12:34:56"])
        assert dt.floor(a, "days").equal(dt.floor(a, "day"))
        assert dt.floor(a, "y").equal(dt.floor(a, "year"))
        assert dt.floor(a, "mo").equal(dt.floor(a, "month"))
        assert dt.floor(a, "min").equal(dt.floor(a, "minute"))
        with pytest.raises(ValueError, match="Ambiguous"):
            dt.floor(a, "m")
        with pytest.raises(ValueError, match="Bad"):
            dt.floor(a, "")
        with pytest.raises(ValueError, match="Bad"):
            dt.floor(a, "dayss")

    def test_from_components_date(self):
        a = dt.from_components(Vector([2022, 2023, np.nan]), 6, 30)
        assert a.dtype == np.dtype("datetime64[D]")
        assert a.equal(dt.new(["2022-06-30", "2023-06-30", NaT]))

    def test_from_components_datetime(self):
        a = dt.from_components([1969, 2000], 12, 31, 23, 59, 59, [999999, 1])
        assert a.dtype == np.dtype("datetime64[us]")
        assert a.equal(dt.new(["1969-12-31T23:59:59.999999", "2000-12-31T23:59:59.000001"]))
        assert dt.from_components([2000], 1, 1, minute=30).equal(dt.new(["2000-01-01T00:30:00"]))

    def test_from_components_errors_replace(self):
        a = dt.from_components([2024, 2023, 2023, 2023], [2, 2, 13, 1], [29, 29, 1, 1.5], errors="replace")
        assert a.equal(dt.new(["2024-02-29", NaT, NaT, NaT]))
        a = dt.from_components([2023, 2023], 1, 1, [23, 24], errors="replace")
        assert a.equal(dt.new(["2023-01-01T23:00:00", NaT]))

    def test_from_components_errors_strict(self):
        with pytest.raises(ValueError, match="day=29"):
            dt.from_components([2024, 2023], 2, 29)

    def test_from_components_scalar(self):
        assert dt.from_components(2022, 10, 15) == np.datetime64("2022-10-15")
        assert dt.from_components(2022, 10, 15, 12) == np.datetime64("2022-10-15T12:00:00")
        assert np.isnat(dt.from_components(2023, 2, 29, errors="replace"))

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

    def test_from_string_errors_replace(self):
        a = Vector(["14.11.2022", "32.11.2022"], str)
        b = dt.from_string(a, "%d.%m.%Y", errors="replace")
        assert b.dtype == np.dtype("datetime64[D]")
        assert b[0] == np.datetime64("2022-11-14")
        assert np.isnat(b[1])

    def test_from_string_sorted_runs(self):
        # https://github.com/numpy/numpy/issues/33089
        a = dt.new(["2022-01-01"]).repeat(128) + np.arange(128)
        b = dt.from_string(dt.isoformat(a.concat(a)), "%Y-%m-%d")
        assert b.equal(a.concat(a))

    def test_from_string_errors_strict(self):
        a = Vector(["14.11.2022", "32.11.2022"], str)
        with pytest.raises(ValueError):
            dt.from_string(a, "%d.%m.%Y")

    def test_from_string_type_error(self):
        a = dt.new(["2022-10-15"])
        with pytest.raises(TypeError, match="Expected string, got datetime64"):
            dt.from_string(a, "%Y-%m-%d")

    def test_hour(self):
        a = dt.new(["2022-10-15T12:34:56", NaT])
        assert dt.hour(a).tolist() == [12, None]

    def test_isoformat_date(self):
        a = np.array(["2022-10-15", NaT], np.datetime64)
        assert dt.isoformat(a).tolist() == ["2022-10-15", None]

    def test_isoformat_date_scalar(self):
        x = np.datetime64("2022-10-15")
        assert dt.isoformat(x) == "2022-10-15"

    def test_isoformat_datetime(self):
        a = np.array(["2022-10-15T12:34:56.789", NaT], np.datetime64)
        assert dt.isoformat(a).tolist() == ["2022-10-15T12:34:56.789", None]

    def test_isoformat_datetime_coarsest_unit(self):
        a = np.array(["2022-10-15T12:34", NaT], "datetime64[ns]")
        assert dt.isoformat(a).tolist() == ["2022-10-15T12:34:00", None]
        a = np.array(["2022-10-15T12:34:56", "2022-10-15T12:34:56.5"], "datetime64[us]")
        assert dt.isoformat(a).tolist() == ["2022-10-15T12:34:56.000", "2022-10-15T12:34:56.500"]

    def test_isoformat_datetime_repetitive(self):
        a = np.array(["2022-10-15T12:34:56.5", NaT, "2022-10-15T12:34"] * 10, np.datetime64)
        b = ["2022-10-15T12:34:56.500", None, "2022-10-15T12:34:00.000"] * 10
        assert dt.isoformat(a).tolist() == b

    def test_isoformat_datetime_scalar(self):
        x = np.datetime64("2022-10-15T12:34:56.789")
        assert dt.isoformat(x) == "2022-10-15T12:34:56.789"

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

    def test_replace_all(self):
        a = dt.new(["2022-10-15", NaT])
        b = dt.new(["2025-02-03", NaT])
        assert dt.replace(a, year=2025, month=2, day=3).equal(b)
        assert dt.replace(a, year=[2025, 2025], month=2, day=3).equal(b)

    def test_replace_nanoseconds(self):
        a = Vector(["2022-10-15T12:34:56", NaT], "datetime64[ns]")
        b = Vector(["2022-01-01T12:34:56", NaT], "datetime64[ns]")
        assert dt.replace(a, month=1, day=1).equal(b)
        assert dt.replace(a, month=[1, 1], day=1).equal(b)

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

    def test_sequence_date(self):
        a = dt.sequence(dt.new("2022-10-15"), dt.new("2022-10-18"), "day")
        assert a.equal(dt.new(["2022-10-15", "2022-10-16", "2022-10-17", "2022-10-18"]))
        a = dt.sequence(dt.new("2022-10-10"), dt.new("2022-10-30"), "week")
        assert a.equal(dt.new(["2022-10-10", "2022-10-17", "2022-10-24"]))
        a = dt.sequence(dt.new("2022-10-18"), dt.new("2022-10-15"), "day")
        assert a.dtype == np.dtype("datetime64[D]")
        assert len(a) == 0

    def test_sequence_datetime(self):
        a = dt.sequence(dt.new("2022-10-15T22:30"), dt.new("2022-10-16T01:00"), "hour")
        assert a.dtype == np.dtype("datetime64[m]")
        assert a.equal(dt.new(["2022-10-15T22:30", "2022-10-15T23:30", "2022-10-16T00:30"]))
        a = dt.sequence(dt.new("2022-10-15"), dt.new("2022-10-15T02:00"), "hour")
        assert a.dtype == np.dtype("datetime64[h]")
        assert a.equal(dt.new(["2022-10-15T00", "2022-10-15T01", "2022-10-15T02"]))

    def test_sequence_month(self):
        a = dt.sequence(dt.new("2022-01-31"), dt.new("2022-04-30"), "month")
        assert a.equal(dt.new(["2022-01-31", "2022-02-28", "2022-03-31", "2022-04-30"]))
        a = dt.sequence(dt.new("2022-11-30"), dt.new("2023-06-01"), "quarter")
        assert a.equal(dt.new(["2022-11-30", "2023-02-28", "2023-05-30"]))
        a = dt.sequence(dt.new("2024-02-29"), dt.new("2026-03-01"), "year")
        assert a.equal(dt.new(["2024-02-29", "2025-02-28", "2026-02-28"]))
        a = dt.sequence(dt.new("2022-01-31T12:00"), dt.new("2022-03-31"), "month")
        assert a.equal(dt.new(["2022-01-31T12:00", "2022-02-28T12:00"]))

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

    def test_to_string_na(self):
        a = dt.new([NaT, NaT])
        b = dt.to_string(a, "%d.%m.%Y")
        assert b.is_string()
        assert b.tolist() == [None, None]

    def test_to_string_nanoseconds(self):
        a = np.array(["2022-11-14T22:49:00", NaT], "datetime64[ns]")
        assert dt.to_string(a, "%Y%m%d-%H%M%S").tolist() == ["20221114-224900", None]

    @pytest.mark.parametrize("format", ["%Y", "%b %Y", "%a", "%G-W%V", "%H %p", "%H:%M", "%S.%f", "%%Y", "const"])
    def test_to_string_floor(self, format):
        a = ["2022-01-01T00:00:00", "2022-11-14T22:49:00.5", "2023-01-01T13:01:59"]
        b = dt.to_string(dt.new(a), format)
        assert b.tolist() == [datetime.datetime.fromisoformat(x).strftime(format) for x in a]

    def test_today(self):
        assert isinstance(dt.today(), np.datetime64)

    def test_total_seconds(self):
        a = dt.new(["2022-10-17", NaT]) - dt.new("2022-10-15")
        b = dt.new(["2022-10-15T12:00:00.5", NaT]) - dt.new("2022-10-15")
        assert dt.total_seconds(a).tolist() == [172800, None]
        assert dt.total_seconds(b).tolist() == [43200.5, None]
        assert dt.total_seconds(a).is_float()

    def test_total_seconds_scalar(self):
        assert dt.total_seconds(np.timedelta64(2, "m")) == 120
        assert dt.total_seconds(datetime.timedelta(hours=1)) == 3600

    def test_total_seconds_type_error(self):
        a = dt.new(["2022-10-15"])
        with pytest.raises(TypeError, match="Expected timedelta, got datetime64"):
            dt.total_seconds(a)

    def test_weekday(self):
        a = dt.new(["2022-10-15", NaT])
        assert dt.weekday(a).tolist() == [5, None]

    def test_year(self):
        a = dt.new(["2022-10-15", NaT])
        assert dt.year(a).tolist() == [2022, None]

    def test_year_type_error(self):
        a = Vector(["2022-10-15"], str)
        with pytest.raises(TypeError, match="Expected datetime, got string"):
            dt.year(a)
