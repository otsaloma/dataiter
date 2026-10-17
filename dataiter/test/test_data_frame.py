# -*- coding: utf-8 -*-

# Copyright (c) 2019 Osmo Salomaa
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

import dataiter as di
import datetime
import json
import numpy as np
import pytest

from dataiter import DataFrame
from dataiter import DataFrameColumn
from dataiter import test
from dataiter import Vector
from unittest.mock import patch

# See dataiter/dtypes.py for the equivalent non-test ignore.
IGNORE_NPZ_PICKLE_WARNING = "ignore:Custom dtypes are saved as python objects using the pickle protocol"

class TestDataFrameColumn:

    def test___init___given_array(self):
        column = DataFrameColumn(np.array([1, 2, 3]))
        assert column.tolist() == [1, 2, 3]

    def test___init___given_iterator(self):
        column = DataFrameColumn(range(3))
        assert column.tolist() == [0, 1, 2]
        column = DataFrameColumn(x for x in range(3))
        assert column.tolist() == [0, 1, 2]
        column = DataFrameColumn(map(lambda x: x, range(3)))
        assert column.tolist() == [0, 1, 2]

    def test___init___given_list(self):
        column = DataFrameColumn([1, 2, 3])
        assert column.tolist() == [1, 2, 3]

    def test__init___given_scalar(self):
        column = DataFrameColumn(1)
        assert column.tolist() == [1]

    def test___init___with_dtype(self):
        column = DataFrameColumn([1, 2, 3], dtype=float)
        assert column.is_float()

    def test__init___with_nrow(self):
        column = DataFrameColumn([1], nrow=3)
        assert column.tolist() == [1, 1, 1]

    def test_nrow(self):
        assert DataFrameColumn([1, 2, 3]).nrow == 3

class TestDataFrame:

    def test___init___broadcast(self):
        data = DataFrame(a=[1, 2, 3], b=[1], c=1)
        assert data.a.tolist() == [1, 2, 3]
        assert data.b.tolist() == [1, 1, 1]
        assert data.c.tolist() == [1, 1, 1]

    def test___init___broadcast_error(self):
        with pytest.raises(ValueError, match="'b'.* length 2 to 3 rows"):
            DataFrame(a=[1, 2, 3], b=[1, 2])

    def test___init___empty(self):
        data = DataFrame()
        assert data.nrow == 0
        assert data.ncol == 0
        assert not data.columns
        assert not data.colnames

    def test___init___given_data_frame_column(self):
        data = DataFrame(a=DataFrameColumn([1, 2, 3]))
        assert data.a.tolist() == [1, 2, 3]

    def test___init___given_list(self):
        data = DataFrame(a=[1, 2, 3])
        assert data.a.tolist() == [1, 2, 3]

    def test___init___reserved_names(self):
        # Columns names that conflict with method names should
        # not be made attributes, only accessible as keys.
        data = DataFrame(items=[1, 2, 3])
        assert callable(data.items)
        assert not callable(data["items"])

    def test___delattr__(self):
        data = DataFrame(a=[1, 2, 3])
        assert "a" in data
        del data.a
        assert "a" not in data

    def test___eq__(self):
        data = test.data_frame("vehicles.csv")
        assert data == data.copy()
        assert data == data.deepcopy()

    def test___getattr__(self):
        data = test.data_frame("vehicles.csv")
        assert data.make is data["make"]

    def test___dir__(self):
        data = DataFrame(a=[1, 2, 3], items=[1, 2, 3])
        data["b c"] = 1
        names = dir(data)
        assert "a" in names
        assert "b c" not in names
        assert names.count("items") == 1
        assert "left_join" in names

    def test___setattr__(self):
        data = test.data_frame("vehicles.csv")
        assert "test" not in data
        data.test = 1
        assert "test" in data

    def test___setitem__(self):
        data = test.data_frame("vehicles.csv")
        assert "test" not in data
        data["test"] = 1
        assert "test" in data

    def test___setitem___broadcast_error(self):
        data = DataFrame(a=[1, 2, 3])
        with pytest.raises(ValueError, match="'b'.* length 2 to 3 rows"):
            data["b"] = [1, 2]
        with pytest.raises(ValueError, match="'b'.* length 2 to 3 rows"):
            data.modify(b=[1, 2])

    def test_aggregate(self):
        data = test.data_frame("vehicles.csv")
        data = data.filter_out(data.cyl.is_na())
        data = data.filter_out(data.displ.is_na())
        stat = (data
                .group_by("make", "model")
                .aggregate(cyl=di.median("cyl"),
                           displ=di.mean("displ")))

        assert stat.nrow == 3240
        assert stat.ncol == 4
        assert stat.sort(make=1, model=1) == stat
        assert np.isclose(np.sum(stat.cyl), 19964.5, atol=0.1)
        assert np.isclose(np.sum(stat.displ), 11430.1, atol=0.1)

    def test_aggregate_ungrouped(self):
        data = test.data_frame("downloads.csv")
        stat = data.aggregate(n=di.count(), downloads=lambda x: x.downloads.sum())
        assert stat.nrow == 1
        assert stat.n[0] == data.nrow
        assert stat.downloads[0] == data.downloads.sum()

    def test_anti_join(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        data = orig.anti_join(holidays, "date")
        assert data.nrow == 870
        assert data.ncol == orig.ncol
        assert np.sum(data.downloads) == 523109256

    def test_cbind(self):
        data1 = test.data_frame("vehicles.csv")
        data2 = test.data_frame("vehicles.csv")
        data1.colnames = [x + "1" for x in data1.colnames]
        data2.colnames = [x + "2" for x in data2.colnames]
        data = data1.cbind(data2)
        assert data.nrow == data1.nrow
        assert data.ncol == data1.ncol + data2.ncol

    def test_cbind_broadcast(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.cbind(DataFrame(test=1))
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol + 1
        assert np.all(data.test == 1)
        assert data.unselect("test") == orig

    def test_clear(self):
        data = test.data_frame("vehicles.csv")
        assert not data.clear()

    def test_colnames(self):
        data = test.data_frame("downloads.csv")
        assert data.colnames == ["category", "date", "downloads"]

    def test_colnames_set(self):
        data = test.data_frame("downloads.csv")
        data.colnames = ["a", "b", "c"]
        assert data.colnames == ["a", "b", "c"]

    def test_colnames_set_again(self):
        data = test.data_frame("downloads.csv")
        data.colnames = ["a", "b", "c"]
        assert data.colnames == ["a", "b", "c"]
        data.colnames = ["a", "b", "c"]
        assert data.colnames == ["a", "b", "c"]

    def test_colnames_set_reserved(self):
        # One should be able to use column names that clash with
        # method names without any error setting or deleting attributes.
        data = test.data_frame("downloads.csv")
        data.colnames = ["a", "count", "update"]
        assert data.colnames == ["a", "count", "update"]
        assert callable(data.count)
        assert callable(data.update)
        data.colnames = ["a", "b", "c"]
        assert data.colnames == ["a", "b", "c"]
        assert callable(data.count)
        assert callable(data.update)

    def test_columns(self):
        data = test.data_frame("downloads.csv")
        assert data.columns == [data.category, data.date, data.downloads]

    def test_compare(self):
        old = test.data_frame("vehicles.csv")
        new = old.modify(hwy=lambda x: np.minimum(100, x.hwy))
        added, removed, changed = new.compare(old, "id")
        assert added.nrow == 0
        assert removed.nrow == 0
        assert changed.nrow == 9
        assert all(changed.column == "hwy")

    def test_compare_equal(self):
        data = test.data_frame("vehicles.csv")
        for value in data.compare(data, "id"):
            assert isinstance(value, DataFrame)
            assert value.nrow == 0
            assert not value

    def test_compare_na(self):
        old = DataFrame(a=[1, 1, 2], b=["x", "", ""], c=[1, 2, 3])
        new = old.modify(c=[1, 5, 3])
        added, removed, changed = new.compare(old, "a", "b")
        assert added.nrow == 0
        assert removed.nrow == 0
        assert changed.nrow == 1
        assert changed.a.tolist() == [1]
        assert changed.b.tolist() == [None]
        assert changed.xvalue.tolist() == [5]

    def test_copy(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.copy()
        assert data == orig
        assert data is not orig

    def test_count(self):
        data = test.data_frame("listings.csv")
        stat = data.count("hood")
        assert stat.hood.equal(data.hood.unique().sort())
        assert stat.n.sum() == data.nrow

    def test_count_n_exists(self):
        data = test.data_frame("listings.csv")
        stat1 = data.count("zipcode")
        stat2 = data.count("zipcode").count("n")
        assert stat1.colnames == ["zipcode", "n"]
        assert stat2.colnames == ["n_", "n"]
        assert stat2.nrow == len(stat1.n.unique())
        assert sum(stat2.n) == stat1.nrow

    def test_deepcopy(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.copy()
        assert data == orig
        assert data is not orig

    def test_drop_na(self):
        data = test.data_frame("listings.csv")
        data = data.drop_na("sqft")
        assert data.nrow == 396
        assert not (data.sqft.is_na()).any()

    def test_drop_na_all(self):
        data = DataFrame(a=[1, None, 3], b=["x", "y", ""])
        assert data.drop_na().a.tolist() == [1]

    def test_expand_grid(self):
        data = DataFrame.expand_grid(x=[1, 2], y=["a", "b", "c"])
        assert data.colnames == ["x", "y"]
        assert data.x.tolist() == [1, 1, 1, 2, 2, 2]
        assert data.y.tolist() == ["a", "b", "c", "a", "b", "c"]

    def test_explode_given_sep(self):
        data = DataFrame(id=[1, 2, 3], x=["a,b", "", "c"])
        for sep in [",", ", "]:
            data.x = data.x.str.replace(",", sep)
            result = data.explode("x", sep=sep)
            assert result.id.tolist() == [1, 1, 2, 3]
            assert result.x.tolist() == ["a", "b", None, "c"]
            assert result.x.is_string()

    def test_explode_given_sequences(self):
        data = DataFrame(id=[1, 2, 3, 4], x=["[1, 2]", "null", "[]", "[3, null]"])
        data.x = data.x.map(json.loads)
        result = data.explode("x")
        assert result.id.tolist() == [1, 1, 2, 3, 4, 4]
        assert result.x.is_float()
        assert np.array_equal(result.x, [1, 2, np.nan, np.nan, 3, np.nan], equal_nan=True)
        assert data.explode("x", dtype=int).x.is_float()
        assert data.filter(id=1).explode("x", dtype=int).x.tolist() == [1, 2]

    def test_filter_given_rows(self):
        data = test.data_frame("vehicles.csv")
        data = data.filter(data.make == "Saab")
        assert data.nrow == 424
        assert np.all(data.make == "Saab")
        assert np.sum(data.hwy) == 10672

    def test_filter_given_colname_value_pairs(self):
        data = test.data_frame("vehicles.csv")
        data = data.filter(make="Saab")
        assert data.nrow == 424
        assert np.all(data.make == "Saab")
        assert np.sum(data.hwy) == 10672

    def test_filter_given_colname_sequence_pairs(self):
        data = DataFrame(x=["a", "b", "c", "d"], y=[1, 2, 3, 4])
        for value in [["a", "c"], ("a", "c"), {"a", "c"}, np.array(["a", "c"]), Vector(["a", "c"])]:
            assert data.filter(x=value).y.tolist() == [1, 3]
        assert data.filter(x=["a", "c"], y=3).y.tolist() == [3]
        assert data.filter_out(x=["a", "c"]).y.tolist() == [2, 4]

    def test_filter_given_colname_long_sequence_pairs(self):
        data = test.data_frame("vehicles.csv")
        makes = data.make.unique()[:20].tolist()
        expected = data.filter(np.isin(data.make, makes))
        assert data.filter(make=makes) == expected
        assert data.filter(make=iter(makes)) == expected

    def test_filter_group_wise(self):
        data = DataFrame(g=[1, 3, 3, 2, 3, 2], x=[1, 5, 9, 2, 3, 7])
        data = data.group_by("g").filter(lambda x: x.x == x.x.max())
        assert data.g.tolist() == [1, 3, 2]
        assert data.x.tolist() == [1, 9, 7]

    def test_filter_group_wise_scalar(self):
        data = DataFrame(g=[1, 3, 3, 2, 3, 2], x=[1, 5, 9, 2, 3, 7])
        data = data.group_by("g").filter(lambda x: x.nrow > 1)
        assert data.g.tolist() == [3, 3, 2, 3, 2]

    def test_filter_out_group_wise(self):
        data = DataFrame(g=[1, 3, 3, 2, 3, 2], x=[1, 5, 9, 2, 3, 7])
        data = data.group_by("g").filter_out(lambda x: x.x == x.x.max())
        assert data.g.tolist() == [3, 2, 3]
        assert data.x.tolist() == [5, 2, 3]

    def test_filter_out_given_rows(self):
        data = test.data_frame("vehicles.csv")
        data = data.filter_out(data.make == "Saab")
        assert data.nrow == 33018
        assert np.all(data.make != "Saab")
        assert np.sum(data.hwy) == 776930

    def test_filter_out_given_colname_value_pairs(self):
        data = test.data_frame("vehicles.csv")
        data = data.filter_out(make="Saab")
        assert data.nrow == 33018
        assert np.all(data.make != "Saab")
        assert np.sum(data.hwy) == 776930

    def test_from_arrow(self):
        import pyarrow as pa
        orig = test.data_frame("vehicles.csv")
        data = orig.to_arrow()
        assert isinstance(data, pa.Table)
        assert data.num_rows == orig.nrow
        assert data.num_columns == orig.ncol
        data = DataFrame.from_arrow(data)
        assert data == orig

    def test_from_arrow_decimal(self):
        import decimal
        import pyarrow as pa
        values = [decimal.Decimal("1.10"), None, decimal.Decimal("123456789.25")]
        table = pa.table({"a": pa.array(values, pa.decimal128(12, 2))})
        data = DataFrame.from_arrow(table)
        assert data.a.is_float()
        assert data.a.tolist() == [1.1, None, 123456789.25]
        data = DataFrame.from_arrow(table, dtypes={"a": object})
        assert data.a.tolist() == values

    def test_from_arrow_list(self):
        import pyarrow as pa
        orig = DataFrame(a=[[1, 2], None, []], b=[[1, 2], [3, 4], [5, 6]])
        data = DataFrame.from_arrow(orig.to_arrow())
        assert data.a.is_object()
        assert data.a.tolist() == [[1, 2], None, []]
        assert data.b.tolist() == [[1, 2], [3, 4], [5, 6]]
        array = pa.array([[1, 2], [3, 4]], pa.list_(pa.int64(), 2))
        data = DataFrame.from_arrow(pa.table({"a": array}))
        assert data.a.tolist() == [[1, 2], [3, 4]]

    def test_from_arrow_null(self):
        import pyarrow as pa
        table = pa.table({"a": pa.nulls(2)})
        data = DataFrame.from_arrow(table)
        assert data.a.is_float()
        assert data.a.is_na().all()
        data = DataFrame.from_arrow(table, dtypes={"a": object})
        assert data.a.is_object()

    def test_from_arrow_bytes_missing(self):
        orig = DataFrame(a=[b"x", None])
        data = DataFrame.from_arrow(orig.to_arrow())
        assert data.a.is_bytes()
        assert data.a.tolist() == [b"x", None]

    def test_from_arrow_string_first_missing(self):
        orig = DataFrame(a=["", "x"])
        data = DataFrame.from_arrow(orig.to_arrow())
        assert data.a.is_string()
        assert data.a.tolist() == [None, "x"]

    def test_from_arrow_time(self):
        import pyarrow as pa
        table = pa.table({"a": pa.array([datetime.time(16, 6, 30), None], pa.time32("s"))})
        data = DataFrame.from_arrow(table)
        assert data.a.is_string()
        assert data.a.tolist() == ["16:06:30", None]

    def test_from_arrow_time_zeros(self):
        import pyarrow as pa
        a = pa.array([datetime.time(16, 6), datetime.time(8, 0, 30)], pa.time64("us"))
        b = pa.array([datetime.time(16, 6), datetime.time(8, 0)], pa.time64("us"))
        data = DataFrame.from_arrow(pa.table({"a": a, "b": b}))
        assert data.a.tolist() == ["16:06:00", "08:00:30"]
        assert data.b.tolist() == ["16:06", "08:00"]

    def test_from_arrow_timezone(self):
        import pyarrow as pa
        values = [datetime.datetime(2025, 1, 1, 10), datetime.datetime(2025, 7, 1, 10), None]
        a = pa.array(values, pa.timestamp("s", tz="Europe/Helsinki"))
        b = pa.array(values, pa.timestamp("s", tz="+03:00"))
        table = pa.table({"a": a, "b": b})
        data = DataFrame.from_arrow(table)
        assert data.a.tolist() == [datetime.datetime(2025, 1, 1, 12), datetime.datetime(2025, 7, 1, 13), None]
        assert data.b.tolist() == [datetime.datetime(2025, 1, 1, 13), datetime.datetime(2025, 7, 1, 13), None]
        data = DataFrame.from_arrow(table, timezone="utc")
        assert data.a.tolist() == values
        assert data.b.tolist() == values
        with pytest.raises(ValueError):
            DataFrame.from_arrow(table, timezone="local")

    def test_from_json(self):
        orig = test.data_frame("downloads.json")
        text = orig.to_json()
        data = DataFrame.from_json(text)
        assert data == orig

    def test_from_pandas(self):
        import pandas as pd
        orig = test.data_frame("vehicles.csv")
        data = orig.to_pandas()
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] == orig.nrow
        assert data.shape[1] == orig.ncol
        data = DataFrame.from_pandas(data)
        assert data == orig

    def test_from_pandas_date(self):
        orig = test.data_frame("downloads.csv")
        assert orig.date.dtype == np.dtype("datetime64[D]")
        data = DataFrame.from_pandas(orig.to_pandas())
        assert data.date.dtype == np.dtype("datetime64[D]")
        assert data == orig

    def test_full_join(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        data = orig.full_join(holidays, "date")
        assert data.nrow == 930
        assert data.ncol == 4
        assert sum(~data.holiday.is_na()) == 60
        assert sum(data.downloads.is_na()) == 25
        assert np.nansum(data.downloads) == 541335745

    def test_full_join_multiple(self):
        a = DataFrame(k=[1, 2, 3], x=[1, 2, 3])
        b = DataFrame(k=[2, 1, 2, 4], y=[1, 2, 3, 4])
        data = a.full_join(b, "k", multiple="all")
        assert data.k.tolist() == [1, 2, 2, 3, 4]
        assert data.x.tolist() == [1, 2, 2, 3, None]
        assert data.y.tolist() == [2, 1, 3, None, 4]

    def test_head(self):
        data = test.data_frame("vehicles.csv")
        assert data.head(10) == data.slice(list(range(10)))

    def test_head_group_wise(self):
        data = DataFrame(g=[2, 1, 2, 1, 2], x=[1, 2, 3, 4, 5])
        data = data.group_by("g").head(2)
        assert data.g.tolist() == [1, 1, 2, 2]
        assert data.x.tolist() == [2, 4, 1, 3]

    def test_head_negative(self):
        data = DataFrame(g=[2, 1, 2, 1, 2], x=[1, 2, 3, 4, 5])
        assert data.head(-2).x.tolist() == [1, 2, 3]
        assert data.head(-9).x.tolist() == []
        assert data.group_by("g").head(-1).x.tolist() == [2, 1, 3]

    def test_inner_join(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        data = orig.inner_join(holidays, "date")
        assert data.nrow == 35
        assert data.ncol == orig.ncol + 1
        assert np.sum(data.holiday.is_na()) == 0
        assert np.sum(data.downloads) == 18226489

    def test_inner_join_multiple(self):
        a = DataFrame(k=[1, 2, 3], x=[1, 2, 3])
        b = DataFrame(k=[2, 1, 2], y=[1, 2, 3])
        assert a.inner_join(b, "k", multiple="first").y.tolist() == [2, 1]
        assert a.inner_join(b, "k", multiple="last").y.tolist() == [2, 3]
        data = a.inner_join(b, "k", multiple="all")
        assert data.x.tolist() == [1, 2, 2]
        assert data.y.tolist() == [2, 1, 3]

    def test_left_join(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        data = orig.left_join(holidays, "date")
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol + 1
        assert np.sum(~data.holiday.is_na()) == 35
        assert np.sum(data.downloads) == 541335745

    def test_left_join_all_found(self):
        a = DataFrame(k=[1, 2])
        b = DataFrame(k=[2, 1], n=[3, 4], f=[True, False])
        data = a.left_join(b, "k")
        assert data.n.is_integer()
        assert data.f.is_boolean()
        assert data.n.tolist() == [4, 3]
        assert data.f.tolist() == [False, True]

    def test_left_join_by_tuple(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        holidays = holidays.rename(holiday_date="date")
        data = orig.left_join(holidays, ("date", "holiday_date"))
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol + 1
        assert np.sum(~data.holiday.is_na()) == 35
        assert np.sum(data.downloads) == 541335745

    def test_left_join_incompatible_types(self):
        a = DataFrame(k=[1, 2])
        b = DataFrame(k=["1", "2"], v=[3, 4])
        with pytest.raises(TypeError):
            a.left_join(b, "k")
        b = DataFrame(k=[1.0, 2.0], v=[3, 4])
        assert a.left_join(b, "k").v.tolist() == [3, 4]

    def test_left_join_multiple(self):
        a = DataFrame(k=[1, 2, 3, np.nan], x=[1, 2, 3, 4])
        b = DataFrame(k=[2, 1, 2, np.nan], y=[1, 2, 3, 4])
        assert a.left_join(b, "k", multiple="first").y.tolist() == [2, 1, None, None]
        assert a.left_join(b, "k", multiple="last").y.tolist() == [2, 3, None, None]
        data = a.left_join(b, "k", multiple="all")
        assert data.x.tolist() == [1, 2, 2, 3, 4]
        assert data.y.tolist() == [2, 1, 3, None, None]

    def test_left_join_multiple_strings(self):
        # Test both self smaller and larger than other.
        a = DataFrame(k=["b", "a", "c", ""], x=[1, 2, 3, 4])
        b = DataFrame(k=["a", "b", "a", "", "d", "e"], y=[1, 2, 3, 4, 5, 6])
        assert a.left_join(b, "k", multiple="first").y.tolist() == [2, 1, None, None]
        assert a.left_join(b, "k", multiple="last").y.tolist() == [2, 3, None, None]
        data = a.left_join(b, "k", multiple="all")
        assert data.x.tolist() == [1, 2, 2, 3, 4]
        assert data.y.tolist() == [2, 1, 3, None, None]
        assert b.left_join(a, "k").x.tolist() == [2, 1, 2, None, None, None]

    def test_left_join_multiple_empty(self):
        a = DataFrame(k=[1, 2], x=[1, 2])
        b = DataFrame(k=[1, 2], y=[1, 2]).head(0)
        assert a.left_join(b, "k", multiple="all").y.tolist() == [None, None]
        assert b.left_join(a, "k", multiple="all").nrow == 0

    def test_map(self):
        data = test.data_frame("vehicles.csv")
        x = data.map(lambda x, i: x.hwy[i]**2)
        y = [x**2 for x in data.hwy]
        assert x == y

    def test_modify(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.modify(test=1)
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol + 1
        assert np.all(data.test == 1)

    def test_modify_function(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.modify(test=lambda x: x.make)
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol + 1
        assert np.all(data.test == data.make)

    def test_modify_group_wise(self):
        data = DataFrame(g=[1, 2, 2, 3, 3, 3])
        data = data.group_by("g").modify(f=lambda x: 1 / x.nrow)
        assert data.f.tolist() == [1, 1/2, 1/2, 1/3, 1/3, 1/3]

    def test_modify_group_wise_unsorted(self):
        data = DataFrame(g=[1, 3, 3, 3, 2, 2])
        data = data.group_by("g").modify(f=lambda x: 1 / x.nrow)
        assert data.f.tolist() == [1, 1/3, 1/3, 1/3, 1/2, 1/2]

    def test_modify_group_wise_sequential(self):
        data = DataFrame(g=[1, 3, 3, 3, 2, 2], x=[1, 2, 3, 4, 5, 6])
        data = data.group_by("g").modify(s=lambda x: x.x.sum(), f=lambda x: x.x / x.s)
        assert data.s.tolist() == [1, 9, 9, 9, 11, 11]
        assert data.f.tolist() == [1, 2/9, 3/9, 4/9, 5/11, 6/11]

    def test_modify_sequential(self):
        data = DataFrame(x=[1, 2, 3])
        data = data.modify(y=lambda x: x.x * 2, x=lambda x: x.y + 1, z=lambda x: x.x)
        assert data.colnames == ["x", "y", "z"]
        assert data.x.tolist() == [3, 5, 7]
        assert data.y.tolist() == [2, 4, 6]
        assert data.z.tolist() == [3, 5, 7]

    def test_ncol(self):
        data = test.data_frame("vehicles.csv")
        assert data.ncol == 12

    def test_nrow(self):
        data = test.data_frame("vehicles.csv")
        assert data.nrow == 33442

    def test_pivot_longer(self):
        wide = test.data_frame("listings.csv")
        wide = wide.sort(id=1).head(10)
        long = wide.pivot_longer(ids="id",
                                 names="name",
                                 values="value")

        assert long.nrow == 50
        assert long.ncol == 3
        assert all(long.name.is_in(wide.colnames))

    def test_pivot_longer_common_type(self):
        wide = DataFrame(id=[1, 2], a=[True, False], b=[3, 4], c=[5.5, 6.5])
        long = wide.pivot_longer(ids="id")
        assert long.value.is_float()
        assert long.value.tolist() == [1.0, 3.0, 5.5, 0.0, 4.0, 6.5]

    def test_pivot_longer_order(self):
        wide = DataFrame(id=["b", "a", "b"], x=[1, 2, 3], y=[4, 5, 6])
        long = wide.pivot_longer(ids="id")
        assert long.id.tolist() == ["b", "b", "a", "a", "b", "b"]
        assert long.name.tolist() == ["x", "y", "x", "y", "x", "y"]
        assert long.value.tolist() == [1, 4, 2, 5, 3, 6]

    def test_pivot_longer_mixed_kinds(self):
        wide = DataFrame(id=[1, 2], a=[3, 4], b=np.array([5, 6], "timedelta64[s]"))
        long = wide.pivot_longer(ids="id")
        assert long.value.is_object()

    def test_pivot_longer_defaults(self):
        wide = DataFrame(id=[1, 2], a=[3, 4], b=[5, 6])
        long = wide.pivot_longer(ids="id")
        assert long.colnames == ["id", "name", "value"]
        assert long.pivot_wider(ids="id") == wide

    def test_pivot_wider(self):
        long = test.data_frame("downloads.csv")
        long = long.sort(date=1).tail(100)
        wide = long.pivot_wider(ids="date",
                                names="category",
                                values="downloads",
                                rename=lambda x: f"dl_{x.lower()}")

        assert wide.nrow == 20
        assert wide.ncol == 6
        total_downloads = sum(
            sum(wide[x]) for x in wide.colnames
            if x.startswith("dl_"))
        assert total_downloads == sum(long.downloads)

    def test_pivot_wider_duplicates(self):
        long = DataFrame(id=[1, 1, 2, 2], m=["a", "b", "a", "a"], value=[1, 2, 3, 4])
        with pytest.raises(ValueError, match=r"\{'id': 2, 'm': 'a'\}"):
            long.pivot_wider(ids="id", names="m", values="value")

    def test_pivot_wider_missing(self):
        long = DataFrame(id=[1, 1, 2], m=["a", "b", "a"], value=[1, 2, 3])
        wide = long.pivot_wider(ids="id", names="m", values="value")
        assert wide.a.tolist() == [1, 3]
        assert wide.a.is_integer()
        assert wide.b.tolist() == [2, None]

    def test_pivot_wider_integer_names(self):
        long = DataFrame(id=[1, 1, 2], m=[2, 10, 1], value=[1, 2, 3])
        wide = long.pivot_wider(ids="id", names="m", values="value")
        assert wide.colnames == ["id", "2", "10", "1"]
        wide = long.pivot_wider(ids="id", names="m", values="value", rename=lambda x: f"m{x}")
        assert wide.colnames == ["id", "m2", "m10", "m1"]

    def test_pivot_wider_order(self):
        long = DataFrame(id=[1, 1, 1, 2, 2], m=["Mar", "", "Jan", "Feb", "Jan"], value=[1, 2, 3, 4, 5])
        wide = long.pivot_wider(ids="id", names="m", values="value")
        assert wide.colnames == ["id", "Mar", "Jan", "Feb", ""]

    def test_pop(self):
        data = test.data_frame("vehicles.csv")
        data.pop("fuel")
        assert "fuel" not in data
        assert not hasattr(data, "fuel")

    def test_popitem(self):
        data = test.data_frame("vehicles.csv")
        key, value = data.popitem()
        assert key not in data
        assert not hasattr(data, key)
        assert isinstance(value, DataFrameColumn)

    @patch("builtins.print")
    def test_print_memory_use(self, mock_print):
        data = test.data_frame("vehicles.csv")
        data.print_memory_use()
        mock_print.assert_called()

    @patch("builtins.print")
    @patch("dataiter.PRINT_MAX_ROWS", 1)
    def test_print_memory_use_all_rows(self, mock_print):
        data = test.data_frame("vehicles.csv")
        data.print_memory_use()
        assert "TOTAL" in mock_print.call_args.args[0]

    @patch("builtins.print")
    def test_print_na_counts(self, mock_print):
        data = test.data_frame("vehicles.csv")
        data.print_na_counts()
        mock_print.assert_called()

    @patch("builtins.print")
    def test_print_na_counts_none(self, mock_print):
        data = test.data_frame("vehicles.csv")
        data = data.select("id", "make", "model")
        data.print_na_counts()
        mock_print.assert_called_once_with("No missing values")

    def test_rbind(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.rbind(orig, orig)
        assert data.nrow == orig.nrow * 3
        assert data.ncol == orig.ncol
        assert data.slice(range(orig.nrow*0, orig.nrow*1)) == orig
        assert data.slice(range(orig.nrow*1, orig.nrow*2)) == orig
        assert data.slice(range(orig.nrow*2, orig.nrow*3)) == orig

    def test_rbind_na(self):
        part1 = test.data_frame("vehicles.csv")
        part2 = test.data_frame("vehicles.csv")
        part1.test1 = 1
        part2.test2 = 2
        data = part1.rbind(part2)
        assert data.nrow == part1.nrow + part2.nrow
        assert data.ncol == part1.ncol + 1
        assert np.all(data.test1[:3] == 1)
        assert np.all(np.isnan(data.test1[-3:]))
        assert np.all(np.isnan(data.test2[:3]))
        assert np.all(data.test2[-3:] == 2)

    def test_rbind_incompatible(self):
        a = DataFrame(x=[1.5])
        b = DataFrame(x=["a"])
        with pytest.raises(TypeError, match="'x'.*float64, string"):
            a.rbind(b)

    def test_rbind_na_all(self):
        # As read_csv gives for blank columns.
        na = Vector([np.nan], float)
        a = DataFrame(s=na, t=na, x=na)
        b = DataFrame(s=["x"], t=[np.datetime64("2025-01-01")], x=na)
        data = a.rbind(b)
        assert data.s.tolist() == [None, "x"]
        assert data.t.tolist() == [None, datetime.date(2025, 1, 1)]
        assert data.x.tolist() == [None, None]
        assert data.s.is_string()
        assert data.t.is_datetime()
        assert data.x.is_float()

    def test_rbind_onto_empty(self):
        data = DataFrame().rbind(DataFrame(a=[1, 2]))
        assert data.a.is_integer()
        data = data.rbind(DataFrame(b=[3]))
        assert data.a.is_float()
        assert data.b.is_float()

    def test_rbind_ignore_empty(self):
        a = DataFrame(s=["x"], n=[1])
        b = DataFrame(n=[2])
        empty = DataFrame(s=[1.5], n=[1.5], x=[1.5]).head(0)
        data = empty.rbind(a, b)
        assert data.s.tolist() == ["x", None]
        assert data.n.tolist() == [1, 2]
        assert data.x.tolist() == [None, None]
        assert data.s.is_string()
        assert data.n.is_integer()
        assert empty.rbind(empty).n.is_float()

    def test_read_csv(self):
        path = str(test.get_data_path("vehicles.csv"))
        data = DataFrame.read_csv(path)
        assert data.nrow == 33442
        assert data.ncol == 12

    def test_read_csv_columns(self):
        path = test.get_data_path("vehicles.csv")
        data = DataFrame.read_csv(path, columns=["make", "model"])
        assert data.colnames == ["make", "model"]

    def test_read_csv_dtypes(self):
        path = test.get_data_path("vehicles.csv")
        dtypes = {"make": object, "model": object}
        data = DataFrame.read_csv(path, dtypes=dtypes)
        assert data.make.is_object()
        assert data.model.is_object()

    def test_read_csv_dtypes_string(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("zip\n02134\n", "utf-8")
        data = DataFrame.read_csv(path, dtypes={"zip": str})
        assert data.zip.tolist() == ["02134"]

    def test_read_csv_fractional_seconds(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("time\n2026-01-02 05:36:24.539\n", "utf-8")
        data = DataFrame.read_csv(path)
        assert data.time.dtype == np.dtype("datetime64[us]")
        assert data.time.dt.hour().tolist() == [5]

    def test_read_csv_empty_column(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("a,b\n1,\n2,\n", "utf-8")
        data = DataFrame.read_csv(path)
        assert data.b.is_float()
        assert data.b.is_na().all()

    def test_read_csv_time(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("time\n16:06\n08:00\n", "utf-8")
        data = DataFrame.read_csv(path)
        assert data.time.is_string()
        assert data.time.tolist() == ["16:06", "08:00"]

    def test_read_csv_no_header_columns(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("1,2,3\n", "utf-8")
        data = DataFrame.read_csv(path, header=False, columns=["c", "a"])
        assert data.colnames == ["c", "a"]
        assert data.c.tolist() == [3]

    def test_read_csv_no_header_dtypes(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("02134,007\n", "utf-8")
        data = DataFrame.read_csv(path, header=False, dtypes={"b": str})
        assert data.b.tolist() == ["007"]

    def test_read_csv_missing_blank(self, tmp_path):
        text = """
int,float,string,bool,date
,2.5,"test",TRUE,2025-01-01
1,,"test",TRUE,2025-01-01
1,2.5,"",TRUE,2025-01-01
1,2.5,"test",,2025-01-01
1,2.5,"test",TRUE,
""".strip()
        path = tmp_path / "test.csv"
        path.write_text(text, "utf-8")
        data = DataFrame.read_csv(path)
        for i, column in enumerate(data.columns):
            assert column.is_na().sum() == 1
            assert column.is_na()[i]

    def test_read_csv_missing_excel(self, tmp_path):
        text = """
int,float,string,bool,date
#N/A,2.5,"test",TRUE,2025-01-01
1,#N/A,"test",TRUE,2025-01-01
1,2.5,"",TRUE,2025-01-01
1,2.5,"test",#N/A,2025-01-01
1,2.5,"test",TRUE,#N/A
""".strip()
        path = tmp_path / "test.csv"
        path.write_text(text, "utf-8")
        data = DataFrame.read_csv(path)
        for i, column in enumerate(data.columns):
            assert column.is_na().sum() == 1
            assert column.is_na()[i]

    def test_read_csv_missing_na(self, tmp_path):
        text = """
int,float,string,bool,date
NA,2.5,"test",TRUE,2025-01-01
1,NA,"test",TRUE,2025-01-01
1,2.5,"",TRUE,2025-01-01
1,2.5,"test",NA,2025-01-01
1,2.5,"test",TRUE,NA
""".strip()
        path = tmp_path / "test.csv"
        path.write_text(text, "utf-8")
        data = DataFrame.read_csv(path)
        for i, column in enumerate(data.columns):
            assert column.is_na().sum() == 1
            assert column.is_na()[i]

    def test_read_csv_na(self, tmp_path):
        text = """
int,float,string,bool,date,other
\\N,2.5,test,TRUE,2025-01-01,NA
1,\\N,test,TRUE,2025-01-01,NA
1,2.5,\\N,TRUE,2025-01-01,NA
1,2.5,test,\\N,2025-01-01,NA
1,2.5,test,TRUE,\\N,NA
""".strip()
        path = tmp_path / "test.csv"
        path.write_text(text, "utf-8")
        data = DataFrame.read_csv(path, na="\\N", dtypes={"other": str})
        for i, column in enumerate(data.columns[:-1]):
            assert column.is_na().sum() == 1
            assert column.is_na()[i]
        assert data.int.is_float()
        assert data.string.is_string()
        assert data.other.tolist() == ["NA"] * 5

    def test_read_csv_quote(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text('a\tb\n"x\t1\n', "utf-8")
        data = DataFrame.read_csv(path, sep="\t", quote=None)
        assert data.a.tolist() == ['"x']
        assert data.b.tolist() == [1]

    def test_read_csv_missing_null(self, tmp_path):
        text = """
int,float,string,bool,date
NULL,2.5,"test",TRUE,2025-01-01
1,NULL,"test",TRUE,2025-01-01
1,2.5,"",TRUE,2025-01-01
1,2.5,"test",NULL,2025-01-01
1,2.5,"test",TRUE,NULL
""".strip()
        path = tmp_path / "test.csv"
        path.write_text(text, "utf-8")
        data = DataFrame.read_csv(path)
        for i, column in enumerate(data.columns):
            assert column.is_na().sum() == 1
            assert column.is_na()[i]

    def test_read_json(self):
        path = str(test.get_data_path("downloads.json"))
        data = DataFrame.read_json(path)
        assert data.nrow == 905
        assert data.ncol == 3

    def test_read_json_columns(self):
        path = str(test.get_data_path("vehicles.json"))
        data = DataFrame.read_json(path, columns=["make", "model"])
        assert data.colnames == ["make", "model"]

    def test_read_json_dtypes(self):
        path = str(test.get_data_path("vehicles.json"))
        dtypes = {"make": object, "model": object}
        data = DataFrame.read_json(path, dtypes=dtypes)
        assert data.make.is_object()
        assert data.model.is_object()

    def test_read_json_path(self):
        DataFrame.read_json(test.get_data_path("vehicles.json"))

    @pytest.mark.filterwarnings(IGNORE_NPZ_PICKLE_WARNING)
    def test_read_npz(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = str(tmp_path / "test.npz")
        orig.write_npz(path)
        data = DataFrame.read_npz(path)
        assert data == orig

    @pytest.mark.filterwarnings(IGNORE_NPZ_PICKLE_WARNING)
    def test_read_npz_path(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = tmp_path / "test.npz"
        orig.write_npz(path)
        DataFrame.read_npz(path)

    def test_read_parquet(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = str(tmp_path / "test.parquet")
        orig.write_parquet(path)
        data = DataFrame.read_parquet(path)
        assert data == orig

    def test_read_pickle(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = str(tmp_path / "test.pkl")
        orig.write_pickle(path)
        data = DataFrame.read_pickle(path)
        assert data == orig

    def test_read_pickle_path(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = tmp_path / "test.pkl"
        orig.write_pickle(path)
        DataFrame.read_pickle(path)

    def test_rename(self):
        orig = test.data_frame("downloads.csv")
        assert orig.colnames == ["category", "date", "downloads"]
        data = orig.rename(ymd="date")
        assert data.colnames == ["category", "ymd", "downloads"]
        assert data.category.equal(orig.category)
        assert data.ymd.equal(orig.date)
        assert data.downloads.equal(orig.downloads)

    def test_rename_missing(self):
        data = test.data_frame("downloads.csv")
        with pytest.raises(KeyError):
            data.rename(ymd="nope")

    def test_replace_na(self):
        orig = DataFrame(a=[1, None], b=["", "x"], c=[None, 2])
        data = orig.replace_na(a=0, b="-")
        assert data.colnames == ["a", "b", "c"]
        assert data.a.tolist() == [1, 0]
        assert data.b.tolist() == ["-", "x"]
        assert data.c.tolist() == [None, 2]
        assert orig.a.tolist() == [1, None]

    def test_sample(self):
        data = test.data_frame("vehicles.csv")
        assert data.sample(10).nrow == 10

    def test_select(self):
        orig = test.data_frame("vehicles.csv")
        orig_colnames = list(orig.colnames)
        data = orig.select("make", "model")
        assert data.colnames == ["make", "model"]
        assert orig.colnames == orig_colnames

    def test_semi_join(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        data = orig.semi_join(holidays, "date")
        assert data.nrow == 35
        assert data.ncol == orig.ncol
        assert np.sum(data.downloads) == 18226489

    def test_slice_boolean(self):
        data = DataFrame(x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9])
        data = data.slice(rows=[True, False, True], cols=np.array([True, False, True]))
        assert data.colnames == ["x", "z"]
        assert data.x.tolist() == [1, 3]

    def test_slice_given_both(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.slice(rows=[0, 1, 2], cols=[0, 1, 2])
        assert data.nrow == 3
        assert data.ncol == 3
        assert data.colnames == orig.colnames[:3]

    def test_slice_given_cols(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.slice(cols=[0, 1, 2])
        assert data.nrow == orig.nrow
        assert data.ncol == 3
        assert data.colnames == orig.colnames[:3]

    def test_slice_given_rows(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.slice(rows=[0, 1, 2])
        assert data.nrow == 3
        assert data.ncol == orig.ncol
        assert data.colnames == orig.colnames

    def test_slice_off_boolean(self):
        data = DataFrame(x=[1, 2, 3], y=[4, 5, 6], z=[7, 8, 9])
        data = data.slice_off(rows=[True, False, True], cols=np.array([True, False, True]))
        assert data.colnames == ["y"]
        assert data.y.tolist() == [5]

    def test_slice_off_given_both(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.slice_off(rows=[0, 1, 2], cols=[0, 1, 2])
        assert data.nrow == orig.nrow - 3
        assert data.ncol == orig.ncol - 3
        assert data.colnames == orig.colnames[3:]

    def test_slice_off_given_cols(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.slice_off(cols=[0, 1, 2])
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol - 3
        assert data.colnames == orig.colnames[3:]

    def test_slice_off_given_rows(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.slice_off(rows=[0, 1, 2])
        assert data.nrow == orig.nrow - 3
        assert data.ncol == orig.ncol
        assert data.colnames == orig.colnames

    def test_sort(self):
        orig = test.data_frame("vehicles.csv").sort(id=1)
        data = orig.sort(year=-1, make=1, model=1)
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol
        assert data.year.tolist() == sorted(data.year, reverse=True)

    def test_sort_object(self):
        orig1 = test.data_frame("vehicles.csv").sort(id=1)
        data1 = orig1.sort(year=-1, make=1, model=1)
        orig2 = test.data_frame("vehicles.csv").sort(id=1)
        orig2.make = orig2.make.as_object()
        orig2.model = orig2.model.as_object()
        data2 = orig2.sort(year=-1, make=1, model=1)
        assert (data1.year == data2.year).all()
        assert (data1.make == data2.make).all()
        assert (data1.model == data2.model).all()

    def test_sort_descending_unsigned(self):
        data = DataFrame(x=np.array([0, 3, 1], np.uint8))
        assert data.sort(x=-1).x.tolist() == [3, 1, 0]

    def test_sort_string_fixed_na(self):
        orig = DataFrame(x=np.array(["b", "", "a"]))
        data = orig.sort(x=1)
        assert orig.x.tolist() == ["b", None, "a"]
        assert data.x.tolist() == ["a", "b", None]

    @pytest.mark.parametrize("values", [
        np.array(["b", "", "a"]),
        Vector(["b", None, "a"]),
        Vector(["b", None, "a"], object),
        Vector([2.0, None, 1.0]),
        Vector([np.datetime64("2020-01-02"), None, np.datetime64("2020-01-01")]),
    ])
    def test_sort_na(self, values):
        data = DataFrame(g=[1, 1, 1], x=values)
        assert data.sort(x=1).x.is_na().tolist() == [False, False, True]
        assert data.sort(x=-1).x.is_na().tolist() == [False, False, True]
        assert data.sort(g=1, x=-1).x.is_na().tolist() == [False, False, True]
        assert data.sort(x=-1).x[0] == values[0]

    def test_split(self):
        data = DataFrame(
            x=[1, 2, 2, 3, 3, 3],
            y=[1, 1, 1, 1, 1, 2],
        )
        rows = data.split("x", "y")
        rows = [x.tolist() for x in rows]
        assert rows == [[0], [1, 2], [3, 4], [5]]

    def test_tail(self):
        data = test.data_frame("vehicles.csv")
        assert data.tail(10) == data.slice(list(range(data.nrow - 10, data.nrow)))

    def test_tail_group_wise(self):
        data = DataFrame(g=[2, 1, 2, 1, 2], x=[1, 2, 3, 4, 5])
        data = data.group_by("g").tail(2)
        assert data.g.tolist() == [1, 1, 2, 2]
        assert data.x.tolist() == [2, 4, 3, 5]

    def test_tail_negative(self):
        data = DataFrame(g=[2, 1, 2, 1, 2], x=[1, 2, 3, 4, 5])
        assert data.tail(-2).x.tolist() == [3, 4, 5]
        assert data.tail(-9).x.tolist() == []
        assert data.group_by("g").tail(-1).x.tolist() == [4, 3, 5]

    def test_to_json(self):
        orig = test.data_frame("downloads.json")
        text = orig.to_json()
        data = DataFrame.from_json(text)
        assert data == orig

    def test_to_list_of_dicts(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.to_list_of_dicts()
        assert len(data) == orig.nrow

    def test_to_arrow(self):
        import pyarrow as pa
        orig = test.data_frame("vehicles.csv")
        data = orig.to_arrow()
        assert isinstance(data, pa.Table)
        assert data.num_rows == orig.nrow
        assert data.num_columns == orig.ncol

    def test_to_pandas(self):
        import pandas as pd
        orig = test.data_frame("vehicles.csv")
        data = orig.to_pandas()
        assert isinstance(data, pd.DataFrame)
        assert data.shape[0] == orig.nrow
        assert data.shape[1] == orig.ncol

    def test_to_string(self):
        data = test.data_frame("vehicles.csv")
        assert data.head(0).to_string()
        assert data.head(5).to_string()

    def test_unique(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.unique("make", "year", "displ")
        assert data.nrow == 6415
        by = list(zip(data.make, data.year, data.displ))
        assert len(set(by)) == len(by)

    def test_unique_by_all(self):
        orig = test.data_frame("vehicles.csv")
        orig = orig.rbind(orig.tail(100))
        data = orig.unique()
        assert data.nrow == orig.nrow - 100

    def test_unique_by_one(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.unique("make")
        assert data.nrow == 128
        by = list(zip(data.make))
        assert len(set(by)) == len(by)

    def test_unique_by_same_dtype(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.unique("make", "model")
        assert data.nrow == 3264
        by = list(zip(data.make, data.model))
        assert len(set(by)) == len(by)

    def test_unique_preserve_order(self):
        orig = test.data_frame("vehicles.csv")
        orig.order = np.arange(orig.nrow)
        assert not orig.year.sort().equal(orig.year)
        assert not orig.cyl.sort().equal(orig.cyl)
        data = orig.unique("year", "cyl")
        assert np.all(np.diff(data.order) > 0)

    def test_unselect(self):
        orig = test.data_frame("vehicles.csv")
        orig_colnames = list(orig.colnames)
        data = orig.unselect("make", "model")
        assert data.colnames == [x for x in orig_colnames if not x in ["make", "model"]]
        assert orig.colnames == orig_colnames

    def test_unselect_missing(self):
        data = test.data_frame("vehicles.csv")
        with pytest.raises(KeyError):
            data.unselect("make", "nope")

    def test_update(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.update({"make": "Talbot", "test": 1})
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol + 1
        assert np.all(data.make == "Talbot")
        assert np.all(data.test == 1)

    def test_write_csv(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = str(tmp_path / "test.csv")
        orig.write_csv(path)
        data = DataFrame.read_csv(path)
        assert data == orig

    @pytest.mark.parametrize("extension", ["bz2", "gz", "xz"])
    def test_write_csv_compressed(self, tmp_path, extension):
        orig = test.data_frame("vehicles.csv")
        path = str(tmp_path / f"test.csv.{extension}")
        orig.write_csv(path)
        data = DataFrame.read_csv(path)
        assert data == orig

    def test_write_csv_path(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = tmp_path / "test.csv"
        orig.write_csv(path)

    def test_write_csv_timedelta(self, tmp_path):
        orig = DataFrame(a=np.array([615, 1440, "NaT"], "timedelta64[m]"))
        path = tmp_path / "test.csv"
        orig.write_csv(path)
        assert path.read_text("utf-8") == '"a"\n"10:15:00"\n"1 day 00:00:00"\n\n'
        assert orig.a.is_timedelta()

    def test_write_json(self, tmp_path):
        orig = test.data_frame("downloads.json")
        path = str(tmp_path / "test.json")
        orig.write_json(path)
        data = DataFrame.read_json(path)
        assert data == orig

    def test_write_json_path(self, tmp_path):
        orig = test.data_frame("downloads.json")
        path = tmp_path / "test.json"
        orig.write_json(path)

    @pytest.mark.filterwarnings(IGNORE_NPZ_PICKLE_WARNING)
    def test_write_npz(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = str(tmp_path / "test.npz")
        orig.write_npz(path)
        data = DataFrame.read_npz(path)
        assert data == orig

    @pytest.mark.filterwarnings(IGNORE_NPZ_PICKLE_WARNING)
    def test_write_npz_path(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = tmp_path / "test.npz"
        orig.write_npz(path)

    def test_write_parquet(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = str(tmp_path / "test.parquet")
        orig.write_parquet(path)
        data = DataFrame.read_parquet(path)
        assert data == orig

    def test_write_pickle(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = str(tmp_path / "test.pkl")
        orig.write_pickle(path)
        data = DataFrame.read_pickle(path)
        assert data == orig

    def test_write_pickle_path(self, tmp_path):
        orig = test.data_frame("vehicles.csv")
        path = tmp_path / "test.pkl"
        orig.write_pickle(path)
