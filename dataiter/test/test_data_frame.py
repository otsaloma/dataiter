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

import numpy as np
import tempfile

from dataiter import DataFrame
from dataiter import DataFrameColumn
from dataiter import test
from pathlib import Path
from unittest.mock import patch


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

    def test___delattr__(self):
        data = DataFrame(a=DataFrameColumn([1, 2, 3]))
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

    def test_aggregate(self):
        data = test.data_frame("vehicles.csv")
        # Avoid RuntimeWarning: All-NaN slice encountered.
        data = data.filter_out(np.isnan(data.cyl))
        data = data.filter_out(np.isnan(data.displ))
        stat = (data
                .group_by("make", "model")
                .aggregate(cyl=lambda x: np.median(x.cyl),
                           displ=lambda x: np.mean(x.displ)))

        assert stat.nrow == 3240
        assert stat.ncol == 4
        assert stat.sort(make=1, model=1) == stat
        assert np.isclose(np.sum(stat.cyl), 19964.5, atol=0.1)
        assert np.isclose(np.sum(stat.displ), 11430.1, atol=0.1)

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

    def test_colnames(self):
        data = test.data_frame("downloads.csv")
        assert data.colnames == ["category", "date", "downloads"]

    def test_colnames_set(self):
        data = test.data_frame("downloads.csv")
        data.colnames = ["a", "b", "c"]
        assert data.colnames == ["a", "b", "c"]

    def test_columns(self):
        data = test.data_frame("downloads.csv")
        assert data.columns == [data.category, data.date, data.downloads]

    def test_copy(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.copy()
        assert data == orig
        assert data is not orig

    def test_deepcopy(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.copy()
        assert data == orig
        assert data is not orig

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

    def test_full_join(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        data = orig.full_join(holidays, "date")
        assert data.nrow > orig.nrow
        assert data.ncol == orig.ncol + 1
        assert sum(data.holiday != "") == 60
        assert np.nansum(data.downloads) == 541335745

    def test_head(self):
        data = test.data_frame("vehicles.csv")
        assert data.head(10) == data.slice(list(range(10)))

    def test_inner_join(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        data = orig.inner_join(holidays, "date")
        assert data.nrow == 35
        assert data.ncol == orig.ncol + 1
        assert all(data.holiday != "")
        assert np.sum(data.downloads) == 18226489

    def test_left_join(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        data = orig.left_join(holidays, "date")
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol + 1
        assert sum(data.holiday != "") == 35
        assert np.sum(data.downloads) == 541335745

    def test_left_join_by_tuple(self):
        orig = test.data_frame("downloads.csv")
        holidays = test.data_frame("holidays.csv")
        holidays = holidays.rename(holiday_date="date")
        data = orig.left_join(holidays, ("date", "holiday_date"))
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol + 1
        assert sum(data.holiday != "") == 35
        assert np.sum(data.downloads) == 541335745

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

    def test_ncol(self):
        data = test.data_frame("vehicles.csv")
        assert data.ncol == 12

    def test_nrow(self):
        data = test.data_frame("vehicles.csv")
        assert data.nrow == 33442

    @patch("builtins.print")
    def test_print_memory_use(self, mock_print):
        data = test.data_frame("vehicles.csv")
        data.print_memory_use()
        mock_print.assert_called()

    @patch("builtins.print")
    def test_print_missing_counts(self, mock_print):
        data = test.data_frame("vehicles.csv")
        data.print_missing_counts()
        mock_print.assert_called()

    @patch("builtins.print")
    def test_print_missing_counts_none(self, mock_print):
        data = test.data_frame("vehicles.csv")
        data = data.select("id", "make", "model")
        data.print_missing_counts()
        mock_print.assert_not_called()

    def test_rbind(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.rbind(orig, orig)
        assert data.nrow == orig.nrow * 3
        assert data.ncol == orig.ncol
        assert data.slice(range(orig.nrow*0, orig.nrow*1)) == orig
        assert data.slice(range(orig.nrow*1, orig.nrow*2)) == orig
        assert data.slice(range(orig.nrow*2, orig.nrow*3)) == orig

    def test_rbind_missing(self):
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

    def test_read_csv(self):
        data = test.data_frame("vehicles.csv")
        assert data.nrow == 33442
        assert data.ncol == 12

    def test_read_csv_columns(self):
        path = str(test.get_data_path("vehicles.csv"))
        data = DataFrame.read_csv(path, columns=["make", "model"])
        assert data.colnames == ["make", "model"]

    def test_read_csv_path(self):
        DataFrame.read_csv(test.get_data_path("vehicles.csv"))

    def test_read_json(self):
        data = test.data_frame("downloads.json")
        assert data.nrow == 905
        assert data.ncol == 3

    def test_read_json_path(self):
        DataFrame.read_json(test.get_data_path("vehicles.json"))

    def test_read_npz(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".npz")
        orig.write_npz(path)
        data = DataFrame.read_npz(path)
        assert data == orig

    def test_read_npz_path(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".npz")
        orig.write_npz(path)
        DataFrame.read_npz(Path(path))

    def test_read_pickle(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".pkl")
        orig.write_pickle(path)
        data = DataFrame.read_pickle(path)
        assert data == orig

    def test_read_pickle_path(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".pkl")
        orig.write_pickle(path)
        DataFrame.read_pickle(Path(path))

    def test_rename(self):
        orig = test.data_frame("downloads.csv")
        assert orig.colnames == ["category", "date", "downloads"]
        data = orig.rename(ymd="date")
        assert data.colnames == ["category", "ymd", "downloads"]
        assert data.category.equal(orig.category)
        assert data.ymd.equal(orig.date)
        assert data.downloads.equal(orig.downloads)

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

    def test_sort(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.sort(year=-1, make=1, model=1)
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol
        assert data.year.tolist() == sorted(data.year, reverse=True)

    def test_tail(self):
        data = test.data_frame("vehicles.csv")
        assert data.tail(10) == data.slice(list(range(data.nrow - 10, data.nrow)))

    def test_to_json(self):
        orig = test.data_frame("downloads.json")
        text = orig.to_json()
        data = DataFrame.from_json(text)
        assert data == orig

    def test_to_list_of_dicts(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.to_list_of_dicts()
        assert len(data) == orig.nrow

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

    def test_update(self):
        orig = test.data_frame("vehicles.csv")
        data = orig.update({"make": "Talbot", "test": 1})
        assert data.nrow == orig.nrow
        assert data.ncol == orig.ncol + 1
        assert np.all(data.make == "Talbot")
        assert np.all(data.test == 1)

    def test_write_csv(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".csv")
        orig.write_csv(path)
        data = DataFrame.read_csv(path)
        assert data == orig

    def test_write_csv_path(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".csv")
        orig.write_csv(Path(path))

    def test_write_json(self):
        orig = test.data_frame("downloads.json")
        handle, path = tempfile.mkstemp(".json")
        orig.write_json(path)
        data = DataFrame.read_json(path)
        assert data == orig

    def test_write_json_path(self):
        orig = test.data_frame("downloads.json")
        handle, path = tempfile.mkstemp(".json")
        orig.write_json(Path(path))

    def test_write_npz(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".npz")
        orig.write_npz(path)
        data = DataFrame.read_npz(path)
        assert data == orig

    def test_write_npz_path(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".npz")
        orig.write_npz(Path(path))

    def test_write_pickle(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".pkl")
        orig.write_pickle(path)
        data = DataFrame.read_pickle(path)
        assert data == orig

    def test_write_pickle_path(self):
        orig = test.data_frame("vehicles.csv")
        handle, path = tempfile.mkstemp(".pkl")
        orig.write_pickle(Path(path))
