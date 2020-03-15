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


class TestDataFrameColumn:

    def test___init___given_array(self):
        column = DataFrameColumn(np.array([1, 2, 3]))
        assert isinstance(column, DataFrameColumn)
        assert isinstance(column, np.ndarray)

    def test___init___given_dtype(self):
        column = DataFrameColumn([1, 2, 3], dtype="float64")
        assert column.dtype is np.dtype("float64")

    def test___init___given_list(self):
        column = DataFrameColumn([1, 2, 3])
        assert isinstance(column, DataFrameColumn)
        assert isinstance(column, np.ndarray)

    def test__init___given_nrow(self):
        column = DataFrameColumn([1], nrow=3)
        assert column.tolist() == [1, 1, 1]

    def test__init___given_scalar(self):
        column = DataFrameColumn(1)
        assert column.tolist() == [1]

    def test_equal_given_bool(self):
        a = DataFrameColumn([True, False, np.nan])
        b = DataFrameColumn([True, False, np.nan])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_given_float(self):
        a = DataFrameColumn([1.1, 2.2, np.nan])
        b = DataFrameColumn([1.1, 2.2, np.nan])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_given_int(self):
        a = DataFrameColumn([1, 2, 3])
        b = DataFrameColumn([1, 2, 3])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_given_str(self):
        a = DataFrameColumn(["a", "b", "c"])
        b = DataFrameColumn(["a", "b", "c"])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_is_boolean(self):
        assert DataFrameColumn([True]).is_boolean
        assert not DataFrameColumn([1.1]).is_boolean
        assert not DataFrameColumn([1]).is_boolean
        assert not DataFrameColumn(["a"]).is_boolean

    def test_is_float(self):
        assert not DataFrameColumn([True]).is_float
        assert DataFrameColumn([1.1]).is_float
        assert not DataFrameColumn([1]).is_float
        assert not DataFrameColumn(["a"]).is_float

    def test_is_integer(self):
        assert not DataFrameColumn([True]).is_integer
        assert not DataFrameColumn([1.1]).is_integer
        assert DataFrameColumn([1]).is_integer
        assert not DataFrameColumn(["a"]).is_integer

    def test_is_number(self):
        assert not DataFrameColumn([True]).is_number
        assert DataFrameColumn([1.1]).is_number
        assert DataFrameColumn([1]).is_number
        assert not DataFrameColumn(["a"]).is_number

    def test_is_string(self):
        assert not DataFrameColumn([True]).is_string
        assert not DataFrameColumn([1.1]).is_string
        assert not DataFrameColumn([1]).is_string
        assert DataFrameColumn(["a"]).is_string

    def test_nrow(self):
        assert DataFrameColumn([1, 2, 3]).nrow == 3


class TestDataFrame:

    def from_file(self, fname):
        fname = test.get_data_filename(fname)
        extension = fname.split(".")[-1]
        read = getattr(DataFrame, "read_{}".format(extension))
        return read(fname)

    def test___init___broadcast(self):
        data = DataFrame(a=[1, 2, 3], b=[1], c=1)
        assert data.a.tolist() == [1, 2, 3]
        assert data.b.tolist() == [1, 1, 1]
        assert data.c.tolist() == [1, 1, 1]

    def test___init___given_data_frame_column(self):
        data = DataFrame(a=DataFrameColumn([1, 2, 3]))
        assert data.ncol == 1
        assert data.nrow == 3

    def test___init___given_list(self):
        data = DataFrame(a=[1, 2, 3])
        assert data.ncol == 1
        assert data.nrow == 3

    def test___delattr__(self):
        data = DataFrame(a=DataFrameColumn([1, 2, 3]))
        assert "a" in data.colnames
        del data.a
        assert "a" not in data.colnames

    def test___eq__(self):
        data = self.from_file("vehicles.csv")
        assert data == data.copy()
        assert data == data.deepcopy()

    def test___getattr__(self):
        data = self.from_file("vehicles.csv")
        assert data.make is data["make"]

    def test___setattr__(self):
        data = self.from_file("vehicles.csv")
        assert "test" not in data.colnames
        data.test = 1
        assert "test" in data.colnames

    def test___setitem__(self):
        data = self.from_file("vehicles.csv")
        assert "test" not in data.colnames
        data["test"] = 1
        assert "test" in data.colnames

    def test_aggregate(self):
        pass

    def test_colnames(self):
        data = self.from_file("downloads.csv")
        assert data.colnames == ["category", "date", "downloads"]

    def test_columns(self):
        data = self.from_file("downloads.csv")
        assert data.columns == [data.category, data.date, data.downloads]

    def test_copy(self):
        orig = self.from_file("vehicles.csv")
        data = orig.copy()
        assert data == orig
        assert data is not orig

    def test_deepcopy(self):
        orig = self.from_file("vehicles.csv")
        data = orig.copy()
        assert data == orig
        assert data is not orig

    def test_filter(self):
        pass

    def test_filter_out(self):
        pass

    def test_from_json(self):
        orig = self.from_file("downloads.json")
        text = orig.to_json()
        data = DataFrame.from_json(text)
        assert data == orig

    def test_group_by(self):
        pass

    def test_head(self):
        pass

    def test_join(self):
        pass

    def test_ncol(self):
        data = self.from_file("downloads.csv")
        assert data.ncol == 3

    def test_nrow(self):
        data = self.from_file("downloads.csv")
        assert data.nrow == 905

    def test_read_csv(self):
        data = self.from_file("vehicles.csv")
        assert data.ncol == 12
        assert data.nrow == 33442

    def test_read_json(self):
        data = self.from_file("downloads.json")
        assert data.ncol == 3
        assert data.nrow == 905

    def test_rename(self):
        pass

    def test_select(self):
        orig = self.from_file("vehicles.csv")
        orig_colnames = list(orig.colnames)
        data = orig.select("make", "model")
        assert data.colnames == ["make", "model"]
        assert orig.colnames == orig_colnames

    def test_slice(self):
        pass

    def test_sort(self):
        pass

    def test_tail(self):
        pass

    def test_to_json(self):
        orig = self.from_file("downloads.json")
        text = orig.to_json()
        data = DataFrame.from_json(text)
        assert data == orig

    def test_to_list_of_dicts(self):
        orig = self.from_file("vehicles.csv")
        data = orig.to_list_of_dicts()
        assert len(data) == orig.nrow

    def test_to_pandas(self):
        orig = self.from_file("vehicles.csv")
        data = orig.to_pandas()
        assert data.shape[0] == orig.nrow
        assert data.shape[1] == orig.ncol

    def test_unique(self):
        pass

    def test_unselect(self):
        orig = self.from_file("vehicles.csv")
        orig_colnames = list(orig.colnames)
        data = orig.unselect("make", "model")
        assert data.colnames == [x for x in orig_colnames if not x in ["make", "model"]]
        assert orig.colnames == orig_colnames

    def test_write_csv(self):
        orig = self.from_file("vehicles.csv")
        handle, fname = tempfile.mkstemp(".csv")
        orig.write_csv(fname)
        data = DataFrame.read_csv(fname)
        assert data == orig

    def test_write_json(self):
        orig = self.from_file("downloads.json")
        handle, fname = tempfile.mkstemp(".json")
        orig.write_json(fname)
        data = DataFrame.read_json(fname)
        assert data == orig
