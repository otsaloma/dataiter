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

import tempfile

from dataiter import DataFrame
from dataiter import DataFrameColumn
from dataiter import test


class TestDataFrameColumn:

    def test___init__(self):
        column = DataFrameColumn([1, 2, 3])
        assert isinstance(column, DataFrameColumn)
        assert isinstance(column, np.ndarray)

    def test___init___given_dtype(self):
        column = DataFrameColumn([1, 2, 3], dtype="float64")
        assert column.dtype is np.dtype("float64")

    def test__init___given_nrow(self):
        column = DataFrameColumn([1], nrow=3)
        assert column.tolist() == [1, 1, 1]

    def test__init___given_scalar(self):
        column = DataFrameColumn(1, nrow=1)
        assert column.tolist() == [1]
        column = DataFrameColumn(1, nrow=3)
        assert column.tolist() == [1, 1, 1]

    def test_nrow(self):
        column = DataFrameColumn([1, 2, 3])
        assert column.nrow == 3


class TestDataFrame:

    def setup_method(self, method):
        self.data = DataFrame.read_json(test.get_json_filename())
        self.data_backup = self.data.deepcopy()

    def test___init___broadcast(self):
        data = DataFrame(a=[1, 2, 3], b=1)
        assert data.a.tolist() == [1, 2, 3]
        assert data.b.tolist() == [1, 1, 1]

    def test___init___given_data_frame_column(self):
        data = DataFrame(a=DataFrameColumn([1, 2, 3]))
        assert data.ncol == 1
        assert data.nrow == 3

    def test___init___given_list(self):
        data = DataFrame(a=[1, 2, 3])
        assert data.ncol == 1
        assert data.nrow == 3

    def test___delattr__(self):
        del self.data.date
        assert "date" not in self.data.colnames

    def test___eq__(self):
        assert self.data == self.data_backup

    def test___getattr__(self):
        assert self.data.date is self.data["date"]

    def test___setattr__(self):
        self.data.new = 1
        assert "new" in self.data.colnames

    def test___setitem__(self):
        self.data["new"] = 1
        assert "new" in self.data.colnames

    def test___str__(self):
        assert str(self.data)

    def test_aggregate(self):
        # TODO:
        pass

    def test_colnames(self):
        assert self.data.colnames == ["category", "date", "downloads"]

    def test_columns(self):
        assert self.data.columns == [self.data.category, self.data.date, self.data.downloads]

    def test_copy(self):
        data = self.data.copy()
        assert data == self.data
        assert data is not self.data

    def test_deepcopy(self):
        data = self.data.copy()
        assert data == self.data
        assert data is not self.data

    def test_filter(self):
        # TODO:
        pass

    def test_filter_out(self):
        # TODO:
        pass

    def test_from_json(self):
        string = self.data.to_json()
        data = DataFrame.from_json(string)
        assert data == self.data

    def test_group_by(self):
        # TODO:
        pass

    def test_join(self):
        # TODO:
        pass

    def test_ncol(self):
        assert self.data.ncol == 3

    def test_nrow(self):
        assert self.data.nrow == 137

    def test_read_csv(self):
        # XXX: pandas.read_csv reads "null" as NaN.
        data = DataFrame.read_csv(test.get_csv_filename())
        data.category[data.category == "nan"] = "null"
        assert data == self.data

    def test_read_json(self):
        data = DataFrame.read_json(test.get_json_filename())
        assert data == self.data

    def test_rename(self):
        # TODO:
        pass

    def test_select(self):
        # TODO:
        pass

    def test_sort(self):
        # TODO:
        pass

    def test_to_json(self):
        string = self.data.to_json()
        data = DataFrame.from_json(string)
        assert data == self.data

    def test_to_list_of_dicts(self):
        self.data.to_list_of_dicts()

    def test_to_pandas(self):
        self.data.to_pandas()

    def test_unique(self):
        # TODO:
        pass

    def test_unselect(self):
        # TODO:
        pass

    def test_write_csv(self):
        # XXX: pandas.read_csv reads "null" as NaN.
        handle, fname = tempfile.mkstemp(".csv")
        self.data.write_csv(fname)
        data = DataFrame.read_csv(fname)
        data.category[data.category == "nan"] = "null"
        assert data == self.data

    def test_write_json(self):
        handle, fname = tempfile.mkstemp(".json")
        self.data.write_json(fname)
        data = DataFrame.read_json(fname)
        assert data == self.data
