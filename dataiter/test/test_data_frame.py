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

from dataiter import DataFrame
from dataiter import test


class TestDataFrame:

    def setup_method(self, method):
        self.data = DataFrame.read_json(test.get_json_filename())
        self.data_backup = self.data.deepcopy()

    def test___init__(self):
        # TODO:
        pass

    def test___copy__(self):
        # TODO:
        pass

    def test___deepcopy__(self):
        # TODO:
        pass

    def test___delattr__(self):
        # TODO:
        pass

    def test___getattr__(self):
        # TODO:
        pass

    def test___setattr__(self):
        # TODO:
        pass

    def test___setitem__(self):
        # TODO:
        pass

    def test___str__(self):
        print(self.data)

    def test_aggregate(self):
        # TODO:
        pass

    def test_colnames(self):
        # TODO:
        pass

    def test_columns(self):
        # TODO:
        pass

    def test_copy(self):
        # TODO:
        pass

    def test_deepcopy(self):
        # TODO:
        pass

    def test_filter(self):
        # TODO:
        pass

    def test_filter_out(self):
        # TODO:
        pass

    def test_from_json(self):
        # TODO:
        pass

    def test_group_by(self):
        # TODO:
        pass

    def test_join(self):
        # TODO:
        pass

    def test_ncol(self):
        # TODO:
        pass

    def test_nrow(self):
        # TODO:
        pass

    def test_read_csv(self):
        # TODO:
        pass

    def test_read_json(self):
        # TODO:
        pass

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
        # TODO:
        pass

    def test_unique(self):
        # TODO:
        pass

    def test_unselect(self):
        # TODO:
        pass

    def test_write_csv(self):
        # TODO:
        pass

    def test_write_json(self):
        # TODO:
        pass
