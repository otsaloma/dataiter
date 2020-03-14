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

from dataiter import DataFrameColumn


class TestDataFrameColumn:

    def test___init__(self):
        column = DataFrameColumn([1, 2, 3])
        assert isinstance(column, DataFrameColumn)
        assert isinstance(column, np.ndarray)

    def test___init__given_dtype(self):
        column = DataFrameColumn([1, 2, 3], dtype="float64")
        assert column.dtype is np.dtype("float64")

    def test__init___given_nrow(self):
        column = DataFrameColumn(1, nrow=3)
        assert column.tolist() == [1, 1, 1]

    def test_nrow(self):
        column = DataFrameColumn([1, 2, 3])
        assert column.nrow == 3
