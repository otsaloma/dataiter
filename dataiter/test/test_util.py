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

import numpy as np

from dataiter import util


class TestUtil:

    def test_count_decimals(self):
        assert util.count_decimals(1.1) == 1
        assert util.count_decimals(1.11) == 2

    def test_generate_colnames(self):
        colnames = util.generate_colnames(1000)
        assert len(colnames) == 1000
        assert len(set(colnames)) == 1000

    def test_length(self):
        assert util.length(1) == 1
        assert util.length([1]) == 1
        assert util.length([1, 2]) == 2

    def test_pad(self):
        assert util.pad(["a", "aa", "aaa"], align="right") == ["  a", " aa", "aaa"]
        assert util.pad(["a", "aa", "aaa"], align="left")  == ["a  ", "aa ", "aaa"]

    def test_quote(self):
        assert util.quote("hello") == '"hello"'
        assert util.quote('"hello"') == '"\\"hello\\""'

    def test_unique_keys(self):
        assert util.unique_keys([1, 2, 3]) == [1, 2, 3]
        assert util.unique_keys([1, 2, 3, 1]) == [1, 2, 3]

    def test_unique_types(self):
        assert util.unique_types([1, 2, 3.3, np.nan, None]) == set((int, float))
