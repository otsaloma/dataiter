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

    def test_get_colnames(self):
        colnames = util.get_colnames(1000)
        assert len(colnames) == 1000
        assert len(set(colnames)) == 1000

    def test_length(self):
        assert util.length(1) == 1
        assert util.length([1]) == 1
        assert util.length([1, 2]) == 2

    def test_np_to_string_bool(self):
        assert util.np_to_string(np.array([True])[0]) == "True"

    def test_np_to_string_date(self):
        assert util.np_to_string(np.datetime64("2020-01-01")) == "2020-01-01"

    def test_np_to_string_float(self):
        assert util.np_to_string(np.array([1/3])[0]) == "0.333333"

    def test_np_to_string_int(self):
        assert util.np_to_string(np.array([1])[0]) == "1"

    def test_np_to_string_str(self):
        assert util.np_to_string(np.array(["a"])[0]) == "a"
