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

import math
import numpy as np
import tempfile

from dataiter import util


class TestUtil:

    def test_count_digits(self):
        assert util.count_digits(0) == (0, 0)
        assert util.count_digits(0.1) == (0, 1)
        assert util.count_digits(1.000) == (1, 0)
        assert util.count_digits(123.456) == (3, 3)

    def test_count_digits_special(self):
        assert util.count_digits(np.nan) == (0, 0)
        assert util.count_digits(math.inf) == (0, 0)

    def test_format_floats_1(self):
        a = [1/1000000000, 1/1000000, 1/1000, np.nan]
        b = ["1e-09", "1e-06", "1e-03", "nan"]
        assert util.format_floats(a) == b

    def test_format_floats_2(self):
        a = [0.000123456, 0.123456, 0, np.nan]
        b = ["0.000123", "0.123456", "0.000000", "nan"]
        assert util.format_floats(a) == b

    def test_format_floats_3(self):
        a = [0.123456, 1, 123.456, np.nan]
        b = ["0.123", "1.000", "123.456", "nan"]
        assert util.format_floats(a) == b

    def test_format_floats_4(self):
        a = [123.456789, 123456.789, 123456789, np.nan]
        b = ["123", "123457", "123456789", "nan"]
        assert util.format_floats(a) == b

    def test_format_floats_5(self):
        a = [12345678, 1234567812345678, 123456781234567812345678, np.nan]
        b = ["1.234568e+07", "1.234568e+15", "1.234568e+23", "nan"]
        assert util.format_floats(a) == b

    def test_format_floats_inf(self):
        a = [-math.inf, 0, math.inf, np.nan]
        b = ["-inf", "0e+00", "inf", "nan"]
        assert util.format_floats(a) == b

    def test_format_floats_integer(self):
        a = [1.0, 2.0, 3.0, np.nan]
        b = ["1", "2", "3", "nan"]
        assert util.format_floats(a) == b

    def test_generate_colnames(self):
        colnames = util.generate_colnames(1000)
        assert len(colnames) == 1000
        assert len(set(colnames)) == 1000

    def test_get_print_width(self):
        assert 0 < util.get_print_width() < 1000

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

    def test_ulen(self):
        assert util.ulen("asdf") == 4
        assert util.ulen("asdf\u200b") == 4
        assert util.ulen("asdf\u200b\u200b") == 4

    def test_unique_keys(self):
        assert util.unique_keys([1, 2, 3]) == [1, 2, 3]
        assert util.unique_keys([1, 2, 3, 1]) == [1, 2, 3]

    def test_unique_types(self):
        assert util.unique_types([1, 2, 3.3, np.nan, None]) == {int, float}

    def test_xopen_bz2(self):
        text = "test åäö"
        handle, path = tempfile.mkstemp(".bz2")
        with util.xopen(path, "wt") as f:
            f.write(text)
        with util.xopen(path, "rt") as f:
            assert f.read() == text

    def test_xopen_gz(self):
        text = "test åäö"
        handle, path = tempfile.mkstemp(".gz")
        with util.xopen(path, "wt") as f:
            f.write(text)
        with util.xopen(path, "rt") as f:
            assert f.read() == text

    def test_xopen_txt(self):
        text = "test åäö"
        handle, path = tempfile.mkstemp(".txt")
        with util.xopen(path, "wt") as f:
            f.write(text)
        with util.xopen(path, "rt") as f:
            assert f.read() == text

    def test_xopen_xz(self):
        text = "test åäö"
        handle, path = tempfile.mkstemp(".xz")
        with util.xopen(path, "wt") as f:
            f.write(text)
        with util.xopen(path, "rt") as f:
            assert f.read() == text
