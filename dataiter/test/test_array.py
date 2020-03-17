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

from dataiter import Array


class TestArray:

    def test_equal_given_bool(self):
        a = Array([True, False, np.nan])
        b = Array([True, False, np.nan])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_given_float(self):
        a = Array([1.1, 2.2, np.nan])
        b = Array([1.1, 2.2, np.nan])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_given_int(self):
        a = Array([1, 2, 3])
        b = Array([1, 2, 3])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_equal_given_str(self):
        a = Array(["a", "b", "c"])
        b = Array(["a", "b", "c"])
        assert a.equal(b)
        assert not a.equal(b[::-1])

    def test_is_boolean(self):
        assert Array([True]).is_boolean
        assert not Array([1.1]).is_boolean
        assert not Array([1]).is_boolean
        assert not Array(["a"]).is_boolean

    def test_is_float(self):
        assert not Array([True]).is_float
        assert Array([1.1]).is_float
        assert not Array([1]).is_float
        assert not Array(["a"]).is_float

    def test_is_integer(self):
        assert not Array([True]).is_integer
        assert not Array([1.1]).is_integer
        assert Array([1]).is_integer
        assert not Array(["a"]).is_integer

    def test_is_number(self):
        assert not Array([True]).is_number
        assert Array([1.1]).is_number
        assert Array([1]).is_number
        assert not Array(["a"]).is_number

    def test_is_string(self):
        assert not Array([True]).is_string
        assert not Array([1.1]).is_string
        assert not Array([1]).is_string
        assert Array(["a"]).is_string
