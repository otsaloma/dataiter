# -*- coding: utf-8 -*-

# Copyright (c) 2022 Osmo Salomaa
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

import inspect

from dataiter import DataFrame
from dataiter import GeoJSON
from dataiter import io
from dataiter import ListOfDicts

class TestIO:

    def test_read_csv(self):
        s1 = inspect.signature(io.read_csv)
        s2 = inspect.signature(DataFrame.read_csv)
        assert s1 == s2

    def test_read_geojson(self):
        s1 = inspect.signature(io.read_geojson)
        s2 = inspect.signature(GeoJSON.read)
        assert s1 == s2

    def test_read_json(self):
        s1 = inspect.signature(io.read_json)
        s2 = inspect.signature(ListOfDicts.read_json)
        assert s1 == s2

    def test_read_npz(self):
        s1 = inspect.signature(io.read_npz)
        s2 = inspect.signature(DataFrame.read_npz)
        assert s1 == s2
