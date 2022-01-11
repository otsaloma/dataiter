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

import dataiter as di
import numpy as np
import pytest
import unittest.mock

patch = unittest.mock.patch
skipif = pytest.mark.skipif

def isclose(a, b):
    return np.isclose(a, b, equal_nan=True).all()


class TestUtil:

    def setup_method(self, method):
        self.g = di.Vector([1, 1, 2, 2, 3, 3, 4, 4, 5, 5], int)
        self.a = di.Vector([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, np.nan, np.nan, np.nan], float)
        self.data = di.DataFrame(g=self.g, a=self.a)

    def test_mean(self):
        assert isclose(di.mean(self.a), 0.4)
        assert np.isnan(di.mean(self.a, dropna=False))

    @patch("dataiter.USE_NUMBA", False)
    def test_mean_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.mean("a"))
        assert isclose(stat.a, [0.15, 0.35, 0.55, 0.7, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_mean_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.mean("a"))
        assert isclose(stat.a, [0.15, 0.35, 0.55, 0.7, np.nan])

    def test_ncol(self):
        assert di.ncol(self.data) == 2

    def test_nrow(self):
        assert di.nrow(self.data) == 10
