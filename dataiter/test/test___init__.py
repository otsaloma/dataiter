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

from dataiter import Vector

T = True
F = False
NaN = np.nan

# If Numba is available, parametrize tests to run aggregation
# functions both with and without Numba.
USE_NUMBA_PARAMS = [False, True] if di.USE_NUMBA else [False]

parametrize = pytest.mark.parametrize
patch = unittest.mock.patch
skipif = pytest.mark.skipif

def isclose(a, b):
    return np.isclose(a, b, equal_nan=True).all()


class TestUtil:

    def get_data(self, a=None, g=None):
        a = Vector(a or [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        g = Vector(g or [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        return di.DataFrame(g=g, a=a)

    def test_all(self):
        assert     di.all(Vector([T, T]))
        assert not di.all(Vector([T, F]))
        assert not di.all(Vector([F, F]))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_all_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([T, T, T, T, T, F, F, F, F, F])
            stat = data.group_by("g").aggregate(a=di.all("a"))
            assert stat.a.equal(Vector([T, T, F, F, F]))

    def test_any(self):
        assert     di.any(Vector([T, T]))
        assert     di.any(Vector([T, F]))
        assert not di.any(Vector([F, F]))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_any_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([T, T, T, T, T, F, F, F, F, F])
            stat = data.group_by("g").aggregate(a=di.any("a"))
            assert stat.a.equal(Vector([T, T, T, F, F]))

    def test_count(self):
        assert di.count(Vector([])) == 0
        assert di.count(Vector([1])) == 1
        assert di.count(Vector([1, 2])) == 2

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_count_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            stat = data.group_by("g").aggregate(a=di.count())
            assert stat.a.equal(Vector([2, 2, 2, 2, 2]))
            stat = data.group_by("g").aggregate(a=di.count("a"))
            assert stat.a.equal(Vector([2, 2, 2, 2, 2]))

    def test_count_unique(self):
        assert di.count_unique(Vector([1, 2])) == 2
        assert di.count_unique(Vector([1, 2, 2])) == 2
        assert di.count_unique(Vector([1, 2, 2, 3])) == 3

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_count_unique_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 1, 2, 3, 4, 4, 5, 6, 7, 7])
            stat = data.group_by("g").aggregate(a=di.count_unique("a"))
            assert stat.a.equal(Vector([1, 2, 1, 2, 1]))

    def test_first(self):
        assert di.first(Vector([1, 2, 3])) == 1
        assert np.isnan(di.first(Vector([])))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_first_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            stat = data.group_by("g").aggregate(a=di.first("a"))
            assert stat.a.equal(Vector([1, 3, 5, 7, 9]))

    def test_last(self):
        assert di.last(Vector([1, 2, 3])) == 3
        assert np.isnan(di.last(Vector([])))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_last_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            stat = data.group_by("g").aggregate(a=di.last("a"))
            assert stat.a.equal(Vector([2, 4, 6, 8, 10]))

    def test_max(self):
        assert di.max(Vector([1, 3, 2])) == 3
        assert di.max(Vector([1, 3, NaN])) == 3
        assert np.isnan(di.max(Vector([1, 3, NaN]), dropna=False))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_max_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, NaN, NaN, NaN])
            stat = data.group_by("g").aggregate(a=di.max("a"))
            assert stat.a.equal(Vector([2, 4, 6, 7, NaN]))

    def test_mean(self):
        assert isclose(di.mean(Vector([1, 2, 10])), 4.333333)
        assert np.isnan(di.mean(Vector([1, 2, NaN]), dropna=False))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_mean_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, NaN, NaN, NaN])
            stat = data.group_by("g").aggregate(a=di.mean("a"))
            assert stat.a.equal(Vector([1.5, 3.5, 5.5, 7, NaN]))

    def test_median(self):
        assert isclose(di.median(Vector([1, 4, 6, 8, 5])), 5)
        assert np.isnan(di.median(Vector([1, 4, NaN]), dropna=False))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_median_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, NaN, NaN, NaN])
            stat = data.group_by("g").aggregate(a=di.median("a"))
            assert stat.a.equal(Vector([1.5, 3.5, 5.5, 7, NaN]))

    def test_min(self):
        assert di.min(Vector([3, 2, 1])) == 1
        assert di.min(Vector([3, 2, NaN])) == 2
        assert np.isnan(di.min(Vector([3, 2, NaN]), dropna=False))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_min_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, NaN, NaN, NaN])
            stat = data.group_by("g").aggregate(a=di.min("a"))
            assert stat.a.equal(Vector([1, 3, 5, 7, NaN]))

    def test_mode(self):
        assert di.mode(Vector([1, 2, 2, 3, 3, 3])) == 3
        assert np.isnan(di.mode(Vector([NaN, 1]), dropna=False))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_mode_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 1, 3, 4, 5, 5, 7, NaN, NaN, NaN])
            stat = data.group_by("g").aggregate(a=di.mode("a"))
            assert stat.a.equal(Vector([1, 3, 5, 7, NaN]))

    def test_ncol(self):
        assert di.ncol(self.get_data()) == 2

    def test_nrow(self):
        assert di.nrow(self.get_data()) == 10

    def test_nth(self):
        assert di.nth(Vector([1, 2, 3]), 1) == 2
        assert np.isnan(di.nth(Vector([1, 2, 3]), 3))
        assert np.isnan(di.nth(Vector([1, 2, 3]), -4))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_nth_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            stat = data.group_by("g").aggregate(a=di.nth("a", 1))
            assert stat.a.equal(Vector([2, 4, 6, 8, 10]))

    def test_quantile(self):
        assert isclose(di.quantile(Vector([1, 5, 6, 7, 8]), 0.5), 6)
        assert np.isnan(di.quantile(Vector([1, 5, NaN]), 0.5, dropna=False))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_quantile_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, NaN, NaN, NaN])
            stat = data.group_by("g").aggregate(a=di.quantile("a", 0.5))
            assert stat.a.equal(Vector([1.5, 3.5, 5.5, 7, NaN]))

    def test_std(self):
        assert isclose(di.std(Vector([3, 6, 7])), 1.699673)
        assert np.isnan(di.std(Vector([3, 6, NaN]), dropna=False))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_std_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, NaN, NaN, NaN])
            stat = data.group_by("g").aggregate(a=di.std("a"))
            assert stat.a.equal(Vector([0.5, 0.5, 0.5, NaN, NaN]))

    def test_sum(self):
        assert di.sum(Vector([1, 2, 3])) == 6
        assert np.isnan(di.sum(Vector([1, 2, NaN]), dropna=False))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_sum_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, NaN, NaN, NaN])
            stat = data.group_by("g").aggregate(a=di.sum("a"))
            assert stat.a.equal(Vector([3, 7, 11, 7, np.nan]))

    def test_var(self):
        assert isclose(di.var(Vector([3, 6, 7])), 2.888889)
        assert np.isnan(di.var(Vector([3, 6, NaN]), dropna=False))

    @parametrize("use_numba", USE_NUMBA_PARAMS)
    def test_var_aggregate(self, use_numba):
        with patch("dataiter.USE_NUMBA", use_numba):
            data = self.get_data([1, 2, 3, 4, 5, 6, 7, NaN, NaN, NaN])
            stat = data.group_by("g").aggregate(a=di.var("a"))
            assert stat.a.equal(Vector([0.25, 0.25, 0.25, NaN, NaN]))
