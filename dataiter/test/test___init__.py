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
        self.b = di.Vector([True, True, True, True, True, False, False, False, False, False], bool)
        self.data = di.DataFrame(g=self.g, a=self.a, b=self.b)

    def test_all(self):
        assert not di.all(self.b)

    @patch("dataiter.USE_NUMBA", False)
    def test_all_aggregate(self):
        stat = self.data.group_by("g").aggregate(b=di.all("b"))
        assert stat.b.equal(di.Vector([True, True, False, False, False]))

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_all_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(b=di.all("b"))
        assert stat.b.equal(di.Vector([True, True, False, False, False]))

    def test_any(self):
        assert di.any(self.b)

    @patch("dataiter.USE_NUMBA", False)
    def test_any_aggregate(self):
        stat = self.data.group_by("g").aggregate(b=di.any("b"))
        assert stat.b.equal(di.Vector([True, True, True, False, False]))

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_any_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(b=di.any("b"))
        assert stat.b.equal(di.Vector([True, True, True, False, False]))

    def test_first(self):
        assert isclose(di.first(self.a), 0.1)

    @patch("dataiter.USE_NUMBA", False)
    def test_first_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.first("a"))
        assert isclose(stat.a, [0.1, 0.3, 0.5, 0.7, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_first_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.first("a"))
        assert isclose(stat.a, [0.1, 0.3, 0.5, 0.7, np.nan])

    def test_last(self):
        assert np.isnan(di.last(self.a))

    @patch("dataiter.USE_NUMBA", False)
    def test_last_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.last("a"))
        assert isclose(stat.a, [0.2, 0.4, 0.6, np.nan, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_last_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.last("a"))
        assert isclose(stat.a, [0.2, 0.4, 0.6, np.nan, np.nan])

    def test_max(self):
        assert isclose(di.max(self.a), 0.7)
        assert np.isnan(di.max(self.a, dropna=False))

    @patch("dataiter.USE_NUMBA", False)
    def test_max_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.max("a"))
        assert isclose(stat.a, [0.2, 0.4, 0.6, 0.7, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_max_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.max("a"))
        assert isclose(stat.a, [0.2, 0.4, 0.6, 0.7, np.nan])

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

    def test_median(self):
        assert isclose(di.median(self.a), 0.4)
        assert np.isnan(di.median(self.a, dropna=False))

    @patch("dataiter.USE_NUMBA", False)
    def test_median_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.median("a"))
        assert isclose(stat.a, [0.15, 0.35, 0.55, 0.7, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_median_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.median("a"))
        assert isclose(stat.a, [0.15, 0.35, 0.55, 0.7, np.nan])

    def test_min(self):
        assert isclose(di.min(self.a), 0.1)
        assert np.isnan(di.min(self.a, dropna=False))

    @patch("dataiter.USE_NUMBA", False)
    def test_min_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.min("a"))
        assert isclose(stat.a, [0.1, 0.3, 0.5, 0.7, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_min_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.min("a"))
        assert isclose(stat.a, [0.1, 0.3, 0.5, 0.7, np.nan])

    def test_n(self):
        assert di.n(self.a) == 10
        assert di.n(self.a, dropna=True) == 7

    @patch("dataiter.USE_NUMBA", False)
    def test_n_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.n())
        assert isclose(stat.a, [2, 2, 2, 2, 2])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_n_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.n())
        assert isclose(stat.a, [2, 2, 2, 2, 2])

    def test_ncol(self):
        assert di.ncol(self.data) == 3

    def test_nrow(self):
        assert di.nrow(self.data) == 10

    def test_nth(self):
        assert isclose(di.nth(self.a, 1), 0.2)
        assert np.isnan(di.nth(self.a, 999))

    @patch("dataiter.USE_NUMBA", False)
    def test_nth_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.nth("a", 1))
        assert isclose(stat.a, [0.2, 0.4, 0.6, np.nan, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_nth_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.nth("a", 1))
        assert isclose(stat.a, [0.2, 0.4, 0.6, np.nan, np.nan])

    def test_nunique(self):
        assert di.nunique(self.a) == 10
        assert di.nunique(self.a, dropna=True) == 7

    @patch("dataiter.USE_NUMBA", False)
    def test_nunique_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.nunique("a"))
        assert isclose(stat.a, [2, 2, 2, 2, 2])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_nunique_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.nunique("a"))
        assert isclose(stat.a, [2, 2, 2, 2, 2])

    def test_quantile(self):
        assert isclose(di.quantile(self.a, 0.5), 0.4)
        assert np.isnan(di.quantile(self.a, 0.5, dropna=False))

    @patch("dataiter.USE_NUMBA", False)
    def test_quantile_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.quantile("a", 0.5))
        assert isclose(stat.a, [0.15, 0.35, 0.55, 0.7, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_quantile_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.quantile("a", 0.5))
        assert isclose(stat.a, [0.15, 0.35, 0.55, 0.7, np.nan])

    def test_std(self):
        assert isclose(di.std(self.a), 0.2)
        assert np.isnan(di.std(self.a, dropna=False))

    @patch("dataiter.USE_NUMBA", False)
    def test_std_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.std("a"))
        assert isclose(stat.a, [0.05, 0.05, 0.05, np.nan, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_std_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.std("a"))
        assert isclose(stat.a, [0.05, 0.05, 0.05, np.nan, np.nan])

    def test_sum(self):
        assert isclose(di.sum(self.a), 2.8)
        assert np.isnan(di.sum(self.a, dropna=False))

    @patch("dataiter.USE_NUMBA", False)
    def test_sum_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.sum("a"))
        assert isclose(stat.a, [0.3, 0.7, 1.1, 0.7, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_sum_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.sum("a"))
        assert isclose(stat.a, [0.3, 0.7, 1.1, 0.7, np.nan])

    def test_var(self):
        assert isclose(di.var(self.a), 0.04)
        assert np.isnan(di.var(self.a, dropna=False))

    @patch("dataiter.USE_NUMBA", False)
    def test_var_aggregate(self):
        stat = self.data.group_by("g").aggregate(a=di.var("a"))
        assert isclose(stat.a, [0.0025, 0.0025, 0.0025, np.nan, np.nan])

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_var_aggregate_numba(self):
        stat = self.data.group_by("g").aggregate(a=di.var("a"))
        assert isclose(stat.a, [0.0025, 0.0025, 0.0025, np.nan, np.nan])
