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

import dataiter
import datetime
import numpy as np
import pytest

from dataiter import DataFrame
from dataiter import Vector
from dataiter.aggregate import all
from dataiter.aggregate import any
from dataiter.aggregate import count
from dataiter.aggregate import count_unique
from dataiter.aggregate import first
from dataiter.aggregate import last
from dataiter.aggregate import max
from dataiter.aggregate import mean
from dataiter.aggregate import median
from dataiter.aggregate import min
from dataiter.aggregate import mode
from dataiter.aggregate import nth
from dataiter.aggregate import quantile
from dataiter.aggregate import std
from dataiter.aggregate import sum
from dataiter.aggregate import var
from unittest.mock import patch

T = True
F = False

D1 = datetime.date.today()
D2 = D1 + datetime.timedelta(days=1)
D3 = D1 + datetime.timedelta(days=2)
D4 = D1 + datetime.timedelta(days=3)
D5 = D1 + datetime.timedelta(days=4)
D6 = D1 + datetime.timedelta(days=5)
D7 = D1 + datetime.timedelta(days=6)

NaN = np.nan
NaT = np.datetime64("NaT")

EMPTY_VECTOR = Vector([], float)
GROUPS = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

nth0 = lambda x: nth(x, 0)
quantile05 = lambda x: quantile(x, 0.5)

TEST_MATRIX = [

    # NaNs evaluate to true, because they are not equal to zero.
    # https://numpy.org/doc/stable/reference/generated/numpy.all.html
    (all, [T, T, T, T, T, F, F, F, F, F], [T, T, F, F, F]),
    (all, [1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [T, T, F, F, F]),
    (all, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, NaN, NaN, NaN], [F, F, T, T, T]),
    (all, [D1, D2, D3, D4, D5, NaT, NaT, NaT, NaT, NaT], [T, T, T, T, T]),
    (all, ["a", "b", "c", "d", "e", "", "", "", "", ""], [T, T, T, T, T]),

    # NaNs evaluate to true, because they are not equal to zero.
    # https://numpy.org/doc/stable/reference/generated/numpy.any.html
    (any, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (any, [1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [T, T, T, F, F]),
    (any, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, NaN, NaN, NaN], [F, T, T, T, T]),
    (any, [D1, D2, D3, D4, D5, NaT, NaT, NaT, NaT, NaT], [T, T, T, T, T]),
    (any, ["a", "b", "c", "d", "e", "", "", "", "", ""], [T, T, T, T, T]),

    (count, [T, T, T, T, T, F, F, F, F, F], [2, 2, 2, 2, 2]),
    (count, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 2, 2, 2, 2]),
    (count, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [2, 2, 2, 2, 2]),
    (count, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [2, 2, 2, 2, 2]),
    (count, ["a", "b", "c", "d", "e", "f", "g", "", "", ""], [2, 2, 2, 2, 2]),

    # NaN is not considered equal to itself and thus all are counted here.
    (count_unique, [T, T, T, T, T, F, F, F, F, F], [1, 1, 2, 1, 1]),
    (count_unique, [1, 1, 3, 3, 5, 6, 7, 8, 9, 10], [1, 1, 2, 2, 2]),
    (count_unique, [1.0, 1.0, 3.0, 3.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1, 1, 2, 2, 2]),
    (count_unique, [D1, D1, D3, D3, D5, D6, D7, NaT, NaT, NaT], [1, 1, 2, 2, 2]),
    (count_unique, ["a", "a", "c", "c", "e", "f", "g", "", "", ""], [1, 1, 2, 2, 1]),

    (first, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (first, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 3, 5, 7, 9]),
    (first, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.0, 3.0, 5.0, 7.0, NaN]),
    (first, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D1, D3, D5, D7, NaT]),
    (first, ["a", "b", "c", "d", "e", "f", "g", "", "", ""], ["a", "c", "e", "g", ""]),

    (last, [T, T, T, T, T, F, F, F, F, F], [T, T, F, F, F]),
    (last, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6, 8, 10]),
    (last, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [2.0, 4.0, 6.0, NaN, NaN]),
    (last, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D2, D4, D6, NaT, NaT]),
    (last, ["a", "b", "c", "d", "e", "f", "g", "", "", ""], ["b", "d", "f", "", ""]),

    (max, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (max, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6, 8, 10]),
    (max, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [2.0, 4.0, 6.0, 7.0, NaN]),
    (max, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D2, D4, D6, D7, NaT]),

    (mean, [T, T, T, T, T, F, F, F, F, F], [1.0, 1.0, 0.5, 0.0, 0.0]),
    (mean, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.5, 3.5, 5.5, 7.5, 9.5]),
    (mean, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.5, 3.5, 5.5, 7.0, NaN]),

    (median, [T, T, T, T, T, F, F, F, F, F], [1.0, 1.0, 0.5, 0.0, 0.0]),
    (median, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.5, 3.5, 5.5, 7.5, 9.5]),
    (median, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.5, 3.5, 5.5, 7.0, NaN]),

    (min, [T, T, T, T, T, F, F, F, F, F], [T, T, F, F, F]),
    (min, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 3, 5, 7, 9]),
    (min, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.0, 3.0, 5.0, 7.0, NaN]),
    (min, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D1, D3, D5, D7, NaT]),

    (mode, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (mode, [1, 1, 3, 3, 5, 6, 7, 8, 9, 10], [1, 3, 5, 7, 9]),
    (mode, [1.0, 1.0, 3.0, 3.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.0, 3.0, 5.0, 7.0, NaN]),
    (mode, [D1, D1, D3, D3, D5, D6, D7, NaT, NaT, NaT], [D1, D3, D5, D7, NaT]),
    (mode, ["a", "a", "c", "c", "e", "f", "g", "", "", ""], ["a", "c", "e", "g", ""]),

    (nth0, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (nth0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 3, 5, 7, 9]),
    (nth0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.0, 3.0, 5.0, 7.0, NaN]),
    (nth0, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D1, D3, D5, D7, NaT]),
    (nth0, ["a", "b", "c", "d", "e", "f", "g", "", "", ""], ["a", "c", "e", "g", ""]),

    (quantile05, [T, T, T, T, T, F, F, F, F, F], [1.0, 1.0, 0.5, 0.0, 0.0]),
    (quantile05, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.5, 3.5, 5.5, 7.5, 9.5]),
    (quantile05, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.5, 3.5, 5.5, 7.0, NaN]),

    (std, [T, T, T, T, T, F, F, F, F, F], [0.0, 0.0, 0.5, 0.0, 0.0]),
    (std, [1, 1, 2, 3, 5, 7, 8, 12, 13, 21], [0.0, 0.5, 1.0, 2.0, 4.0]),
    (std, [1.0, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, NaN, NaN, NaN], [0.0, 0.5, 1.0, NaN, NaN]),

    (sum, [T, T, T, T, T, F, F, F, F, F], [2, 2, 1, 0, 0]),
    (sum, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 7, 11, 15, 19]),
    (sum, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [3.0, 7.0, 11.0, 7.0, 0.0]),

    (var, [T, T, T, T, T, F, F, F, F, F], [0.0, 0.0, 0.25, 0.0, 0.0]),
    (var, [1, 1, 2, 3, 5, 7, 8, 12, 13, 21], [0.0, 0.25, 1.0, 4.0, 16.0]),
    (var, [1.0, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, NaN, NaN, NaN], [0.0, 0.25, 1.0, NaN, NaN]),

]

class TestAggregate:

    @pytest.mark.parametrize("use_numba", [False, True])
    @pytest.mark.parametrize("function,input,output", TEST_MATRIX)
    def test_aggregate(self, function, input, output, use_numba):
        if use_numba and not dataiter.USE_NUMBA:
            pytest.skip("No Numba")
        with patch("dataiter.USE_NUMBA", use_numba):
            data = DataFrame(g=GROUPS, a=input)
            stat = data.group_by("g").aggregate(a=function("a"))
            expected = Vector(output)
            try:
                assert stat.a.equal(expected)
            except AssertionError:
                print("")
                print(data)
                print("Expected:")
                print(expected)
                print("Got:")
                print(stat.a)
                raise

    @pytest.mark.parametrize("use_numba", [False, True])
    def test_aggregate_count(self, use_numba):
        if use_numba and not dataiter.USE_NUMBA:
            pytest.skip("No Numba")
        with patch("dataiter.USE_NUMBA", use_numba):
            data = DataFrame(g=GROUPS)
            stat = data.group_by("g").aggregate(n=count())
            assert (stat.n == 2).all()

    def test_all(self):
        assert all(EMPTY_VECTOR)
        assert all(Vector([T, T]))
        assert not all(Vector([T, F]))
        assert not all(Vector([F, F]))

    def test_any(self):
        assert not any(EMPTY_VECTOR)
        assert any(Vector([T, T]))
        assert any(Vector([T, F]))
        assert not any(Vector([F, F]))

    def test_count(self):
        assert count(EMPTY_VECTOR) == 0
        assert count(Vector([1])) == 1
        assert count(Vector([1, 2])) == 2
        assert count(Vector([1, 2, NaN])) == 3
        assert count(Vector([1, 2, NaN]), drop_na=True) == 2

    def test_count_unique(self):
        assert count_unique(EMPTY_VECTOR) == 0
        assert count_unique(Vector([1])) == 1
        assert count_unique(Vector([1, 1])) == 1
        assert count_unique(Vector([1, 1, 2])) == 2
        assert count_unique(Vector([1, 1, 2, NaN])) == 3
        assert count_unique(Vector([1, 1, 2, NaN]), drop_na=True) == 2

    def test_first(self):
        assert first(Vector([1, 2, 3])) == 1
        assert first(Vector([NaN, 1, 2]), drop_na=True) == 1

    def test_first_nan(self):
        assert np.isnan(first(EMPTY_VECTOR))
        assert np.isnan(first(Vector([NaN, 1, 2])))

    def test_last(self):
        assert last(Vector([1, 2, 3])) == 3
        assert last(Vector([1, 2, NaN]), drop_na=True) == 2

    def test_last_nan(self):
        assert np.isnan(last(EMPTY_VECTOR))
        assert np.isnan(last(Vector([1, 2, NaN])))

    def test_max(self):
        assert max(Vector([3, 2, 1])) == 3
        assert max(Vector([3, 2, NaN])) == 3

    def test_max_nan(self):
        assert np.isnan(max(EMPTY_VECTOR))
        assert np.isnan(max(Vector([3, 2, NaN]), drop_na=False))

    def test_mean(self):
        assert np.isclose(mean(Vector([1, 2, 10])), 4.333333)
        assert np.isclose(mean(Vector([1, 2, NaN])), 1.5)

    def test_mean_nan(self):
        assert np.isnan(mean(EMPTY_VECTOR))
        assert np.isnan(mean(Vector([1, 2, NaN]), drop_na=False))

    def test_median(self):
        assert median(Vector([1, 4, 6, 8, 5])) == 5
        assert median(Vector([1, 4, 6, NaN, NaN])) == 4

    def test_median_nan(self):
        assert np.isnan(median(EMPTY_VECTOR))
        assert np.isnan(median(Vector([1, 4, NaN]), drop_na=False))

    def test_min(self):
        assert min(Vector([3, 2, 1])) == 1
        assert min(Vector([3, 2, NaN])) == 2

    def test_min_nan(self):
        assert np.isnan(min(EMPTY_VECTOR))
        assert np.isnan(min(Vector([3, 2, NaN]), drop_na=False))

    def test_mode(self):
        assert mode(Vector([1])) == 1
        assert mode(Vector([1, 2])) == 1
        assert mode(Vector([1, 2, 2])) == 2
        assert mode(Vector([1, 2, 2, NaN])) == 2
        assert mode(Vector([1, 2, 2, NaN]), drop_na=False) == 2

    def test_mode_nan(self):
        assert np.isnan(mode(EMPTY_VECTOR))
        assert np.isnan(mode(Vector([NaN, NaN], float), drop_na=False))

    def test_nth(self):
        assert nth(Vector([1, 2, 3]), 0) == 1
        assert nth(Vector([NaN, 1, 2]), 0, drop_na=True) == 1

    def test_nth_nan(self):
        assert np.isnan(nth(EMPTY_VECTOR, 0))
        assert np.isnan(nth(Vector([NaN, 1, 2]), 0))

    def test_quantile(self):
        assert quantile(Vector([1, 4, 6, 8, 5]), 0.5) == 5
        assert quantile(Vector([1, 4, 6, NaN, NaN]), 0.5) == 4

    def test_quantile_nan(self):
        assert np.isnan(quantile(EMPTY_VECTOR, 0.5))
        assert np.isnan(quantile(Vector([1, 4, NaN]), 0.5, drop_na=False))

    def test_std(self):
        assert np.isclose(std(Vector([3, 6, 7])), 1.699673)
        assert np.isclose(std(Vector([3, 6, NaN])), 1.5)

    def test_std_nan(self):
        assert np.isnan(std(EMPTY_VECTOR))
        assert np.isnan(std(Vector([1])))
        assert np.isnan(std(Vector([3, 6, NaN]), drop_na=False))

    def test_sum(self):
        assert sum(EMPTY_VECTOR) == 0
        assert sum(Vector([1])) == 1
        assert sum(Vector([1, 2])) == 3
        assert sum(Vector([1, 2, NaN])) == 3

    def test_sum_nan(self):
        assert np.isnan(sum(Vector([1, 2, NaN]), drop_na=False))

    def test_var(self):
        assert np.isclose(var(Vector([3, 6, 7])), 2.888889)
        assert np.isclose(var(Vector([3, 6, NaN])), 2.25)

    def test_var_nan(self):
        assert np.isnan(var(EMPTY_VECTOR))
        assert np.isnan(var(Vector([1])))
        assert np.isnan(var(Vector([3, 6, NaN]), drop_na=False))
