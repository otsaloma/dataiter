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
from dataiter.aggregate import prod
from dataiter.aggregate import quantile
from dataiter.aggregate import std
from dataiter.aggregate import sum
from dataiter.aggregate import var

T = True
F = False

D1 = datetime.date.today()
D2 = D1 + datetime.timedelta(days=1)
D3 = D1 + datetime.timedelta(days=2)
D4 = D1 + datetime.timedelta(days=3)
D5 = D1 + datetime.timedelta(days=4)
D6 = D1 + datetime.timedelta(days=5)
D7 = D1 + datetime.timedelta(days=6)

S0 = np.timedelta64(0, "s")
S1 = np.timedelta64(1, "s")
S2 = np.timedelta64(2, "s")
S3 = np.timedelta64(3, "s")
S4 = np.timedelta64(4, "s")
S5 = np.timedelta64(5, "s")
S6 = np.timedelta64(6, "s")
S7 = np.timedelta64(7, "s")
S11 = np.timedelta64(11, "s")

NaN = np.nan
NaT = np.datetime64("NaT", "D")
NaTd = np.timedelta64("NaT", "s")

EMPTY_VECTOR = Vector([], float)
GROUPS = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

nth0 = lambda x: nth(x, 0)
quantile05 = lambda x: quantile(x, 0.5)

TEST_MATRIX = [

    (all, [T, T, T, T, T, F, F, F, F, F], [T, T, F, F, F]),
    (all, [1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [T, T, F, F, F]),
    (all, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, NaN, NaN, NaN], [F, F, T, T, T]),
    (all, [D1, D2, D3, D4, D5, NaT, NaT, NaT, NaT, NaT], [T, T, T, T, T]),
    (all, ["a", "b", "c", "d", "e", "", "", "", "", ""], [T, T, T, T, T]),

    (any, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (any, [1, 2, 3, 4, 5, 0, 0, 0, 0, 0], [T, T, T, F, F]),
    (any, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, NaN, NaN, NaN], [F, T, T, T, F]),
    (any, [D1, D2, D3, D4, D5, NaT, NaT, NaT, NaT, NaT], [T, T, T, F, F]),
    (any, ["a", "b", "c", "d", "e", "", "", "", "", ""], [T, T, T, F, F]),

    (count, [T, T, T, T, T, F, F, F, F, F], [2, 2, 2, 2, 2]),
    (count, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 2, 2, 2, 2]),
    (count, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [2, 2, 2, 2, 2]),
    (count, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [2, 2, 2, 2, 2]),
    (count, [S1, S2, S3, S4, S5, S6, S7, NaTd, NaTd, NaTd], [2, 2, 2, 2, 2]),
    (count, ["a", "b", "c", "d", "e", "f", "g", "", "", ""], [2, 2, 2, 2, 2]),

    (count_unique, [T, T, T, T, T, F, F, F, F, F], [1, 1, 2, 1, 1]),
    (count_unique, [1, 1, 3, 3, 5, 6, 7, 8, 9, 10], [1, 1, 2, 2, 2]),
    (count_unique, [1.0, 1.0, 3.0, 3.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1, 1, 2, 2, 1]),
    (count_unique, [D1, D1, D3, D3, D5, D6, D7, NaT, NaT, NaT], [1, 1, 2, 2, 1]),
    (count_unique, ["a", "a", "c", "c", "e", "f", "g", "", "", ""], [1, 1, 2, 2, 1]),

    (first, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (first, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 3, 5, 7, 9]),
    (first, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.0, 3.0, 5.0, 7.0, NaN]),
    (first, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D1, D3, D5, D7, NaT]),
    (first, [S1, S2, S3, S4, S5, S6, S7, NaTd, NaTd, NaTd], [S1, S3, S5, S7, NaTd]),
    (first, ["a", "b", "c", "d", "e", "f", "g", "", "", ""], ["a", "c", "e", "g", ""]),

    (last, [T, T, T, T, T, F, F, F, F, F], [T, T, F, F, F]),
    (last, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6, 8, 10]),
    (last, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [2.0, 4.0, 6.0, NaN, NaN]),
    (last, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D2, D4, D6, NaT, NaT]),
    (last, [S1, S2, S3, S4, S5, S6, S7, NaTd, NaTd, NaTd], [S2, S4, S6, NaTd, NaTd]),
    (last, ["a", "b", "c", "d", "e", "f", "g", "", "", ""], ["b", "d", "f", "", ""]),

    (max, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (max, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 4, 6, 8, 10]),
    (max, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [2.0, 4.0, 6.0, 7.0, NaN]),
    (max, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D2, D4, D6, D7, NaT]),
    (max, [S1, S2, S3, S4, S5, S6, S7, NaTd, NaTd, NaTd], [S2, S4, S6, S7, NaTd]),

    (mean, [T, T, T, T, T, F, F, F, F, F], [1.0, 1.0, 0.5, 0.0, 0.0]),
    (mean, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.5, 3.5, 5.5, 7.5, 9.5]),
    (mean, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.5, 3.5, 5.5, 7.0, NaN]),
    (mean, [D1, D3, D3, D5, D5, D7, D7, NaT, NaT, NaT], [D2, D4, D6, D7, NaT]),
    (mean, [S1, S3, S3, S5, S5, S7, S7, NaTd, NaTd, NaTd], [S2, S4, S6, S7, NaTd]),

    (median, [T, T, T, T, T, F, F, F, F, F], [1.0, 1.0, 0.5, 0.0, 0.0]),
    (median, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.5, 3.5, 5.5, 7.5, 9.5]),
    (median, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.5, 3.5, 5.5, 7.0, NaN]),
    (median, [D1, D3, D3, D5, D5, D7, D7, NaT, NaT, NaT], [D2, D4, D6, D7, NaT]),
    (median, [S1, S3, S3, S5, S5, S7, S7, NaTd, NaTd, NaTd], [S2, S4, S6, S7, NaTd]),

    (min, [T, T, T, T, T, F, F, F, F, F], [T, T, F, F, F]),
    (min, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 3, 5, 7, 9]),
    (min, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.0, 3.0, 5.0, 7.0, NaN]),
    (min, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D1, D3, D5, D7, NaT]),
    (min, [S1, S2, S3, S4, S5, S6, S7, NaTd, NaTd, NaTd], [S1, S3, S5, S7, NaTd]),

    (mode, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (mode, [1, 1, 3, 3, 5, 6, 7, 8, 9, 10], [1, 3, 5, 7, 9]),
    (mode, [1.0, 1.0, 3.0, 3.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.0, 3.0, 5.0, 7.0, NaN]),
    (mode, [D1, D1, D3, D3, D5, D6, D7, NaT, NaT, NaT], [D1, D3, D5, D7, NaT]),
    (mode, [S1, S1, S3, S3, S5, S6, S7, NaTd, NaTd, NaTd], [S1, S3, S5, S7, NaTd]),
    (mode, ["a", "a", "c", "c", "e", "f", "g", "", "", ""], ["a", "c", "e", "g", ""]),

    (nth0, [T, T, T, T, T, F, F, F, F, F], [T, T, T, F, F]),
    (nth0, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 3, 5, 7, 9]),
    (nth0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.0, 3.0, 5.0, 7.0, NaN]),
    (nth0, [D1, D2, D3, D4, D5, D6, D7, NaT, NaT, NaT], [D1, D3, D5, D7, NaT]),
    (nth0, [S1, S2, S3, S4, S5, S6, S7, NaTd, NaTd, NaTd], [S1, S3, S5, S7, NaTd]),
    (nth0, ["a", "b", "c", "d", "e", "f", "g", "", "", ""], ["a", "c", "e", "g", ""]),

    (prod, [T, T, T, T, T, F, F, F, F, F], [1, 1, 0, 0, 0]),
    (prod, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [2, 12, 30, 56, 90]),
    (prod, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [2.0, 12.0, 30.0, 7.0, 1.0]),

    (quantile05, [T, T, T, T, T, F, F, F, F, F], [1.0, 1.0, 0.5, 0.0, 0.0]),
    (quantile05, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.5, 3.5, 5.5, 7.5, 9.5]),
    (quantile05, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [1.5, 3.5, 5.5, 7.0, NaN]),
    (quantile05, [D1, D3, D3, D5, D5, D7, D7, NaT, NaT, NaT], [D2, D4, D6, D7, NaT]),
    (quantile05, [S1, S3, S3, S5, S5, S7, S7, NaTd, NaTd, NaTd], [S2, S4, S6, S7, NaTd]),

    (std, [T, T, T, T, T, F, F, F, F, F], [0.0, 0.0, 0.5, 0.0, 0.0]),
    (std, [1, 1, 2, 3, 5, 7, 8, 12, 13, 21], [0.0, 0.5, 1.0, 2.0, 4.0]),
    (std, [1.0, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, NaN, NaN, NaN], [0.0, 0.5, 1.0, 0.0, NaN]),

    (sum, [T, T, T, T, T, F, F, F, F, F], [2, 2, 1, 0, 0]),
    (sum, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [3, 7, 11, 15, 19]),
    (sum, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, NaN, NaN, NaN], [3.0, 7.0, 11.0, 7.0, 0.0]),
    (sum, [S1, S2, S3, S4, S5, S6, S7, NaTd, NaTd, NaTd], [S3, S7, S11, S7, S0]),

    (var, [T, T, T, T, T, F, F, F, F, F], [0.0, 0.0, 0.25, 0.0, 0.0]),
    (var, [1, 1, 2, 3, 5, 7, 8, 12, 13, 21], [0.0, 0.25, 1.0, 4.0, 16.0]),
    (var, [1.0, 1.0, 2.0, 3.0, 5.0, 7.0, 8.0, NaN, NaN, NaN], [0.0, 0.25, 1.0, 0.0, NaN]),

]

class TestAggregate:

    @pytest.mark.parametrize("function,input,output", TEST_MATRIX)
    def test_aggregate(self, function, input, output):
        data = DataFrame(g=GROUPS, a=input)
        stat = data.group_by("g").aggregate(a=function("a"))
        expected = Vector(output)
        try:
            assert stat.a.equal(expected)
        except AssertionError:
            print("")
            print("Function:", function.__name__)
            print(data)
            print("Expected:")
            print(expected)
            print("Got:")
            print(stat.a)
            raise

    def test_aggregate_count(self):
        data = DataFrame(g=GROUPS)
        stat = data.group_by("g").aggregate(n=count())
        assert (stat.n == 2).all()

    def test_all(self):
        assert all(EMPTY_VECTOR)
        assert all(Vector([T, T]))
        assert not all(Vector([T, F]))
        assert not all(Vector([F, F]))

    def test_all_na(self):
        assert all(Vector([1.0, NaN])) is True
        assert all(Vector([1.0, NaN]), drop_na=False) is None
        assert all(Vector([0.0, NaN]), drop_na=False) is False

    def test_all_na_grouped(self):
        data = DataFrame(g=[1, 1, 2, 2, 3, 3], a=[1.0, NaN, 0.0, NaN, 1.0, 1.0])
        stat = data.group_by("g").aggregate(a=all("a", drop_na=False))
        assert stat.a.tolist() == [None, False, True]

    def test_any(self):
        assert not any(EMPTY_VECTOR)
        assert any(Vector([T, T]))
        assert any(Vector([T, F]))
        assert not any(Vector([F, F]))

    def test_any_na(self):
        assert any(Vector([0.0, NaN])) is False
        assert any(Vector([0.0, NaN]), drop_na=False) is None
        assert any(Vector([1.0, NaN]), drop_na=False) is True

    def test_any_na_grouped(self):
        data = DataFrame(g=[1, 1, 2, 2, 3, 3], a=[0.0, NaN, 1.0, NaN, 0.0, 0.0])
        stat = data.group_by("g").aggregate(a=any("a", drop_na=False))
        assert stat.a.tolist() == [None, True, False]

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
        assert count_unique(Vector([1, 1, 2, NaN, NaN])) == 3
        assert count_unique(Vector([1, 1, 2, NaN]), drop_na=True) == 2

    def test_first(self):
        assert first(Vector([1, 2, 3])) == 1
        assert first(Vector([NaN, 1, 2]), drop_na=True) == 1
        assert first(Vector(["a", "b"])) == "a"

    def test_first_nan(self):
        assert np.isnan(first(EMPTY_VECTOR))
        assert np.isnan(first(Vector([NaN, 1, 2])))

    def test_last(self):
        assert last(Vector([1, 2, 3])) == 3
        assert last(Vector([1, 2, NaN]), drop_na=True) == 2
        assert last(Vector(["a", "b"])) == "b"

    def test_last_nan(self):
        assert np.isnan(last(EMPTY_VECTOR))
        assert np.isnan(last(Vector([1, 2, NaN])))

    def test_max(self):
        assert max(Vector([3, 2, 1])) == 3
        assert max(Vector([3, 2, NaN])) == 3
        assert max(Vector(["b", "c", "a"])) == "c"

    def test_max_nanoseconds(self):
        x = Vector([D1, D2]).as_datetime("ns")
        assert max(x) == datetime.datetime.combine(D2, datetime.time())
        assert max(x - x[0]) == datetime.timedelta(days=1)

    def test_max_nan(self):
        assert np.isnan(max(EMPTY_VECTOR))
        assert np.isnan(max(Vector([3, 2, NaN]), drop_na=False))

    def test_mean(self):
        assert np.isclose(mean(Vector([1, 2, 10])), 4.333333)
        assert np.isclose(mean(Vector([1, 2, NaN])), 1.5)

    def test_mean_date(self):
        assert mean(Vector([D1, D3, NaT])) == D2

    def test_mean_nan(self):
        assert np.isnan(mean(EMPTY_VECTOR))
        assert np.isnan(mean(Vector([1, 2, NaN]), drop_na=False))

    def test_mean_weights(self):
        assert np.isclose(mean(Vector([1, 2, 10]), weights=Vector([3, 2, 1])), 17 / 6)
        assert mean(Vector([D1, D3]), weights=Vector([1, 1])) == D2

    def test_mean_weights_grouped(self):
        data = DataFrame(g=[1, 1, 2, 2], x=[1.0, 3.0, 5.0, 7.0], w=[1, 3, 0, 0])
        stat = data.group_by("g").aggregate(x=mean("x", weights="w"))
        assert stat.x.equal(Vector([2.5, NaN]))

    def test_mean_weights_nan(self):
        assert mean(Vector([1, 2, NaN]), weights=Vector([1, NaN, 1])) == 1
        assert np.isnan(mean(Vector([1, 2]), weights=Vector([1, NaN]), drop_na=False))
        assert np.isnan(mean(Vector([1, 2]), weights=Vector([0, 0])))

    def test_median(self):
        assert median(Vector([1, 4, 6, 8, 5])) == 5
        assert median(Vector([1, 4, 6, NaN, NaN])) == 4

    def test_median_date(self):
        assert median(Vector([D1, D2, D7, NaT])) == D2

    def test_median_nan(self):
        assert np.isnan(median(EMPTY_VECTOR))
        assert np.isnan(median(Vector([1, 4, NaN]), drop_na=False))

    def test_min(self):
        assert min(Vector([3, 2, 1])) == 1
        assert min(Vector([3, 2, NaN])) == 2
        assert min(Vector(["b", "c", "a"])) == "a"

    def test_min_nan(self):
        assert np.isnan(min(EMPTY_VECTOR))
        assert np.isnan(min(Vector([3, 2, NaN]), drop_na=False))

    def test_mode(self):
        assert mode(Vector([1])) == 1
        assert mode(Vector([1, 2])) == 1
        assert mode(Vector([1, 2, 2])) == 2
        assert mode(Vector([1, 2, 2, NaN])) == 2
        assert mode(Vector([1, 2, 2, NaN]), drop_na=False) == 2
        assert type(mode(Vector([1.5]))) is float

    def test_mode_nan(self):
        assert np.isnan(mode(EMPTY_VECTOR))
        assert np.isnan(mode(Vector([NaN, NaN], float), drop_na=False))

    def test_nth(self):
        assert nth(Vector([1, 2, 3]), 0) == 1
        assert nth(Vector([NaN, 1, 2]), 0, drop_na=True) == 1
        assert nth(Vector(["a", "b"]), 1) == "b"

    def test_nth_grouped(self):
        data = DataFrame(g=[2, 1, 2, 1, 2], a=[1, 2, 3, 4, 5])
        stat = data.group_by("g").aggregate(a=nth("a", 2), b=max("a"), c=nth("a", 0), d=nth("a", -1, drop_na=True))
        assert stat.a.equal(Vector([NaN, 5.0]))
        assert stat.b.tolist() == [4, 5]
        assert stat.c.tolist() == [2, 1]
        assert stat.d.tolist() == [4, 5]

    def test_nth_nan(self):
        assert np.isnan(nth(EMPTY_VECTOR, 0))
        assert np.isnan(nth(Vector([NaN, 1, 2]), 0))

    def test_prod(self):
        assert prod(EMPTY_VECTOR) == 1
        assert prod(Vector([2])) == 2
        assert prod(Vector([2, 3])) == 6
        assert prod(Vector([2, 3, NaN])) == 6

    def test_prod_date(self):
        with pytest.raises(TypeError):
            prod(Vector([D1, D2]))
        with pytest.raises(TypeError):
            DataFrame(g=[1, 1], a=[D1, D2]).group_by("g").aggregate(a=prod("a"))

    def test_prod_nan(self):
        assert np.isnan(prod(Vector([2, 3, NaN]), drop_na=False))

    def test_quantile(self):
        assert quantile(Vector([1, 4, 6, 8, 5]), 0.5) == 5
        assert quantile(Vector([1, 4, 6, NaN, NaN]), 0.5) == 4

    def test_quantile_date(self):
        assert quantile(Vector([D1, D3, NaT]), 0.5) == D2

    def test_quantile_nan(self):
        assert np.isnan(quantile(EMPTY_VECTOR, 0.5))
        assert np.isnan(quantile(Vector([1, 4, NaN]), 0.5, drop_na=False))

    def test_std(self):
        assert np.isclose(std(Vector([3, 6, 7])), 1.699673)
        assert np.isclose(std(Vector([3, 6, NaN])), 1.5)
        assert std(Vector([3])) == 0

    def test_std_date(self):
        with pytest.raises(TypeError):
            std(Vector([D1, D2]))
        with pytest.raises(TypeError):
            DataFrame(g=[1, 1], a=[D1, D2]).group_by("g").aggregate(a=std("a"))

    def test_std_nan(self):
        assert np.isnan(std(EMPTY_VECTOR))
        assert np.isnan(std(Vector([1]), ddof=1))
        assert np.isnan(std(Vector([3, 6, NaN]), drop_na=False))

    def test_sum(self):
        assert sum(EMPTY_VECTOR) == 0
        assert sum(Vector([1])) == 1
        assert sum(Vector([1, 2])) == 3
        assert sum(Vector([1, 2, NaN])) == 3

    def test_sum_date(self):
        with pytest.raises(TypeError):
            sum(Vector([D1, D2]))
        with pytest.raises(TypeError):
            DataFrame(g=[1, 1], a=[D1, D2]).group_by("g").aggregate(a=sum("a"))

    def test_sum_nan(self):
        assert np.isnan(sum(Vector([1, 2, NaN]), drop_na=False))

    def test_var(self):
        assert np.isclose(var(Vector([3, 6, 7])), 2.888889)
        assert np.isclose(var(Vector([3, 6, NaN])), 2.25)
        assert var(Vector([3])) == 0

    def test_var_date(self):
        with pytest.raises(TypeError):
            var(Vector([D1, D2]))
        with pytest.raises(TypeError):
            DataFrame(g=[1, 1], a=[D1, D2]).group_by("g").aggregate(a=var("a"))

    def test_var_nan(self):
        assert np.isnan(var(EMPTY_VECTOR))
        assert np.isnan(var(Vector([1]), ddof=1))
        assert np.isnan(var(Vector([3, 6, NaN]), drop_na=False))
