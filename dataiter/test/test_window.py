# -*- coding: utf-8 -*-

# Copyright (c) 2026 Osmo Salomaa
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
import pytest

from dataiter import DataFrame
from dataiter import Vector
from dataiter.window import cummax
from dataiter.window import cummin
from dataiter.window import cumsum
from dataiter.window import rank
from dataiter.window import row_number
from dataiter.window import shift

NaN = np.nan

class TestWindow:

    def test_cummax(self):
        assert cummax(Vector([1, 3, 2, 4])).tolist() == [1, 3, 3, 4]
        assert cummax(Vector([1.0, NaN, 2.0])).equal(Vector([1.0, NaN, NaN]))

    def test_cummax_grouped(self):
        data = DataFrame(g=[1, 2, 1, 2, 1, 2], x=[1, 3, 0, 4, 2, 2])
        data = data.group_by("g").modify(y=cummax("x"))
        assert data.y.tolist() == [1, 3, 1, 4, 2, 4]

    def test_cummin(self):
        assert cummin(Vector([3, 1, 2, 0])).tolist() == [3, 1, 1, 0]
        assert cummin(Vector([1.0, NaN, 0.0])).equal(Vector([1.0, NaN, NaN]))

    def test_cummin_grouped(self):
        data = DataFrame(g=[1, 2, 1, 2, 1, 2], x=[1, 3, 0, 4, 2, 2])
        data = data.group_by("g").modify(y=cummin("x"))
        assert data.y.tolist() == [1, 3, 0, 3, 0, 2]

    def test_cumsum(self):
        assert cumsum(Vector([1, 2, 3])).tolist() == [1, 3, 6]
        assert cumsum(Vector([True, False, True])).tolist() == [1, 1, 2]
        assert cumsum(Vector([1.0, NaN, 2.0])).equal(Vector([1.0, NaN, NaN]))
        with pytest.raises(TypeError):
            cumsum(Vector(["a", "b"]))

    def test_cumsum_grouped(self):
        data = DataFrame(g=[1, 2, 1, 2, 1, 2], x=[1, 2, 3, 4, 5, 6])
        data = data.group_by("g").modify(y=cumsum("x"))
        assert data.y.tolist() == [1, 2, 4, 6, 9, 12]

    def test_cumsum_upcast(self):
        x = Vector(np.full(3, 2**30, np.int32))
        assert cumsum(x).tolist() == [2**30, 2**31, 3 * 2**30]
        assert cumsum(x).dtype == np.int64
        assert cumsum(x.astype(np.uint8)).dtype == np.uint64

    def test_cumsum_grouped_large(self):
        # Cover both looping over groups and scanning.
        for n in [10, 1000]:
            g = np.repeat(np.arange(n), 10_000 // n)
            x = np.random.random(len(g))
            data = DataFrame(g=g, x=x).group_by("g").modify(y=cumsum("x"))
            expected = np.concatenate([np.cumsum(x) for x in np.split(x, n)])
            assert np.allclose(data.y, expected, rtol=1e-14)

    @pytest.mark.parametrize("dtype", [bool, np.int16, np.uint16, np.int64])
    def test_cumsum_grouped_large_integers(self, dtype):
        for n in [10, 1000]:
            g = np.repeat(np.arange(n), 10_000 // n)
            x = np.random.randint(0, 2, len(g)).astype(dtype)
            data = DataFrame(g=g, x=x).group_by("g").modify(y=cumsum("x"))
            expected = np.concatenate([np.cumsum(x) for x in np.split(x, n)])
            assert data.y.tolist() == expected.tolist()

    def test_cumsum_grouped_overflow(self):
        # Sum of all overflows, sums within groups don't.
        data = DataFrame(g=[1, 1, 2, 2], x=[2**62, 2**62, 5, 7])
        data = data.group_by("g").modify(y=cumsum("x"))
        assert data.y.tolist() == [2**62, 2**63 - 2**64, 5, 12]

    def test_cumsum_grouped_empty(self):
        data = DataFrame(g=[1], x=[1]).head(0)
        data = data.group_by("g").modify(y=cumsum("x"))
        assert data.nrow == 0
        assert "y" in data

    @pytest.mark.parametrize("method", ["min", "max", "dense", "ordinal"])
    @pytest.mark.parametrize("dir", [1, -1])
    def test_rank_grouped(self, method, dir):
        data = DataFrame(g=[1, 2, 1, 2, 1, 2, 1, 2], x=[2, 1, NaN, 3, 2, 1, 1, NaN])
        data = data.group_by("g").modify(
            a=rank("x", method=method, dir=dir),
            b=lambda x: x.x.rank(method=method, dir=dir))
        assert data.a.tolist() == data.b.tolist()

    def test_rank(self):
        assert rank(Vector([3, 1, 1, 2])).tolist() == [4, 1, 1, 3]
        with pytest.raises(ValueError):
            DataFrame(g=[1, 2], x=[1, 2]).group_by("g").modify(r=rank("x", method="foo"))

    def test_row_number(self):
        data = DataFrame(x=[10, 20, 30])
        data = data.modify(i=row_number())
        assert data.i.equal(Vector([1, 2, 3]))

    def test_row_number_grouped(self):
        data = DataFrame(g=[2, 1, 2, 1, 2, 3], x=[1, 2, 3, 4, 5, 6])
        data = data.group_by("g").modify(i=row_number(), j=row_number(0))
        assert data.i.equal(Vector([1, 1, 2, 2, 3, 1]))
        assert data.j.equal(Vector([0, 0, 1, 1, 2, 0]))

    def test_row_number_grouped_empty(self):
        data = DataFrame(g=[1, 2]).filter_out(g=[1, 2])
        data = data.group_by("g").modify(i=row_number())
        assert data.nrow == 0

    def test_shift(self):
        assert shift(Vector([1, 2, 3])).equal(Vector([NaN, 1, 2]))
        assert shift(Vector([1, 2, 3]), -1).equal(Vector([2, 3, NaN]))

    def test_shift_grouped(self):
        data = DataFrame(g=[1, 2, 1, 2, 1, 2], x=[1, 2, 3, 4, 5, 6])
        data = data.group_by("g").modify(a=shift("x"), b=shift("x", -2), c=shift("x", 5))
        assert data.a.equal(Vector([NaN, NaN, 1, 2, 3, 4]))
        assert data.b.equal(Vector([5, 6, NaN, NaN, NaN, NaN]))
        assert data.c.equal(Vector([NaN] * 6))
