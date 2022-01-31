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

from dataiter import aggregate
from dataiter import Vector

skipif = pytest.mark.skipif


class TestAggregate:

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_drop_missing_datetime(self):
        x = Vector([None, "2022-01-01", "2022-01-02", "2022-01-03"]).as_date()
        g = np.repeat(1, len(x))
        y = aggregate.nth_numba(x, g, 0, drop_missing=True, default=x.missing_value)
        assert y[0] == x[1]

    @skipif(not di.USE_NUMBA, reason="No Numba")
    def test_drop_missing_float(self):
        x = Vector([None, 1, 2, 3])
        g = np.repeat(1, len(x))
        y = aggregate.nth_numba(x, g, 0, drop_missing=True, default=x.missing_value)
        assert y[0] == x[1]
