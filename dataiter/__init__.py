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

import contextlib

from dataiter import util

__version__ = "0.47"

DEFAULT_PEEK_ELEMENTS = 10
DEFAULT_PEEK_ITEMS = 3
DEFAULT_PEEK_ROWS = 10
PRINT_FLOAT_PRECISION = 6
PRINT_MAX_ELEMENTS = 100
PRINT_MAX_ITEMS = 10
PRINT_MAX_ROWS = 100
PRINT_MAX_WIDTH = 80
PRINT_THOUSAND_SEPARATOR = ""
PRINT_TRUNCATE_WIDTH = 36
USE_NUMBA = False
USE_NUMBA_CACHE = True

with contextlib.suppress(LookupError):
    USE_NUMBA_CACHE = util.parse_env_boolean("DATAITER_USE_NUMBA_CACHE")

try:
    # Force Numba on or off if environment variable defined.
    USE_NUMBA = util.parse_env_boolean("DATAITER_USE_NUMBA")
except LookupError:
    with contextlib.suppress(Exception):
        # Use Numba automatically if found
        # and calling a trivial function works.
        import numba
        try:
            @numba.njit(cache=USE_NUMBA_CACHE)
            def check(x):
                return x**2
            assert check(10) == 100
            USE_NUMBA = True
        except Exception as error:
            print(f"Numba found, but disabled due to error: {error!s}")

globals().pop("check", None)
globals().pop("contextlib", None)
globals().pop("numba", None)
globals().pop("util", None)

from dataiter.vector import Vector # noqa
from dataiter.data_frame import DataFrame # noqa
from dataiter.data_frame import DataFrameColumn # noqa
from dataiter.geojson import GeoJSON # noqa
from dataiter.list_of_dicts import ListOfDicts # noqa
from dataiter import dt # noqa

from dataiter.aggregate import all # noqa
from dataiter.aggregate import any # noqa
from dataiter.aggregate import count # noqa
from dataiter.aggregate import count_unique # noqa
from dataiter.aggregate import first # noqa
from dataiter.aggregate import last # noqa
from dataiter.aggregate import max # noqa
from dataiter.aggregate import mean # noqa
from dataiter.aggregate import median # noqa
from dataiter.aggregate import min # noqa
from dataiter.aggregate import mode # noqa
from dataiter.aggregate import nth # noqa
from dataiter.aggregate import quantile # noqa
from dataiter.aggregate import std # noqa
from dataiter.aggregate import sum # noqa
from dataiter.aggregate import var # noqa

from dataiter.io import read_csv # noqa
from dataiter.io import read_geojson # noqa
from dataiter.io import read_json # noqa
from dataiter.io import read_npz # noqa
from dataiter.io import read_parquet # noqa
