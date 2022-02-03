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

__version__ = "0.29.1"

DEFAULT_PEEK_ELEMENTS = 10
DEFAULT_PEEK_ITEMS = 3
DEFAULT_PEEK_ROWS = 10
PRINT_FLOAT_PRECISION = 6
PRINT_MAX_ELEMENTS = 100
PRINT_MAX_ITEMS = 10
PRINT_MAX_ROWS = 100
PRINT_MAX_WIDTH = 80
USE_NUMBA = False

with contextlib.suppress(Exception):
    # Use Numba automatically if found.
    import numba
    USE_NUMBA = True
    del numba

with contextlib.suppress(LookupError):
    # Force Numba on or off via an environment variable.
    USE_NUMBA = util.parse_env_boolean("DATAITER_USE_NUMBA")

del contextlib
del util

# Classes API
from dataiter.vector import Vector # noqa
from dataiter.data_frame import DataFrame # noqa
from dataiter.data_frame import DataFrameColumn # noqa
from dataiter.geojson import GeoJSON # noqa
from dataiter.list_of_dicts import ListOfDicts # noqa

# Functions API
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
from dataiter.aggregate import nrow
from dataiter.aggregate import nth
from dataiter.aggregate import quantile
from dataiter.aggregate import std
from dataiter.aggregate import sum
from dataiter.aggregate import var
from dataiter.io import read_csv
from dataiter.io import read_geojson
from dataiter.io import read_json
from dataiter.io import read_npz

for f in [
        all,
        any,
        count,
        count_unique,
        first,
        last,
        max,
        mean,
        median,
        min,
        mode,
        nrow,
        nth,
        quantile,
        read_csv,
        read_geojson,
        read_json,
        read_npz,
        std,
        sum,
        var,
]:
    # Patch module to include f in API documentation.
    f.__module__ = "dataiter"
