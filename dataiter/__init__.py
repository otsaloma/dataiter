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

__version__ = "1.4"

#: Default amount of elements to return from :meth:`Vector.head`,
#: :meth:`Vector.sample` and :meth:`Vector.tail`.
DEFAULT_PEEK_ELEMENTS = 10

#: Default amount of items to return from :meth:`ListOfDicts.head`,
#: :meth:`ListOfDicts.sample` and :meth:`ListOfDicts.tail`.
DEFAULT_PEEK_ITEMS = 3

#: Default amount of rows to return from :meth:`DataFrame.head`,
#: :meth:`DataFrame.sample` and :meth:`DataFrame.tail`.
DEFAULT_PEEK_ROWS = 10

#: Amount of significant digits to print floats with, all values of a
#: column or vector using the decimals that the largest value gets.
PRINT_FLOAT_PRECISION = 6

#: Maximum amount of elements to show when printing a :class:`Vector`.
PRINT_MAX_ELEMENTS = 100

#: Maximum amount of items to show when printing a :class:`ListOfDicts`.
PRINT_MAX_ITEMS = 10

#: Maximum amount of rows to show when printing a :class:`DataFrame`.
PRINT_MAX_ROWS = 20

#: Maximum amount of columns to wrap print output to. This is only a fallback
#: in case Python's ``shutil.get_terminal_size`` fails to detect the width of
#: your terminal. By default the detected full width is used.
PRINT_MAX_WIDTH = 80

#: Thousand separator to use when printing numbers. By default this is blank,
#: meaning no thousand separators are rendered.
PRINT_THOUSAND_SEPARATOR = ""

#: Maximum width to truncate string columns to in :class:`DataFrame` print
#: output. When this is exceeded, strings will be cut and an ellipsis (``…``)
#: rendered at the cut point.
PRINT_TRUNCATE_WIDTH = 36

from dataiter import dtypes
from dataiter.vector import Vector
from dataiter.vector import of_type
from dataiter.data_frame import DataFrame
from dataiter.data_frame import DataFrameColumn
from dataiter.geojson import GeoJSON
from dataiter.list_of_dicts import ListOfDicts
from dataiter import dt
from dataiter import regex

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

from dataiter.io import read_csv
from dataiter.io import read_geojson
from dataiter.io import read_json
from dataiter.io import read_npz
from dataiter.io import read_parquet

from dataiter import util

#: Special value to refer to the next element/item.
#: Used by ``replace_na`` to replace missing values with
#: the next non-missing value, i.e. to fill backward.
NEXT = util.Sentinel("NEXT")

#: Special value to refer to the previous element/item.
#: Used by ``replace_na`` to replace missing values with
#: the previous non-missing value, i.e. to fill forward.
PREVIOUS = util.Sentinel("PREVIOUS")
