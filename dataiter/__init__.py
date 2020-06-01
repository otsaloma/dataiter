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

from dataiter.vector import Vector # noqa
from dataiter.data_frame import DataFrame # noqa
from dataiter.data_frame import DataFrameColumn # noqa
from dataiter.list_of_dicts import ListOfDicts # noqa
from dataiter.list_of_dicts import ObsoleteError # noqa
from dataiter.list_of_dicts import ObsoleteListOfDicts # noqa

__version__ = "0.11"

DEFAULT_PEEK_ELEMENTS = 10
DEFAULT_PEEK_ITEMS = 3
DEFAULT_PEEK_ROWS = 10
PRINT_FLOAT_PRECISION = 6
PRINT_MAX_ELEMENTS = 100
PRINT_MAX_ITEMS = 10
PRINT_MAX_ROWS = 100
PRINT_MAX_WIDTH = 80

def ncol(data):
    return data.ncol

def nrow(data):
    return data.nrow
