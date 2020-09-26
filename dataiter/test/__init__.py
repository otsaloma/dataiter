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

import os

from dataiter import DataFrame
from dataiter import GeoJSON
from dataiter import ListOfDicts


def data_frame(fname):
    fname = get_data_filename(fname)
    extension = fname.split(".")[-1]
    read = getattr(DataFrame, f"read_{extension}")
    return read(fname)

def geojson(fname):
    fname = get_data_filename(fname)
    return GeoJSON.read(fname)

def get_data_filename(fname):
    directory = os.path.dirname(__file__)
    return os.path.join(directory, "..", "..", "data", fname)

def list_of_dicts(fname):
    fname = get_data_filename(fname)
    extension = fname.split(".")[-1]
    read = getattr(ListOfDicts, f"read_{extension}")
    return read(fname)
