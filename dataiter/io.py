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

from dataiter import DataFrame
from dataiter import GeoJSON
from dataiter import ListOfDicts
from dataiter import util
from math import inf


def read_csv(path, *, encoding="utf-8", sep=",", header=True, columns=[], strings_as_object=inf, dtypes={}):
    return DataFrame.read_csv(path,
                              encoding=encoding,
                              sep=sep,
                              header=header,
                              columns=columns,
                              strings_as_object=strings_as_object,
                              dtypes=dtypes)

def read_geojson(path, *, encoding="utf-8", columns=[], strings_as_object=inf, dtypes={}, **kwargs):
    return GeoJSON.read(path,
                        encoding=encoding,
                        columns=columns,
                        strings_as_object=strings_as_object,
                        dtypes=dtypes,
                        **kwargs)

def read_json(path, *, encoding="utf-8", keys=[], types={}, **kwargs):
    return ListOfDicts.read_json(path,
                                 encoding=encoding,
                                 keys=keys,
                                 types=types,
                                 **kwargs)

def read_npz(path, *, allow_pickle=True):
    return DataFrame.read_npz(path, allow_pickle=allow_pickle)

def read_parquet(path, *, columns=[], strings_as_object=inf, dtypes={}):
    return DataFrame.read_parquet(path, columns=[], strings_as_object=inf, dtypes={})

read_csv.__doc__ = util.format_alias_doc(read_csv, DataFrame.read_csv)
read_geojson.__doc__ = util.format_alias_doc(read_geojson, GeoJSON.read)
read_json.__doc__ = util.format_alias_doc(read_json, ListOfDicts.read_json)
read_npz.__doc__ = util.format_alias_doc(read_npz, DataFrame.read_npz)
read_parquet.__doc__ = util.format_alias_doc(read_parquet, DataFrame.read_parquet)
