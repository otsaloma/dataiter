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

import json

from attd import AttributeDict
from dataiter import DataFrame
from dataiter import DataFrameColumn
from dataiter import util
from dataiter import Vector
from math import inf


class GeoJSON(DataFrame):

    """
    A class for GeoJSON data.

    GeoJSON is a simple wrapper class that reads GeoJSON features into a
    :class:`.DataFrame`. Any operations on the data are thus done with
    methods provided by the data frame class. Geometry is available in the
    "geometry" column, but no special geometric operations are supported.

    All other data is available in the "metadata" attribute as an
    ``attd.AttributeDict``.
    """

    # List of names that are actual attributes, not columns
    ATTRIBUTES = DataFrame.ATTRIBUTES + ["metadata"]

    # Lists of supported GeoJSON keys and types
    FEATURE_KEYS = ["type", "properties", "geometry"]
    FEATURE_TYPES = ["Feature"]
    PROPERTY_TYPES = [bool, int, float, str, type(None)]
    TOP_LEVEL_TYPES = ["FeatureCollection"]

    def __init__(self, *args, **kwargs):
        """
        Return a new GeoJSON object.

        `args` and `kwargs` are like for ``dict``.

        https://docs.python.org/3/library/stdtypes.html#dict
        """
        super().__init__(*args, **kwargs)
        self.metadata = AttributeDict(type="FeatureCollection")

    @classmethod
    def _check_raw_data(cls, data):
        if data.type not in cls.TOP_LEVEL_TYPES:
            raise TypeError(f"Top-level type {data.type!r} not supported")
        warned_feature_keys = []
        for feature in data.features:
            cls._check_raw_feature(feature, warned_feature_keys)

    @classmethod
    def _check_raw_feature(cls, feature, warned_feature_keys):
        if feature.type not in cls.FEATURE_TYPES:
            raise TypeError(f"Feature type {feature.type!r} not supported")
        for key in set(feature) - set(cls.FEATURE_KEYS):
            if key in warned_feature_keys: continue
            print(f"Warning: Ignoring feature key {key!r}")
            warned_feature_keys.append(key)
        for key, value in feature.properties.items():
            if isinstance(value, tuple(cls.PROPERTY_TYPES)): continue
            raise TypeError(f"Property type {type(value)} of {key!r} not supported")

    @classmethod
    def read(cls, path, *, encoding="utf-8", columns=[], strings_as_object=inf, dtypes={}, **kwargs):
        """
        Return data from GeoJSON file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz``.

        `columns` is an optional list of columns to limit to.

        `strings_as_object` is a cutoff point. If any row has more characters
        than that, the whole column will use the object data type. This is
        intended to help limit memory use as NumPy strings are fixed-length and
        can take a huge amount of memory if even a single row is long. If set,
        `dtypes` overrides this.

        `dtypes` is an optional dict mapping column names to NumPy datatypes.

        `kwargs` are passed to ``json.load``.
        """
        if (not isinstance(strings_as_object, (int, float)) or
            isinstance(strings_as_object, bool)):
            raise TypeError("Expected a number for strings_as_object")
        with util.xopen(path, "rt", encoding=encoding) as f:
            raw = AttributeDict(json.load(f, **kwargs))
        cls._check_raw_data(raw)
        data = {}
        for feature in raw.features:
            for key in feature.properties:
                data.setdefault(key, [])
        if columns:
            data = {k: v for k, v in data.items() if k in columns}
        for feature in raw.features:
            for key in data:
                value = feature.properties.get(key, None)
                data[key].append(value)
        data["geometry"] = [x.geometry for x in raw.features]
        dtypes = dtypes.copy()
        if strings_as_object < inf:
            for name in data:
                if (data[name] and
                    name not in dtypes and
                    any(isinstance(x, str) for x in data[name]) and
                    any(len(x) > strings_as_object for x in data[name] if isinstance(x, str))):
                    dtypes[name] = object
        for name, dtype in dtypes.items():
            data[name] = DataFrameColumn(data[name], dtype)
        data = cls(**data)
        del raw.features
        data.metadata = raw
        return data

    def to_data_frame(self, drop_geometry=False):
        """
        Return GeoJSON converted to a regular data frame.
        """
        data = dict.copy(self)
        if drop_geometry:
            data.pop("geometry", None)
        return DataFrame(**data)

    def to_string(self, *, max_rows=None, max_width=None):
        data = self
        if "geometry" in data.colnames:
            geometry = [f"<{x['type']}>" for x in data.geometry]
            data = data.modify(geometry=Vector.fast(geometry, object))
        return DataFrame.to_string(data, max_rows=max_rows, max_width=max_width)

    def write(self, path, *, encoding="utf-8", **kwargs):
        """
        Write data to GeoJSON file `path`.

        Will automatically compress if `path` ends in ``.bz2|.gz|.xz``.

        `kwargs` are passed to ``json.dump``.
        """
        kwargs.setdefault("default", str)
        kwargs.setdefault("ensure_ascii", False)
        indent_width = kwargs.pop("indent", 2) or 0
        indent1 = " " * indent_width * 1
        indent2 = " " * indent_width * 2
        if "geometry" not in self:
            raise ValueError("Geometry missing")
        data = self.to_list_of_dicts()
        util.makedirs_for_file(path)
        with util.xopen(path, "wt", encoding=encoding) as f:
            f.write("{\n")
            for key, value in self.metadata.items():
                blob = json.dumps(value, **kwargs)
                f.write(f'{indent1}"{key}": {blob},\n')
            f.write(f'{indent1}"features": [\n')
            for i, item in enumerate(data):
                geometry = item.pop("geometry")
                blob = {"type": "Feature", "properties": item, "geometry": geometry}
                blob = json.dumps(blob, **kwargs)
                comma = "," if i < len(data) - 1 else ""
                f.write(f"{indent2}{blob}{comma}\n")
            f.write(f"{indent1}]\n")
            f.write("}\n")
