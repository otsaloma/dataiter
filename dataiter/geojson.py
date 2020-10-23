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
import warnings

from attd import AttributeDict
from dataiter import DataFrame
from dataiter import Vector
from dataiter import util


class GeoJSON(DataFrame):

    """
    A class for GeoJSON data.

    GeoJSON is a simple wrapper class that reads GeoJSON features into a
    :class:`dataiter.DataFrame`. Any operations on the data are thus done with
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
            raise Exception(f"Top-level type {data.type!r} not supported")
        for feature in data.features:
            cls._check_raw_feature(feature)

    @classmethod
    def _check_raw_feature(cls, feature):
        if feature.type not in cls.FEATURE_TYPES:
            raise Exception(f"Feature type {feature.type!r} not supported")
        for key in set(feature) - set(cls.FEATURE_KEYS):
            warnings.warn(f"Ignoring feature key {key!r}")
        for key, value in feature.properties.items():
            if isinstance(value, tuple(cls.PROPERTY_TYPES)): continue
            raise Exception(f"Property type {type(value)} of {key!r} not supported")

    @classmethod
    def read(cls, fname, encoding="utf_8", **kwargs):
        """
        Return data from GeoJSON file `fname`.

        `kwargs` are passed to ``json.load``.

        >>> data = di.GeoJSON.read("data/neighbourhoods.geojson")
        >>> data.head()
        """
        with open(fname, "r", encoding=encoding) as f:
            raw = AttributeDict(json.load(f, **kwargs))
        cls._check_raw_data(raw)
        data = {}
        for feature in raw.features:
            for key in feature.properties:
                data.setdefault(key, [])
        for feature in raw.features:
            for key in data:
                value = feature.properties.get(key, None)
                data[key].append(value)
        data["geometry"] = [x.geometry for x in raw.features]
        data = cls(**data)
        del raw.features
        data.metadata = raw
        return data

    def to_string(self, max_rows=None, max_width=None):
        geometry = [f"<{x['type']}>" for x in self.geometry]
        data = self.modify(geometry=Vector.fast(geometry, object))
        return DataFrame.to_string(data, max_rows, max_width)

    def write(self, fname, encoding="utf_8", **kwargs):
        """
        Write data to GeoJSON file `fname`.

        `kwargs` are passed to ``json.dumps``.
        """
        kwargs.setdefault("default", str)
        kwargs.setdefault("ensure_ascii", False)
        indent_width = kwargs.pop("indent", 2) or 0
        indent1 = " " * indent_width * 1
        indent2 = " " * indent_width * 2
        if "geometry" not in self:
            raise Exception("Geometry missing")
        data = self.to_list_of_dicts()
        util.makedirs_for_file(fname)
        with open(fname, "w", encoding=encoding) as f:
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
