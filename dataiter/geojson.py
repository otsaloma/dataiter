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

class GeoJSON(DataFrame):

    """
    A class for GeoJSON data.

    GeoJSON is a simple wrapper class that reads GeoJSON features into a
    :class:`.DataFrame`. Any operations on the data are thus done with methods
    provided by the data frame class. Geometry is available in the "geometry"
    column, but no special geometric operations are supported. Feature
    identifiers, if present, are available in the "feature_id" column. All
    other data is available in the "metadata" attribute as an
    ``attd.AttributeDict``.
    """

    # List of names that are actual attributes, not columns
    ATTRIBUTES = DataFrame.ATTRIBUTES + ["metadata"]

    # Lists of supported GeoJSON keys and types
    FEATURE_KEYS = ["type", "id", "properties", "geometry"]
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
    def read(cls, path, *, encoding="utf-8", select=[], dtypes={}, **kwargs):
        """
        Return data from GeoJSON file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz|.zst``.
        `select` is an optional list of columns or glob patterns to limit to,
        as in :meth:`.DataFrame.select`. `dtypes` is an optional dict mapping
        column names to NumPy datatypes. `kwargs` are passed to ``json.load``.
        """
        with util.xopen(path, "rt", encoding=encoding) as f:
            raw = AttributeDict(json.load(f, **kwargs))
        cls._check_raw_data(raw)
        keys = util.unique_keys(key for x in raw.features for key in x.properties)
        has_ids = any("id" in x for x in raw.features)
        if has_ids and "feature_id" in keys:
            print("Warning: Ignoring feature key 'id' in favor of property 'feature_id'")
            has_ids = False
        names = ["feature_id", *keys] if has_ids else keys
        if select:
            # Geometry is always included last, but can be selected too.
            names = [x for x in util.glob_all(select, [*names, "geometry"]) if x != "geometry"]
        data = {}
        for name in names:
            if has_ids and name == "feature_id":
                data[name] = [x.get("id") for x in raw.features]
            else:
                data[name] = [x.properties.get(name, None) for x in raw.features]
        data["geometry"] = [x.geometry for x in raw.features]
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

    def to_string(self, *, max_rows=None, max_width=None, truncate_width=None):
        if "geometry" in self.colnames:
            geometry = [f"<{x['type']}>" for x in self.geometry]
            self = self.modify(geometry=Vector.fast(geometry, object))
        return DataFrame.to_string(self, max_rows=max_rows, max_width=max_width, truncate_width=truncate_width)

    def write(self, path, *, encoding="utf-8", **kwargs):
        """
        Write data to GeoJSON file `path`.

        Will automatically compress if `path` ends in ``.bz2|.gz|.xz|.zst``.
        `kwargs` are passed to ``json.dumps``.
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
                id = item.pop("feature_id", None)
                blob = {"type": "Feature", "id": id, "properties": item, "geometry": geometry}
                if id is None:
                    del blob["id"]
                blob = json.dumps(blob, **kwargs)
                comma = "," if i < len(data) - 1 else ""
                f.write(f"{indent2}{blob}{comma}\n")
            f.write(f"{indent1}]\n")
            f.write("}\n")
