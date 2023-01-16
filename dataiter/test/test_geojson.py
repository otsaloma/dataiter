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

import tempfile

from dataiter import GeoJSON
from dataiter import test
from pathlib import Path


class TestGeoJSON:

    path = "neighbourhoods.geojson"

    def test_read(self):
        path = str(test.get_data_path(self.path))
        data = GeoJSON.read(path)
        assert data.nrow == 233
        assert data.ncol == 3

    def test_read_columns(self):
        path = test.get_data_path(self.path)
        data = GeoJSON.read(path, columns=["neighbourhood"])
        assert data.colnames == ["neighbourhood", "geometry"]

    def test_read_dtypes(self):
        path = test.get_data_path(self.path)
        dtypes = {"neighbourhood": object, "neighbourhood_group": object}
        data = GeoJSON.read(path, dtypes=dtypes)
        assert data.neighbourhood.is_object()
        assert data.neighbourhood_group.is_object()

    def test_read_strings_as_object(self):
        path = test.get_data_path(self.path)
        data = GeoJSON.read(path, strings_as_object=8)
        assert data.neighbourhood.is_object()
        assert data.neighbourhood_group.is_object()

    def test_to_data_frame(self):
        orig = test.geojson(self.path)
        data = orig.to_data_frame()
        assert data.ncol == orig.ncol
        assert data.nrow == orig.nrow
        assert not isinstance(data, GeoJSON)

    def test_to_data_frame_drop_geometry(self):
        orig = test.geojson(self.path)
        data = orig.to_data_frame(drop_geometry=True)
        assert data.ncol == orig.ncol - 1
        assert data.nrow == orig.nrow
        assert not isinstance(data, GeoJSON)
        assert "geometry" not in data.colnames

    def test_to_string(self):
        data = test.geojson(self.path)
        assert data.head(0).to_string()
        assert data.head(5).to_string()

    def test_to_string_no_geometry(self):
        data = test.geojson(self.path)
        del data.geometry
        assert data.head(0).to_string()
        assert data.head(5).to_string()

    def test_write(self):
        orig = test.geojson(self.path)
        handle, path = tempfile.mkstemp(".geojson")
        orig.write(path)
        data = GeoJSON.read(path)
        assert data == orig
        assert data.metadata == orig.metadata

    def test_write_path(self):
        orig = test.geojson(self.path)
        handle, path = tempfile.mkstemp(".geojson")
        orig.write(Path(path))
