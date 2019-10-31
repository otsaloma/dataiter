# -*- coding: utf-8 -*-

# Copyright (c) 2019 Osmo Salomaa
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
import os

from dataiter import ListOfDicts


class TestListOfDicts:

    def assert_common_keys_match(self, a, b):
        for key in set(a) & set(b):
            assert a[key] == b[key]

    def setup_method(self, method):
        # https://pypistats.org/api/packages/attd/system
        fname = os.path.splitext(__file__)[0] + ".json"
        with open(fname, "r") as f:
            data = json.load(f)["data"]
        self.data = ListOfDicts(data)

    def test___getitem____dict(self):
        data = self.data[0]
        assert isinstance(data, dict)

    def test___getitem____list(self):
        data = self.data[:3]
        assert isinstance(data, ListOfDicts)

    def test_aggregate(self):
        data = self.data.group_by("category").aggregate(**{
            "date_min":  lambda x: min(x.pluck("date")),
            "date_max":  lambda x: max(x.pluck("date")),
            "downloads": lambda x: sum(x.pluck("downloads")),
        })
        assert data == [{
            "category": "Darwin",
            "date_min": "2019-04-12",
            "date_max": "2019-06-04",
            "downloads": 4,
        }, {
            "category": "Linux",
            "date_min": "2019-04-12",
            "date_max": "2019-09-26",
            "downloads": 102,
        }, {
            "category": "Windows",
            "date_min": "2019-07-03",
            "date_max": "2019-07-03",
            "downloads": 1,
        }, {
            "category": "null",
            "date_min": "2019-04-02",
            "date_max": "2019-09-24",
            "downloads": 328,
        }]

    def test_deepcopy(self):
        data = self.data.deepcopy()
        assert data == self.data
        assert data is not self.data
        for item, orig in zip(data, self.data):
            assert item == orig
            assert item is not orig

    def test_filter__function(self):
        data = self.data.filter(lambda x: x.category == "Linux")
        assert len(data) == 38
        for item in data:
            assert item.category == "Linux"
            assert item in self.data

    def test_filter__key_value_pairs(self):
        data = self.data.filter(category="Linux")
        assert len(data) == 38
        for item in data:
            assert item.category == "Linux"
            assert item in self.data

    def test_filter_out__function(self):
        data = self.data.filter_out(lambda x: x.category == "Linux")
        assert len(data) == 99
        for item in data:
            assert item.category != "Linux"
            assert item in self.data

    def test_filter_out__key_value_pairs(self):
        data = self.data.filter_out(category="Linux")
        assert len(data) == 99
        for item in data:
            assert item.category != "Linux"
            assert item in self.data

    def test_from_json(self):
        string = json.dumps(self.data)
        data = ListOfDicts.from_json(string)
        assert data == self.data

    def test_join(self):
        other = ListOfDicts([
            {"category": "Darwin",  "path_separator": "/" },
            {"category": "Linux",   "path_separator": "/" },
            {"category": "Windows", "path_separator": "\\"},
        ])
        data = self.data.join(other, "category")
        assert len(data) == len(self.data)
        for item, orig in zip(data, self.data):
            if item.category == "Darwin":
                assert item.path_separator == "/"
            if item.category == "Linux":
                assert item.path_separator == "/"
            if item.category == "Windows":
                assert item.path_separator == "\\"
            self.assert_common_keys_match(item, orig)

    def test_modify(self):
        data = self.data.modify(year=lambda x: int(x.date[:4]))
        assert len(data) == len(self.data)
        for item, orig in zip(data, self.data):
            assert item.year == int(orig.date[:4])
            self.assert_common_keys_match(item, orig)

    def test_pluck(self):
        dates = self.data.pluck("date")
        assert len(dates) == len(self.data)
        for date, orig in zip(dates, self.data):
            assert date == orig.date

    def test_rename(self):
        data = self.data.rename(ymd="date")
        assert len(data) == len(self.data)
        for item, orig in zip(data, self.data):
            assert "ymd" in item
            assert "date" not in item
            self.assert_common_keys_match(item, orig)

    def test_select(self):
        data = self.data.select("date", "downloads")
        assert len(data) == len(self.data)
        for item, orig in zip(data, self.data):
            assert len(item) == 2
            assert "date" in item
            assert "downloads" in item
            self.assert_common_keys_match(item, orig)

    def test_sort(self):
        data = self.data.sort("date", "category")
        assert len(data) == len(self.data)
        for item in data:
            assert item in self.data

    def test_to_json(self):
        string = self.data.to_json()
        data = ListOfDicts.from_json(string)
        assert data == self.data

    def test_unique(self):
        data = self.data.unique("date")
        assert len(data) == 114
        for item in data:
            assert item in self.data

    def test_unselect(self):
        data = self.data.unselect("date", "downloads")
        assert len(data) == len(self.data)
        for item, orig in zip(data, self.data):
            assert "date" not in item
            assert "downloads" not in item
            self.assert_common_keys_match(item, orig)
