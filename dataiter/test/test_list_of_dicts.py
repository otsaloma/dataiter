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

import tempfile

from attd import AttributeDict
from dataiter import ListOfDicts
from dataiter import test


class TestListOfDicts:

    def is_list_of_dicts(self, data):
        return (isinstance(data, ListOfDicts) and
                all(isinstance(x, AttributeDict) for x in data))

    def test___init__(self):
        item = dict(a=1, b=2, c=3)
        data = ListOfDicts([item])
        assert len(data) == 1
        assert data[0] == item
        assert data[0] is not item

    def test___init___empty(self):
        data = ListOfDicts()
        assert len(data) == 0

    def test___add__(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig + orig
        assert isinstance(data, ListOfDicts)
        assert len(data) == len(orig) * 2
        assert data[:len(orig)] == orig
        assert data[-len(orig):] == orig

    def test___getitem__(self):
        data = test.list_of_dicts("downloads.json")
        assert isinstance(data[0], AttributeDict)
        assert isinstance(data[:100], ListOfDicts)

    def test___mul__(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig * 2
        assert isinstance(data, ListOfDicts)
        assert len(data) == len(orig) * 2
        assert data[:len(orig)] == orig
        assert data[-len(orig):] == orig

    def test___rmul__(self):
        orig = test.list_of_dicts("downloads.json")
        data = 2 * orig
        assert isinstance(data, ListOfDicts)
        assert len(data) == len(orig) * 2
        assert data[:len(orig)] == orig
        assert data[-len(orig):] == orig

    def test___setitem__(self):
        data = test.list_of_dicts("downloads.json")
        item = dict(date="1970-01-01")
        data[0] = item
        assert isinstance(data[0], AttributeDict)
        assert data[0] == item
        assert data[0] is not item

    def test_aggregate(self):
        data = test.list_of_dicts("downloads.json")
        stat = data.group_by("category").aggregate(
            date_min =lambda x: min(x.pluck("date")),
            date_max =lambda x: max(x.pluck("date")),
            downloads=lambda x: sum(x.pluck("downloads")),
        )
        assert stat == [{
            "category": "Darwin",
            "date_min": "2019-09-16",
            "date_max": "2020-03-14",
            "downloads": 6928129,
        }, {
            "category": "Linux",
            "date_min": "2019-09-16",
            "date_max": "2020-03-14",
            "downloads": 510902781,
        }, {
            "category": "Windows",
            "date_min": "2019-09-16",
            "date_max": "2020-03-14",
            "downloads": 13024960,
        }, {
            "category": "null",
            "date_min": "2019-09-16",
            "date_max": "2020-03-14",
            "downloads": 10421576,
        }, {
            "category": "other",
            "date_min": "2019-09-16",
            "date_max": "2020-03-14",
            "downloads": 58299,
        }]

    def test_anti_join(self):
        orig = test.list_of_dicts("downloads.json")
        holidays = test.list_of_dicts("holidays.json")
        data = orig.anti_join(holidays, "date")
        assert len(data) == 870
        assert sum(data.pluck("downloads")) == 523109256

    def test_append(self):
        orig = test.list_of_dicts("downloads.json")
        item = dict(date="3000-01-01")
        data = orig.append(item)
        assert len(data) == len(orig) + 1
        assert isinstance(data[-1], AttributeDict)
        assert data[-1] == item

    def test_clear(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.clear()
        assert len(data) == 0

    def test_copy(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.copy()
        assert data == orig
        assert data is not orig
        for a, b in zip(data, orig):
            assert a == b
            assert a is b

    def test_copy_handle_predecessor(self):
        a = test.list_of_dicts("downloads.json")
        b = a.select("date")
        c = b.copy()
        assert a._predecessor is None
        assert b._predecessor is a
        assert c._predecessor is b

    def test_deepcopy(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.deepcopy()
        assert data == orig
        assert data is not orig
        for a, b in zip(data, orig):
            assert a == b
            assert a is not b

    def test_deepcopy_handle_predecessor(self):
        a = test.list_of_dicts("downloads.json")
        b = a.select("date")
        c = b.deepcopy()
        assert a._predecessor is None
        assert b._predecessor is a
        assert c._predecessor is None

    def test_extend(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.extend(orig)
        assert isinstance(data, ListOfDicts)
        assert len(data) == len(orig) * 2
        assert data[:len(orig)] == orig
        assert data[-len(orig):] == orig

    def test_filter_given_function(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.filter(lambda x: x.category == "Linux")
        assert len(data) == 181
        assert sum(data.pluck("downloads")) == 510902781

    def test_filter_given_key_value_pairs(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.filter(category="Linux")
        assert len(data) == 181
        assert sum(data.pluck("downloads")) == 510902781

    def test_filter_out_given_function(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.filter_out(lambda x: x.category == "Linux")
        assert len(data) == 724
        assert sum(data.pluck("downloads")) == 30432964

    def test_filter_out_given_key_value_pairs(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.filter_out(category="Linux")
        assert len(data) == 724
        assert sum(data.pluck("downloads")) == 30432964

    def test_from_json(self):
        orig = test.list_of_dicts("downloads.json")
        text = orig.to_json()
        data = ListOfDicts.from_json(text)
        assert data == orig

    def test_full_join(self):
        orig = test.list_of_dicts("downloads.json")
        holidays = test.list_of_dicts("holidays.json")
        data = orig.full_join(holidays, "date")
        assert len(data) == 930
        assert sum("holiday" in x for x in data) == 60
        assert sum(data.pluck("downloads", 0)) == 541335745

    def test_head(self):
        data = test.list_of_dicts("downloads.json")
        assert data.head(10) == data[:10]

    def test_inner_join(self):
        orig = test.list_of_dicts("downloads.json")
        holidays = test.list_of_dicts("holidays.json")
        data = orig.inner_join(holidays, "date")
        assert len(data) == 35
        assert all("holiday" in x for x in data)
        assert sum(data.pluck("downloads")) == 18226489

    def test_insert(self):
        orig = test.list_of_dicts("downloads.json")
        item = dict(date="3000-01-01")
        data = orig.insert(100, item)
        assert len(data) == len(orig) + 1
        assert isinstance(data[100], AttributeDict)
        assert data[100] == item

    def test_left_join(self):
        orig = test.list_of_dicts("downloads.json")
        holidays = test.list_of_dicts("holidays.json")
        data = orig.left_join(holidays, "date")
        assert len(data) == 905
        assert sum("holiday" in x for x in data) == 35
        assert sum(data.pluck("downloads")) == 541335745

    def test__mark_obsolete_after_multiple_modify(self):
        data = test.list_of_dicts("downloads.json")
        data = data.modify(a=lambda x: 1)
        data = data.modify(b=lambda x: 2)
        data = data.modify(c=lambda x: 3)

    def test_modify(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.modify(year=lambda x: int(x.date[:4]))
        assert len(data) == len(orig)
        assert all("year" in x for x in data)
        assert sum(data.pluck("year")) == 1827565
        assert orig._obsolete

    def test_modify_if(self):
        orig = test.list_of_dicts("downloads.json")
        predicate = lambda x: x.category == "Linux"
        data = orig.modify_if(predicate, year=lambda x: int(x.date[:4]))
        assert len(data) == len(orig)
        assert sum("year" in x for x in data) == 181
        assert sum(data.pluck("year", 0)) == 365513
        assert orig._obsolete

    def test_pluck(self):
        data = test.list_of_dicts("downloads.json")
        downloads = data.pluck("downloads")
        assert len(downloads) == len(data)
        assert sum(downloads) == 541335745

    def test_pluck_with_missing(self):
        data = test.list_of_dicts("downloads.json")
        del data[0].downloads
        downloads = data.pluck("downloads", 0)
        assert len(downloads) == len(data)
        assert sum(downloads) == 541300639

    def test_read_csv(self):
        data = test.list_of_dicts("vehicles.csv")
        assert len(data) == 33442
        assert all(len(x) == 12 for x in data)

    def test_read_json(self):
        data = test.list_of_dicts("downloads.json")
        assert len(data) == 905
        assert all(len(x) == 3 for x in data)

    def test_read_pickle(self):
        orig = test.list_of_dicts("downloads.json")
        handle, fname = tempfile.mkstemp(".pkl")
        orig.write_pickle(fname)
        data = ListOfDicts.read_pickle(fname)
        assert data == orig

    def test_rename(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.rename(ymd="date")
        assert len(data) == len(orig)
        assert all("ymd" in x for x in data)
        assert all("date" not in x for x in data)
        assert orig._obsolete

    def test_reverse(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.reverse()
        assert data == orig[::-1]

    def test_sample(self):
        data = test.list_of_dicts("downloads.json")
        assert len(data.sample(10)) == 10

    def test_select(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.select("date", "downloads")
        assert self.is_list_of_dicts(data)
        assert len(data) == len(orig)
        assert all(len(x) == 2 for x in data)
        assert all("date" in x for x in data)
        assert all("downloads" in x for x in data)
        assert orig._obsolete

    def test_select_order(self):
        data1 = test.list_of_dicts("downloads.json").select("date", "downloads")
        data2 = test.list_of_dicts("downloads.json").select("downloads", "date")
        assert list(data1[0].keys()) == ["date", "downloads"]
        assert list(data2[0].keys()) == ["downloads", "date"]

    def test_semi_join(self):
        orig = test.list_of_dicts("downloads.json")
        holidays = test.list_of_dicts("holidays.json")
        data = orig.semi_join(holidays, "date")
        assert len(data) == 35
        assert sum(data.pluck("downloads")) == 18226489

    def test_sort_ascending(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.sort(date=1, category=1)
        assert len(data) == len(orig)
        assert all(x in orig for x in data)
        assert data.pluck("date") == sorted(data.pluck("date"))
        assert data[0].date == min(data.pluck("date"))
        assert data[-1].date == max(data.pluck("date"))
        assert data[0].category == min(data.pluck("category"))
        assert data[-1].category == max(data.pluck("category"))

    def test_sort_descending(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.sort(date=-1, category=-1)
        assert len(data) == len(orig)
        assert all(x in orig for x in data)
        assert data.pluck("date") == sorted(data.pluck("date"), reverse=True)
        assert data[0].date == max(data.pluck("date"))
        assert data[-1].date == min(data.pluck("date"))
        assert data[0].category == max(data.pluck("category"))
        assert data[-1].category == min(data.pluck("category"))

    def test_sort_with_none_ascending(self):
        # Nones should be sorted last.
        orig = test.list_of_dicts("downloads.json")
        orig[0].category = None
        data = orig.sort(category=1)
        assert data[-1] is orig[0]

    def test_sort_with_none_descending(self):
        # Nones should be sorted last.
        orig = test.list_of_dicts("downloads.json")
        orig[0].category = None
        data = orig.sort(category=-1)
        assert data[-1] is orig[0]

    def test_sort_with_none_multiple_keys_ascending(self):
        # Nones should be sorted group-wise last.
        orig = test.list_of_dicts("downloads.json")
        orig[0].category = None
        orig[1].date = None
        orig[2].category = None
        orig[2].date = None
        data = orig.sort(date=1, category=1)
        assert data[ 4] is orig[0]
        assert data[-2] is orig[1]
        assert data[-1] is orig[2]

    def test_sort_with_none_multiple_keys_descending(self):
        # Nones should be sorted group-wise last.
        orig = test.list_of_dicts("downloads.json")
        orig[0].category = None
        orig[1].date = None
        orig[2].category = None
        orig[2].date = None
        data = orig.sort(date=-1, category=-1)
        assert data[-3] is orig[0]
        assert data[-2] is orig[1]
        assert data[-1] is orig[2]

    def test_tail(self):
        data = test.list_of_dicts("downloads.json")
        assert data.tail(10) == data[-10:]

    def test_to_data_frame(self):
        orig = test.list_of_dicts("vehicles.csv")
        data = orig.to_data_frame()
        assert data.nrow == len(orig)
        assert data.ncol == len(orig[0])

    def test_to_json(self):
        orig = test.list_of_dicts("downloads.json")
        text = orig.to_json()
        data = ListOfDicts.from_json(text)
        assert data == orig

    def test_to_pandas(self):
        orig = test.list_of_dicts("vehicles.csv")
        data = orig.to_pandas()
        assert data.shape[0] == len(orig)
        assert data.shape[1] == len(orig[0])

    def test_to_string(self):
        data = test.list_of_dicts("downloads.json")
        assert data.head(0).to_string()
        assert data.head(5).to_string()

    def test_unique(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.unique("date")
        assert len(data) == 181
        by = data.pluck("date")
        assert len(set(by)) == len(by)

    def test_unique_by_all(self):
        orig = test.list_of_dicts("downloads.json")
        orig = orig.append(orig[-1])
        data = orig.unique()
        assert len(data) == len(orig) - 1

    def test_unselect(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.unselect("date", "downloads")
        assert len(data) == len(orig)
        assert all(len(x) == 1 for x in data)
        assert all("date" not in x for x in data)
        assert all("downloads" not in x for x in data)
        assert orig._obsolete

    def test_write_csv(self):
        orig = test.list_of_dicts("vehicles.csv")
        handle, fname = tempfile.mkstemp(".csv")
        orig.write_csv(fname)
        data = ListOfDicts.read_csv(fname)
        assert data == orig

    def test_write_json(self):
        orig = test.list_of_dicts("downloads.json")
        handle, fname = tempfile.mkstemp(".json")
        orig.write_json(fname)
        data = ListOfDicts.read_json(fname)
        assert data == orig

    def test_write_pickle(self):
        orig = test.list_of_dicts("downloads.json")
        handle, fname = tempfile.mkstemp(".pkl")
        orig.write_pickle(fname)
        data = ListOfDicts.read_pickle(fname)
        assert data == orig
