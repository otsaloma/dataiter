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

import datetime
import numpy as np

from attd import AttributeDict
from dataiter import ListOfDicts
from dataiter import test
from dataiter import Vector
from unittest.mock import patch

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

    def test_aggregate_ungrouped(self):
        data = test.list_of_dicts("downloads.json")
        stat = data.aggregate(n=len, downloads=lambda x: sum(x.pluck("downloads")))
        assert stat == [{"n": len(data), "downloads": sum(data.pluck("downloads"))}]

    def test_aggregate_ungrouped_empty(self):
        data = ListOfDicts([])
        assert data.aggregate(n=len, x=lambda x: x[0].x) == []

    def test_aggregate_ungroups(self):
        data = ListOfDicts(dict(g=g) for g in [2, 1, 2, 3, 2])
        stat = data.group_by("g").aggregate(n=len).sort(n=-1).head(1)
        assert stat == [{"g": 2, "n": 3}]

    def test_aggregate_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        stat = orig.group_by("b").aggregate(n=len)
        assert stat == [{"b": 2, "n": 1}, {"b": None, "n": 2}]

    def test_anti_join(self):
        orig = test.list_of_dicts("downloads.json")
        holidays = test.list_of_dicts("holidays.json")
        data = orig.anti_join(holidays, "date")
        assert len(data) == 870
        assert sum(data.pluck("downloads")) == 523109256

    def test_anti_join_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        other = ListOfDicts([{"b": 2, "c": 1}, {"c": 2}])
        data = orig.anti_join(other, "b")
        assert data.pluck("a") == [2, 3]

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
        b = a.head(10)
        c = b.copy()
        assert a._predecessor is None
        assert b._predecessor is a
        assert c._predecessor is b

    def test_count(self):
        data = test.list_of_dicts("listings.json")
        stat = data.count("hood")
        assert stat.pluck("hood") == sorted(set(data.pluck("hood")))
        assert sum(stat.pluck("n")) == len(data)

    def test_count_n_exists(self):
        data = test.list_of_dicts("listings.json")
        stat1 = data.count("zipcode")
        stat2 = data.count("zipcode").count("n")
        assert list(stat1.keys()) == ["zipcode", "n"]
        assert list(stat2.keys()) == ["n_", "n"]
        assert len(stat2) == len(set(stat1.pluck("n")))
        assert sum(stat2.pluck("n")) == len(stat1)

    def test_count_ungrouped(self):
        data = test.list_of_dicts("listings.json")
        assert data.count() == [{"n": len(data)}]
        assert ListOfDicts([]).count() == []

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
        b = a.head(10)
        c = b.deepcopy()
        assert a._predecessor is None
        assert b._predecessor is a
        assert c._predecessor is None

    def test_drop_na(self):
        data = test.list_of_dicts("listings.json")
        data = data.drop_na("sqft")
        assert len(data) == 396
        assert None not in data.pluck("sqft")

    def test_drop_na_all(self):
        data = ListOfDicts([{"a": 1, "b": 1}, {"a": None, "b": 2}, {"a": 3}])
        assert data.drop_na().pluck("a") == [1]

    def test_expand_grid(self):
        data = ListOfDicts.expand_grid(x=[1, 2], y=["a", "b", "c"])
        assert data.pluck("x") == [1, 1, 1, 2, 2, 2]
        assert data.pluck("y") == ["a", "b", "c", "a", "b", "c"]

    def test_extend(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.extend(orig)
        assert isinstance(data, ListOfDicts)
        assert len(data) == len(orig) * 2
        assert data[:len(orig)] == orig
        assert data[-len(orig):] == orig

    def test_fill_missing_keys(self):
        orig = test.list_of_dicts("downloads.json")
        for i, item in enumerate(orig):
            if i % 2 == 0:
                del item.downloads
        data = orig.deepcopy().fill_missing_keys(downloads=0)
        assert sum("downloads" in x for x in orig) == len(orig) // 2
        assert sum("downloads" in x for x in data) == len(orig)

    def test_fill_missing_keys_all(self):
        orig = test.list_of_dicts("downloads.json")
        for i, item in enumerate(orig):
            if i % 2 == 0: del item.category
            if i % 4 == 0: del item.date
            if i % 8 == 0: del item.downloads
        data = orig.deepcopy().fill_missing_keys()
        for item in data:
            assert "category" in item
            assert "date" in item
            assert "downloads" in item

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

    def test_filter_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        data = orig.filter(b=None)
        assert data.pluck("a") == [2, 3]

    def test_filter_given_key_sequence_pairs(self):
        data = ListOfDicts([{"x": "a", "y": 1}, {"x": "b", "y": 2}, {"x": "c", "y": 3}, {"y": 4}])
        for value in [["a", "c"], ("a", "c"), {"a", "c"}, np.array(["a", "c"]), Vector(["a", "c"])]:
            assert data.filter(x=value).pluck("y") == [1, 3]
        assert data.filter(x=["a", "c"], y=3).pluck("y") == [3]
        assert data.filter(x=["a", None]).pluck("y") == [1, 4]
        assert data.filter_out(x=["a", "c"]).pluck("y") == [2, 4]

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

    def test_filter_out_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        data = orig.filter_out(b=None)
        assert data.pluck("a") == [1]

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
        assert max(len(x) for x in data) == 4
        assert sum("holiday" in x for x in data) == 60
        assert sum("downloads" not in x for x in data) == 25
        assert sum(data.pluck("downloads", 0)) == 541335745

    def test_full_join_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        other = ListOfDicts([{"b": 2, "c": 1}, {"c": 2}])
        data = orig.full_join(other, "b")
        assert data.pluck("c") == [1, None, None, 2]

    def test_group_by_next_operation_only(self):
        data = ListOfDicts(dict(g=g, x=x) for g, x in zip([2, 1, 2, 1, 2], [1, 2, 3, 4, 5]))
        assert data.group_by("g").sort(x=-1).head(1).pluck("x") == [5]
        assert data.group_by("g").deepcopy().head(1).pluck("x") == [1]

    def test_head(self):
        data = test.list_of_dicts("downloads.json")
        assert data.head(10) == data[:10]

    def test_head_group_wise(self):
        data = ListOfDicts(dict(g=g, x=x) for g, x in zip([2, 1, 2, 1, 2], [1, 2, 3, 4, 5]))
        data = data.group_by("g").head(2)
        assert data.pluck("g") == [1, 1, 2, 2]
        assert data.pluck("x") == [2, 4, 1, 3]

    def test_head_group_wise_ungroups(self):
        data = ListOfDicts(dict(g=g, x=x) for g, x in zip([2, 1, 2, 1, 2], [1, 2, 3, 4, 5]))
        assert data.group_by("g").head(2).head(1).pluck("x") == [2]

    def test_head_negative(self):
        data = ListOfDicts(dict(g=g, x=x) for g, x in zip([2, 1, 2, 1, 2], [1, 2, 3, 4, 5]))
        assert data.head(-2).pluck("x") == [1, 2, 3]
        assert data.head(-9).pluck("x") == []
        assert data.group_by("g").head(-1).pluck("x") == [2, 1, 3]

    def test_inner_join(self):
        orig = test.list_of_dicts("downloads.json")
        holidays = test.list_of_dicts("holidays.json")
        data = orig.inner_join(holidays, "date")
        assert len(data) == 35
        assert all("holiday" in x for x in data)
        assert sum(data.pluck("downloads")) == 18226489

    def test_inner_join_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        other = ListOfDicts([{"b": 2, "c": 1}, {"c": 2}])
        data = orig.inner_join(other, "b")
        assert data.pluck("a") == [1]

    def test_insert(self):
        orig = test.list_of_dicts("downloads.json")
        item = dict(date="3000-01-01")
        data = orig.insert(100, item)
        assert len(data) == len(orig) + 1
        assert isinstance(data[100], AttributeDict)
        assert data[100] == item

    def test_keys(self):
        data = test.list_of_dicts("listings.json")
        assert list(data.keys()) == ["id", "hood", "zipcode", "guests", "sqft", "price"]

    def test_left_join(self):
        orig = test.list_of_dicts("downloads.json")
        holidays = test.list_of_dicts("holidays.json")
        data = orig.left_join(holidays, "date")
        assert len(data) == 905
        assert sum("holiday" in x for x in data) == 35
        assert sum(data.pluck("downloads")) == 541335745

    def test_left_join_by_tuple(self):
        orig = test.list_of_dicts("downloads.json")
        holidays = test.list_of_dicts("holidays.json")
        holidays = holidays.rename(holiday_date="date")
        data = orig.left_join(holidays, ("date", "holiday_date"))
        assert len(data) == 905
        assert sum("holiday" in x for x in data) == 35
        assert sum("holiday_date" in x for x in data) == 0
        assert sum(data.pluck("downloads")) == 541335745

    def test_left_join_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        other = ListOfDicts([{"b": 2, "c": 1}, {"c": 2}])
        data = orig.left_join(other, "b")
        assert data.pluck("c") == [1, None, None]

    def test_join_common_keys(self):
        a = ListOfDicts([{"id": 1, "x": 1}, {"id": 2, "x": 2}])
        b = ListOfDicts([{"id": 1, "x": 3}, {"id": 3, "x": 4}])
        assert a.deepcopy().left_join(b, "id").pluck("x") == [1, 2]
        assert a.deepcopy().inner_join(b, "id").pluck("x") == [1]
        assert a.deepcopy().full_join(b, "id").pluck("x") == [1, 2, 4]

    def test_map(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.map(lambda x: {**x, "year": int(x.date[:4])})
        assert len(data) == len(orig)
        assert all("year" in x for x in data)
        assert sum(data.pluck("year")) == 1827565

    def test__mark_obsolete_after_multiple_modify(self):
        data = test.list_of_dicts("downloads.json")
        data = data.modify(a=lambda x: 1)
        data = data.modify(b=lambda x: 2)
        data = data.modify(c=lambda x: 3)

    def test__mark_obsolete_not_after_new_dicts(self):
        orig = test.list_of_dicts("downloads.json")
        orig.select("date").modify(a=lambda x: 1)
        orig.rename(ymd="date").modify(a=lambda x: 1)
        orig.group_by("category").aggregate(n=len).modify(a=lambda x: 1)
        assert not orig._obsolete

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

    def test_pluck_with_na(self):
        data = test.list_of_dicts("downloads.json")
        del data[0].downloads
        downloads = data.pluck("downloads", 0)
        assert len(downloads) == len(data)
        assert sum(downloads) == 541300639

    @patch("builtins.print")
    def test_print_memory_use(self, mock_print):
        data = test.list_of_dicts("vehicles.json")
        data.print_memory_use()
        mock_print.assert_called()

    @patch("builtins.print")
    def test_print_na_counts(self, mock_print):
        data = test.list_of_dicts("vehicles.json")
        data.print_na_counts()
        mock_print.assert_called()

    @patch("builtins.print")
    def test_print_na_counts_none(self, mock_print):
        data = test.list_of_dicts("vehicles.json")
        data = data.select("id", "make", "model")
        data.print_na_counts()
        mock_print.assert_called_once_with("No missing values")

    def test_read_csv(self):
        path = str(test.get_data_path("vehicles.csv"))
        data = ListOfDicts.read_csv(path)
        assert len(data) == 33442
        assert all(len(x) == 12 for x in data)

    def test_read_csv_keys(self):
        path = str(test.get_data_path("vehicles.csv"))
        data = ListOfDicts.read_csv(path, keys=["make", "model"])
        assert list(data[0].keys()) == ["make", "model"]
        assert list(data[100].keys()) == ["make", "model"]

    def test_read_csv_na(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text("a,b\n1,x\n\\N,\\N\n", "utf-8")
        data = ListOfDicts.read_csv(path, na="\\N", types={"a": int})
        assert data.pluck("a") == [1, None]
        assert data.pluck("b") == ["x", None]

    def test_read_csv_path(self):
        ListOfDicts.read_csv(test.get_data_path("vehicles.csv"))

    def test_read_csv_quote(self, tmp_path):
        path = tmp_path / "test.csv"
        path.write_text('a\tb\n"x\t1\n', "utf-8")
        data = ListOfDicts.read_csv(path, sep="\t", quote=None)
        assert data.pluck("a") == ['"x']
        assert data.pluck("b") == ["1"]

    def test_read_csv_types(self):
        path = str(test.get_data_path("vehicles.csv"))
        types = {"hwy": int, "cty": int}
        data = ListOfDicts.read_csv(path, types=types)
        assert isinstance(data[0].hwy, int)
        assert isinstance(data[0].cty, int)
        assert isinstance(data[100].hwy, int)
        assert isinstance(data[100].cty, int)

    def test_read_json(self):
        path = str(test.get_data_path("downloads.json"))
        data = ListOfDicts.read_json(path)
        assert len(data) == 905
        assert all(len(x) == 3 for x in data)

    def test_read_json_keys(self):
        path = str(test.get_data_path("downloads.json"))
        data = ListOfDicts.read_json(path, keys=["date", "downloads"])
        assert list(data[0].keys()) == ["date", "downloads"]
        assert list(data[100].keys()) == ["date", "downloads"]

    def test_read_json_path(self):
        ListOfDicts.read_csv(test.get_data_path("vehicles.json"))

    def test_read_json_types(self):
        path = str(test.get_data_path("downloads.json"))
        # datetime.date.fromisoformat requires Python >= 3.7.
        date = lambda x: datetime.date(*map(int, x.split("-")))
        types = {"date": date, "downloads": int}
        data = ListOfDicts.read_json(path, types=types)
        assert isinstance(data[0].date, datetime.date)
        assert isinstance(data[0].downloads, int)
        assert isinstance(data[100].date, datetime.date)
        assert isinstance(data[100].downloads, int)

    def test_read_pickle(self, tmp_path):
        orig = test.list_of_dicts("downloads.json")
        path = str(tmp_path / "test.pkl")
        orig.write_pickle(path)
        data = ListOfDicts.read_pickle(path)
        assert data == orig

    def test_read_pickle_path(self, tmp_path):
        orig = test.list_of_dicts("downloads.json")
        path = tmp_path / "test.pkl"
        orig.write_pickle(path)
        ListOfDicts.read_pickle(path)

    def test_rename(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.rename(ymd="date")
        assert len(data) == len(orig)
        assert all("ymd" in x for x in data)
        assert all("date" not in x for x in data)
        assert not orig._obsolete

    def test_replace_na(self):
        orig = ListOfDicts([{"a": 1, "b": None}, {"b": 2}])
        data = orig.replace_na(a=0, b=-1)
        assert data.pluck("a") == [1, 0]
        assert data.pluck("b") == [-1, 2]
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
        assert not orig._obsolete

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

    def test_semi_join_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        other = ListOfDicts([{"b": 2, "c": 1}, {"c": 2}])
        data = orig.semi_join(other, "b")
        assert data.pluck("a") == [1]

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

    def test_sort_missing_key(self):
        orig = ListOfDicts([{"a": 1}, {"a": 2, "b": 1}])
        data = orig.sort(b=1)
        assert data.pluck("a") == [2, 1]

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

    def test_split(self):
        data = ListOfDicts(
            {"x": x, "y": y} for x, y in zip(
                [1, 2, 2, 3, 3, 3],
                [1, 1, 1, 1, 1, 2],
            )
        )
        rows = data.split("x", "y")
        assert rows == [[0], [1, 2], [3, 4], [5]]

    def test_split_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        assert orig.split("b") == [[0], [1, 2]]

    def test_tail(self):
        data = test.list_of_dicts("downloads.json")
        assert data.tail(10) == data[-10:]

    def test_tail_group_wise(self):
        data = ListOfDicts(dict(g=g, x=x) for g, x in zip([2, 1, 2, 1, 2], [1, 2, 3, 4, 5]))
        data = data.group_by("g").tail(2)
        assert data.pluck("g") == [1, 1, 2, 2]
        assert data.pluck("x") == [2, 4, 3, 5]

    def test_tail_group_wise_ungroups(self):
        data = ListOfDicts(dict(g=g, x=x) for g, x in zip([2, 1, 2, 1, 2], [1, 2, 3, 4, 5]))
        assert data.group_by("g").tail(2).tail(1).pluck("x") == [5]

    def test_tail_negative(self):
        data = ListOfDicts(dict(g=g, x=x) for g, x in zip([2, 1, 2, 1, 2], [1, 2, 3, 4, 5]))
        assert data.tail(0).pluck("x") == []
        assert data.tail(-2).pluck("x") == [3, 4, 5]
        assert data.tail(-9).pluck("x") == []
        assert data.group_by("g").tail(-1).pluck("x") == [4, 3, 5]

    def test_to_data_frame(self):
        orig = test.list_of_dicts("vehicles.csv")
        data = orig.to_data_frame()
        assert data.nrow == len(orig)
        assert data.ncol == len(orig[0])

    def test_to_data_frame_missing_keys(self):
        orig = ListOfDicts([{"a": 1}, {"a": 2, "b": 3}])
        data = orig.to_data_frame()
        assert data.colnames == ["a", "b"]
        assert data.b.is_na().tolist() == [True, False]

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

    def test_to_pandas_missing_keys(self):
        orig = ListOfDicts([{"a": 1}, {"a": 2, "b": 3}])
        data = orig.to_pandas()
        assert data.columns.tolist() == ["a", "b"]

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

    def test_unique_missing_key(self):
        orig = ListOfDicts([{"a": 1, "b": 2}, {"a": 2}, {"a": 3, "b": None}])
        data = orig.unique("b")
        assert data.pluck("a") == [1, 2]

    def test_unselect(self):
        orig = test.list_of_dicts("downloads.json")
        data = orig.unselect("date", "downloads")
        assert len(data) == len(orig)
        assert all(len(x) == 1 for x in data)
        assert all("date" not in x for x in data)
        assert all("downloads" not in x for x in data)
        assert orig._obsolete

    def test_write_csv(self, tmp_path):
        orig = test.list_of_dicts("vehicles.csv")
        path = str(tmp_path / "test.csv")
        orig.write_csv(path)
        data = ListOfDicts.read_csv(path)
        assert data == orig

    def test_write_csv_path(self, tmp_path):
        orig = test.list_of_dicts("vehicles.csv")
        path = tmp_path / "test.csv"
        orig.write_csv(path)

    def test_write_json(self, tmp_path):
        orig = test.list_of_dicts("downloads.json")
        path = str(tmp_path / "test.json")
        orig.write_json(path)
        data = ListOfDicts.read_json(path)
        assert data == orig

    def test_write_json_path(self, tmp_path):
        orig = test.list_of_dicts("downloads.json")
        path = tmp_path / "test.json"
        orig.write_json(path)

    def test_write_pickle(self, tmp_path):
        orig = test.list_of_dicts("downloads.json")
        path = str(tmp_path / "test.pkl")
        orig.write_pickle(path)
        data = ListOfDicts.read_pickle(path)
        assert data == orig

    def test_write_pickle_path(self, tmp_path):
        orig = test.list_of_dicts("downloads.json")
        path = tmp_path / "test.pkl"
        orig.write_pickle(path)
