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

import copy
import csv
import itertools
import json
import operator
import os
import pandas as pd

from attd import AttributeDict
from dataiter import deco
from dataiter import util


class ListOfDicts(list):

    def __init__(self, dicts, group_keys=None, shallow=False, predecessor=None):
        super().__init__(dicts if shallow else (AttributeDict(x) for x in dicts))
        self._group_keys = group_keys or []
        self._predecessor = predecessor

    @deco.new_from_generator
    def __add__(self, other):
        assert isinstance(other, self.__class__)
        yield from itertools.chain(self, other)

    def __copy__(self):
        return self.__class__(self,
                              group_keys=self._group_keys[:],
                              shallow=True,
                              predecessor=self)

    def __deepcopy__(self, memo=None):
        return self.__class__(map(copy.deepcopy, self),
                              group_keys=self._group_keys[:],
                              shallow=True,
                              predecessor=None)

    def __getitem__(self, index):
        # Needed so that slicing gives a ListOfDicts, not a list.
        value = super().__getitem__(index)
        return self._new(value) if isinstance(value, list) else value

    def __setitem__(self, index, value):
        return super().__setitem__(index, AttributeDict(value))

    @deco.new_from_generator
    def aggregate(self, **key_function_pairs):
        by = self._group_keys
        groups = self.unique(*by).deepcopy().select(*by)
        key_function_pairs = key_function_pairs.items()
        for group in groups.sort(*by):
            items = self.filter(**group)
            for key, function in key_function_pairs:
                group[key] = function(items)
            yield group

    @deco.new_from_generator
    def anti_join(self, other, *by):
        extract = operator.itemgetter(*by)
        other_ids = set(map(extract, other))
        for item in self:
            if extract(item) not in other_ids:
                yield item

    @deco.new_from_generator
    def append(self, item):
        yield from itertools.chain(self, [AttributeDict(item)])

    def copy(self):
        return self.__copy__()

    def deepcopy(self):
        return self.__deepcopy__()

    @deco.new_from_generator
    def filter(self, function=None, **key_value_pairs):
        if callable(function):
            for item in self:
                if function(item):
                    yield item
        elif key_value_pairs:
            extract = operator.itemgetter(*key_value_pairs.keys())
            values = tuple(key_value_pairs.values())
            values = values[0] if len(values) == 1 else values
            for item in self:
                if extract(item) == values:
                    yield item

    @deco.new_from_generator
    def filter_out(self, function=None, **key_value_pairs):
        if callable(function):
            for item in self:
                if not function(item):
                    yield item
        elif key_value_pairs:
            extract = operator.itemgetter(*key_value_pairs.keys())
            values = tuple(key_value_pairs.values())
            values = values[0] if len(values) == 1 else values
            for item in self:
                if extract(item) != values:
                    yield item

    @classmethod
    def from_json(cls, string, **kwargs):
        obj = json.loads(string, **kwargs)
        if not isinstance(obj, list):
            raise TypeError("Not a list")
        return cls(obj)

    @deco.obsoletes
    @deco.new_from_generator
    def full_join(self, other, *by):
        counter = itertools.count(start=1)
        x = self.deepcopy().modify(_xid=lambda x: next(counter))
        y = other.deepcopy().modify(_yid=lambda x: next(counter))
        return (self.__class__(x.left_join(y, *by) + y.left_join(x, *by))
                .modify(_xid=lambda x: x.get("_xid", -1))
                .modify(_yid=lambda x: x.get("_yid", -1))
                .unique("_xid", "_yid")
                .unselect("_xid", "_yid"))

    def group_by(self, *keys):
        self._group_keys = keys[:]
        return self

    def head(self, n):
        return self._new(self[:n])

    @deco.obsoletes
    @deco.new_from_generator
    def inner_join(self, other, *by):
        extract = operator.itemgetter(*by)
        other_by_id = {extract(x): x for x in reversed(other)}
        for item in self:
            id = extract(item)
            if id in other_by_id:
                item.update(other_by_id[id])
                yield item

    @deco.obsoletes
    @deco.new_from_generator
    def left_join(self, other, *by):
        extract = operator.itemgetter(*by)
        other_by_id = {extract(x): x for x in reversed(other)}
        for item in self:
            item.update(other_by_id.get(extract(item), {}))
            yield item

    def _mark_obsolete(self):
        if isinstance(self._predecessor, ListOfDicts):
            self._predecessor._mark_obsolete()
        self.__class__ = ObsoleteListOfDicts

    @deco.obsoletes
    @deco.new_from_generator
    def modify(self, **key_function_pairs):
        key_function_pairs = key_function_pairs.items()
        for item in self:
            for key, function in key_function_pairs:
                item[key] = function(item)
            yield item

    @deco.obsoletes
    @deco.new_from_generator
    def modify_if(self, predicate, **key_function_pairs):
        key_function_pairs = key_function_pairs.items()
        for item in self:
            if predicate(item):
                for key, function in key_function_pairs:
                    item[key] = function(item)
            yield item

    def _new(self, dicts):
        return self.__class__(dicts,
                              group_keys=self._group_keys[:],
                              shallow=True,
                              predecessor=self)

    def pluck(self, key):
        return [x[key] for x in self]

    @classmethod
    def read_csv(cls, fname, encoding="utf_8", header=True, sep=","):
        with open(fname, "r", encoding=encoding) as f:
            rows = list(csv.reader(f, dialect="unix", delimiter=sep))
            if not rows: return cls([])
            keys = rows.pop(0) if header else util.generate_colnames(len(rows[0]))
            return cls(dict(zip(keys, x)) for x in rows)

    @classmethod
    def read_json(cls, fname, encoding="utf_8", **kwargs):
        with open(fname, "r", encoding=encoding) as f:
            return cls.from_json(f.read(), **kwargs)

    @deco.obsoletes
    @deco.new_from_generator
    def rename(self, **to_from_pairs):
        to_from_pairs = to_from_pairs.items()
        for item in self:
            for to, fm in to_from_pairs:
                item[to] = item.pop(fm)
            yield item

    @deco.obsoletes
    @deco.new_from_generator
    def select(self, *keys):
        keys = set(keys)
        for item in self:
            for key in set(item) - keys:
                del item[key]
            yield item

    @deco.new_from_generator
    def semi_join(self, other, *by):
        extract = operator.itemgetter(*by)
        other_ids = set(map(extract, other))
        for item in self:
            if extract(item) in other_ids:
                yield item

    def sort(self, *keys, reverse=False):
        def sort_key(item):
            return tuple((item[x] is None, item[x]) for x in keys)
        return self._new(sorted(self, key=sort_key, reverse=reverse))

    def tail(self, n):
        return self._new(self[-n:])

    def _to_columns(self):
        return {k: self.pluck(k) for k in self[0]} if self else {}

    def to_data_frame(self):
        from dataiter import DataFrame
        return DataFrame(**self._to_columns())

    def to_json(self, **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self, **kwargs)

    def to_pandas(self):
        return pd.DataFrame(self._to_columns())

    @deco.new_from_generator
    def unique(self, *keys):
        found_ids = set()
        extract = operator.itemgetter(*keys)
        for item in self:
            id = extract(item)
            if id not in found_ids:
                found_ids.add(id)
                yield item

    @deco.obsoletes
    @deco.new_from_generator
    def unselect(self, *keys):
        for item in self:
            for key in keys:
                if key in item:
                    del item[key]
            yield item

    def write_csv(self, fname, encoding="utf_8", header=True, sep=","):
        if not self:
            raise Exception("Cannot write empty CSV file")
        keys = list(self[0].keys())
        for item in self:
            if set(item.keys()) != set(keys):
                raise Exception("Keys differ between dicts")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w", encoding=encoding) as f:
            writer = csv.DictWriter(f, keys, dialect="unix", delimiter=sep)
            writer.writeheader() if header else None
            for item in self:
                writer.writerow(item)

    def write_json(self, fname, encoding="utf_8", **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w", encoding=encoding) as f:
            encoder = json.JSONEncoder(**kwargs)
            for chunk in encoder.iterencode(self):
                f.write(chunk)
            f.write("\n")


class ObsoleteError(Exception):

    pass


class ObsoleteListOfDicts(list):

    def __getattr__(self, name):
        raise ObsoleteError("Cannot act on a ListOfDicts object whose successor has modified the shared dicts")
