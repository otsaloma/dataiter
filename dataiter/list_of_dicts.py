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
import dataiter
import itertools
import json
import operator
import os
import pickle
import random

from attd import AttributeDict
from dataiter import deco
from dataiter import util


class ListOfDicts(list):

    def __init__(self, dicts=(), group_keys=None, predecessor=None, as_is=False):
        super().__init__(dicts if as_is else map(AttributeDict, dicts))
        self._group_keys = tuple(group_keys or ())
        self._predecessor = predecessor

    @deco.new_from_generator
    def __add__(self, other):
        if not isinstance(other, ListOfDicts):
            raise TypeError("Not a ListOfDicts")
        yield from itertools.chain(self, other)

    def __copy__(self):
        return self._new(self)

    def __deepcopy__(self, memo=None):
        return self.__class__(map(copy.deepcopy, self),
                              group_keys=self._group_keys,
                              predecessor=None,
                              as_is=True)

    def __getitem__(self, index):
        # Needed so that slicing gives a ListOfDicts, not a list.
        value = super().__getitem__(index)
        return self._new(value) if isinstance(value, list) else value

    @deco.new_from_generator
    def __mul__(self, other):
        if not isinstance(other, int):
            raise TypeError("Multiplier not an integer")
        for i in range(other):
            yield from self

    def __repr__(self):
        return self.to_string()

    def __rmul__(self, other):
        return self.__mul__(other)

    def __setitem__(self, index, value):
        if not isinstance(value, AttributeDict):
            value = AttributeDict(value)
        return super().__setitem__(index, value)

    def __str__(self):
        return self.to_string()

    @deco.new_from_generator
    def aggregate(self, **key_function_pairs):
        by = self._group_keys
        groups = self.unique(*by).deepcopy().select(*by)
        extract = operator.itemgetter(*by)
        items_by_group = {}
        for item in self:
            id = extract(item)
            items_by_group.setdefault(id, []).append(item)
        key_function_pairs = key_function_pairs.items()
        for group in groups.sort(**dict.fromkeys(by, 1)):
            id = extract(group)
            items = ListOfDicts(items_by_group[id])
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
        if not isinstance(item, AttributeDict):
            item = AttributeDict(item)
        yield from itertools.chain(self, [item])

    def clear(self):
        return self._new([])

    def copy(self):
        return self.__copy__()

    def deepcopy(self):
        return self.__deepcopy__()

    @deco.new_from_generator
    def extend(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        yield from itertools.chain(self, other)

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

    def full_join(self, other, *by):
        counter = itertools.count(start=1)
        other = other.deepcopy().modify(_id_=lambda x: next(counter))
        # This obsoletes self, @deco.obsoletes not needed.
        a = self.left_join(other, *by)
        found_ids = set(x.get("_id_", -1) for x in a)
        b = other.filter_out(lambda x: x._id_ in found_ids)
        return (a + b).unselect("_id_")

    def group_by(self, *keys):
        self._group_keys = tuple(keys)
        return self

    def head(self, n=None):
        if n is None:
            n = dataiter.DEFAULT_PEEK_ITEMS
        n = min(len(self), n)
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

    @deco.new_from_generator
    def insert(self, index, item):
        if not isinstance(item, AttributeDict):
            item = AttributeDict(item)
        for i in range(len(self)):
            if i == index:
                yield item
            yield self[i]

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
                              group_keys=self._group_keys,
                              predecessor=self,
                              as_is=True)

    def pluck(self, key, default=None):
        return [x.get(key, default) for x in self]

    def print_(self, max_items=None):
        print(self.to_string(max_items))

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

    @classmethod
    def read_pickle(cls, fname):
        with open(fname, "rb") as f:
            return cls(pickle.load(f))

    @deco.obsoletes
    @deco.new_from_generator
    def rename(self, **to_from_pairs):
        to_from_pairs = to_from_pairs.items()
        for item in self:
            for to, fm in to_from_pairs:
                item[to] = item.pop(fm)
            yield item

    @deco.new_from_generator
    def reverse(self):
        yield from reversed(self)

    @deco.new_from_generator
    def sample(self, n=None):
        if n is None:
            n = dataiter.DEFAULT_PEEK_ITEMS
        n = min(len(self), n)
        for i in sorted(random.sample(range(len(self)), n)):
            yield self[i]

    @deco.obsoletes
    @deco.new_from_generator
    def select(self, *keys):
        for item in self:
            yield AttributeDict({x: item[x] for x in keys if x in item})

    @deco.new_from_generator
    def semi_join(self, other, *by):
        extract = operator.itemgetter(*by)
        other_ids = set(map(extract, other))
        for item in self:
            if extract(item) in other_ids:
                yield item

    def sort(self, **key_dir_pairs):
        key_dir_pairs = key_dir_pairs.items()
        for key, dir in key_dir_pairs:
            if dir not in [1, -1]:
                raise ValueError("dir should be 1 or -1")
        def flip(value, dir):
            # XXX: This only supports numeric types of value.
            if dir < 0 and isinstance(value, (bool, int, float)):
                return -value
            return value
        @deco.tuplefy
        def sort_key(item):
            for key, dir in key_dir_pairs:
                yield (item[key] is None, flip(item[key], dir))
        return self._new(sorted(self, key=sort_key))

    def tail(self, n=None):
        if n is None:
            n = dataiter.DEFAULT_PEEK_ITEMS
        n = min(len(self), n)
        return self._new(self[-n:])

    def _to_columns(self):
        return {k: self.pluck(k) for k in self[0]} if self else {}

    def to_data_frame(self):
        from dataiter import DataFrame
        return DataFrame(**self._to_columns())

    def to_json(self, **kwargs):
        kwargs.setdefault("default", str)
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self, **kwargs)

    def to_pandas(self):
        import pandas as pd
        return pd.DataFrame(self._to_columns())

    def to_string(self, max_items=None):
        if max_items is None:
            max_items = dataiter.PRINT_MAX_ITEMS
        string = self.head(max_items).to_json()
        if max_items < len(self):
            string += f" ... {len(self)} items total"
        return string

    @deco.new_from_generator
    def unique(self, *keys):
        if not self: return
        if not keys:
            # If keys not given, use all common keys.
            keys = set(self[0])
            for item in self:
                keys &= set(item)
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
            raise ValueError("Cannot write empty CSV file")
        # Take a superset of all keys and fill in missing as None.
        keys = util.unique_keys(list(itertools.chain(*self)))
        data = [{**dict.fromkeys(keys), **x} for x in self]
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w", encoding=encoding) as f:
            writer = csv.DictWriter(f, keys, dialect="unix", delimiter=sep)
            writer.writeheader() if header else None
            for item in data:
                writer.writerow(item)

    def write_json(self, fname, encoding="utf_8", **kwargs):
        kwargs.setdefault("default", str)
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w", encoding=encoding) as f:
            encoder = json.JSONEncoder(**kwargs)
            for chunk in encoder.iterencode(self):
                f.write(chunk)
            f.write("\n")

    def write_pickle(self, fname):
        with open(fname, "wb") as f:
            out = [dict(x) for x in self]
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)


class ObsoleteError(Exception):

    pass


class ObsoleteListOfDicts(list):

    def __getattr__(self, name):
        raise ObsoleteError("Cannot act on a ListOfDicts object whose successor has modified the shared dicts")
