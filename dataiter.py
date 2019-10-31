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

"""Classes for data manipulation."""

import copy
import functools
import json
import operator

from attd import AttributeDict

__version__ = "0.2"


def _new_from_generator(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        return args[0]._new(function(*args, **kwargs))
    return wrapper


class ListOfDicts(list):

    """Data as a list of dictionaries."""

    def __init__(self, dicts, group_keys=None, shallow=False):
        super().__init__(dicts if shallow else (AttributeDict(x) for x in dicts))
        self._group_keys = group_keys or []

    def __getitem__(self, key):
        # Needed so that slicing gives a ListOfDicts, not a list.
        value = super().__getitem__(key)
        return self._new(value) if isinstance(value, list) else value

    @_new_from_generator
    def aggregate(self, **key_function_pairs):
        by = self._group_keys
        groups = self.unique(*by).deepcopy().select(*by)
        key_function_pairs = key_function_pairs.items()
        for group in groups.sort(*by):
            items = self.filter(**group)
            for key, function in key_function_pairs:
                group[key] = function(items)
            yield group

    def deepcopy(self):
        return self._new(copy.deepcopy(self))

    @_new_from_generator
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

    @_new_from_generator
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

    def group_by(self, *keys):
        self._group_keys = keys[:]
        return self

    @_new_from_generator
    def join(self, other, *by):
        extract = operator.itemgetter(*by)
        other = {extract(x): x for x in reversed(other)}
        for item in self:
            item.update(other.get(extract(item), {}))
            yield item

    @_new_from_generator
    def modify(self, **key_function_pairs):
        key_function_pairs = key_function_pairs.items()
        for item in self:
            for key, function in key_function_pairs:
                item[key] = function(item)
            yield item

    def _new(self, dicts):
        return self.__class__(dicts, self._group_keys[:], shallow=True)

    def pluck(self, key):
        return [x[key] for x in self]

    @_new_from_generator
    def rename(self, **to_from_pairs):
        to_from_pairs = to_from_pairs.items()
        for item in self:
            for to, fm in to_from_pairs:
                item[to] = item.pop(fm)
            yield item

    @_new_from_generator
    def select(self, *keys):
        keys = set(keys)
        for item in self:
            for key in set(item) - keys:
                del item[key]
            yield item

    def sort(self, *keys, reverse=False):
        extract = operator.itemgetter(*keys)
        return self._new(sorted(self, key=extract, reverse=reverse))

    def to_json(self, **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self, **kwargs)

    @_new_from_generator
    def unique(self, *keys):
        found_ids = set()
        extract = operator.itemgetter(*keys)
        for item in self:
            id = extract(item)
            if id not in found_ids:
                found_ids.add(id)
                yield item

    @_new_from_generator
    def unselect(self, *keys):
        for item in self:
            for key in keys:
                if key in item:
                    del item[key]
            yield item
