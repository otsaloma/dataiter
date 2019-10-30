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

from attd import AttributeDict

__version__ = "0.1"


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
        groups = [{k: x[k] for k in self._group_keys} for x in self]
        groups = self.__class__(groups).unique(*self._group_keys)
        for group in groups.sort(*self._group_keys):
            items = self.filter(**group)
            for key, function in key_function_pairs.items():
                group[key] = function(items)
            yield group

    def deepcopy(self):
        return self._new(copy.deepcopy(self))

    @_new_from_generator
    def filter(self, function=None, **key_value_pairs):
        if function is None and key_value_pairs:
            function = lambda x: all(
                x[k] == v for k, v in key_value_pairs.items())
        for item in self:
            if function(item):
                yield item

    @_new_from_generator
    def filter_out(self, function=None, **key_value_pairs):
        if function is None and key_value_pairs:
            function = lambda x: all(
                x[k] == v for k, v in key_value_pairs.items())
        for item in self:
            if not function(item):
                yield item

    @classmethod
    def from_json(cls, string, **kwargs):
        obj = json.loads(string, **kwargs)
        if not isinstance(obj, list):
            raise TypeError("Not a list")
        return cls(obj)

    def group_by(self, *keys):
        return self.__class__(self, keys[:])

    @_new_from_generator
    def join(self, other, *by):
        other = {tuple(x[k] for k in by): x for x in other}
        for item in self:
            item = item.copy()
            id = tuple(item[k] for k in by)
            item.update(other.get(id, {}))
            yield item

    @_new_from_generator
    def modify(self, **key_function_pairs):
        for item in self:
            item = item.copy()
            for key, function in key_function_pairs.items():
                item[key] = function(item)
            yield item

    def _new(self, dicts):
        return self.__class__(dicts, self._group_keys[:], shallow=True)

    def pluck(self, key):
        return [x[key] for x in self]

    @_new_from_generator
    def rename(self, **to_from_pairs):
        for item in self:
            item = item.copy()
            for to, fm in to_from_pairs.items():
                item[to] = item.pop(fm)
            yield item

    @_new_from_generator
    def select(self, *keys):
        for item in self:
            item = item.copy()
            for key in set(item) - set(keys):
                del item[key]
            yield item

    def sort(self, *keys, reverse=False):
        function = lambda x: tuple(x[k] for k in keys)
        return self._new(sorted(self, key=function, reverse=reverse))

    def to_json(self, **kwargs):
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self, **kwargs)

    @_new_from_generator
    def unique(self, *keys):
        found_ids = set()
        for item in self:
            id = tuple(item[k] for k in keys)
            if id not in found_ids:
                found_ids.add(id)
                yield item

    @_new_from_generator
    def unselect(self, *keys):
        for item in self:
            item = item.copy()
            for key in set(item) & set(keys):
                del item[key]
            yield item
