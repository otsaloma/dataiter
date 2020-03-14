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
import csv
import functools
import json
import numpy as np
import operator
import os

from attd import AttributeDict

__version__ = "0.7"

DATA_FRAME_PRINT_MAX_ROWS = 10


def _modifies_dicts(function):
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        value = function(self, *args, **kwargs)
        self._mark_obsolete()
        return value
    return wrapper

def _new_from_generator(function):
    @functools.wraps(function)
    def wrapper(self, *args, **kwargs):
        return self._new(function(self, *args, **kwargs))
    return wrapper

def _translate_error(fm, to):
    def outer_wrapper(function):
        @functools.wraps(function)
        def inner_wrapper(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except fm as error:
                raise to(str(error))
        return inner_wrapper
    return outer_wrapper


class DataFrame(dict):

    """A table (dictionary of :class:`DataFrameColumn`s)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if not isinstance(value, DataFrameColumn):
                super().__setitem__(key, DataFrameColumn(value))

    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo=None):
        return self.__class__({k: v.copy() for k, v in self.items()})

    @_translate_error(KeyError, AttributeError)
    def __delattr__(self, name):
        return self.__delitem__(name)

    @_translate_error(KeyError, AttributeError)
    def __getattr__(self, name):
        return self.__getitem__(name)

    def __setattr__(self, name, value):
        return self.__setitem__(name, value)

    def __setitem__(self, key, value):
        nrow = self.nrow if self else None
        value = DataFrameColumn(value, nrow=nrow)
        return super().__setitem__(key, value)

    def __str__(self):
        # TODO: Wrap columns like R's print.data.frame.
        # TODO: Use better type-specific formatting.
        rows = []
        rows.append(self.colnames)
        rows.append([str(x.dtype) for x in self.columns])
        for i in range(min(self.nrow, DATA_FRAME_PRINT_MAX_ROWS)):
            rows.append([str(x[i]) for x in self.columns])
        for i in range(len(rows[0])):
            width = max(len(x[i]) for x in rows)
            for row in rows:
                padding = width - len(row[i])
                row[i] = " " * padding + row[i]
        rows = [" ".join(x) for x in rows]
        return "\n" + "\n".join(rows)

    def aggregate(self, **name_function_pairs):
        raise NotImplementedError

    @property
    def colnames(self):
        return list(self)

    @property
    def columns(self):
        return list(self.values())

    def copy(self):
        return self.__copy__()

    def deepcopy(self):
        return self.__deepcopy__()

    def filter(self, function=None, **name_value_pairs):
        raise NotImplementedError

    def filter_out(self, function=None, **name_value_pairs):
        raise NotImplementedError

    @classmethod
    def from_json(cls, string, **kwargs):
        obj = json.loads(string, **kwargs)
        if not isinstance(obj, list):
            raise TypeError("Not a list")
        columns = {}
        for item in obj:
            for key, value in item.items():
                columns.setdefault(key, []).append(value)
        return cls(**columns)

    def group_by(self, *names):
        raise NotImplementedError

    def join(self, other, *by):
        raise NotImplementedError

    @property
    def ncol(self):
        return len(self)

    @property
    def nrow(self):
        if not self: return 0
        return self[next(iter(self))].nrow

    @classmethod
    def read_csv(cls, fname, encoding="utf_8", **kwargs):
        raise NotImplementedError

    @classmethod
    def read_json(cls, fname, encoding="utf_8"):
        with open(fname, "r", encoding=encoding) as f:
            return cls.from_json(f.read())

    def rename(self, **to_from_pairs):
        raise NotImplementedError

    def select(self, *names):
        raise NotImplementedError

    def sort(self, *names, reverse=False):
        raise NotImplementedError

    def to_json(self, **kwargs):
        raise NotImplementedError

    def unique(self, *names):
        raise NotImplementedError

    def unselect(self, *names):
        raise NotImplementedError

    def write_csv(self, fname, encoding="utf_8", **kwargs):
        raise NotImplementedError

    def write_json(self, fname, encoding="utf_8", **kwargs):
        raise NotImplementedError


class DataFrameColumn(np.ndarray):

    """A vector (one-dimensional Numpy array)."""

    def __new__(cls, object, dtype=None, nrow=None):
        column = np.array(object, dtype)
        if nrow is not None and nrow != column.size:
            if not (column.size == 1 and nrow > 1):
                raise ValueError("Incompatible object and nrow for broadcast")
            column = column.repeat(nrow)
        return column.view(cls)

    def __init__(self, object, dtype=None, nrow=None):
        self.__check_dimensions()

    def __check_dimensions(self):
        if self.ndim == 1: return
        raise ValueError("Bad dimensions: {!r}".format(self.ndim))

    @property
    def nrow(self):
        self.__check_dimensions()
        return self.size


class ListOfDicts(list):

    """A list of dictionaries."""

    def __init__(self, dicts, group_keys=None, shallow=False, predecessor=None):
        super().__init__(dicts if shallow else (AttributeDict(x) for x in dicts))
        self._group_keys = group_keys or []
        self._predecessor = predecessor

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

    def copy(self):
        return self.__copy__()

    def deepcopy(self):
        return self.__deepcopy__()

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

    @_modifies_dicts
    @_new_from_generator
    def join(self, other, *by):
        extract = operator.itemgetter(*by)
        other = {extract(x): x for x in reversed(other)}
        for item in self:
            item.update(other.get(extract(item), {}))
            yield item

    def _mark_obsolete(self):
        if isinstance(self._predecessor, ListOfDicts):
            self._predecessor._mark_obsolete()
        self.__class__ = ObsoleteListOfDicts

    @_modifies_dicts
    @_new_from_generator
    def modify(self, **key_function_pairs):
        key_function_pairs = key_function_pairs.items()
        for item in self:
            for key, function in key_function_pairs:
                item[key] = function(item)
            yield item

    @_modifies_dicts
    @_new_from_generator
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
    def read_csv(cls, fname, encoding="utf_8", **kwargs):
        kwargs.setdefault("dialect", "unix")
        with open(fname, "r", encoding=encoding) as f:
            return cls(csv.DictReader(f, **kwargs))

    @classmethod
    def read_json(cls, fname, encoding="utf_8"):
        with open(fname, "r", encoding=encoding) as f:
            return cls.from_json(f.read())

    @_modifies_dicts
    @_new_from_generator
    def rename(self, **to_from_pairs):
        to_from_pairs = to_from_pairs.items()
        for item in self:
            for to, fm in to_from_pairs:
                item[to] = item.pop(fm)
            yield item

    @_modifies_dicts
    @_new_from_generator
    def select(self, *keys):
        keys = set(keys)
        for item in self:
            for key in set(item) - keys:
                del item[key]
            yield item

    def sort(self, *keys, reverse=False):
        def sort_key(item):
            return tuple((item[x] is None, item[x]) for x in keys)
        return self._new(sorted(self, key=sort_key, reverse=reverse))

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

    @_modifies_dicts
    @_new_from_generator
    def unselect(self, *keys):
        for item in self:
            for key in keys:
                if key in item:
                    del item[key]
            yield item

    def write_csv(self, fname, encoding="utf_8", **kwargs):
        if not self:
            raise Exception("Cannot write empty CSV file")
        kwargs.setdefault("dialect", "unix")
        keys = list(self[0].keys())
        for item in self:
            if set(item.keys()) != set(keys):
                raise Exception("Keys differ between dicts")
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        with open(fname, "w", encoding=encoding) as f:
            writer = csv.DictWriter(f, keys, **kwargs)
            writer.writeheader()
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
