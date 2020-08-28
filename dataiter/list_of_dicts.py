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
import pickle
import random

from attd import AttributeDict
from dataiter import deco
from dataiter import util


class ListOfDicts(list):

    """
    A class for data as a list of dicts.

    Most of the data-modifying methods return **shallow** copies, that is a new
    list of dicts that contains the same dict objects. To avoid surprises with
    modifying the same dicts in different objects, list of dicts marks the
    previous object "obsolete" upon returning a modified copy. Any attempted
    operations on the obsolete object will print a warning once per object.
    Usually, if you see this warning, you'll want to call :meth:`deepcopy`
    to create a new, completely independent object.

    Contained dicts are upon initialization converted to
    ``attd.AttributeDict``, which is a simple subclass of ``dict`` that
    provides attribute access to dict keys. This means that you can access keys
    as e.g. ``data[0].x`` in addition to ``data[0]["x"]``. In most cases,
    attribute access should be more convenient and is the way recommended by
    dataiter. You'll still need to use the bracket notation for any keys that
    are not valid identifiers, such as keys with spaces, or ones that conflict
    with dict methods, such as "items".

    https://github.com/otsaloma/attd
    """

    def __init__(self, dicts=(), as_is=False):
        """
        Return a new list of dicts.

        `dicts` is the data to hold, any kind of a sequence of dicts.

        `as_is` can be set to ``True`` to not convert the dicts to
        ``attd.AttributeDict``. This conversion can be skipped for a small
        speed gain if you know that `dicts` are already attribute dicts. Note
        that regular dicts will not work, the conversion needs to be done at
        some point.
        """
        super().__init__(dicts if as_is else map(AttributeDict, dicts))
        self._group_keys = ()
        self._obsolete = False
        self._obsolete_warned = False
        self._predecessor = None

    @deco.new_from_generator
    def __add__(self, other):
        if not isinstance(other, ListOfDicts):
            raise TypeError("Not a ListOfDicts")
        yield from itertools.chain(self, other)

    def __copy__(self):
        return self._new(self)

    def __deepcopy__(self, memo=None):
        new = self.__class__(map(copy.deepcopy, self), as_is=True)
        new._group_keys = self._group_keys
        return new

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if ("obsolete" not in name and
            callable(value) and
            self._obsolete and
            not self._obsolete_warned):
            print("Warning: A successor has modified the shared dicts")
            self._obsolete_warned = True
        return value

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
        """
        Return group-wise calculated summaries.

        Usually aggregation is preceded by grouping, which can be conveniently
        written via method chaining as ``data.group_by(...).aggregate(...)``.

        In `key_function_pairs`, `function` receives as an argument a list of
        dicts object, a group-wise subset of all items. It can return any kind
        of value, it will end up as-is in the output.

        >>> from statistics import mean
        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.group_by("hood").aggregate(n=len, price=lambda x: mean(x.pluck("price")))
        """
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
        """
        Return items with no matches in `other`.

        `by` are keys, by which to look for matching items.

        >>> # All listings that don't have reviews
        >>> listings = di.ListOfDicts.read_json("data/listings.json")
        >>> reviews = di.ListOfDicts.read_json("data/listings-reviews.json")
        >>> listings.anti_join(reviews, "id")
        """
        extract = operator.itemgetter(*by)
        other_ids = set(map(extract, other))
        for item in self:
            if extract(item) not in other_ids:
                yield item

    @deco.new_from_generator
    def append(self, item):
        """
        Return list with `item` added to the end.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data = data.append(dict.fromkeys(data[0].keys()))
        >>> data.tail()
        """
        if not isinstance(item, AttributeDict):
            item = AttributeDict(item)
        yield from itertools.chain(self, [item])

    def clear(self):
        """
        Return list with all items removed.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.clear()
        """
        return self._new([])

    def copy(self):
        """
        Return a shallow copy.
        """
        return self.__copy__()

    def deepcopy(self):
        """
        Return a deep copy.
        """
        return self.__deepcopy__()

    @deco.new_from_generator
    def extend(self, other):
        """
        Return list with items from `other` added to the end.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data = data.extend([dict.fromkeys(data[0].keys())])
        >>> data.tail()
        """
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        yield from itertools.chain(self, other)

    @deco.new_from_generator
    def filter(self, function=None, **key_value_pairs):
        """
        Return items that match condition.

        Filtering can be done either by `function`, which receives an
        individual item as its argument and returns ``True`` or ``False``, or
        by `key_value_pairs`, which are a shorthand to check against a fixed
        value. See the example below of equivalent filtering with both ways.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.filter(lambda x: x.hood == "Manhattan" and x.guests == 2)
        >>> data.filter(hood="Manhattan", guests=2)
        """
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
        """
        Return items that don't match condition.

        Filtering can be done either by `function`, which receives an
        individual item as its argument and returns ``True`` or ``False``, or
        by `key_value_pairs`, which are a shorthand to check against a fixed
        value. See the example below of equivalent filtering with both ways.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.filter_out(lambda x: x.hood == "Manhattan")
        >>> data.filter_out(hood="Manhattan")
        """
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
        """
        Return a new list of dicts from JSON `string`.
        """
        obj = json.loads(string, **kwargs)
        if not isinstance(obj, list):
            raise TypeError("Not a list")
        return cls(obj)

    def full_join(self, other, *by):
        """
        Return list with matching items merged from `self` and `other`.

        `full_join` keeps all items from both lists, merging matching ones. If
        there are multiple matches, the first one will be used. For items, for
        which matches are not found, no keys are added.

        `by` are keys, by which to look for matching items.

        >>> listings = di.ListOfDicts.read_json("data/listings.json")
        >>> reviews = di.ListOfDicts.read_json("data/listings-reviews.json")
        >>> listings.full_join(reviews, "id")
        """
        counter = itertools.count(start=1)
        other = other.deepcopy().modify(_id_=lambda x: next(counter))
        # This obsoletes self, @deco.obsoletes not needed.
        a = self.left_join(other, *by)
        found_ids = set(x.get("_id_", -1) for x in a)
        b = other.filter_out(lambda x: x._id_ in found_ids)
        return (a + b).unselect("_id_")

    def group_by(self, *keys):
        """
        Return list with `keys` set for grouped operations, such as :meth:`aggregate`.
        """
        self._group_keys = tuple(keys)
        return self

    def head(self, n=None):
        """
        Return the first `n` items.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.head(3)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ITEMS
        n = min(len(self), n)
        return self._new(self[:n])

    @deco.obsoletes
    @deco.new_from_generator
    def inner_join(self, other, *by):
        """
        Return list with matching items merged from `self` and `other`.

        `inner_join` keeps only items found in both lists, merging matching
        ones. If there are multiple matches, the first one will be used.

        `by` are keys, by which to look for matching items.

        >>> listings = di.ListOfDicts.read_json("data/listings.json")
        >>> reviews = di.ListOfDicts.read_json("data/listings-reviews.json")
        >>> listings.inner_join(reviews, "id")
        """
        extract = operator.itemgetter(*by)
        other_by_id = {extract(x): x for x in reversed(other)}
        for item in self:
            id = extract(item)
            if id in other_by_id:
                item.update(other_by_id[id])
                yield item

    @deco.new_from_generator
    def insert(self, index, item):
        """
        Return list with `item` inserted at `index`.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data = data.insert(0, dict.fromkeys(data[0].keys()))
        >>> data.head()
        """
        if not isinstance(item, AttributeDict):
            item = AttributeDict(item)
        for i in range(len(self)):
            if i == index:
                yield item
            yield self[i]

    @deco.obsoletes
    @deco.new_from_generator
    def left_join(self, other, *by):
        """
        Return list with matching items merged from `self` and `other`.

        `left_join` keeps all items in `self`, merging matching ones from
        `other`. If there are multiple matches, the first one will be used. For
        items, for which matches are not found, no keys are added.

        `by` are keys, by which to look for matching items.

        >>> listings = di.ListOfDicts.read_json("data/listings.json")
        >>> reviews = di.ListOfDicts.read_json("data/listings-reviews.json")
        >>> listings.left_join(reviews, "id")
        """
        extract = operator.itemgetter(*by)
        other_by_id = {extract(x): x for x in reversed(other)}
        for item in self:
            item.update(other_by_id.get(extract(item), {}))
            yield item

    def _mark_obsolete(self):
        if isinstance(self._predecessor, ListOfDicts):
            self._predecessor._mark_obsolete()
        self._obsolete = True

    @deco.obsoletes
    @deco.new_from_generator
    def modify(self, **key_function_pairs):
        """
        Return list with items modified.

        In `key_function_pairs`, `function` receives as an argument an
        individual item.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.modify(price_per_guest=lambda x: x.price / x.guests)
        """
        key_function_pairs = key_function_pairs.items()
        for item in self:
            for key, function in key_function_pairs:
                item[key] = function(item)
            yield item

    @deco.obsoletes
    @deco.new_from_generator
    def modify_if(self, predicate, **key_function_pairs):
        """
        Return list with items matching `predicate` modified.

        `predicate` is a function that receives an individual item as argument
        and returns ``True`` to modify or ``False`` to not modify.

        In `key_function_pairs`, `function` receives as an argument an
        individual item.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.modify_if(lambda x: x.sqft, price_per_sqft=lambda x: x.price / x.sqft)
        """
        key_function_pairs = key_function_pairs.items()
        for item in self:
            if predicate(item):
                for key, function in key_function_pairs:
                    item[key] = function(item)
            yield item

    def _new(self, dicts):
        new = self.__class__(dicts, as_is=True)
        new._group_keys = self._group_keys
        new._predecessor = self
        return new

    def pluck(self, key, default=None):
        """
        Return a list of the values of `key` in all items.

        `default` is used for items in which `key` is not found.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.pluck("id")[:10]
        """
        return [x.get(key, default) for x in self]

    def print_(self, max_items=None):
        """
        Print list to ``sys.stdout``.

        `print_` does the same as calling Python's builtin ``print`` function,
        but since it's a method, you can use it at the end of a method chain
        instead of wrapping a ``print`` call around the whole chain.

        >>> di.ListOfDicts.read_json("data/listings.json").print_()
        """
        print(self.to_string(max_items))

    @classmethod
    def read_csv(cls, fname, encoding="utf_8", header=True, sep=","):
        """
        Return a new list from CSV file `fname`.
        """
        with open(fname, "r", encoding=encoding) as f:
            rows = list(csv.reader(f, dialect="unix", delimiter=sep))
            if not rows: return cls([])
            keys = rows.pop(0) if header else util.generate_colnames(len(rows[0]))
            return cls(dict(zip(keys, x)) for x in rows)

    @classmethod
    def read_json(cls, fname, encoding="utf_8", **kwargs):
        """
        Return a new list from JSON file `fname`.

        `kwargs` are passed to :meth:`from_json`.
        """
        with open(fname, "r", encoding=encoding) as f:
            return cls.from_json(f.read(), **kwargs)

    @classmethod
    def read_pickle(cls, fname):
        """
        Return a new list from Pickle file `fname`.
        """
        with open(fname, "rb") as f:
            return cls(pickle.load(f))

    @deco.obsoletes
    @deco.new_from_generator
    def rename(self, **to_from_pairs):
        """
        Return items with keys renamed.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.rename(listing_id="id")
        """
        renames = {v: k for k, v in to_from_pairs.items()}
        for item in self:
            keys = [renames.get(x, x) for x in item.keys()]
            yield AttributeDict(zip(keys, item.values()))

    @deco.new_from_generator
    def reverse(self):
        """
        Return items in reverse order.
        """
        yield from reversed(self)

    @deco.new_from_generator
    def sample(self, n=None):
        """
        Return randomly chosen `n` items.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.sample(3)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ITEMS
        n = min(len(self), n)
        for i in sorted(random.sample(range(len(self)), n)):
            yield self[i]

    @deco.obsoletes
    @deco.new_from_generator
    def select(self, *keys):
        """
        Return items, keeping only `keys`.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.select("id", "hood", "zipcode")
        """
        for item in self:
            yield AttributeDict({x: item[x] for x in keys if x in item})

    @deco.new_from_generator
    def semi_join(self, other, *by):
        """
        Return items with matches in `other`.

        `by` are keys, by which to look for matching items.

        >>> # All listings that have reviews
        >>> listings = di.ListOfDicts.read_json("data/listings.json")
        >>> reviews = di.ListOfDicts.read_json("data/listings-reviews.json")
        >>> listings.semi_join(reviews, "id")
        """
        extract = operator.itemgetter(*by)
        other_ids = set(map(extract, other))
        for item in self:
            if extract(item) in other_ids:
                yield item

    def sort(self, **key_dir_pairs):
        """
        Return items in sorted order.

        `key_dir_pairs` defines the sort order by key with `dir` being ``1``
        for ascending sort, ``-1`` for descending.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.sort(hood=1, zipcode=1)
        """
        data = self
        # Sort one key at a time to handle reverse and Nones correct.
        # https://stackoverflow.com/a/55866810
        for key, dir in list(key_dir_pairs.items())[::-1]:
            if dir not in [1, -1]:
                raise ValueError("dir should be 1 or -1")
            def sort_key(item):
                return ((item[key] is None, item[key]) if dir > 0 else
                        (item[key] is not None, item[key]))
            data = sorted(data, key=sort_key, reverse=dir < 0)
        return self._new(data)

    def tail(self, n=None):
        """
        Return the last `n` items.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.tail(3)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ITEMS
        n = min(len(self), n)
        return self._new(self[-n:])

    def _to_columns(self):
        return {k: self.pluck(k) for k in self[0]} if self else {}

    def to_data_frame(self):
        """
        Return list converted to a :class:`dataiter.DataFrame`.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.to_data_frame()
        """
        from dataiter import DataFrame
        return DataFrame(**self._to_columns())

    def to_json(self, **kwargs):
        """
        Return list converted to a JSON string.

        `kwargs` are passed to ``json.dumps``.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.to_json()[:100]
        """
        kwargs.setdefault("default", str)
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self, **kwargs)

    def to_pandas(self):
        """
        Return list converted to a ``pandas.DataFrame``.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.to_pandas()
        """
        import pandas as pd
        return pd.DataFrame(self._to_columns())

    def to_string(self, max_items=None):
        """
        Return list as a string formatted for display.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.to_string()
        """
        if max_items is None:
            max_items = dataiter.PRINT_MAX_ITEMS
        string = self.head(max_items).to_json()
        if max_items < len(self):
            string += f" ... {len(self)} items total"
        return string

    @deco.new_from_generator
    def unique(self, *keys):
        """
        Return unique items by `keys`.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.unique("hood")
        """
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
        """
        Return items, dropping `keys`.

        >>> data = di.ListOfDicts.read_json("data/listings.json")
        >>> data.unselect("guests", "sqft", "price")
        """
        for item in self:
            for key in keys:
                if key in item:
                    del item[key]
            yield item

    def write_csv(self, fname, encoding="utf_8", header=True, sep=","):
        """
        Write list to CSV file `fname`.
        """
        if not self:
            raise ValueError("Cannot write empty CSV file")
        # Take a superset of all keys and fill in missing as None.
        keys = util.unique_keys(list(itertools.chain(*self)))
        data = [{**dict.fromkeys(keys), **x} for x in self]
        util.makedirs_for_file(fname)
        with open(fname, "w", encoding=encoding) as f:
            writer = csv.DictWriter(f, keys, dialect="unix", delimiter=sep)
            writer.writeheader() if header else None
            for item in data:
                writer.writerow(item)

    def write_json(self, fname, encoding="utf_8", **kwargs):
        """
        Write list to JSON file `fname`.

        `kwargs` are passed to ``json.JSONEncoder``.
        """
        kwargs.setdefault("default", str)
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        util.makedirs_for_file(fname)
        with open(fname, "w", encoding=encoding) as f:
            encoder = json.JSONEncoder(**kwargs)
            for chunk in encoder.iterencode(self):
                f.write(chunk)
            f.write("\n")

    def write_pickle(self, fname):
        """
        Write list to Pickle file `fname`.
        """
        util.makedirs_for_file(fname)
        with open(fname, "wb") as f:
            out = [dict(x) for x in self]
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
