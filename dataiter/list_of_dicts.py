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
from math import inf


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

    List of dicts is a subclass of list. This means that if you need fast
    in-place methods instead of the regular ones that return shallow copies,
    you can use those from the list baseclass. A common example is appending
    items one by one in a for loop: instead of ``data = data.append(item)``,
    you can do ``list.append(data, item)``.

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

    def __init__(self, dicts=(), *, as_is=False):
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
        >>> data = di.read_json("data/listings.json")
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

        `by` are keys, by which to look for matching items, or tuples of keys
        if the correspoding key differs between `self` and `other`.

        >>> # All listings that don't have reviews
        >>> listings = di.read_json("data/listings.json")
        >>> reviews = di.read_json("data/listings-reviews.json")
        >>> listings.anti_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        extract1 = operator.itemgetter(*by1)
        extract2 = operator.itemgetter(*by2)
        other_ids = set(map(extract2, other))
        for item in self:
            if extract1(item) not in other_ids:
                yield item

    @deco.new_from_generator
    def append(self, item):
        """
        Return list with `item` added to the end.

        >>> data = di.read_json("data/listings.json")
        >>> data = data.append(dict.fromkeys(data[0].keys()))
        >>> data.tail()
        """
        if not isinstance(item, AttributeDict):
            item = AttributeDict(item)
        yield from itertools.chain(self, [item])

    def clear(self):
        """
        Return list with all items removed.

        >>> data = di.read_json("data/listings.json")
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

        >>> data = di.read_json("data/listings.json")
        >>> data = data.extend([dict.fromkeys(data[0].keys())])
        >>> data.tail()
        """
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        yield from itertools.chain(self, other)

    @deco.obsoletes
    @deco.new_from_generator
    def fill_missing_keys(self, **key_value_pairs):
        """
        Return list with missing keys added.

        If `key_value_pairs` not given, fill all missing keys with ``None``.

        >>> data = di.read_json("data/listings.json")
        >>> data = data.fill_missing_keys(price=None)
        >>> data = data.fill_missing_keys()
        """
        if not key_value_pairs:
            keys = util.unique_keys(itertools.chain(*self))
            key_value_pairs = dict.fromkeys(keys, None)
        key_value_pairs = key_value_pairs.items()
        for item in self:
            for key, value in key_value_pairs:
                if key not in item:
                    item[key] = value
            yield item

    @deco.new_from_generator
    def filter(self, function=None, **key_value_pairs):
        """
        Return items that match condition.

        Filtering can be done either by `function`, which receives an
        individual item as its argument and returns ``True`` or ``False``, or
        by `key_value_pairs`, which are a shorthand to check against a fixed
        value. See the example below of equivalent filtering with both ways.

        >>> data = di.read_json("data/listings.json")
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

        >>> data = di.read_json("data/listings.json")
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
    def from_json(cls, string, *, keys=[], types={}, **kwargs):
        """
        Return a new list of dicts from JSON `string`.

        `keys` is an optional list of keys to limit to. `types` is an optional
        dict mapping keys to datatypes. `kwargs` are passed to ``json.load``.
        """
        data = json.loads(string, **kwargs)
        if not isinstance(data, list):
            raise TypeError("Not a list")
        if keys:
            keys = set(keys)
            for item in data:
                for key in set(item) - keys:
                    del item[key]
        for key, type in types.items():
            for item in data:
                if key in item:
                    item[key] = type(item[key])
        return cls(data)

    def full_join(self, other, *by):
        """
        Return list with matching items merged from `self` and `other`.

        `full_join` keeps all items from both lists, merging matching ones. If
        there are multiple matches, the first one will be used. For items, for
        which matches are not found, no keys are added.

        `by` are keys, by which to look for matching items, or tuples of keys
        if the correspoding key differs between `self` and `other`.

        >>> listings = di.read_json("data/listings.json")
        >>> reviews = di.read_json("data/listings-reviews.json")
        >>> listings.full_join(reviews, "id")
        """
        acounter = itertools.count(start=1)
        bcounter = itertools.count(start=1)
        a = self.deepcopy().modify(_aid_=lambda x: next(acounter))
        b = other.deepcopy().modify(_bid_=lambda x: next(bcounter))
        ab = a.deepcopy().left_join(b, *by)
        # Fill in missing _bid_ with bogus values.
        ab = ab.fill_missing_keys(_bid_=next(bcounter))
        # Check which items of b were not joined into a,
        # if no items remain, full join is the same as left join ab.
        b = b.anti_join(ab, "_bid_")
        if len(b) == 0:
            return ab.unselect("_aid_", "_bid_")
        ba = b.left_join(a, *by)
        # Fill in missing _aid_ with bogus values.
        ba = ba.fill_missing_keys(_aid_=next(acounter))
        return (ab + ba).sort(_aid_=1, _bid_=1).unselect("_aid_", "_bid_")

    def group_by(self, *keys):
        """
        Return list with `keys` set for grouped operations, such as :meth:`aggregate`.
        """
        self._group_keys = tuple(keys)
        return self

    def head(self, n=None):
        """
        Return the first `n` items.

        >>> data = di.read_json("data/listings.json")
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

        `by` are keys, by which to look for matching items, or tuples of keys
        if the correspoding key differs between `self` and `other`.

        >>> listings = di.read_json("data/listings.json")
        >>> reviews = di.read_json("data/listings-reviews.json")
        >>> listings.inner_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        extract1 = operator.itemgetter(*by1)
        extract2 = operator.itemgetter(*by2)
        other_by_id = {extract2(x): x for x in reversed(other)}
        for item in self:
            id = extract1(item)
            if id in other_by_id:
                new = other_by_id[id]
                new = {k: v for k, v in new.items() if k not in by2}
                item.update(new)
                yield item

    @deco.new_from_generator
    def insert(self, index, item):
        """
        Return list with `item` inserted at `index`.

        >>> data = di.read_json("data/listings.json")
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

        `by` are keys, by which to look for matching items, or tuples of keys
        if the correspoding key differs between `self` and `other`.

        >>> listings = di.read_json("data/listings.json")
        >>> reviews = di.read_json("data/listings-reviews.json")
        >>> listings.left_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        extract1 = operator.itemgetter(*by1)
        extract2 = operator.itemgetter(*by2)
        other_by_id = {extract2(x): x for x in reversed(other)}
        for item in self:
            new = other_by_id.get(extract1(item), {})
            new = {k: v for k, v in new.items() if k not in by2}
            item.update(new)
            yield item

    def map(self, function):
        """
        Apply `function` to each item in list.

        If `function` returns a dict, then the return value will be coerced to
        a :class:`ListOfDicts` instance, otherwise the return value will be a
        list of whatever `function` returns.

        >>> data = di.read_json("data/listings.json")
        >>> data.map(lambda x: (x.guests, x.price))
        """
        new = list(map(function, self))
        coerce = all(isinstance(x, dict) for x in new)
        return self.__class__(new) if coerce else new

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

        >>> data = di.read_json("data/listings.json")
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

        >>> data = di.read_json("data/listings.json")
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

        >>> data = di.read_json("data/listings.json")
        >>> data.pluck("id")[:10]
        """
        return [x.get(key, default) for x in self]

    def print_(self, *, max_items=None):
        """
        Print list to ``sys.stdout``.

        `print_` does the same as calling Python's builtin ``print`` function,
        but since it's a method, you can use it at the end of a method chain
        instead of wrapping a ``print`` call around the whole chain.

        >>> di.read_json("data/listings.json").print_()
        """
        print(self.to_string(max_items=max_items))

    def print_na_counts(self):
        """
        Print counts of missing values by key.

        Both keys entirely missing and keys with a value of ``None`` are
        considered missing.

        >>> data = di.read_json("data/listings.json")
        >>> data.print_na_counts()
        """
        print("Missing counts:")
        for key in util.unique_keys(itertools.chain(*self)):
            n = sum(x.get(key, None) is None for x in self)
            if n == 0: continue
            pc = 100 * n / len(self)
            print(f"... {key}: {n} ({pc:.1f}%)")

    @classmethod
    def read_csv(cls, path, *, encoding="utf-8", sep=",", header=True, keys=[], types={}):
        """
        Return a new list from CSV file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz``.

        `keys` is an optional list of keys to limit to. `types` is an optional
        dict mapping keys to datatypes.
        """
        with util.xopen(path, "rt", encoding=encoding) as f:
            rows = list(csv.reader(f, dialect="unix", delimiter=sep))
            if not rows: return cls([])
            colnames = rows.pop(0) if header else util.generate_colnames(len(rows[0]))
            if keys:
                # Drop all keys except the requested ones.
                drop = [i for i in range(len(rows[0])) if colnames[i] not in keys]
                for row in rows:
                    for i in reversed(drop):
                        del row[i]
                colnames = keys
            data = cls(dict(zip(colnames, x)) for x in rows)
            for key, type in types.items():
                for item in data:
                    if key in item:
                        item[key] = type(item[key])
            return data

    @classmethod
    def read_json(cls, path, *, encoding="utf-8", keys=[], types={}, **kwargs):
        """
        Return a new list from JSON file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz``.

        `keys` is an optional list of keys to limit to. `types` is an optional
        dict mapping keys to datatypes. `kwargs` are passed to ``json.load``.
        """
        with util.xopen(path, "rt", encoding=encoding) as f:
            return cls.from_json(f.read(), keys=keys, types=types, **kwargs)

    @classmethod
    def read_pickle(cls, path):
        """
        Return a new list from Pickle file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz``.
        """
        with util.xopen(path, "rb") as f:
            return cls(pickle.load(f))

    @deco.obsoletes
    @deco.new_from_generator
    def rename(self, **to_from_pairs):
        """
        Return items with keys renamed.

        >>> data = di.read_json("data/listings.json")
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

        >>> data = di.read_json("data/listings.json")
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

        >>> data = di.read_json("data/listings.json")
        >>> data.select("id", "hood", "zipcode")
        """
        for item in self:
            yield AttributeDict({x: item[x] for x in keys if x in item})

    @deco.new_from_generator
    def semi_join(self, other, *by):
        """
        Return items with matches in `other`.

        `by` are keys, by which to look for matching items, or tuples of keys
        if the correspoding key differs between `self` and `other`.

        >>> # All listings that have reviews
        >>> listings = di.read_json("data/listings.json")
        >>> reviews = di.read_json("data/listings-reviews.json")
        >>> listings.semi_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        extract1 = operator.itemgetter(*by1)
        extract2 = operator.itemgetter(*by2)
        other_ids = set(map(extract2, other))
        for item in self:
            if extract1(item) in other_ids:
                yield item

    def sort(self, **key_dir_pairs):
        """
        Return items in sorted order.

        `key_dir_pairs` defines the sort order by key with `dir` being ``1``
        for ascending sort, ``-1`` for descending.

        >>> data = di.read_json("data/listings.json")
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

    def split(self, *by):
        """
        Split list into groups and return a list of their indices.

        >>> data = di.ListOfDicts({"x": x} for x in [1, 2, 2, 3, 3, 3])
        >>> data.split("x")
        """
        extract = operator.itemgetter(*by)
        indices_by_group = {}
        for i, item in enumerate(self):
            id = extract(item)
            indices_by_group.setdefault(id, []).append(i)
        return list(indices_by_group.values())

    def _split_join_by(self, *by):
        by1 = [x if isinstance(x, str) else x[0] for x in by]
        by2 = [x if isinstance(x, str) else x[1] for x in by]
        return by1, by2

    def tail(self, n=None):
        """
        Return the last `n` items.

        >>> data = di.read_json("data/listings.json")
        >>> data.tail(3)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ITEMS
        n = min(len(self), n)
        return self._new(self[-n:])

    def _to_columns(self):
        return {k: self.pluck(k) for k in self[0]} if self else {}

    def to_data_frame(self, strings_as_object=inf):
        """
        Return list converted to a :class:`.DataFrame`.

        `strings_as_object` is a cutoff point. If any row has more characters
        than that, the whole column will use the object data type. This is
        intended to help limit memory use as NumPy strings are fixed-length and
        can take a huge amount of memory if even a single row is long.

        >>> data = di.read_json("data/listings.json")
        >>> data.to_data_frame()
        """
        from dataiter import DataFrame
        from dataiter import DataFrameColumn
        data = self._to_columns()
        if strings_as_object < inf:
            for name in data:
                if (data[name] and
                    any(isinstance(x, str) for x in data[name]) and
                    any(len(x) > strings_as_object for x in data[name] if isinstance(x, str))):
                    data[name] = DataFrameColumn(data[name], object)
        return DataFrame(**data)

    def to_json(self, **kwargs):
        """
        Return list converted to a JSON string.

        `kwargs` are passed to ``json.dump``.

        >>> data = di.read_json("data/listings.json")
        >>> data.to_json()[:100]
        """
        kwargs.setdefault("default", str)
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self, **kwargs)

    def to_pandas(self):
        """
        Return list converted to a ``pandas.DataFrame``.

        >>> data = di.read_json("data/listings.json")
        >>> data.to_pandas()
        """
        import pandas as pd
        return pd.DataFrame(self._to_columns())

    def to_string(self, *, max_items=None):
        """
        Return list as a string formatted for display.

        >>> data = di.read_json("data/listings.json")
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

        >>> data = di.read_json("data/listings.json")
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

        >>> data = di.read_json("data/listings.json")
        >>> data.unselect("guests", "sqft", "price")
        """
        for item in self:
            for key in keys:
                if key in item:
                    del item[key]
            yield item

    def write_csv(self, path, *, encoding="utf-8", header=True, sep=","):
        """
        Write list to CSV file `path`.

        Will automatically compress if `path` ends in ``.bz2|.gz|.xz``.
        """
        if not self:
            raise ValueError("Cannot write empty CSV file")
        # Take a superset of all keys.
        keys = util.unique_keys(itertools.chain(*self))
        util.makedirs_for_file(path)
        with util.xopen(path, "wt", encoding=encoding) as f:
            writer = csv.DictWriter(f,
                                    keys,
                                    dialect="unix",
                                    delimiter=sep,
                                    quoting=csv.QUOTE_MINIMAL)

            writer.writeheader() if header else None
            for item in self:
                # Fill in missing as None.
                item = {**dict.fromkeys(keys), **item}
                writer.writerow(item)

    def write_json(self, path, *, encoding="utf-8", **kwargs):
        """
        Write list to JSON file `path`.

        Will automatically compress if `path` ends in ``.bz2|.gz|.xz``.

        `kwargs` are passed to ``json.JSONEncoder``.
        """
        kwargs.setdefault("default", str)
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        util.makedirs_for_file(path)
        with util.xopen(path, "wt", encoding=encoding) as f:
            encoder = json.JSONEncoder(**kwargs)
            for chunk in encoder.iterencode(self):
                f.write(chunk)
            f.write("\n")

    def write_pickle(self, path):
        """
        Write list to Pickle file `path`.

        Will automatically compress if `path` ends in ``.bz2|.gz|.xz``.
        """
        util.makedirs_for_file(path)
        with util.xopen(path, "wb") as f:
            out = [dict(x) for x in self]
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
