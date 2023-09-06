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

import contextlib
import dataiter
import functools
import itertools
import json
import numpy as np
import pickle

from dataiter import deco
from dataiter import util
from dataiter import Vector
from math import inf


class DataFrameColumn(Vector):

    """
    A column in a data frame.

    DataFrameColumn is a subclass of :class:`.Vector`. See the vector
    documentation for relevant properties and methods.
    """

    def __new__(cls, object, dtype=None, nrow=None):
        object = util.sequencify(object)
        column = Vector(object, dtype)
        if nrow is not None and nrow != column.length:
            if column.length != 1 or nrow < 1:
                raise ValueError("Bad arguments for broadcast")
            column = column.repeat(nrow)
        return column.view(cls)

    def __init__(self, object, dtype=None, nrow=None):
        """
        Return a new data frame column.

        `dtype` is the NumPy-compatible data type for the vector. Providing
        `dtype` will make creating the vector faster, otherwise the appropriate
        data type will be guessed by introspecting the elements of `object`,
        which is potentially slow, especially for large objects.

        If provided, `nrow` is the row count to produce, i.e. the length to
        which `object` will be broadcast.

        >>> di.DataFrameColumn([1, 2, 3], int)
        >>> di.DataFrameColumn([1], int, nrow=10)
        """
        super().__init__(object, dtype)

    @property
    def nrow(self):
        """
        Return the amount of rows.
        """
        return self.length


class DataFrame(dict):

    """
    A class for tabular data.

    DataFrame is a subclass of ``dict``, with columns being
    :class:`.DataFrameColumn`, which are :class:`.Vector`, which are NumPy
    ``ndarray``. This means that basic ``dict`` methods, such as ``items()``,
    ``keys()`` and ``values()`` can be used iterate over and manage the data as
    a whole and NumPy functions and array methods can be used for fast
    vectorized computations on the data.

    Columns can be accessed by attribute notation, e.g. ``data.x`` in addition
    to ``data["x"]``. In most cases, attribute access should be more convenient
    and is the way recommended by dataiter. You'll still need to use the
    bracket notation for any column names that are not valid identifiers, such
    as ones with spaces, or ones that conflict with dict methods, such as
    "items".

    DataFrame does not support indexing directly as the bracket notation is
    used to refer to dict keys, i.e. columns by name. If you want to index the
    whole data frame object, use the method :meth:`slice`. Individual columns
    are indexed the same as NumPy arrays.
    """

    # List of names that are actual attributes, not columns
    ATTRIBUTES = ["colnames", "_group_colnames"]

    # Use dummy attributes corresponding to dictionary keys so that
    # Tab completion of column names at a Python shell would work.
    COLUMN_PLACEHOLDER = type("COLUMN_PLACEHOLDER", (), {})

    def __init__(self, *args, **kwargs):
        """
        Return a new data frame.

        `args` and `kwargs` are like for ``dict``.

        https://docs.python.org/3/library/stdtypes.html#dict
        """
        super().__init__(*args, **kwargs)
        nrow = max(map(util.length, self.values()), default=0)
        for key, value in self.items():
            if (isinstance(value, DataFrameColumn) and
                value.nrow == nrow): continue
            column = DataFrameColumn(value, nrow=nrow)
            super().__setitem__(key, column)
        for key in self:
            if not self.__hasattr(key) and key.isidentifier():
                super().__setattr__(key, self.COLUMN_PLACEHOLDER)
        # Check that we have a uniform table.
        self._check_dimensions()
        self._group_colnames = ()

    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo=None):
        return self.__class__({k: v.copy() for k, v in self.items()})

    def __delattr__(self, name):
        if name in self:
            return self.__delitem__(name)
        return super().__delattr__(name)

    def __delitem__(self, key):
        # Note that this is not called for some methods,
        # at least pop, popitem and clear.
        if self[key] is self.COLUMN_PLACEHOLDER:
            super().__delattr__(key)
        return super().__delitem__(key)

    def __eq__(self, other):
        return (isinstance(other, DataFrame) and
                self.nrow == other.nrow and
                self.ncol == other.ncol and
                set(self.colnames) == set(other.colnames) and
                all(self[x].equal(other[x]) for x in self))

    def __getattr__(self, name):
        if name in self:
            return self.__getitem__(name)
        raise AttributeError(name)

    def __getattribute__(self, name):
        value = super().__getattribute__(name)
        if name == "COLUMN_PLACEHOLDER":
            return value
        if value is self.COLUMN_PLACEHOLDER and name in self:
            return self[name]
        return value

    def __hasattr(self, name):
        # Return True if attribute exists and is not a column.
        return hasattr(self, name) and not isinstance(getattr(self, name), DataFrameColumn)

    @classmethod
    def __is_builtin_attr(cls, name):
        return name in cls.__list_builtin_attrs()

    @classmethod
    @functools.lru_cache(None)
    def __list_builtin_attrs(cls):
        return set(dir(cls()))

    def __setattr__(self, name, value):
        if name in self.ATTRIBUTES:
            return super().__setattr__(name, value)
        return self.__setitem__(name, value)

    def __setitem__(self, key, value):
        value = self._reconcile_column(value)
        if not self.__hasattr(key) and key.isidentifier():
            super().__setattr__(key, self.COLUMN_PLACEHOLDER)
        return super().__setitem__(key, value)

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    def aggregate(self, **colname_function_pairs):
        """
        Return group-wise calculated summaries.

        Usually aggregation is preceded by grouping, which can be conveniently
        written via method chaining as ``data.group_by(...).aggregate(...)``.

        In `colname_function_pairs`, `function` receives as an argument a data
        frame object, a group-wise subset of all rows. It should return a
        scalar value. Common aggregation functions have shorthand helpers
        available under :mod:`dataiter`, see the guide on :doc:`aggregation
        </aggregation>` for details.

        >>> data = di.read_csv("data/listings.csv")
        >>> # The below aggregations are identical. Usually you'll get by
        >>> # with the shorthand helpers, but for complicated calculations,
        >>> # you might need custom lambda functions.
        >>> data.group_by("hood").aggregate(n=di.count(), price=di.mean("price"))
        >>> data.group_by("hood").aggregate(n=lambda x: x.nrow, price=lambda x: x.price.mean())
        """
        group_colnames = self._group_colnames
        data = self.sort(**dict.fromkeys(group_colnames, 1))
        data._index_ = np.arange(data.nrow)
        stat = data.unique(*group_colnames).select("_index_", *group_colnames)
        indices = np.split(data._index_, stat._index_[1:])
        group_aware = [getattr(x, "group_aware", False) for x in colname_function_pairs.values()]
        if any(group_aware):
            groups = Vector.fast(range(len(indices)), int)
            n = Vector.fast(map(len, indices), int)
            data._group_ = np.repeat(groups, n)
        slices = None
        for colname, function in colname_function_pairs.items():
            if getattr(function, "group_aware", False):
                # function might leave Nones in its output,
                # once those are replaced with the proper default
                # we can do a fast conversion to DataFrameColumn.
                column = function(data)
                default = function.default
                for i in range(len(column)):
                    if column[i] is None:
                        column[i] = default
                assert len(column) == stat.nrow
                column = DataFrameColumn.fast(column)
                stat[colname] = column
            else:
                # When using an arbitrary function, we cannot know
                # what special values to expect and thus we end up
                # needing to use the slow Vector.__init__.
                if slices is None:
                    slices = [data._view_rows(x) for x in indices]
                stat[colname] = [function(x) for x in slices]
        return stat.unselect("_index_", "_group_")

    @deco.new_from_generator
    def anti_join(self, other, *by):
        """
        Return rows with no matches in `other`.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        >>> # All listings that don't have reviews
        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.anti_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        other = other.drop_na(*by2).unique(*by2)
        found, src = self._get_join_indices(other, by1, by2)
        for colname, column in self.items():
            yield colname, np.delete(column, found)

    @deco.new_from_generator
    def cbind(self, *others):
        """
        Return data frame with columns from `others` added.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.cbind(di.DataFrame(x=1))
        """
        found_colnames = set()
        data_frames = [self] + list(others)
        for i, data in enumerate(data_frames):
            for colname, column in data.items():
                if colname in found_colnames: continue
                found_colnames.add(colname)
                column = self._reconcile_column(column)
                yield colname, column.copy()

    def _check_dimensions(self):
        if not self: return
        nrows = [x.nrow for x in self.columns]
        if len(set(nrows)) == 1: return
        raise ValueError(f"Bad dimensions: {nrows!r}")

    def clear(self):
        return self._new()

    @property
    def colnames(self):
        """
        Get or set column names as a list.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.head()
        >>> data.colnames
        >>> data.colnames = ["a", "b", "c", "d", "e", "f"]
        >>> data.head()
        """
        return list(self)

    @colnames.setter
    def colnames(self, colnames):
        for fm, to in zip(list(self.keys()), colnames):
            self[to] = self.pop(fm)

    @property
    def columns(self):
        """
        Return columns as a list.
        """
        return list(self.values())

    def compare(self, other, *by, ignore_columns=[], max_changed=inf):
        """
        Find differences against another data frame.

        `by` are identifier columns which are used to uniquely identify rows
        and match them between `self` and `other`. `compare` will not work if
        your data lacks suitable identifiers. `ignore_columns` is an optional
        list of columns, differences in which to ignore.

        `compare` returns three data frames: added rows, removed rows and
        changed values. The first two are basically subsets of the rows of
        `self` and `other`, respectively. Changed values are returned as a data
        frame with one row per differing value (not per differing row). Listing
        changes will terminate once `max_changed` is reached.

        .. warning:: `compare` is experimental, do not rely on it reporting all
                     of the differences correctly. Do not try to give it two
                     huge data frames with very little in common, unless also
                     giving some sensible value for `max_changed`.

        >>> old = di.read_csv("data/vehicles.csv")
        >>> new = old.modify(hwy=lambda x: np.minimum(100, x.hwy))
        >>> added, removed, changed = new.compare(old, "id")
        >>> changed
        """
        if self.unique(*by).nrow < self.nrow:
            raise ValueError(f"self not unique by {by}")
        if other.unique(*by).nrow < other.nrow:
            raise ValueError(f"other not unique by {by}")
        added = self.anti_join(other, *by)
        removed = other.anti_join(self, *by)
        x = self.modify(_i_=range(self.nrow))
        y = other.modify(_j_=range(other.nrow))
        z = x.inner_join(y.select("_j_", *by), *by)
        colnames = util.unique_keys(self.colnames + other.colnames)
        colnames = [x for x in colnames if x not in ignore_columns]
        changed = []
        for i, j in zip(z._i_, z._j_):
            if len(changed) >= max_changed:
                print(f"max_changed={max_changed} reached, terminating")
                break
            for colname in colnames:
                if len(changed) >= max_changed: break
                # XXX: How to make a distinction between
                # a missing column and a missing value?
                xvalue = x[colname][i] if colname in x else None
                yvalue = y[colname][j] if colname in y else None
                if (xvalue != yvalue and
                    not Vector([xvalue, yvalue]).is_na().all()):
                    # XXX: We could have a name clash here.
                    byrow = {k: x[k][i] for k in by}
                    changed.append(dict(**byrow,
                                        column=colname,
                                        xvalue=xvalue,
                                        yvalue=yvalue))

        added = added if added.nrow > 0 else None
        removed = removed if removed.nrow > 0 else None
        changed = self.from_json(changed) if changed else None
        return added, removed, changed

    def copy(self):
        """
        Return a shallow copy.
        """
        return self.__copy__()

    def count(self, *colnames):
        """
        Return row counts grouped by `colnames`.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.count("hood")
        """
        return self.copy().group_by(*colnames).aggregate(n=dataiter.count())

    def deepcopy(self):
        """
        Return a deep copy.
        """
        return self.__deepcopy__()

    def drop_na(self, *colnames):
        """
        Return data frame without rows that have missing values in `colnames`.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.drop_na("sqft")
        """
        drop = Vector.fast([False], bool).repeat(self.nrow)
        for colname in colnames:
            drop = drop | self[colname].is_na()
        return self.filter_out(drop)

    @deco.new_from_generator
    def filter(self, rows=None, **colname_value_pairs):
        """
        Return rows that match condition.

        Filtering can be done by either `rows` or `colname_value_pairs`. `rows`
        can be either a boolean vector or a function that receives the data
        frame as argument and returns a boolean vector. The latter is
        especially useful in a method chaining context where you don't have
        direct access to the data frame in question. Alternatively,
        `colname_value_pairs` provides a shorthand to check against a fixed
        value. See the example below of equivalent filtering all three ways.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.filter((data.hood == "Manhattan") & (data.guests == 2))
        >>> data.filter(lambda x: (x.hood == "Manhattan") & (x.guests == 2))
        >>> data.filter(hood="Manhattan", guests=2)
        """
        if rows is not None:
            if callable(rows):
                rows = rows(self)
        elif colname_value_pairs:
            rows = Vector.fast([True], bool).repeat(self.nrow)
            for colname, value in colname_value_pairs.items():
                rows = rows & (self[colname] == value)
        rows = self._parse_rows_from_boolean(rows)
        for colname, column in self.items():
            yield colname, np.take(column, rows)

    @deco.new_from_generator
    def filter_out(self, rows=None, **colname_value_pairs):
        """
        Return rows that don't match condition.

        Filtering can be done by either `rows` or `colname_value_pairs`. `rows`
        can be either a boolean vector or a function that receives the data
        frame as argument and returns a boolean vector. The latter is
        especially useful in a method chaining context where you don't have
        direct access to the data frame in question. Alternatively,
        `colname_value_pairs` provides a shorthand to check against a fixed
        value. See the example below of equivalent filtering all three ways.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.filter_out(data.hood == "Manhattan")
        >>> data.filter_out(lambda x: x.hood == "Manhattan")
        >>> data.filter_out(hood="Manhattan")
        """
        if rows is not None:
            if callable(rows):
                rows = rows(self)
        elif colname_value_pairs:
            rows = Vector.fast([True], bool).repeat(self.nrow)
            for colname, value in colname_value_pairs.items():
                rows = rows & (self[colname] == value)
        rows = self._parse_rows_from_boolean(rows)
        for colname, column in self.items():
            yield colname, np.delete(column, rows)

    @classmethod
    def from_arrow(cls, data, *, strings_as_object=inf, dtypes={}):
        """
        Return a new data frame from ``pyarrow.Table`` `data`.

        `strings_as_object` is a cutoff point. If any row has more characters
        than that, the whole column will use the object data type. This is
        intended to help limit memory use as NumPy strings are fixed-length and
        can take a huge amount of memory if even a single row is long. If set,
        `dtypes` overrides this.

        `dtypes` is an optional dict mapping column names to NumPy datatypes.
        """
        # Arrow's 'to_numpy' is "limited to primitive types for which NumPy has
        # the same physical representation as Arrow, and assuming the Arrow
        # data has no nulls." Using Pandas is easier and probably good enough.
        return cls.from_pandas(data.to_pandas(),
                               strings_as_object=strings_as_object,
                               dtypes=dtypes)

    @classmethod
    def from_json(cls, string, *, columns=[], dtypes={}, **kwargs):
        """
        Return a new data frame from JSON `string`.

        `columns` is an optional list of columns to limit to. `dtypes` is an
        optional dict mapping column names to NumPy datatypes. `kwargs` are
        passed to ``json.load``.
        """
        data = string
        if isinstance(data, str):
            data = json.loads(data, **kwargs)
        if not isinstance(data, list):
            raise TypeError("Not a list")
        keys = util.unique_keys(itertools.chain(*data))
        if columns:
            keys = [x for x in keys if x in columns]
        data = {k: [x.get(k, None) for x in data] for k in keys}
        for name, dtype in dtypes.items():
            data[name] = DataFrameColumn(data[name], dtype)
        return cls(**data)

    @classmethod
    def from_pandas(cls, data, *, strings_as_object=inf, dtypes={}):
        """
        Return a new data frame from ``pandas.DataFrame`` `data`.

        `strings_as_object` is a cutoff point. If any row has more characters
        than that, the whole column will use the object data type. This is
        intended to help limit memory use as NumPy strings are fixed-length and
        can take a huge amount of memory if even a single row is long. If set,
        `dtypes` overrides this.

        `dtypes` is an optional dict mapping column names to NumPy datatypes.
        """
        if (not isinstance(strings_as_object, (int, float)) or
            isinstance(strings_as_object, bool)):
            raise TypeError("Expected a number for strings_as_object")
        dtypes = dtypes.copy()
        from pandas.api.types import is_object_dtype
        if strings_as_object < inf:
            for name in data.columns:
                if name not in dtypes and is_object_dtype(data[name]):
                    with contextlib.suppress(AttributeError):
                        if data[name].str.len().max() > strings_as_object:
                            dtypes[name] = object
        data = {x: data[x].to_numpy(copy=True) for x in data.columns}
        for name, value in data.items():
            # Pandas object columns are likely to be strings,
            # convert to list to force type guessing in Vector.__init__.
            if np.issubdtype(value.dtype, np.object_):
                data[name] = data[name].tolist()
        for name, dtype in dtypes.items():
            data[name] = DataFrameColumn(data[name], dtype)
        return cls(**data)

    def full_join(self, other, *by):
        """
        Return data frame with matching rows merged from `self` and `other`.

        `full_join` keeps all rows from both data frames, merging matching
        ones. If there are multiple matches, the first one will be used. For
        rows, for which matches are not found, missing values are added.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.full_join(reviews, "id")
        """
        a = self.modify(_aid_=np.arange(self.nrow))
        b = other.modify(_bid_=np.arange(other.nrow))
        ab = a.left_join(b, *by)
        # Check which rows of b were not joined into a.
        # If no rows remain, full join is the same as left join ab.
        b = b.anti_join(ab, "_bid_")
        if b.nrow == 0:
            return ab.unselect("_aid_", "_bid_")
        # Reverse the by-tuples for the reverse join ba,
        # so that the data frame and by orders match.
        by_reverse = [
            tuple(reversed(x)) if isinstance(x, (list, tuple))
            else x
            for x in by]
        ba = b.left_join(a, *by_reverse)
        for item in by:
            # For identifiers in by whose name differs in a and b,
            # rename and keep the variant found in a.
            if isinstance(item, (list, tuple)):
                ba[item[0]] = ba.pop(item[1])
        return ab.rbind(ba).sort(_aid_=1, _bid_=1).unselect("_aid_", "_bid_")

    def _get_join_indices(self, other, by1, by2):
        other_ids = list(zip(*[other[x] for x in by2]))
        other_by_id = {other_ids[i]: i for i in range(other.nrow)}
        self_ids = zip(*[self[x] for x in by1])
        src = map(lambda x: other_by_id.get(x, -1), self_ids)
        src = np.fromiter(src, int, count=self.nrow)
        found = np.where(src > -1)
        return found, src

    def group_by(self, *colnames):
        """
        Return data frame with `colnames` set for grouped operations, such as :meth:`aggregate`.
        """
        self._group_colnames = tuple(colnames)
        return self

    def head(self, n=None):
        """
        Return the first `n` rows.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.head(5)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ROWS
        n = min(self.nrow, n)
        return self.slice(np.arange(n))

    @deco.new_from_generator
    def inner_join(self, other, *by):
        """
        Return data frame with matching rows merged from `self` and `other`.

        `inner_join` keeps only rows found in both data frames, merging
        matching ones. If there are multiple matches, the first one will be
        used.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.inner_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        other = other.drop_na(*by2).unique(*by2)
        found, src = self._get_join_indices(other, by1, by2)
        for colname, column in self.items():
            yield colname, column[found].copy()
        for colname, column in other.items():
            if colname in by2: continue
            if colname in self: continue
            yield colname, column[src[found]].copy()

    @deco.new_from_generator
    def left_join(self, other, *by):
        """
        Return data frame with matching rows merged from `self` and `other`.

        `left_join` keeps all rows in `self`, merging matching ones. If there
        are multiple matches, the first one will be used. For rows, for which
        matches are not found, missing values are added.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.left_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        other = other.drop_na(*by2).unique(*by2)
        found, src = self._get_join_indices(other, by1, by2)
        for colname, column in self.items():
            yield colname, column.copy()
        for colname, column in other.items():
            if colname in by2: continue
            if colname in self: continue
            value = column.na_value
            dtype = column.na_dtype
            new = DataFrameColumn(value, dtype, self.nrow)
            new[found] = column[src[found]]
            yield colname, new.copy()

    def map(self, function):
        """
        Apply `function` to each row in data.

        `function` receives as arguments the full data frame and the loop
        index. The return value will be a list of whatever `function` returns.

        Note that `map` is an inefficient method as it iterates over rows
        instead of doing vectorized computation. `map` is mostly intended for
        complicated conditional cases that are difficult to express in
        vectorized form.

        >>> data = di.read_csv("data/listings-reviews.csv")
        >>> data.map(lambda x, i: (x.reviews[i], x.rating[i]))
        """
        return [function(self, i) for i in range(self.nrow)]

    @deco.new_from_generator
    def modify(self, **colname_value_pairs):
        """
        Return data frame with columns modified.

        In `colname_value_pairs`, `value` can be either a vector or a function
        that receives the data frame as argument and returns a vector. See the
        example below of equivalent modification with both ways.

        Note that column modification can often be done simpler with a plain
        assignment, such as ``data.price_per_guest = data.price /
        data.guests``. `modify` just allows you to do the same in a method
        chain context.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.modify(price_per_guest=data.price/data.guests)
        >>> data.modify(price_per_guest=lambda x: x.price / x.guests)

        If the data frame is grouped, then `colname_value_pairs` need to be
        functions, which are applied to group-wise subsets of the data frame.
        A common use for this is calculating group-wise fractions.

        >>> data = di.DataFrame(g=[1, 2, 2, 3, 3, 3])
        >>> data.group_by("g").modify(f=lambda x: 1 / x.nrow)
        """
        for colname, column in self.items():
            yield colname, column.copy()
        if self._group_colnames:
            slices = self.split(*self._group_colnames)
            # Mapping over slices will produce contiguous groups in order
            # of self._group_colnames. Calculate and apply indexing that
            # will restore the original order.
            restore_indices = np.argsort(np.concatenate(slices))
            slices = [self._view_rows(x) for x in slices]
            for colname, function in colname_value_pairs.items():
                if not callable(function):
                    raise ValueError(f"{colname} argument not callable")
                column = [DataFrameColumn(function(x), nrow=x.nrow) for x in slices]
                yield colname, np.concatenate(column)[restore_indices]
        else:
            for colname, value in colname_value_pairs.items():
                value = value(self) if callable(value) else value
                yield colname, self._reconcile_column(value).copy()

    @property
    def ncol(self):
        """
        Return the amount of columns.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.ncol
        """
        self._check_dimensions()
        return len(self)

    def _new(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    @property
    def nrow(self):
        """
        Return the amount of rows.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.nrow
        """
        if not self: return 0
        self._check_dimensions()
        return self[next(iter(self))].nrow

    def _parse_cols_from_boolean(self, cols):
        cols = Vector.fast(cols, bool)
        if len(cols) != self.ncol:
            raise ValueError("Bad length for boolean cols")
        return Vector.fast(np.nonzero(cols)[0], int)

    def _parse_cols_from_integer(self, cols):
        return Vector.fast(cols, int)

    def _parse_rows_from_boolean(self, rows):
        rows = Vector.fast(rows, bool)
        if len(rows) != self.nrow:
            raise ValueError("Bad length for boolean rows")
        return Vector.fast(np.nonzero(rows)[0], int)

    def _parse_rows_from_integer(self, rows):
        return Vector.fast(rows, int)

    def pop(self, key, *args, **kwargs):
        value = super().pop(key, *args, **kwargs)
        if hasattr(self, key):
            if not self.__is_builtin_attr(key):
                super().__delattr__(key)
        return value

    def popitem(self):
        key, value = super().popitem()
        if hasattr(self, key):
            if not self.__is_builtin_attr(key):
                super().__delattr__(key)
        return key, value

    def print_(self, *, max_rows=None, max_width=None, truncate_width=None):
        """
        Print data frame to ``sys.stdout``.

        `print_` does the same as calling Python's builtin ``print`` function,
        but since it's a method, you can use it at the end of a method chain
        instead of wrapping a ``print`` call around the whole chain.

        >>> di.read_csv("data/listings.csv").print_()
        """
        print(self.to_string(max_rows=max_rows,
                             max_width=max_width,
                             truncate_width=truncate_width))

    def print_memory_use(self):
        """
        Print memory use by column and total.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.print_memory_use()
        """
        mem = DataFrame()
        for name, column in self.items():
            new = DataFrame(column=name)
            new.dtype = str(column.dtype)
            new.item_size = column.itemsize
            new.total_size = column.get_memory_use()
            mem = mem.rbind(new)
        new = DataFrame(column="TOTAL")
        new.dtype = "--"
        new.item_size = mem.item_size.sum()
        new.total_size = mem.total_size.sum()
        mem = mem.rbind(new)
        # Format sizes into sensible values for display.
        mem.item_size = [f"{x:.0f} B" for x in mem.item_size]
        mem.total_size = [f"{x/1024**2:,.0f} MB" for x in mem.total_size]
        mem.colnames = [x.upper() for x in mem.colnames]
        print(mem)

    def print_na_counts(self):
        """
        Print counts of missing values by column.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.print_na_counts()
        """
        nas = DataFrame()
        for name in self.colnames:
            n = self[name].is_na().sum()
            if n == 0: continue
            nas = nas.rbind(DataFrame(column=name, nna=n))
        if not nas: return
        nas.pna = [f"{100*x/self.nrow:.1f}%" for x in nas.nna]
        nas.colnames = [x.upper() for x in nas.colnames]
        print(nas)

    @deco.new_from_generator
    def rbind(self, *others):
        """
        Return data frame with rows from `others` added.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.rbind(data)
        """
        data_frames = [self] + list(others)
        colnames = util.unique_keys(itertools.chain(*data_frames))
        def get_part(data, colname):
            if colname in data:
                return data[colname]
            for ref in data_frames:
                if colname not in ref: continue
                value = ref[colname].na_value
                dtype = ref[colname].na_dtype
                return Vector.fast([value], dtype).repeat(data.nrow)
        for colname in colnames:
            parts = [get_part(x, colname) for x in data_frames]
            total = DataFrameColumn(np.concatenate(parts))
            yield colname, total

    @classmethod
    def read_csv(cls, path, *, encoding="utf-8", sep=",", header=True, columns=[], strings_as_object=inf, dtypes={}):
        """
        Return a new data frame from CSV file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz``.

        `columns` is an optional list of columns to limit to.

        `strings_as_object` is a cutoff point. If any row has more characters
        than that, the whole column will use the object data type. This is
        intended to help limit memory use as NumPy strings are fixed-length and
        can take a huge amount of memory if even a single row is long. If set,
        `dtypes` overrides this.

        `dtypes` is an optional dict mapping column names to NumPy datatypes.
        """
        import pandas as pd
        data = pd.read_csv(path,
                           sep=sep,
                           header=0 if header else None,
                           usecols=columns or None,
                           dtype=dtypes,
                           parse_dates=False,
                           encoding=encoding,
                           low_memory=False)

        if not header:
            data.columns = util.generate_colnames(len(data.columns))
        return cls.from_pandas(data, strings_as_object=strings_as_object, dtypes=dtypes)

    @classmethod
    def read_json(cls, path, *, encoding="utf-8", columns=[], dtypes={}, **kwargs):
        """
        Return a new data frame from JSON file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz``.

        `columns` is an optional list of columns to limit to. `dtypes` is an
        optional dict mapping column names to NumPy datatypes. `kwargs` are
        passed to ``json.load``.
        """
        with util.xopen(path, "rt", encoding=encoding) as f:
            return cls.from_json(f.read(), columns=columns, dtypes=dtypes, **kwargs)

    @classmethod
    def read_npz(cls, path, *, allow_pickle=True):
        """
        Return a new data frame from NumPy file `path`.

        See `numpy.load` for an explanation of `allow_pickle`:
        https://numpy.org/doc/stable/reference/generated/numpy.load.html
        """
        with np.load(path, allow_pickle=allow_pickle) as data:
            return cls(**data)

    @classmethod
    def read_parquet(cls, path, *, columns=[], strings_as_object=inf, dtypes={}):
        """
        Return a new data frame from Parquet file `path`.

        `columns` is an optional list of columns to limit to.

        `strings_as_object` is a cutoff point. If any row has more characters
        than that, the whole column will use the object data type. This is
        intended to help limit memory use as NumPy strings are fixed-length and
        can take a huge amount of memory if even a single row is long. If set,
        `dtypes` overrides this.

        `dtypes` is an optional dict mapping column names to NumPy datatypes.
        """
        import pyarrow.parquet as pq
        columns = columns or None
        data = pq.read_table(path, columns=columns)
        return cls.from_arrow(data, strings_as_object=strings_as_object, dtypes=dtypes)

    @classmethod
    def read_pickle(cls, path):
        """
        Return a new data frame from Pickle file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz``.
        """
        with util.xopen(path, "rb") as f:
            return cls(pickle.load(f))

    def _reconcile_column(self, column):
        if isinstance(column, DataFrameColumn):
            if column.nrow == self.nrow:
                return column
        nrow = self.nrow if self else None
        return DataFrameColumn(column, nrow=nrow)

    @deco.new_from_generator
    def rename(self, **to_from_pairs):
        """
        Return data frame with columns renamed.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.rename(listing_id="id")
        """
        from_to_pairs = {v: k for k, v in to_from_pairs.items()}
        for fm in self.colnames:
            to = from_to_pairs.get(fm, fm)
            yield to, self[fm].copy()

    def sample(self, n=None):
        """
        Return randomly chosen `n` rows.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.sample(5)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ROWS
        n = min(self.nrow, n)
        rows = np.random.choice(self.nrow, n, replace=False)
        return self.slice(np.sort(rows))

    @deco.new_from_generator
    def select(self, *colnames):
        """
        Return data frame, keeping only `colnames`.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.select("id", "hood", "zipcode")
        """
        for colname in colnames:
            yield colname, self[colname].copy()

    @deco.new_from_generator
    def semi_join(self, other, *by):
        """
        Return rows with matches in `other`.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        >>> # All listings that have reviews
        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.semi_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        other = other.unique(*by2)
        found, src = self._get_join_indices(other, by1, by2)
        for colname, column in self.items():
            yield colname, column[found].copy()

    @deco.new_from_generator
    def slice(self, rows=None, cols=None):
        """
        Return a row-wise and/or column-wise subset of data frame.

        Both `rows` and `cols` should be integer vectors correspoding to the
        indices of the rows or columns to keep.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.slice(rows=[0, 1, 2])
        >>> data.slice(cols=[0, 1, 2])
        >>> data.slice(rows=[0, 1, 2], cols=[0, 1, 2])
        """
        rows = np.arange(self.nrow) if rows is None else rows
        cols = np.arange(self.ncol) if cols is None else cols
        rows = self._parse_rows_from_integer(rows)
        cols = self._parse_cols_from_integer(cols)
        for colname in (self.colnames[x] for x in cols):
            yield colname, self[colname][rows].copy()

    @deco.new_from_generator
    def slice_off(self, rows=None, cols=None):
        """
        Return a row-wise and/or column-wise negative subset of data frame.

        Both `rows` and `cols` should be integer vectors correspoding to the
        indices of the rows or columns to drop.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.slice_off(rows=[0, 1, 2])
        >>> data.slice_off(cols=[0, 1, 2])
        >>> data.slice_off(rows=[0, 1, 2], cols=[0, 1, 2])
        """
        rows = [] if rows is None else rows
        cols = [] if cols is None else cols
        rows = self._parse_rows_from_integer(rows)
        cols = self._parse_cols_from_integer(cols)
        for i, colname in enumerate(self.colnames):
            if i in cols: continue
            yield colname, np.delete(self[colname], rows)

    @deco.new_from_generator
    def sort(self, **colname_dir_pairs):
        """
        Return rows in sorted order.

        `colname_dir_pairs` defines the sort order by column name with `dir`
        being ``1`` for ascending sort, ``-1`` for descending.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.sort(hood=1, zipcode=1)
        """
        @deco.tuplefy
        def sort_key():
            pairs = colname_dir_pairs.items()
            for colname, dir in reversed(list(pairs)):
                if dir not in [1, -1]:
                    raise ValueError("dir should be 1 or -1")
                column = self[colname]
                if column.is_object():
                    # Avoid TypeError trying to compare different types.
                    column = column.as_string()
                if dir < 0 and not (column.is_boolean() or column.is_number()):
                    # Use rank for non-numeric types so that we can sort descending.
                    column = column.rank(method="min")
                yield column if dir >= 0 else -column
        indices = np.lexsort(sort_key())
        for colname, column in self.items():
            yield colname, column[indices].copy()

    def split(self, *by):
        """
        Split data frame into groups and return a list of their rows.

        >>> data = di.DataFrame(x=[1, 2, 2, 3, 3, 3])
        >>> data.split("x")
        """
        data = self.select(*by)
        data._index_ = np.arange(data.nrow)
        data = data.sort(**dict.fromkeys(by, 1))
        data._sorted_index_ = np.arange(data.nrow)
        stat = data.unique(*by)
        return np.split(data._index_, stat._sorted_index_[1:])

    def _split_join_by(self, *by):
        by1 = [x if isinstance(x, str) else x[0] for x in by]
        by2 = [x if isinstance(x, str) else x[1] for x in by]
        return by1, by2

    def tail(self, n=None):
        """
        Return the last `n` rows.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.tail(5)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ROWS
        n = min(self.nrow, n)
        return self.slice(np.arange(self.nrow - n, self.nrow))

    def to_arrow(self):
        """
        Return data frame converted to a ``pyarrow.Table``.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.to_arrow()
        """
        import pyarrow as pa
        data = [pa.array(self[x].tolist()) for x in self.colnames]
        return pa.table(data, names=self.colnames)

    def to_json(self, **kwargs):
        """
        Return data frame converted to a JSON string.

        `kwargs` are passed to ``json.dump``.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.to_json()[:100]
        """
        return self.to_list_of_dicts().to_json(**kwargs)

    def to_list_of_dicts(self):
        """
        Return data frame converted to a :class:`.ListOfDicts`.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.to_list_of_dicts()
        """
        from dataiter import ListOfDicts
        data = [{} for i in range(self.nrow)]
        for colname in self.colnames:
            for i, value in enumerate(self[colname].tolist()):
                data[i][colname] = value
        return ListOfDicts(data)

    def to_pandas(self):
        """
        Return data frame converted to a ``pandas.DataFrame``.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.to_pandas()
        """
        import pandas as pd
        return pd.DataFrame({x: self[x].tolist() for x in self.colnames})

    def to_string(self, *, max_rows=None, max_width=None, truncate_width=None):
        """
        Return data frame as a string formatted for display.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.to_string()
        """
        if not self: return ""
        max_rows = max_rows or dataiter.PRINT_MAX_ROWS
        max_width = max_width or util.get_print_width()
        truncate_width = truncate_width or dataiter.PRINT_TRUNCATE_WIDTH
        n = min(self.nrow, max_rows)
        columns = {colname: util.pad(
            [colname] +
            [str(column.dtype)] +
            [str(x) for x in column[:n].to_strings(
                quote=False, pad=True, truncate_width=truncate_width)]
        ) for colname, column in self.items()}
        for column in columns.values():
            column.insert(2, "─" * util.ulen(column[0]))
        row_numbers = [str(i) for i in range(n)]
        row_numbers = util.pad(["", "", ""] + row_numbers)
        # If the length of rows exceeds max_width, split to
        # batches of columns (like R's print.data.frame).
        rows_to_print = []
        while columns:
            first = next(iter(columns.keys()))
            batch_rows = [" ".join(x) for x in zip(
                row_numbers, columns.pop(first))]
            for colname, column in list(columns.items()):
                width = util.ulen(batch_rows[0] + column[0]) + 1
                if width > max_width: break
                for i in range(len(column)):
                    batch_rows[i] += " "
                    batch_rows[i] += column[i]
                del columns[colname]
            rows_to_print.append("" if rows_to_print else ".")
            rows_to_print += batch_rows
        rows_to_print.append(".")
        if max_rows < self.nrow:
            rows_to_print.append(f"... {self.nrow} rows total")
        return "\n".join(rows_to_print)

    @deco.new_from_generator
    def unique(self, *colnames):
        """
        Return unique rows by `colnames`.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.unique("hood")
        """
        colnames = colnames or self.colnames
        if (len(colnames) == 1 and
            not self[colnames[0]].is_object()):
            # Use a single column directly.
            by = self[colnames[0]]
        elif (len(set(self[x].dtype for x in colnames)) == 1 and
              not self[colnames[0]].is_object()):
            # Stack matching dtypes directly in a new array.
            by = np.column_stack([self[x] for x in colnames])
        else:
            # Use rank for differing dtypes.
            by = np.column_stack([self[x].rank(method="min") for x in colnames])
        indices = np.sort(np.unique(by, return_index=True, axis=0)[1])
        for colname, column in self.items():
            yield colname, column[indices].copy()

    @deco.new_from_generator
    def unselect(self, *colnames):
        """
        Return data frame, dropping `colnames`.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.unselect("guests", "sqft", "price")
        """
        for colname in self.colnames:
            if colname not in colnames:
                yield colname, self[colname].copy()

    @deco.new_from_generator
    def update(self, other):
        """
        Return data frame with columns from `other` added.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.update(di.DataFrame(x=1))
        """
        for colname, column in self.items():
            if colname in other: continue
            yield colname, column.copy()
        for colname, column in other.items():
            column = self._reconcile_column(column)
            yield colname, column.copy()

    def _view_rows(self, rows):
        # Initialize a blank instance and use base class update
        # to bypass __init__ and __setitem__ checks for speed.
        data = self.__class__()
        dict.update(data, {x: self[x][rows] for x in self})
        return data

    def write_csv(self, path, *, encoding="utf-8", header=True, sep=","):
        """
        Write data frame to CSV file `path`.

        Will automatically compress if `path` ends in ``.bz2|.gz|.xz``.
        """
        data = self.to_pandas()
        util.makedirs_for_file(path)
        data.to_csv(path, sep=sep, header=header, index=False, encoding=encoding)

    def write_json(self, path, *, encoding="utf-8", **kwargs):
        """
        Write data frame to JSON file `path`.

        Will automatically compress if `path` ends in ``.bz2|.gz|.xz``.

        `kwargs` are passed to ``json.JSONEncoder``.
        """
        return self.to_list_of_dicts().write_json(path, encoding=encoding, **kwargs)

    def write_npz(self, path, *, compress=False):
        """
        Write data frame to NumPy file `path`.
        """
        util.makedirs_for_file(path)
        savez = np.savez_compressed if compress else np.savez
        savez(path, **self)

    def write_parquet(self, path, **kwargs):
        """
        Write data frame to Parquet file `path`.

        `kwargs` are passed to ``pyarrow.parquet.write_table``.
        """
        import pyarrow.parquet as pq
        data = self.to_arrow()
        util.makedirs_for_file(path)
        pq.write_table(data, path, **kwargs)

    def write_pickle(self, path):
        """
        Write data frame to Pickle file `path`.

        Will automatically compress if `path` ends in ``.bz2|.gz|.xz``.
        """
        util.makedirs_for_file(path)
        with util.xopen(path, "wb") as f:
            out = {k: np.array(v, v.dtype) for k, v in self.items()}
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
