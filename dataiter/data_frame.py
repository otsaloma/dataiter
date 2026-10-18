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

import codecs
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
            if column.length != 1 or nrow < 0:
                raise ValueError(f"Can't broadcast length {column.length} to {nrow} rows")
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
    as ones with spaces, or ones that conflict with methods or attributes of
    the data frame, such as "count", "items" or "pop". Note that assignment,
    e.g. ``data.count = ...``, still creates a column, but ``data.count`` gives
    the method.

    DataFrame does not support indexing directly as the bracket notation is
    used to refer to dict keys, i.e. columns by name. If you want to index the
    whole data frame object, use the method :meth:`slice`. Individual columns
    are indexed the same as NumPy arrays.
    """

    # List of names that are actual attributes, not columns
    ATTRIBUTES = ["colnames", "_group_colnames"]

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
            try:
                column = DataFrameColumn(value, nrow=nrow)
            except ValueError as error:
                raise ValueError(f"Column {key!r}: {error}") from None
            super().__setitem__(key, column)
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

    def __dir__(self):
        # Used for tab completion of column names at a Python shell.
        names = super().__dir__()
        return names + [x for x in self if
                        isinstance(x, str) and
                        x.isidentifier() and
                        x not in names]

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

    def __setattr__(self, name, value):
        if name in self.ATTRIBUTES:
            return super().__setattr__(name, value)
        return self.__setitem__(name, value)

    def __setitem__(self, key, value):
        try:
            value = self._reconcile_column(value)
        except ValueError as error:
            raise ValueError(f"Column {key!r}: {error}") from None
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
        data = self.select(*group_colnames)
        data._index_ = np.arange(self.nrow)
        if group_colnames:
            data = data.sort(**dict.fromkeys(group_colnames, 1))
        starts = data._get_group_starts(*group_colnames)
        stat = data.slice(starts)
        group_aware = [getattr(x, "group_aware", False) for x in colname_function_pairs.values()]
        if any(group_aware):
            # Group-aware functions declare the columns they use,
            # sort only those instead of the whole data frame.
            for function in colname_function_pairs.values():
                for colname in getattr(function, "colnames", []):
                    if colname not in data:
                        data[colname] = self[colname][data._index_]
            n = np.diff(starts, append=data.nrow)
            data._group_ = np.repeat(np.arange(len(starts)), n)
        ordered = None
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
                if ordered is None:
                    # Reorder all columns once to make groups contiguous,
                    # so that slices make cheap NumPy views instead of copies.
                    # Create views lazily, as holding hundreds of thousands
                    # of views alive at once makes garbage collection slow.
                    ordered = self._view_rows(data._index_)
                    ends = np.append(starts[1:], self.nrow)
                    slices = [slice(a, b) for a, b in zip(starts.tolist(), ends.tolist())]
                stat[colname] = [function(ordered._view_rows(x)) for x in slices]
        return stat.unselect("_index_")

    @deco.new_from_generator
    def anti_join(self, other, *by):
        """
        Return rows with no matches in `other`.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        Rows with missing values in `by` never match. Note that for strings,
        missing values are empty strings. If you need those to match, use
        e.g. :meth:`replace_na` before joining.

        >>> # All listings that don't have reviews
        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.anti_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        rows, src = self._get_join_indices(other, by1, by2)
        rows = rows[src == -1]
        for colname, column in self.items():
            yield colname, column[rows]

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
        """"""
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
        changes will terminate once `max_changed` is reached. If no differences
        are found, the data frames are empty.

        Changed values are listed one at a time, which is slow if there are
        many of them. When comparing huge data frames with very little in
        common, consider giving some sensible value for `max_changed`.

        >>> old = di.read_csv("data/vehicles.csv")
        >>> new = old.modify(hwy=lambda x: np.minimum(100, x.hwy))
        >>> added, removed, changed = new.compare(old, "id")
        >>> changed
        """
        if self.unique(*by).nrow < self.nrow:
            raise ValueError(f"self not unique by {by}")
        if other.unique(*by).nrow < other.nrow:
            raise ValueError(f"other not unique by {by}")
        # Joins never match missing values, but here missing identifiers
        # should match, so join by codes that treat them as equal.
        keys = self.select(*by).rbind(other.select(*by))
        keys._index_ = np.arange(keys.nrow)
        keys = keys.sort(**dict.fromkeys(by, 1))
        starts = keys._get_group_starts(*by)
        codes = np.empty(keys.nrow, int)
        codes[keys._index_] = np.repeat(np.arange(len(starts)), np.diff(starts, append=keys.nrow))
        x = self.modify(_i_=range(self.nrow), _k_=codes[:self.nrow])
        y = other.modify(_j_=range(other.nrow), _k_=codes[self.nrow:])
        added = x.anti_join(y, "_k_").unselect("_i_", "_k_")
        removed = y.anti_join(x, "_k_").unselect("_j_", "_k_")
        z = x.inner_join(y.select("_j_", "_k_"), "_k_")
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
                    byrow = {k: x[k][i] for k in by}
                    changed.append(dict(**byrow,
                                        _column_=colname,
                                        _xvalue_=xvalue,
                                        _yvalue_=yvalue))

        added = added if added.nrow > 0 else self.__class__()
        removed = removed if removed.nrow > 0 else self.__class__()
        if not changed:
            return added, removed, self.__class__()
        changed = self.from_json(changed)
        for name in ["column", "xvalue", "yvalue"]:
            if name in changed:
                new = name
                while new in changed: new += "_"
                changed = changed.rename(**{new: name})
            changed = changed.rename(**{name: f"_{name}_"})
        return added, removed, changed

    def copy(self):
        """
        Return a shallow copy.
        """
        return self.__copy__()

    def count(self, *colnames):
        """
        Return row counts grouped by `colnames`.

        The count column is always named "n". If `colnames` include "n",
        that column is renamed by appending underscores, e.g. "n_".

        >>> data = di.read_csv("data/listings.csv")
        >>> data.count("hood")
        """
        stat = self.group_by(*colnames).aggregate(_n_=dataiter.count())
        if (name := "n") in stat:
            while name in stat: name += "_"
            stat = stat.rename(**{name: "n"})
        stat.n = stat.pop("_n_")
        return stat

    def deepcopy(self):
        """
        Return a deep copy.
        """
        return self.__deepcopy__()

    def drop_na(self, *colnames):
        """
        Return data frame without rows that have missing values in `colnames`.

        If `colnames` not given, use all columns.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.drop_na("sqft")
        """
        drop = Vector.fast([False], bool).repeat(self.nrow)
        for colname in colnames or self.colnames:
            drop = drop | self[colname].is_na()
        return self.filter_out(drop)

    @classmethod
    def expand_grid(cls, **colname_value_pairs):
        """
        Return a new data frame with all combinations of values.

        Rows are ordered so that the first column varies the slowest. A common
        use is completing data to include missing combinations by following up
        with a :meth:`left_join`.

        >>> data = di.read_csv("data/listings.csv")
        >>> stat = data.group_by("hood", "guests").aggregate(n=di.count())
        >>> grid = di.DataFrame.expand_grid(hood=stat.hood.unique(), guests=stat.guests.unique())
        >>> grid.left_join(stat, "hood", "guests")
        """
        values = [DataFrameColumn(x) for x in colname_value_pairs.values()]
        indices = np.meshgrid(*[np.arange(len(x)) for x in values], indexing="ij")
        return cls(**{
            colname: value[index.ravel()]
            for colname, value, index in zip(colname_value_pairs, values, indices)
        })

    @deco.new_from_generator
    def explode(self, colname, *, sep=None, dtype=None):
        """
        Return data frame with elements of `colname` split to rows.

        `colname` should refer to a column of sequences, such as lists, or a
        column of strings to split by `sep`. Other columns are repeated for
        each element. Missing values and empty sequences give one row with a
        missing value. To split multiple columns, chain calls, which gives all
        combinations.

        `dtype` is the data type of the elements of a sequence column.
        Providing `dtype` will make exploding faster, otherwise the appropriate
        data type will be guessed by introspecting the elements. Split strings
        are always strings, follow up with :meth:`modify` to strip whitespace
        or convert them as needed.

        For JSON strings, first parse them into lists with :meth:`modify` and
        ``json.loads``.

        >>> data = di.DataFrame(id=[1, 2], x=["a, b", "c"], y=["[1, 2]", "[3]"])
        >>> data.explode("x", sep=",").modify(x=lambda x: x.x.str.strip())
        >>> data.modify(y=lambda x: x.y.map(json.loads)).explode("y", dtype=int)
        """
        column = self[colname]
        if column.is_string():
            if sep is None:
                raise ValueError(f"Column {colname!r} is strings, give sep to split")
            counts, values = self._explode_strings(column, sep)
        else:
            counts, values = self._explode_sequences(column, dtype)
        rows = np.repeat(np.arange(self.nrow), counts)
        for name, column in self.items():
            yield name, values if name == colname else column[rows]

    @staticmethod
    def _explode_strings(column, sep):
        if len(sep) > 1:
            # Joining and splitting all at once below would break pieces
            # apart differently for separators that overlap, e.g. ",,".
            return DataFrame._explode_sequences([x.split(sep) for x in column], str)
        # Avoid creating a list per row, which is slow for many rows.
        counts = np.strings.count(column, sep) + 1
        # Unlike tolist, iteration gives missing values as "", not None.
        values = sep.join(list(column)).split(sep) if len(column) else []
        return counts, Vector.fast(values, str)

    @staticmethod
    def _explode_sequences(column, dtype):
        counts = np.fromiter((0 if x is None else len(x) for x in column), np.int64, len(column))
        if (empty := np.flatnonzero(counts == 0)).size:
            column = column.copy()
            for i in empty:
                column[i] = [None]
            counts[empty] = 1
        values = np.fromiter(itertools.chain.from_iterable(column), object, counts.sum())
        na = np.equal(values, None)
        valid = values[~na]
        if dtype is None and len(set(map(type, valid))) == 1:
            dtype = Vector([valid[0]]).dtype
        valid = Vector.fast(valid, dtype) if dtype is not None else Vector(valid.tolist())
        if not na.any():
            return counts, valid
        values = np.full(len(values), valid.na_value, valid.na_dtype)
        values[~na] = valid
        return counts, Vector.fast(values)

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
        value, or if `value` is a sequence, such as a list, against any of its
        elements. See the example below of equivalent filtering all three ways.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.filter((data.hood == "Manhattan") & (data.guests == 2))
        >>> data.filter(lambda x: (x.hood == "Manhattan") & (x.guests == 2))
        >>> data.filter(hood="Manhattan", guests=2)

        If the data frame is grouped, a `rows` function is applied to
        group-wise subsets of the data frame.

        >>> # Listings priced over the mean of their neighbourhood
        >>> data.group_by("hood").filter(lambda x: x.price > x.price.mean())
        """
        if rows is not None:
            if callable(rows):
                rows = (self._modify_group_wise(_rows_=rows)["_rows_"]
                        if self._group_colnames else rows(self))
        elif colname_value_pairs:
            rows = self._match_pairs(colname_value_pairs)
        rows = self._parse_rows_from_boolean(rows)
        for colname, column in self.items():
            yield colname, np.take(column, rows)

    @deco.new_from_generator
    def filter_out(self, rows=None, **colname_value_pairs):
        """
        Return rows that don't match condition.

        `filter_out` is the inverse of :meth:`filter`, see it for details on
        the arguments, which are the same.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.filter_out(hood="Manhattan")
        """
        if rows is not None:
            if callable(rows):
                rows = (self._modify_group_wise(_rows_=rows)["_rows_"]
                        if self._group_colnames else rows(self))
        elif colname_value_pairs:
            rows = self._match_pairs(colname_value_pairs)
        rows = self._parse_rows_from_boolean(rows)
        for colname, column in self.items():
            yield colname, np.delete(column, rows)

    @classmethod
    @deco.new_from_generator
    def from_arrow(cls, data, *, dtypes={}, timezone="keep"):
        """
        Return a new data frame from ``pyarrow.Table`` `data`.

        `dtypes` is an optional dict mapping column names to NumPy datatypes.
        `timezone` determines how to handle timestamps with a timezone, which
        NumPy datetimes can't hold: ``"keep"`` to keep them in the timezone
        they're stored in, ``"utc"`` to convert them to UTC.
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        if timezone not in ["keep", "utc"]:
            raise ValueError(f"Bad {timezone=}")
        for name in dtypes:
            if name not in data.column_names:
                raise KeyError(name)
        for name, column in zip(data.column_names, data.columns):
            req_dtype = dtypes.get(name, None)
            if (timezone == "keep" and
                pa.types.is_timestamp(column.type) and
                column.type.tz is not None):
                column = pc.local_timestamp(column)
            if pa.types.is_time(column.type):
                # NumPy has no time of day type, use strings instead
                # of an object column of datetime.time.
                digits = {"s": 0, "ms": 3, "us": 6, "ns": 9}[column.type.unit]
                column = column.cast(pa.string())
                # Drop fractional seconds and seconds if all are zero,
                # e.g. "16:06" in a CSV file would become "16:06:00".
                suffixes = [":00"] if digits == 0 else ["." + "0" * digits, ":00"]
                for suffix in suffixes:
                    if pc.all(pc.ends_with(column, suffix)).as_py():
                        column = pc.utf8_slice_codeunits(column, 0, -len(suffix))
            if req_dtype is None and pa.types.is_null(column.type):
                # Arrow types columns with no values as null, use float
                # instead of an object column of None, as for numbers.
                column = column.cast(pa.float64())
            if req_dtype is None and pa.types.is_decimal(column.type):
                # NumPy has no decimal type, use float instead
                # of a slow object column of decimal.Decimal.
                column = column.cast(pa.float64())
            if (req_dtype is None and
                (pa.types.is_list(column.type) or
                 pa.types.is_large_list(column.type) or
                 pa.types.is_fixed_size_list(column.type))):
                # to_numpy would give NumPy arrays as elements,
                # use lists instead, as DataFrame(...) does.
                column = column.to_pylist()
                yield name, DataFrameColumn.fast(np.fromiter(column, object, len(column)), object)
                continue
            if (req_dtype is None and
                (pa.types.is_string(column.type) or
                 pa.types.is_large_string(column.type))):
                # Vector.fast assumes object if the first element is missing i.e. None.
                req_dtype = str
            na = column.is_null(nan_is_null=True).to_numpy()
            if (req_dtype is None and
                (pa.types.is_binary(column.type) or
                 pa.types.is_large_binary(column.type))):
                # Vector.fast assumes object if any element is missing i.e. None.
                column = column.fill_null(b"")
            column = column.to_numpy()
            if (req_dtype is None and
                (np.issubdtype(column.dtype, np.datetime64) or
                 np.issubdtype(column.dtype, np.timedelta64)) and
                np.datetime_data(column.dtype)[0] == "ns"):
                # Python's datetime and timedelta don't support nanoseconds,
                # breaking conversions to object, e.g. in dt and tolist.
                column = column.astype(column.dtype.str.replace("ns", "us"))
            if np.issubdtype(column.dtype, np.object_):
                if req_dtype is None or np.dtype(req_dtype) != np.dtype(object):
                    # Likely to be strings, but not necessarily.
                    # Force type-guessing in Vector.fast.
                    column = column.tolist()
            column = DataFrameColumn.fast(column, req_dtype)
            if na.any():
                if column.dtype != column.na_dtype:
                    # Upcast integer to float if needed.
                    column = column.astype(column.na_dtype)
                column[na] = column.na_value
            yield name, column

    @classmethod
    def from_json(cls, string, *, select=[], dtypes={}, **kwargs):
        """
        Return a new data frame from JSON `string`.

        `select` is an optional list of columns to limit to. `dtypes` is an
        optional dict mapping column names to NumPy datatypes. `kwargs` are
        passed to ``json.load``.
        """
        data = string
        if isinstance(data, str):
            data = json.loads(data, **kwargs)
        if not isinstance(data, list):
            raise TypeError("Not a list")
        keys = util.unique_keys(itertools.chain(*data))
        if select:
            keys = [x for x in keys if x in select]
        data = {k: [x.get(k, None) for x in data] for k in keys}
        for name, dtype in dtypes.items():
            data[name] = DataFrameColumn(data[name], dtype)
        return cls(**data)

    @classmethod
    @deco.new_from_generator
    def from_pandas(cls, data, *, dtypes={}):
        """
        Return a new data frame from ``pandas.DataFrame`` `data`.

        `dtypes` is an optional dict mapping column names to NumPy datatypes.
        """
        for name in dtypes:
            if name not in data.columns:
                raise KeyError(name)
        for name in data.columns:
            req_dtype = dtypes.get(name, None)
            na = data[name].isna().to_numpy(copy=True)
            column = data[name].to_numpy(copy=True)
            if np.issubdtype(column.dtype, np.object_):
                if req_dtype is None or np.dtype(req_dtype) != np.dtype(object):
                    # Likely to be strings, but not necessarily.
                    # Force type-guessing in Vector.fast.
                    column = column.tolist()
            column = DataFrameColumn.fast(column, req_dtype)
            if column.is_object() and req_dtype is None:
                # Vector.fast doesn't recognize e.g. dates,
                # try the slower full type-guessing of Vector.
                column[na] = None
                column = DataFrameColumn(column.tolist())
            if na.any():
                if column.dtype != column.na_dtype:
                    # Upcast integer to float if needed.
                    column = column.astype(column.na_dtype)
                column[na] = column.na_value
            yield name, column

    def full_join(self, other, *by, multiple="first"):
        """
        Return data frame with matching rows merged from `self` and `other`.

        `full_join` keeps all rows from both data frames, merging matching
        ones. If there are multiple matches, `multiple` determines which to
        use: ``"first"``, ``"last"`` or ``"all"``. Rows of `other` not used
        are added after rows of `self`. For rows, for which matches are not
        found or not used, missing values are added.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        Rows with missing values in `by` never match. Note that for strings,
        missing values are empty strings. If you need those to match, use
        e.g. :meth:`replace_na` before joining.

        Columns other than `by` found in both `self` and `other` keep their
        values from `self` for rows found in `self`. Rename such columns before
        joining if you need both.

        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.full_join(reviews, "id")
        """
        b = other.modify(_bid_=np.arange(other.nrow))
        ab = self.left_join(b, *by, multiple=multiple)
        b = b.anti_join(ab, "_bid_").unselect("_bid_")
        ab = ab.unselect("_bid_")
        if b.nrow == 0:
            return ab
        for item in by:
            # For identifiers in by whose name differs in self and b,
            # rename and keep the variant found in self.
            if isinstance(item, (list, tuple)):
                b[item[0]] = b.pop(item[1])
        return ab.rbind(b)

    def _get_group_positions(self):
        # Return row indices in group order, the position
        # of each row within its group and its group size.
        group_colnames = self._group_colnames
        data = self.select(*group_colnames)
        data._index_ = np.arange(self.nrow)
        data = data.sort(**dict.fromkeys(group_colnames, 1))
        starts = data._get_group_starts(*group_colnames)
        sizes = np.diff(starts, append=data.nrow)
        positions = np.arange(data.nrow) - np.repeat(starts, sizes)
        return data._index_, positions, np.repeat(sizes, sizes)

    def _get_group_starts(self, *colnames):
        # Assumes data is sorted by colnames, i.e. groups are contiguous.
        change = np.zeros(self.nrow, bool)
        change[:1] = True
        for colname in colnames:
            column = self[colname]
            na = column.is_na()
            change[1:] |= (column[1:] != column[:-1]) & ~(na[1:] & na[:-1])
        return np.flatnonzero(change)

    def _get_join_codes(self, other, by1, by2):
        # Return int64 codes for rows of self and other, equal for equal
        # keys and -1 for rows that can't match, i.e. missing values in by.
        # Non-numeric columns are encoded together via a dict of tuples,
        # the result of which is combined arithmetically with numeric
        # and datetime columns.
        a_na = np.zeros(self.nrow, bool)
        b_na = np.zeros(other.nrow, bool)
        for colname1, colname2 in zip(by1, by2):
            a_na |= self[colname1].is_na()
            b_na |= other[colname2].is_na()
        numeric, hashable = [], []
        for colname1, colname2 in zip(by1, by2):
            pair = self[colname1][~a_na], other[colname2][~b_na]
            is_numeric = all(x.dtype.kind in "biufcmM" for x in pair)
            (numeric if is_numeric else hashable).append(pair)
        if hashable:
            numeric.append(self._get_join_codes_hashable(hashable))
        a, b = self._get_join_codes_numeric(numeric)
        a_codes = np.full(self.nrow, -1)
        b_codes = np.full(other.nrow, -1)
        a_codes[~a_na] = a
        b_codes[~b_na] = b
        return a_codes, b_codes

    @staticmethod
    def _get_join_codes_hashable(pairs):
        # Return one code per row for a list of pairs of columns, using
        # a dict of values or tuples of values for multiple columns,
        # since np.unique can't sort e.g. mixed types.
        # The dict is built from the smaller side, since inserting is slower
        # than looking up. Codes are row indices of first occurrences in the
        # smaller side. Rows of the larger side not found get a shared code
        # past those indices, which can't match. Calling dict methods via map
        # avoids a slow Python-level function call per row.
        if len(pairs) == 1:
            a, b = map(np.ndarray.tolist, pairs[0])
        else:
            a = zip(*[np.ndarray.tolist(x) for x, y in pairs])
            b = zip(*[np.ndarray.tolist(y) for x, y in pairs])
        na, nb = len(pairs[0][0]), len(pairs[0][1])
        swap = na > nb
        (small, n_small), (large, n_large) = ((b, nb), (a, na)) if swap else ((a, na), (b, nb))
        groups = {}
        small = np.fromiter(map(groups.setdefault, small, itertools.count()), np.int64, n_small)
        large = np.fromiter(map(groups.get, large, itertools.repeat(n_small)), np.int64, n_large)
        return (large, small) if swap else (small, large)

    @staticmethod
    def _get_join_codes_numeric(pairs):
        # Return one combined code per row for a list of pairs of numeric or
        # datetime columns, offsetting integers by their minimum and falling
        # back to np.unique for floats or if the range would overflow.
        def encode(x, dense=False):
            # Return integer codes for x and an upper bound for them.
            if not dense and x.dtype.kind in "biumM" and x.dtype != np.uint64 and len(x) > 0:
                x = x.astype(np.int64)
                lo, hi = int(x.min()), int(x.max())
                if hi - lo < 2**62:
                    return x - lo, hi - lo + 1
            codes = np.unique(x, return_inverse=True)[1]
            return codes, int(codes.max(initial=0)) + 1
        codes, size = 0, 1
        for a, b in pairs:
            column, n = encode(np.concatenate([a, b]))
            if size * n >= 2**62:
                codes, size = encode(codes, dense=True)
                column, n = encode(column, dense=True)
            codes, size = codes * n + column, size * n
        n = len(pairs[0][0])
        return codes[:n], codes[n:]

    def _get_join_indices(self, other, by1, by2, multiple="first"):
        # Return aligned indices of rows in self and their matches in other.
        # Rows of self without a match are included once with -1 in other.
        if multiple not in ["first", "last", "all"]:
            raise ValueError(f"Bad {multiple=}")
        kind = lambda x: "number" if x.dtype.kind in "biufc" else x.dtype.kind
        for colname1, colname2 in zip(by1, by2):
            a, b = self[colname1], other[colname2]
            if a.is_object() or b.is_object(): continue
            if kind(a) != kind(b):
                raise TypeError(f"Can't join {colname1!r} with {colname2!r} due to incompatible "
                                f"types: {a.dtype_label} vs. {b.dtype_label}")
        a, b = self._get_join_codes(other, by1, by2)
        left, right, order = self._get_join_ranges(a, b)
        return self._get_join_matches(left, right, order, multiple)

    @staticmethod
    def _get_join_matches(left, right, order, multiple):
        # Return aligned indices of a and b for the first, last or all matches
        # in the ranges, with rows of a without a match once with -1 in b.
        if multiple == "first":
            right = np.minimum(right, left + 1)
        if multiple == "last":
            left = np.maximum(left, right - 1)
        found = right > left
        n = np.maximum(right - left, 1)
        rows = np.repeat(np.arange(len(left)), n)
        offsets = np.arange(len(rows)) - np.repeat(np.cumsum(n) - n, n)
        pos = np.repeat(left, n) + offsets
        found = np.repeat(found, n)
        src = np.full(len(rows), -1)
        src[found] = order[pos[found]]
        return rows, src

    @staticmethod
    def _get_join_ranges(a, b):
        # Return for each code in a the range left:right of equal codes in b,
        # as positions in order, the stable sort order of b. Codes -1 never
        # match: they are left out of order and get an empty range.
        order = np.flatnonzero(b > -1)
        order = order[np.argsort(b[order], kind="stable")]
        left = np.searchsorted(b[order], a, "left")
        right = np.searchsorted(b[order], a, "right")
        return left, right, order

    def _glob(self, colnames):
        matches = {}
        for colname in colnames:
            found = util.glob(colname, self)
            if not found:
                raise KeyError(colname)
            matches.update(dict.fromkeys(found))
        return list(matches)

    def group_by(self, *colnames):
        """
        Return data frame with `colnames` set for grouped operations.

        Supported grouped operations:

        * :meth:`aggregate`
        * :meth:`filter`
        * :meth:`filter_out`
        * :meth:`head`
        * :meth:`modify`
        * :meth:`tail`
        """
        data = self.copy()
        data._group_colnames = tuple(colnames)
        return data

    def head(self, n=None):
        """
        Return the first `n` rows.

        If `n` is negative, return all rows except the last ``-n``. If the
        data frame is grouped, do the same for each group, ordered by group.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.head(5)
        >>> # The two most expensive listings in each neighbourhood
        >>> data.sort(price=-1).group_by("hood").head(2)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ROWS
        if self._group_colnames:
            rows, positions, sizes = self._get_group_positions()
            return self.slice(rows[positions < (n if n >= 0 else sizes + n)])
        n = min(self.nrow, n) if n >= 0 else max(0, self.nrow + n)
        return self.slice(np.arange(n))

    @deco.new_from_generator
    def inner_join(self, other, *by, multiple="first"):
        """
        Return data frame with matching rows merged from `self` and `other`.

        `inner_join` keeps only rows found in both data frames, merging
        matching ones. If there are multiple matches, `multiple` determines
        which to use: ``"first"``, ``"last"`` or ``"all"``.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        Rows with missing values in `by` never match. Note that for strings,
        missing values are empty strings. If you need those to match, use
        e.g. :meth:`replace_na` before joining.

        Columns other than `by` found in both `self` and `other` keep their
        values from `self`. Rename such columns before joining if you need both.

        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.inner_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        rows, src = self._get_join_indices(other, by1, by2, multiple)
        found = src > -1
        rows, src = rows[found], src[found]
        for colname, column in self.items():
            yield colname, column[rows]
        for colname, column in other.items():
            if colname in by2: continue
            if colname in self: continue
            yield colname, column[src]

    @deco.new_from_generator
    def left_join(self, other, *by, multiple="first"):
        """
        Return data frame with matching rows merged from `self` and `other`.

        `left_join` keeps all rows in `self`, merging matching ones. If there
        are multiple matches, `multiple` determines which to use:
        ``"first"``, ``"last"`` or ``"all"``. For rows, for which matches are
        not found, missing values are added.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        Rows with missing values in `by` never match. Note that for strings,
        missing values are empty strings. If you need those to match, use
        e.g. :meth:`replace_na` before joining.

        Columns other than `by` found in both `self` and `other` keep their
        values from `self`. Rename such columns before joining if you need both.

        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.left_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        rows, src = self._get_join_indices(other, by1, by2, multiple)
        found = np.flatnonzero(src > -1)
        for colname, column in self.items():
            # Copying is faster than indexing, rows is a plain
            # range if no row of self matched multiple in other.
            yield colname, column[rows] if len(rows) > self.nrow else column.copy()
        for colname, column in other.items():
            if colname in by2: continue
            if colname in self: continue
            if len(found) == len(src):
                # No missing values needed, keep data type.
                yield colname, column[src]
                continue
            value = column.na_value
            dtype = column.na_dtype
            new = DataFrameColumn(value, dtype, len(src))
            new[found] = column[src[found]]
            yield colname, new

    def map(self, function):
        """
        Apply `function` to each row in data.

        `function` receives as arguments the full data frame and the loop
        index. The return value will be a list of whatever `function` returns.

        Note that `map` is an inefficient method as it iterates over rows
        instead of doing vectorized computation. `map` is mostly intended for
        complicated conditional cases that are difficult to express in
        vectorized form.

        .. warning:: Deprecated since 1.5, please use a list comprehension
                     instead, e.g. ``[function(data, i) for i in
                     range(data.nrow)]``, or :meth:`to_list_of_dicts` and
                     :meth:`.ListOfDicts.map`.

        >>> data = di.read_csv("data/listings-reviews.csv")
        >>> data.map(lambda x, i: (x.reviews[i], x.rating[i]))[:3]
        """
        if not getattr(DataFrame.map, "warning_shown", False):
            print("Warning: DataFrame.map is deprecated, please use a list comprehension instead")
            print("e.g. [function(data, i) for i in range(data.nrow)]")
            DataFrame.map.warning_shown = True
        return [function(self, i) for i in range(self.nrow)]

    @deco.new_from_generator
    def map_columns(self, predicate, function, *args, **kwargs):
        """
        Return data frame with `function` applied to columns matching `predicate`.

        `predicate` is either a function that receives a column as argument and returns ``True``
        or ``False`` or ``None`` to match all columns. :func:`dataiter.of_type`
        is a shorthand for a `predicate` that checks the data type. `function`
        receives a column and `args` and `kwargs` as arguments and should
        return the new column.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.map_columns(di.of_type(float), lambda x: x.replace_na(0))
        >>> data.map_columns(di.of_type(float), lambda x: x.round(2))
        >>> data.map_columns(di.of_type(float), np.pow, 2)
        """
        for colname, column in self.items():
            if predicate is None or predicate(column):
                yield colname, function(column, *args, **kwargs)
            else:
                yield colname, column.copy()

    def _match_pairs(self, colname_value_pairs):
        rows = Vector.fast([True], bool).repeat(self.nrow)
        for colname, value in colname_value_pairs.items():
            if util.is_scalar(value):
                rows = rows & (self[colname] == value)
            else:
                rows = rows & self[colname].is_in(value)
        return rows

    @deco.new_from_generator
    def modify(self, **colname_value_pairs):
        """
        Return data frame with columns modified.

        In `colname_value_pairs`, `value` can be either a vector or a function
        that receives the data frame as argument and returns a vector. See the
        example below of equivalent modification with both ways. Functions are
        evaluated in order and can use columns created by earlier ones.

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
            yield from self._modify_group_wise(**colname_value_pairs).items()
        else:
            data = self.copy()
            for colname, value in colname_value_pairs.items():
                data[colname] = value(data) if callable(value) else value
                yield colname, data[colname].copy()

    def _modify_group_wise(self, **colname_function_pairs):
        group_colnames = self._group_colnames
        data = self.__class__(self)
        data._index_ = np.arange(data.nrow)
        data = data.sort(**dict.fromkeys(group_colnames, 1))
        restore_indices = np.argsort(data.pop("_index_"))
        starts = data._get_group_starts(*group_colnames)
        ends = np.append(starts[1:], data.nrow)
        # Contiguous slices make cheap NumPy views instead of copies.
        # Create them lazily, as holding hundreds of thousands
        # of views alive at once makes garbage collection slow.
        slices = [slice(a, b) for a, b in zip(starts.tolist(), ends.tolist())]
        # Without rows, call functions once with an empty group
        # to get a column of the correct data type.
        slices = slices or [slice(0, 0)]
        columns = {}
        for colname, function in colname_function_pairs.items():
            if not callable(function):
                raise ValueError(f"{colname} argument not callable")
            column = [DataFrameColumn(function(data._view_rows(x)), nrow=x.stop-x.start) for x in slices]
            data[colname] = np.concatenate(column)
            columns[colname] = data[colname][restore_indices]
        return columns

    @property
    def ncol(self):
        """
        Return the amount of columns.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.ncol
        """
        self._check_dimensions()
        return len(self)

    @classmethod
    def _new(cls, *args, **kwargs):
        return cls(*args, **kwargs)

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

    def _parse_cols(self, cols):
        cols = Vector.fast(cols)
        if cols.is_boolean():
            return self._parse_cols_from_boolean(cols)
        return Vector.fast(cols, int)

    def _parse_cols_from_boolean(self, cols):
        cols = Vector.fast(cols, bool)
        if len(cols) != self.ncol:
            raise ValueError("Bad length for boolean cols")
        return Vector.fast(np.nonzero(cols)[0], int)

    def _parse_rows(self, rows):
        rows = Vector.fast(rows)
        if rows.is_boolean():
            return self._parse_rows_from_boolean(rows)
        return Vector.fast(rows, int)

    def _parse_rows_from_boolean(self, rows):
        rows = Vector.fast(rows, bool)
        if len(rows) != self.nrow:
            raise ValueError("Bad length for boolean rows")
        return Vector.fast(np.nonzero(rows)[0], int)

    def pivot_longer(self, *, ids, names="name", values="value"):
        """
        Pivot data from wide to long format.

        `ids` should be the name or a list of names of identifier columns. All
        other columns are considered variable columns and will be pivoted.
        `names` and `values` are names of columns into which the variable names
        and values are put in the result. If all variable columns are of the
        same kind, e.g. numbers, `values` will have their common data type,
        otherwise object.

        >>> wide = di.read_csv("data/listings.csv")
        >>> wide
        >>> wide.pivot_longer(ids="id")
        """
        ids = util.sequencify(ids)
        if not all(x in self for x in ids):
            raise ValueError(f"Bad {ids=}")
        columns = {x: self[x] for x in self if x not in ids}
        # NumPy would also promote e.g. integer and timedelta to timedelta,
        # so check that all are of the same kind before promoting.
        kind = lambda x: "number" if x.dtype.kind in "biuf" else x.dtype.kind
        if len(set(map(kind, columns.values()))) == 1:
            dtype = functools.reduce(np.promote_types, [x.dtype for x in columns.values()])
            columns = {k: v.astype(dtype) for k, v in columns.items()}
        else:
            columns = {k: v.as_object() for k, v in columns.items()}
        long = self.__class__({x: np.repeat(self[x], len(columns)) for x in ids})
        long[names] = np.tile(Vector.fast(list(columns), str), self.nrow)
        long[values] = np.stack(list(columns.values()), axis=1).ravel()
        return long

    def pivot_wider(self, *, ids=None, names="name", values="value", rename=None):
        """
        Pivot data from long to wide format.

        `ids` should be the name or a list of names of identifier columns by
        which the result will be unique. If `ids` is not given, it defaults to
        all columns except `names` and `values`. `names` should be the name of
        the column that contains the names of variables, which become new
        columns. `values` should be the name of the column that contains the
        corresponding values. `rename` is an optional function that you can use
        to e.g. lowercase the new column names or add a prefix or suffix.

        New columns are ordered by first appearance of names. Sort before
        pivoting if you need e.g. alphabetical order. Missing values in names
        raise ``ValueError``, replace them with :meth:`replace_na` or drop them
        with :meth:`drop_na` before pivoting.

        Each combination of `ids` and `names` should occur only once, otherwise
        ``ValueError`` is raised. If you have multiple values per combination,
        first either :meth:`aggregate` or drop duplicates with :meth:`unique`.

        Combinations of `ids` and `names` not found in the data will be missing
        values in the result, which upcasts integers to float. To fill them in,
        e.g. with zero, and if needed, convert back to integer, follow up with
        e.g. ``map_columns(di.of_type(float), lambda x:
        x.replace_na(0).as_integer())``.

        >>> long = di.read_csv("data/downloads.csv")
        >>> long = long.sort(date=1)
        >>> long
        >>> long.pivot_wider(ids="date", names="category", values="downloads", rename=str.lower)
        """
        if names not in self:
            raise ValueError(f"Bad {names=}")
        if values not in self:
            raise ValueError(f"Bad {values=}")
        if not ids:
            ids = [x for x in self if x not in [names, values]]
        ids = util.sequencify(ids)
        if not all(x in self for x in ids):
            raise ValueError(f"Bad {ids=}")
        if self[names].is_na().any():
            raise ValueError(f"Missing values in {names!r}, either replace them "
                             "with data.replace_na(...) or drop them with data.drop_na(...)")
        # Map rows to rows of wide via sorted groups, since unlike
        # joins, this treats missing values in ids as equal.
        keys = self.select(*ids)
        keys._index_ = np.arange(self.nrow)
        keys = keys.sort(**dict.fromkeys(ids, 1))
        starts = keys._get_group_starts(*ids)
        # Sort is stable, so the first row of each group is its first occurrence.
        firsts = keys._index_[starts]
        rank = np.empty(len(starts), int)
        rank[np.argsort(firsts)] = np.arange(len(starts))
        row = np.empty(self.nrow, int)
        row[keys._index_] = np.repeat(rank, np.diff(starts, append=self.nrow))
        wide = self.select(*ids).slice(firsts.sort())
        unique, first, code = np.unique(self[names], return_index=True, return_inverse=True)
        order = np.lexsort((row, code))
        unique_rows = np.ones(len(order), bool)
        unique_rows[1:] = (code[order][1:] != code[order][:-1]) | (row[order][1:] != row[order][:-1])
        if not unique_rows.all():
            i = order[~unique_rows][0]
            example = {x: self[x][[i]].tolist()[0] for x in [*ids, names]}
            raise ValueError(f"Values not unique by ids and names, e.g. {example}, "
                             "either aggregate with group_by(...).aggregate(...) "
                             "or drop duplicates with data.unique(...)")
        bounds = np.searchsorted(code[order], np.arange(len(unique) + 1))
        column = self[values]
        for i in np.argsort(first):
            indices = order[bounds[i]:bounds[i+1]]
            colname = rename(unique[i]) if rename else str(unique[i])
            if len(indices) == wide.nrow:
                # No missing values needed, keep data type.
                wide[colname] = column[indices]
                continue
            new = DataFrameColumn(column.na_value, column.na_dtype, wide.nrow)
            new[row[indices]] = column[indices]
            wide[colname] = new
        return wide

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
        new.item_size = sum(x.itemsize for x in self.values())
        new.total_size = sum(x.get_memory_use() for x in self.values())
        mem = mem.rbind(new)
        # Format sizes into sensible values for display.
        mem.item_size = [f"{x:.0f} B" for x in mem.item_size]
        mem.total_size = [f"{x/1024**2:,.0f} MB" for x in mem.total_size]
        mem.colnames = [x.upper() for x in mem.colnames]
        mem.print_(max_rows=mem.nrow)

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
        if not nas:
            return print("No missing values")
        nas.pna = [f"{100*x/self.nrow:.1f}%" for x in nas.nna]
        nas.colnames = [x.upper() for x in nas.colnames]
        nas.print_(max_rows=nas.nrow)

    @deco.new_from_generator
    def rbind(self, *others):
        """
        Return data frame with rows from `others` added.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.rbind(data)
        """
        data_frames = [self] + list(others)
        colnames = util.unique_keys(itertools.chain(*data_frames))
        # Data types of empty data frames are often guesses, e.g. float64
        # from aggregating zero groups, use them only if all are empty.
        nonempty = [x for x in data_frames if any(map(len, x.values()))] or data_frames
        def get_part(data, colname):
            if colname in data:
                return data[colname]
            for ref in [*nonempty, *data_frames]:
                if colname not in ref: continue
                value = ref[colname].na_value
                dtype = ref[colname].na_dtype
                return Vector.fast([value], dtype).repeat(data.nrow)
        for colname in colnames:
            parts = [get_part(x, colname) for x in nonempty if colname in x or x.nrow > 0]
            try:
                total = DataFrameColumn(np.concatenate(parts))
            except TypeError:
                dtypes = ", ".join(util.unique_keys(x.dtype_label for x in parts))
                raise TypeError(f"Can't combine column {colname!r} of incompatible types: {dtypes}") from None
            yield colname, total

    @classmethod
    def read_csv(cls, path, *, encoding="utf-8", sep=",", quote='"', header=True, colnames=[], na=[], select=[], dtypes={}):
        """
        Return a new data frame from CSV file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz``.
        `quote` is the character used to quote values, ``None`` to disable
        quoting. `header` is whether the first row contains column names.
        `colnames` is an optional list of column names to use instead of those
        in the header row or, if there's no header row, instead of the default
        "a", "b", "c", etc. `na` is an optional string or list of strings to
        read as missing values, in addition to the defaults: blanks in all
        columns and common markers such as "NA" and "NULL" in non-string
        columns. `select` is an optional list of columns to limit to and
        `dtypes` an optional dict mapping column names to NumPy datatypes,
        both using `colnames` if given. Timestamps with UTC offsets, e.g.
        "2025-01-01T12:00:00+02:00", are converted to UTC.
        """
        import pyarrow as pa
        import pyarrow.compute as pc
        from pyarrow import csv
        na = [na] if isinstance(na, str) else list(na)
        # Strings need to be requested already at parse time,
        # otherwise e.g. leading zeros would be lost to type guessing.
        column_types = {k: pa.string() for k, v in dtypes.items() if Vector.fast([], v).is_string()}
        if not header and not colnames and (select or column_types):
            # Translate our generated column names to pyarrow's f0, f1, ...
            names = list(util.yield_colnames())
            to_arrow = {x: f"f{names.index(x)}" for x in [*select, *column_types] if x in names}
            select = [to_arrow.get(x, x) for x in select]
            column_types = {to_arrow.get(k, k): v for k, v in column_types.items()}
        with util.xopen(path, "rb") as f:
            table = csv.read_csv(
                f,
                read_options=csv.ReadOptions(encoding=encoding,
                                             column_names=colnames,
                                             autogenerate_column_names=not header and not colnames,
                                             # Unlike skip_rows, this counts parsed rows,
                                             # handling line breaks in quoted names.
                                             skip_rows_after_names=int(header and bool(colnames))),
                parse_options=csv.ParseOptions(delimiter=sep, quote_char=quote or False, newlines_in_values=True),
                convert_options=csv.ConvertOptions(include_columns=select,
                                                   column_types=column_types,
                                                   null_values=csv.ConvertOptions().null_values + na))

        if not header and not colnames:
            # Translate pyarrow's f0, f1, ... back to our generated column names.
            indices = [int(x[1:]) for x in table.column_names]
            names = util.generate_colnames(max(indices, default=-1) + 1)
            table = table.rename_columns([names[i] for i in indices])
        for i, column in enumerate(table.columns):
            if na and (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
                # pyarrow's strings_can_be_null would apply also the default
                # null_values to strings, so handle our na values separately.
                missing = pc.is_in(column, pa.array(na, column.type))
                column = pc.if_else(missing, pa.scalar(None, column.type), column)
                table = table.set_column(i, table.column_names[i], column)
        return cls.from_arrow(table, dtypes=dtypes)

    @classmethod
    def read_json(cls, path, *, encoding="utf-8", select=[], dtypes={}, **kwargs):
        """
        Return a new data frame from JSON file `path`.

        Will automatically decompress if `path` ends in ``.bz2|.gz|.xz``.
        `select` is an optional list of columns to limit to. `dtypes` is an
        optional dict mapping column names to NumPy datatypes. `kwargs` are
        passed to ``json.load``.
        """
        with util.xopen(path, "rt", encoding=encoding) as f:
            return cls.from_json(f.read(), select=select, dtypes=dtypes, **kwargs)

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
    def read_parquet(cls, path, *, select=[], dtypes={}, timezone="keep"):
        """
        Return a new data frame from Parquet file `path`.

        `select` is an optional list of columns to limit to. `dtypes` is an
        optional dict mapping column names to NumPy datatypes. `timezone`
        determines how to handle timestamps with a timezone, which NumPy
        datetimes can't hold: ``"keep"`` to keep them in the timezone they're
        stored in, ``"utc"`` to convert them to UTC.
        """
        import pyarrow.parquet as pq
        data = pq.read_table(path, columns=select or None)
        return cls.from_arrow(data, dtypes=dtypes, timezone=timezone)

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
    def rename(self, function=None, /, **to_from_pairs):
        """
        Return data frame with columns renamed.

        `function` is an optional function to apply to all column names,
        e.g. ``str.lower``. `to_from_pairs` are applied after `function`,
        referring to columns by their new names.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.rename(listing_id="id")
        >>> data.rename(str.upper)
        """
        names = {x: function(x) if function else x for x in self}
        for fm in to_from_pairs.values():
            if fm not in names.values():
                raise KeyError(fm)
        from_to_pairs = {v: k for k, v in to_from_pairs.items()}
        names = {k: from_to_pairs.get(v, v) for k, v in names.items()}
        if len(set(names.values())) < len(names):
            raise ValueError(f"Duplicate column names after renaming: {list(names.values())}")
        for fm, to in names.items():
            yield to, self[fm].copy()

    @deco.new_from_generator
    def replace_na(self, **colname_value_pairs):
        """
        Return data frame with missing values replaced.

        In `colname_value_pairs`, missing values in column `colname` are
        replaced with `value`.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.replace_na(sqft=0)
        """
        for colname, column in self.items():
            yield colname, column.copy()
        for colname, value in colname_value_pairs.items():
            yield colname, self[colname].replace_na(value)

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

        `colnames` can include glob patterns, e.g. ``"price_*"``, which
        give matching columns in data frame order. Columns are included
        only once, which allows reordering, e.g. ``select("id", "*")``
        moves "id" first.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.select("id", "hood", "zipcode")
        >>> data.select("id", "*")
        """
        for colname in self._glob(colnames):
            yield colname, self[colname].copy()

    @deco.new_from_generator
    def semi_join(self, other, *by):
        """
        Return rows with matches in `other`.

        `by` are column names, by which to look for matching rows, or tuples of
        column names if the correspoding column name differs between `self` and
        `other`.

        Rows with missing values in `by` never match. Note that for strings,
        missing values are empty strings. If you need those to match, use
        e.g. :meth:`replace_na` before joining.

        >>> # All listings that have reviews
        >>> listings = di.read_csv("data/listings.csv")
        >>> reviews = di.read_csv("data/listings-reviews.csv")
        >>> listings.semi_join(reviews, "id")
        """
        by1, by2 = self._split_join_by(*by)
        rows, src = self._get_join_indices(other, by1, by2)
        rows = rows[src > -1]
        for colname, column in self.items():
            yield colname, column[rows]

    @deco.new_from_generator
    def slice(self, rows=None, cols=None):
        """
        Return a row-wise and/or column-wise subset of data frame.

        Both `rows` and `cols` should be either integer vectors of the indices
        of the rows or columns to keep or boolean vectors with ``True`` for the
        rows or columns to keep.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.slice(rows=[0, 1, 2])
        >>> data.slice(cols=[0, 1, 2])
        >>> data.slice(rows=[0, 1, 2], cols=[0, 1, 2])
        """
        rows = np.arange(self.nrow) if rows is None else rows
        cols = np.arange(self.ncol) if cols is None else cols
        rows = self._parse_rows(rows)
        cols = self._parse_cols(cols)
        for colname in (self.colnames[x] for x in cols):
            yield colname, self[colname][rows]

    @deco.new_from_generator
    def slice_off(self, rows=None, cols=None):
        """
        Return a row-wise and/or column-wise negative subset of data frame.

        Both `rows` and `cols` should be either integer vectors of the indices
        of the rows or columns to drop or boolean vectors with ``True`` for the
        rows or columns to drop.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.slice_off(rows=[0, 1, 2])
        >>> data.slice_off(cols=[0, 1, 2])
        >>> data.slice_off(rows=[0, 1, 2], cols=[0, 1, 2])
        """
        rows = [] if rows is None else rows
        cols = [] if cols is None else cols
        rows = self._parse_rows(rows)
        cols = self._parse_cols(cols)
        for i, colname in enumerate(self.colnames):
            if i in cols: continue
            yield colname, np.delete(self[colname], rows)

    @deco.new_from_generator
    def sort(self, **colname_dir_pairs):
        """
        Return rows in sorted order.

        `colname_dir_pairs` defines the sort order by column name with `dir`
        being ``1`` for ascending sort, ``-1`` for descending. Missing values
        are sorted last, regardless of `dir`.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.sort(hood=1, zipcode=1)
        """
        def sort_keys(colname, dir):
            if dir not in [1, -1]:
                raise ValueError("dir should be 1 or -1")
            column = self[colname]
            na = column.is_na()
            yield column.rank(method="min") if column.is_object() else column, dir
            # Keys are sorted by least significant first,
            # so yielding na after the values takes precedence.
            if na.any(): yield na, 1
        # Chained stable argsorts give the same result as np.lexsort,
        # but are faster, especially for strings.
        indices = None
        for key, dir in itertools.chain.from_iterable(
                sort_keys(*x) for x in reversed(colname_dir_pairs.items())):
            if indices is not None:
                key = key[indices]
            if dir > 0:
                order = np.argsort(key, kind="stable")
            else:
                # Sort reversed to keep ties in their original order.
                order = len(key) - 1 - np.argsort(key[::-1], kind="stable")[::-1]
            indices = order if indices is None else indices[order]
        if indices is None:
            indices = np.arange(self.nrow)
        for colname, column in self.items():
            yield colname, column[indices]

    def split(self, *by):
        """
        Split data frame into groups and return a list of their rows.

        >>> data = di.DataFrame(x=[1, 2, 2, 3, 3, 3])
        >>> data.split("x")
        """
        data = self.select(*by)
        data._index_ = np.arange(data.nrow)
        data = data.sort(**dict.fromkeys(by, 1))
        starts = data._get_group_starts(*by)
        return np.split(data._index_, starts[1:]) if len(starts) > 0 else []

    def _split_join_by(self, *by):
        by1 = [x if isinstance(x, str) else x[0] for x in by]
        by2 = [x if isinstance(x, str) else x[1] for x in by]
        return by1, by2

    def tail(self, n=None):
        """
        Return the last `n` rows.

        If `n` is negative, return all rows except the first ``-n``. If the
        data frame is grouped, do the same for each group, ordered by group.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.tail(5)
        >>> # The two cheapest listings in each neighbourhood
        >>> data.sort(price=-1).group_by("hood").tail(2)
        """
        if n is None:
            n = dataiter.DEFAULT_PEEK_ROWS
        if self._group_colnames:
            rows, positions, sizes = self._get_group_positions()
            return self.slice(rows[positions >= (sizes - n if n >= 0 else -n)])
        n = min(self.nrow, n) if n >= 0 else max(0, self.nrow + n)
        return self.slice(np.arange(self.nrow - n, self.nrow))

    def to_arrow(self):
        """
        Return data frame converted to a ``pyarrow.Table``.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.to_arrow()
        """
        import pyarrow as pa
        data = []
        for column in self.columns:
            na = np.asarray(column.is_na())
            try:
                data.append(pa.array(np.asarray(column), mask=na))
            except pa.ArrowNotImplementedError:
                # PyArrow < 26 doesn't support StringDType.
                data.append(pa.array(column.astype(object), mask=na))
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
        return self.to_arrow().to_pandas()

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
        columns = {colname: util.upad(
            [colname] +
            [str(column.dtype_label)] +
            [str(x) for x in column[:n].to_strings(
                quote=False, pad=True, truncate_width=truncate_width)]
        ) for colname, column in self.items()}
        for column in columns.values():
            column.insert(2, "─" * util.ulen(column[0]))
        row_numbers = [str(i) for i in range(n)]
        row_numbers = util.upad(["", "", ""] + row_numbers)
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
            rows_to_print.append("" if rows_to_print else "")
            rows_to_print += batch_rows
        rows_to_print.append("")
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
        columns = [self[x] for x in colnames]
        if all(x.is_number() or
               x.is_boolean() or
               x.is_datetime() or
               x.is_timedelta() for x in columns):
            # np.lexsort is stable, so the first row of each group
            # in sorted order is the first occurrence in self.
            order = np.lexsort(columns[::-1])
            starts = self.select(*colnames).slice(order)._get_group_starts(*colnames)
            keep = np.sort(order[starts])
            for colname, column in self.items():
                yield colname, column[keep]
            return
        # For strings and objects, a Python set is faster than sorting.
        # We just need to avoid NaN and NaT for the 'in' checks to work.
        for i, column in enumerate(columns):
            if not column.is_na().any(): continue
            if column.is_datetime():
                flag = np.nanmin(column) - np.timedelta64(1, "D")
                if np.isnat(flag):
                    flag = np.datetime64("-001-01-01")
                columns[i] = column.replace_na(flag)
            if column.is_timedelta():
                unit = np.datetime_data(column.dtype)[0]
                flag = np.nanmin(column) - np.timedelta64(1, unit)
                if np.isnat(flag):
                    flag = np.timedelta64(0, unit)
                columns[i] = column.replace_na(flag)
            if column.is_float():
                flag = np.nanmin(column) - 1
                if np.isnan(flag):
                    flag = -1
                columns[i] = column.replace_na(flag)
        seen = set()
        keep = []
        for i, row in enumerate(zip(*columns)):
            if row not in seen:
                seen.add(row)
                keep.append(i)
        for colname, column in self.items():
            yield colname, column[keep]

    @deco.new_from_generator
    def unselect(self, *colnames):
        """
        Return data frame, dropping `colnames`.

        `colnames` can include glob patterns, e.g. ``"price_*"``.

        >>> data = di.read_csv("data/listings.csv")
        >>> data.unselect("guests", "sqft", "price")
        """
        colnames = set(self._glob(colnames))
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
        from pyarrow import csv
        data = self.copy()
        for colname, column in data.items():
            if column.is_timedelta():
                # Arrow would write integers with no indication of unit,
                # write as printed instead, e.g. "1 day 04:44:39".
                strings = column.to_strings(quote=False)
                strings[column.is_na()] = ""
                data[colname] = strings
        table = data.to_arrow()
        util.makedirs_for_file(path)
        with util.xopen(path, "wb") as f:
            csv.write_csv(table, f, write_options=csv.WriteOptions(include_header=header,
                                                                   delimiter=sep,
                                                                   quoting_style="needed"))

        # XXX: Arrow's WriteOptions doesn't include encoding.
        # If user requested something else than UTF-8, we need to rewrite!
        if codecs.lookup(encoding) != codecs.lookup("utf-8"):
            with util.xopen(path, "rt", encoding="utf-8") as f: text = f.read()
            with util.xopen(path, "wt", encoding=encoding) as f: f.write(text)

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
