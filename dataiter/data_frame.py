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

import dataiter
import itertools
import json
import numpy as np
import pandas as pd

from dataiter import Array
from dataiter import deco
from dataiter import util


class DataFrameColumn(Array):

    def __new__(cls, object, dtype=None, nrow=None):
        object = [object] if np.isscalar(object) else object
        column = Array(object, dtype)
        if nrow is not None and nrow != column.size:
            if not (column.size == 1 and nrow > 1):
                raise ValueError("Incompatible object and nrow for broadcast")
            column = column.repeat(nrow)
        return column.view(cls)

    def __init__(self, object, dtype=None, nrow=None):
        self.__check_dimensions()

    def __check_dimensions(self):
        if self.ndim == 1: return
        raise ValueError(f"Bad dimensions: {self.ndim!r}")

    @property
    def nrow(self):
        self.__check_dimensions()
        return self.size


class DataFrame(dict):

    ATTRIBUTES = ["_group_colnames"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nrow = max(util.length(x) for x in self.values())
        for key, value in self.items():
            if not isinstance(value, DataFrameColumn) or value.nrow != nrow:
                super().__setitem__(key, DataFrameColumn(value, nrow=nrow))
        # Check that the above broadcasting produced a uniform table.
        self.__check_dimensions()
        self._group_colnames = []

    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo=None):
        return self.__class__({k: v.copy() for k, v in self.items()})

    @deco.translate_error(KeyError, AttributeError)
    def __delattr__(self, name):
        if name in self.ATTRIBUTES:
            return super().__delattr__(name)
        return self.__delitem__(name)

    def __eq__(self, other):
        return (isinstance(other, DataFrame) and
                self.nrow == other.nrow and
                self.ncol == other.ncol and
                set(self.colnames) == set(other.colnames) and
                all(self[x].equal(other[x]) for x in self))

    @deco.translate_error(KeyError, AttributeError)
    def __getattr__(self, name):
        if name in self.ATTRIBUTES:
            return super().__getattr__(name)
        return self.__getitem__(name)

    def __setattr__(self, name, value):
        if name in self.ATTRIBUTES:
            return super().__setattr__(name, value)
        return self.__setitem__(name, value)

    def __setitem__(self, key, value):
        value = self._reconcile_column(value)
        return super().__setitem__(key, value)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        rows = [[""] + self.colnames]
        rows.append([""] + [str(x.dtype) for x in self.columns])
        for i in range(min(self.nrow, dataiter.PRINT_MAX_ROWS)):
            rows.append([str(i)] + [util.np_to_string(x[i]) for x in self.columns])
        for i in range(len(rows[0])):
            width = max(len(x[i]) for x in rows)
            for row in rows:
                padding = width - len(row[i])
                row[i] = " " * padding + row[i]
        # If the length of rows exceeds PRINT_MAX_WIDTH, split to
        # batches of columns (like R's print.data.frame).
        rows_to_print = []
        while rows[0]:
            batch_column_count = 0
            for i in range(len(rows[0])):
                text = " ".join(rows[0][:(i+1)])
                if len(text) <= dataiter.PRINT_MAX_WIDTH:
                    batch_column_count = i + 1
            batch_rows = [""] * len(rows)
            for i, row in enumerate(rows):
                batch_rows[i] += " ".join(row[:batch_column_count])
                del row[:batch_column_count]
            rows_to_print.extend(batch_rows)
        return "\n".join(rows_to_print)

    def aggregate(self, **colname_function_pairs):
        by = np.column_stack(tuple(self[x].astype(bytes) for x in self._group_colnames))
        values, ui, inv = np.unique(by, return_index=True, return_inverse=True, axis=0)
        stat = self.slice(ui).select(*self._group_colnames)
        slice_indices = {}
        for i in range(self.nrow):
            slice_indices.setdefault(inv[i], []).append(i)
        slices = [self.slice(slice_indices[i]) for i in range(stat.nrow)]
        for colname, function in colname_function_pairs.items():
            stat[colname] = [function(x) for x in slices]
        return stat.sort(**dict.fromkeys(self._group_colnames, 1))

    @deco.new_from_generator
    def anti_join(self, other, *by):
        other = other.unique(*by)
        found, src = self._get_join_indices(other, *by)
        for colname, column in self.items():
            yield colname, np.delete(column, found)

    @deco.new_from_generator
    def cbind(self, *others):
        data_frames = [self] + list(others)
        colnames = list(itertools.chain(*data_frames))
        colnames = util.make_unique_names(colnames)
        for data in data_frames:
            for column in data.values():
                column = self._reconcile_column(column)
                yield colnames.pop(0), column.copy()

    def __check_dimensions(self):
        nrows = [x.nrow for x in self.columns]
        if len(set(nrows)) == 1: return
        raise ValueError(f"Bad dimensions: {nrows!r}")

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

    @deco.new_from_generator
    def filter(self, rows):
        rows = self._parse_rows_boolean(rows)
        for colname, column in self.items():
            yield colname, np.take(column, rows)

    @deco.new_from_generator
    def filter_out(self, rows):
        rows = self._parse_rows_boolean(rows)
        for colname, column in self.items():
            yield colname, np.delete(column, rows)

    @classmethod
    def from_json(cls, string, **kwargs):
        obj = json.loads(string, **kwargs)
        if not isinstance(obj, list):
            raise TypeError("Not a list")
        keys = util.unique_keys(itertools.chain(*obj))
        columns = {k: [x.get(k, None) for x in obj] for k in keys}
        return cls(**columns)

    @classmethod
    def from_pandas(cls, data):
        return cls(**{x: data[x].to_numpy().tolist() for x in data.columns})

    def full_join(self, other, *by):
        other = other.modify(_id_=np.arange(other.nrow))
        a = self.left_join(other, *by)
        b = other.filter_out(np.isin(other._id_, a._id_))
        return a.rbind(b).unselect("_id_")

    def _get_join_indices(self, other, *by):
        other_ids = list(zip(*[other[x] for x in by]))
        other_by_id = {other_ids[i]: i for i in range(other.nrow)}
        self_ids = zip(*[self[x] for x in by])
        src = map(lambda x: other_by_id.get(x, -1), self_ids)
        src = np.fromiter(src, np.int, count=self.nrow)
        found = np.where(src > -1)
        return found, src

    def group_by(self, *colnames):
        self._group_colnames = colnames[:]
        return self

    def head(self, n=None):
        n = n or dataiter.DEFAULT_HEAD_TAIL_ROWS
        return self.slice(list(range(n)))

    @deco.new_from_generator
    def inner_join(self, other, *by):
        other = other.unique(*by)
        found, src = self._get_join_indices(other, *by)
        for colname, column in self.items():
            yield colname, column[found].copy()
        for colname, column in other.items():
            if colname in self: continue
            yield colname, column[src[found]].copy()

    @deco.new_from_generator
    def left_join(self, other, *by):
        other = other.unique(*by)
        found, src = self._get_join_indices(other, *by)
        for colname, column in self.items():
            yield colname, column.copy()
        for colname, column in other.items():
            if colname in self: continue
            # NaN not allowed in integer column, use float instead.
            dtype = np.float if column.is_integer else column.dtype
            new = DataFrameColumn(np.nan, dtype=dtype, nrow=self.nrow)
            new[found] = column[src[found]]
            yield colname, new.copy()

    @deco.new_from_generator
    def modify(self, function=None, **colname_value_pairs):
        for colname, column in self.items():
            yield colname, column.copy()
        for colname, value in colname_value_pairs.items():
            value = value(self) if callable(value) else value
            yield colname, self._reconcile_column(value).copy()

    @property
    def ncol(self):
        return len(self)

    def _new(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    @property
    def nrow(self):
        if not self: return 0
        self.__check_dimensions()
        return self[next(iter(self))].nrow

    def _parse_cols_boolean(self, cols):
        cols = Array(cols)
        assert cols.is_boolean
        assert len(cols) == self.ncol
        cols = Array(np.nonzero(cols)[0])
        assert cols.is_integer
        return cols

    def _parse_cols_integer(self, cols):
        cols = Array(cols)
        assert cols.is_integer
        return cols

    def _parse_rows_boolean(self, rows):
        rows = Array(rows)
        assert rows.is_boolean
        assert len(rows) == self.nrow
        rows = Array(np.nonzero(rows)[0])
        assert rows.is_integer
        return rows

    def _parse_rows_integer(self, rows):
        rows = Array(rows)
        assert rows.is_integer
        return rows

    @deco.new_from_generator
    def rbind(self, *others):
        data_frames = [self] + list(others)
        colnames = util.unique_keys(itertools.chain(*data_frames))
        def get_part(data, colname):
            if colname in data:
                return data[colname]
            return Array(np.nan).repeat(data.nrow)
        for colname in colnames:
            parts = [get_part(x, colname) for x in data_frames]
            total = DataFrameColumn(np.concatenate(parts))
            yield colname, total

    @classmethod
    def read_csv(cls, fname, encoding="utf_8", header=True, sep=","):
        data = pd.read_csv(fname, sep=sep, header=0 if header else None, encoding=encoding)
        data.columns = util.generate_colnames(len(data.columns)) if not header else data.columns
        return cls.from_pandas(data)

    @classmethod
    def read_json(cls, fname, encoding="utf_8", **kwargs):
        with open(fname, "r", encoding=encoding) as f:
            return cls.from_json(f.read(), **kwargs)

    def _reconcile_column(self, column):
        return (DataFrameColumn(column, nrow=(self.nrow if self else None))
                if not isinstance(column, DataFrameColumn) or column.nrow != self.nrow
                else column)

    @deco.new_from_generator
    def rename(self, **to_from_pairs):
        from_to_pairs = {v: k for k, v in to_from_pairs.items()}
        for fm in self.colnames:
            to = from_to_pairs.get(fm, fm)
            yield to, self[fm].copy()

    @deco.new_from_generator
    def select(self, *colnames):
        for colname in colnames:
            yield colname, self[colname].copy()

    @deco.new_from_generator
    def semi_join(self, other, *by):
        other = other.unique(*by)
        found, src = self._get_join_indices(other, *by)
        for colname, column in self.items():
            yield colname, column[found].copy()

    @deco.new_from_generator
    def slice(self, rows=None, cols=None):
        rows = np.arange(self.nrow) if rows is None else rows
        cols = np.arange(self.ncol) if cols is None else cols
        rows = self._parse_rows_integer(rows)
        cols = self._parse_cols_integer(cols)
        for colname in (self.colnames[x] for x in cols):
            yield colname, self[colname][rows].copy()

    @deco.new_from_generator
    def sort(self, **colname_dir_pairs):
        @deco.tuplefy
        def sort_key():
            for colname, dir in reversed(colname_dir_pairs.items()):
                column = self[colname]
                if dir < 0 and not (column.is_boolean or column.is_number):
                    # Use rank for non-numeric types so that we can sort descending.
                    column = column.rank()
                yield column if dir >= 0 else -column
        indices = np.lexsort(sort_key())
        for colname, column in self.items():
            yield colname, column[indices].copy()

    def tail(self, n=None):
        n = n or dataiter.DEFAULT_HEAD_TAIL_ROWS
        return self.slice(list(range(self.nrow - n, self.nrow)))

    def to_json(self, **kwargs):
        return json.dumps(self.to_list_of_dicts(), **kwargs)

    def to_list_of_dicts(self):
        from dataiter import ListOfDicts
        data = [{} for i in range(self.nrow)]
        for colname in self.colnames:
            for i, value in enumerate(self[colname].tolist()):
                data[i][colname] = value
        return ListOfDicts(data)

    def to_pandas(self):
        return pd.DataFrame({x: self[x].tolist() for x in self.colnames})

    @deco.new_from_generator
    def unique(self, *colnames):
        colnames = colnames or self.colnames
        by = np.column_stack(tuple(self[x].astype(bytes) for x in colnames))
        values, indices = np.unique(by, return_index=True, axis=0)
        for colname, column in self.items():
            yield colname, column[indices].copy()

    @deco.new_from_generator
    def unselect(self, *colnames):
        for colname in self.colnames:
            if colname not in colnames:
                yield colname, self[colname].copy()

    @deco.new_from_generator
    def update(self, other):
        for colname, column in self.items():
            if colname in other: continue
            yield colname, column.copy()
        for colname, column in other.items():
            column = self._reconcile_column(column)
            yield colname, column.copy()

    def write_csv(self, fname, encoding="utf_8", header=True, sep=","):
        self.to_pandas().to_csv(fname, sep=sep, header=header, index=False, encoding=encoding)

    def write_json(self, fname, encoding="utf_8", **kwargs):
        return self.to_list_of_dicts().write_json(fname, encoding=encoding, **kwargs)
