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
import pickle

from dataiter import deco
from dataiter import util
from dataiter import Vector


class DataFrameColumn(Vector):

    def __new__(cls, object, dtype=None, nrow=None):
        object = [object] if np.isscalar(object) else object
        column = Vector(object, dtype)
        if nrow is not None and nrow != column.length:
            if column.length != 1 or nrow < 1:
                raise ValueError("Bad arguments for broadcast")
            column = column.repeat(nrow)
        return column.view(cls)

    def __init__(self, object, dtype=None, nrow=None):
        super().__init__(object, dtype)

    @property
    def nrow(self):
        return self.length


class DataFrame(dict):

    # List of names that are actual attributes, not columns
    ATTRIBUTES = ["colnames", "_group_colnames"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nrow = max(map(util.length, self.values()), default=0)
        for key, value in self.items():
            if (isinstance(value, DataFrameColumn) and
                value.nrow == nrow): continue
            column = DataFrameColumn(value, nrow=nrow)
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

    def __eq__(self, other):
        return (isinstance(other, DataFrame) and
                self.nrow == other.nrow and
                self.ncol == other.ncol and
                set(self.colnames) == set(other.colnames) and
                all(self[x].equal(other[x]) for x in self))

    def __getattr__(self, name):
        if name in self:
            return self.__getitem__(name)
        return super().__getattr__(name)

    def __setattr__(self, name, value):
        if name in self.ATTRIBUTES:
            return super().__setattr__(name, value)
        return self.__setitem__(name, value)

    def __setitem__(self, key, value):
        value = self._reconcile_column(value)
        return super().__setitem__(key, value)

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    def aggregate(self, **colname_function_pairs):
        by = np.column_stack([self[x].as_bytes() for x in self._group_colnames])
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

    @property
    def colnames(self):
        return list(self)

    @colnames.setter
    def colnames(self, colnames):
        for fm, to in zip(list(self.keys()), colnames):
            self[to] = self.pop(fm)

    @property
    def columns(self):
        return list(self.values())

    def copy(self):
        return self.__copy__()

    def deepcopy(self):
        return self.__deepcopy__()

    @deco.new_from_generator
    def filter(self, rows):
        rows = self._parse_rows_from_boolean(rows)
        for colname, column in self.items():
            yield colname, np.take(column, rows)

    @deco.new_from_generator
    def filter_out(self, rows):
        rows = self._parse_rows_from_boolean(rows)
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
        # It would be a lot faster to skip tolist here,
        # but then we'd need to map some Pandas dtype oddities.
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
        self._group_colnames = tuple(colnames)
        return self

    def head(self, n=None):
        if n is None:
            n = dataiter.DEFAULT_PEEK_ROWS
        n = min(self.nrow, n)
        return self.slice(np.arange(n))

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
            value = column.missing_value
            dtype = column.missing_dtype
            new = DataFrameColumn(value, dtype, self.nrow)
            new[found] = column[src[found]]
            yield colname, new.copy()

    @deco.new_from_generator
    def modify(self, **colname_value_pairs):
        for colname, column in self.items():
            yield colname, column.copy()
        for colname, value in colname_value_pairs.items():
            value = value(self) if callable(value) else value
            yield colname, self._reconcile_column(value).copy()

    @property
    def ncol(self):
        self._check_dimensions()
        return len(self)

    def _new(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    @property
    def nrow(self):
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

    def print_(self, max_rows=None, max_width=None):
        print(self.to_string(max_rows, max_width))

    @deco.new_from_generator
    def rbind(self, *others):
        data_frames = [self] + list(others)
        colnames = util.unique_keys(itertools.chain(*data_frames))
        def get_part(data, colname):
            if colname in data:
                return data[colname]
            for ref in data_frames:
                if colname not in ref: continue
                value = ref[colname].missing_value
                dtype = ref[colname].missing_dtype
                return Vector.fast([value], dtype).repeat(data.nrow)
        for colname in colnames:
            parts = [get_part(x, colname) for x in data_frames]
            total = DataFrameColumn(np.concatenate(parts))
            yield colname, total

    @classmethod
    def read_csv(cls, fname, encoding="utf_8", header=True, sep=","):
        import pandas as pd
        data = pd.read_csv(fname,
                           sep=sep,
                           header=0 if header else None,
                           parse_dates=False,
                           encoding=encoding)

        if not header:
            data.columns = util.generate_colnames(len(data.columns))
        return cls.from_pandas(data)

    @classmethod
    def read_json(cls, fname, encoding="utf_8", **kwargs):
        with open(fname, "r", encoding=encoding) as f:
            return cls.from_json(f.read(), **kwargs)

    @classmethod
    def read_pickle(cls, fname):
        with open(fname, "rb") as f:
            return cls(pickle.load(f))

    def _reconcile_column(self, column):
        if isinstance(column, DataFrameColumn):
            if column.nrow == self.nrow:
                return column
        nrow = self.nrow if self else None
        return DataFrameColumn(column, nrow=nrow)

    @deco.new_from_generator
    def rename(self, **to_from_pairs):
        from_to_pairs = {v: k for k, v in to_from_pairs.items()}
        for fm in self.colnames:
            to = from_to_pairs.get(fm, fm)
            yield to, self[fm].copy()

    def sample(self, n=None):
        if n is None:
            n = dataiter.DEFAULT_PEEK_ROWS
        n = min(self.nrow, n)
        rows = np.random.choice(self.nrow, n, replace=False)
        return self.slice(np.sort(rows))

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
        rows = self._parse_rows_from_integer(rows)
        cols = self._parse_cols_from_integer(cols)
        for colname in (self.colnames[x] for x in cols):
            yield colname, self[colname][rows].copy()

    @deco.new_from_generator
    def sort(self, **colname_dir_pairs):
        @deco.tuplefy
        def sort_key():
            pairs = colname_dir_pairs.items()
            for colname, dir in reversed(list(pairs)):
                if dir not in [1, -1]:
                    raise ValueError("dir should be 1 or -1")
                column = self[colname]
                if dir < 0 and not (column.is_boolean or column.is_number):
                    # Use rank for non-numeric types so that we can sort descending.
                    column = column.rank()
                yield column if dir >= 0 else -column
        indices = np.lexsort(sort_key())
        for colname, column in self.items():
            yield colname, column[indices].copy()

    def tail(self, n=None):
        if n is None:
            n = dataiter.DEFAULT_PEEK_ROWS
        n = min(self.nrow, n)
        return self.slice(np.arange(self.nrow - n, self.nrow))

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
        import pandas as pd
        return pd.DataFrame({x: self[x].tolist() for x in self.colnames})

    def to_string(self, max_rows=None, max_width=None):
        if not self: return ""
        max_rows = dataiter.PRINT_MAX_ROWS if max_rows is None else max_rows
        max_width = dataiter.PRINT_MAX_WIDTH if max_width is None else max_width
        n = min(self.nrow, max_rows)
        columns = {colname: util.pad(
            [colname] +
            [str(column.dtype)] +
            column[:n].to_strings(quote=False, pad=True).tolist()
        ) for colname, column in self.items()}
        for column in columns.values():
            column.insert(2, "-" * len(column[0]))
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
                width = len(batch_rows[0] + column[0]) + 1
                if width > max_width: break
                for i in range(len(column)):
                    batch_rows[i] += " "
                    batch_rows[i] += column[i]
                del columns[colname]
            rows_to_print.append("")
            rows_to_print += batch_rows
        rows_to_print.append("")
        if max_rows < self.nrow:
            rows_to_print.append(f"... {self.nrow} rows total")
        return "\n".join(rows_to_print)

    @deco.new_from_generator
    def unique(self, *colnames):
        colnames = colnames or self.colnames
        by = np.column_stack([self[x].as_bytes() for x in colnames])
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

    def write_pickle(self, fname):
        with open(fname, "wb") as f:
            out = {k: np.array(v, v.dtype) for k, v in self.items()}
            pickle.dump(out, f, pickle.HIGHEST_PROTOCOL)
