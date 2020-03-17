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
        raise ValueError("Bad dimensions: {!r}".format(self.ndim))

    @property
    def nrow(self):
        self.__check_dimensions()
        return self.size


class DataFrame(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nrow = max(util.length(x) for x in self.values())
        for key, value in self.items():
            if not isinstance(value, DataFrameColumn) or value.nrow != nrow:
                super().__setitem__(key, DataFrameColumn(value, nrow=nrow))
        self.__check_dimensions()

    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo=None):
        return self.__class__({k: v.copy() for k, v in self.items()})

    @deco.translate_error(KeyError, AttributeError)
    def __delattr__(self, colname):
        return self.__delitem__(colname)

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.nrow == other.nrow and
                self.ncol == other.ncol and
                set(self.colnames) == set(other.colnames) and
                all(self[x].equal(other[x]) for x in self))

    @deco.translate_error(KeyError, AttributeError)
    def __getattr__(self, colname):
        return self.__getitem__(colname)

    def __setattr__(self, colname, value):
        return self.__setitem__(colname, value)

    def __setitem__(self, key, value):
        nrow = self.nrow if self else None
        value = DataFrameColumn(value, nrow=nrow)
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
        raise NotImplementedError

    def __check_dimensions(self):
        nrows = [x.nrow for x in self.columns]
        if len(set(nrows)) == 1: return
        raise ValueError("Bad dimensions: {!r}".format(nrows))

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

    def filter(self, function=None, **colname_value_pairs):
        raise NotImplementedError

    def filter_out(self, function=None, **colname_value_pairs):
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

    def group_by(self, *colnames):
        raise NotImplementedError

    def head(self, n):
        return self.slice(list(range(n)))

    def join(self, other, *by):
        raise NotImplementedError

    @property
    def ncol(self):
        return len(self)

    def _new(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    @property
    def nrow(self):
        if not self: return 0
        return self[next(iter(self))].nrow

    def _parse_cols_argument(self, cols):
        if cols is None:
            cols = np.arange(self.ncol)
        if np.isscalar(cols):
            cols = [cols]
        cols = Array(cols)
        if cols.is_boolean:
            assert len(cols) == self.ncol
            cols = Array(cols.nonzero()[0])
        assert cols.is_integer
        return cols

    def _parse_rows_argument(self, rows):
        if rows is None:
            rows = np.arange(self.nrow)
        if np.isscalar(rows):
            rows = [rows]
        rows = Array(rows)
        if rows.is_boolean:
            assert len(rows) == self.nrow
            rows = Array(rows.nonzero()[0])
        assert rows.is_integer
        return rows

    @classmethod
    def read_csv(cls, fname, encoding="utf_8", header=True, sep=","):
        data = pd.read_csv(fname, sep=sep, header=0 if header else None, encoding=encoding)
        data.columns = util.get_colnames(len(data.columns)) if not header else data.columns
        return cls(**{x: data[x].to_numpy().tolist() for x in data.columns})

    @classmethod
    def read_json(cls, fname, encoding="utf_8", **kwargs):
        with open(fname, "r", encoding=encoding) as f:
            return cls.from_json(f.read(), **kwargs)

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
    def slice(self, rows=None, cols=None):
        rows = self._parse_rows_argument(rows)
        cols = self._parse_cols_argument(cols)
        for colname in (self.colnames[x] for x in cols):
            yield colname, self[colname][rows].copy()

    def sort(self, *colnames, reverse=False):
        raise NotImplementedError

    def tail(self, n):
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

    def unique(self, *colnames):
        raise NotImplementedError

    @deco.new_from_generator
    def unselect(self, *colnames):
        for colname in self.colnames:
            if colname not in colnames:
                yield colname, self[colname].copy()

    def write_csv(self, fname, encoding="utf_8", header=True, sep=","):
        self.to_pandas().to_csv(fname, sep=sep, header=header, index=False, encoding=encoding)

    def write_json(self, fname, encoding="utf_8", **kwargs):
        return self.to_list_of_dicts().write_json(fname, encoding=encoding, **kwargs)
