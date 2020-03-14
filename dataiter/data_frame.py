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
import pandas as pd

from dataiter import DataFrameColumn
from dataiter import deco
from dataiter import util


class DataFrame(dict):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if not isinstance(value, DataFrameColumn):
                super().__setitem__(key, DataFrameColumn(value))

    def __copy__(self):
        return self.__class__(self)

    def __deepcopy__(self, memo=None):
        return self.__class__({k: v.copy() for k, v in self.items()})

    @deco.translate_error(KeyError, AttributeError)
    def __delattr__(self, colname):
        return self.__delitem__(colname)

    @deco.translate_error(KeyError, AttributeError)
    def __getattr__(self, colname):
        return self.__getitem__(colname)

    def __setattr__(self, colname, value):
        return self.__setitem__(colname, value)

    def __setitem__(self, key, value):
        nrow = self.nrow if self else None
        value = DataFrameColumn(value, nrow=nrow)
        return super().__setitem__(key, value)

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
    def read_csv(cls, fname, encoding="utf_8", header=True, sep=","):
        # XXX: Use Pandas for now as we don't have type guessing.
        data = pd.read_csv(fname, sep=sep, header=0 if header else None, encoding=encoding)
        if not header:
            data.columns = util.get_colnames(len(data.columns))
        return cls(**{x: data[x].to_numpy().tolist() for x in data.columns})

    @classmethod
    def read_json(cls, fname, encoding="utf_8", **kwargs):
        with open(fname, "r", encoding=encoding) as f:
            return cls.from_json(f.read(), **kwargs)

    def rename(self, **to_from_pairs):
        raise NotImplementedError

    def select(self, *colnames):
        raise NotImplementedError

    def sort(self, *colnames, reverse=False):
        raise NotImplementedError

    def to_json(self, **kwargs):
        raise NotImplementedError

    def unique(self, *colnames):
        raise NotImplementedError

    def unselect(self, *colnames):
        raise NotImplementedError

    def write_csv(self, fname, encoding="utf_8", header=True, sep=","):
        raise NotImplementedError

    def write_json(self, fname, encoding="utf_8", **kwargs):
        raise NotImplementedError
