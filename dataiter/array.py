# -*- coding: utf-8 -*-

# Copyright (c) 2020 Osmo Salomaa
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

import numpy as np

from dataiter import util


class Array(np.ndarray):

    def __new__(cls, object, dtype=None):
        object = [object] if np.isscalar(object) else object
        return np.array(object, dtype).view(cls)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return util.np_to_string(self)

    def equal(self, other):
        # XXX: Not really sure what we want to consider equal.
        if self.is_float and other.is_float:
            return np.allclose(self, other, equal_nan=True)
        return np.array_equal(self, other)

    @property
    def is_boolean(self):
        return np.issubdtype(self.dtype, np.bool_)

    @property
    def is_float(self):
        return np.issubdtype(self.dtype, np.floating)

    @property
    def is_integer(self):
        return np.issubdtype(self.dtype, np.integer)

    @property
    def is_number(self):
        return np.issubdtype(self.dtype, np.number)

    @property
    def is_string(self):
        return np.issubdtype(self.dtype, np.character)
