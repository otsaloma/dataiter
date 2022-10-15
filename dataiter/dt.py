# -*- coding: utf-8 -*-

# Copyright (c) 2022 Osmo Salomaa
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

import datetime
import numpy as np

from dataiter import Vector


def day(x):
    """
    Extract day of the month from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.day(x)
    """
    return _pull_int(x, lambda y: y.day)

def hour(x):
    """
    Extract hour from datetime `x`.

    >>> x = dt.new(["2022-10-15T12:34:56"])
    >>> dt.hour(x)
    """
    return _pull_int(x, lambda y: y.hour)

def isoweek(x):
    """
    Extract ISO 8601 week from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.isoweek(x)
    """
    return _pull_int(x, lambda y: y.isocalendar().week)

def isoweekday(x):
    """
    Extract day of the week from datetime `x`.

    Day of the week is an integer between 1 and 7, where 1 is Monday and 7 is
    Sunday.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.isoweekday(x)
    """
    return _pull_int(x, lambda y: y.isoweekday())

def microsecond(x):
    """
    Extract microsecond from datetime `x`.

    >>> x = dt.new(["2022-10-15T12:34:56.789"])
    >>> dt.microsecond(x)
    """
    return _pull_int(x, lambda y: y.microsecond)

def minute(x):
    """
    Extract minute from datetime `x`.

    >>> x = dt.new(["2022-10-15T12:34:56"])
    >>> dt.minute(x)
    """
    return _pull_int(x, lambda y: y.minute)

def month(x):
    """
    Extract month from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.month(x)
    """
    return _pull_int(x, lambda y: y.month)

def new(x):
    """
    Initialize a datetime vector from `x`.

    >>> dt.new(["2022-10-15"])
    >>> dt.new(["2022-10-15T12:00:00"])
    """
    return Vector.fast(map(np.datetime64, x))

def now():
    """
    Return the current local datetime.

    >>> dt.now()
    """
    return np.datetime64(datetime.datetime.now())

def _pull_datetime(x, function):
    assert isinstance(x, Vector)
    assert x.is_datetime()
    out = np.full_like(x, np.nan)
    out = Vector.fast(out)
    na = x.is_na()
    f = np.vectorize(function)
    out[~na] = f(x[~na].as_object())
    return out

def _pull_int(x, function):
    assert isinstance(x, Vector)
    assert x.is_datetime()
    out = np.full_like(x, np.nan, float)
    out = Vector.fast(out)
    na = x.is_na()
    f = np.vectorize(function)
    out[~na] = f(x[~na].as_object())
    return out if na.any() else out.as_integer()

def quarter(x):
    """
    Extract quarter from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.quarter(x)
    """
    return np.ceil(month(x) / 3)

def replace(x, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None):
    """
    Return datetime `x` with given components replaced.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.replace(x, month=1, day=1)
    """
    kwargs = {k: v for k, v in {
        "year": year,
        "month": month,
        "day": day,
        "hour": hour,
        "minute": minute,
        "second": second,
        "microsecond": microsecond,
    }.items() if v is not None}
    return _pull_datetime(x, lambda y: y.replace(**kwargs))

def second(x):
    """
    Extract second from datetime `x`.

    >>> x = dt.new(["2022-10-15T12:34:56"])
    >>> dt.second(x)
    """
    return _pull_int(x, lambda y: y.second)

def today():
    """
    Return the current local date.

    >>> dt.today()
    """
    return np.datetime64(datetime.date.today())

def weekday(x):
    """
    Extract day of the week from datetime `x`.

    Day of the week is an integer between 0 and 6, where 0 is Monday and 6 is
    Sunday.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.weekday(x)
    """
    return _pull_int(x, lambda y: y.weekday())

def year(x):
    """
    Extract year from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.year(x)
    """
    return _pull_int(x, lambda y: y.year)
