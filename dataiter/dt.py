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
import functools
import numpy as np
import re

from dataiter import dtypes
from dataiter import util
from dataiter import Vector
from numpy.dtypes import StringDType

def _vectorized_datetime(function):
    # Call function with a vector of one for a scalar x.
    @functools.wraps(function)
    def wrapper(x, *args, **kwargs):
        if util.is_scalar(x):
            return function(Vector([x], np.datetime64), *args, **kwargs)[0]
        if not (isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.datetime64)):
            raise TypeError(f"Expected datetime, got {_type_label(x)}")
        return function(x, *args, **kwargs)
    return wrapper

def day(x):
    """
    Extract day of the month from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.day(x)
    """
    return _pull_int(x, lambda y: (
        y.astype("datetime64[D]") -
        y.astype("datetime64[M]").astype("datetime64[D]")
    ).astype(int) + 1)

@_vectorized_datetime
def floor(x, unit):
    """
    Round datetime `x` down to the start of `unit`.

    `unit` can be "year", "quarter", "month", "week", "day", "hour", "minute"
    or "second", in singular or plural or abbreviated to any unambiguous
    prefix, e.g. "mo" for month and "min" for minute. Weeks start on
    Monday, as in ISO 8601. The data type of `x` is retained.

    >>> x = dt.new(["2022-10-15", "2022-11-20"])
    >>> dt.floor(x, "month")
    """
    unit = _parse_unit(unit)
    if unit == "quarter":
        out = x.astype("datetime64[M]")
        out -= (out.astype(int) % 3).astype("timedelta64[M]")
    elif unit == "week":
        # 1970-01-01 was a Thursday.
        out = x.astype("datetime64[D]")
        out -= ((out.astype(int) + 3) % 7).astype("timedelta64[D]")
    else:
        codes = dict(year="Y", month="M", day="D", hour="h", minute="m", second="s")
        out = x.astype(f"datetime64[{codes[unit]}]")
    return out.astype(x.dtype).view(Vector)

def from_components(year, month=1, day=1, hour=None, minute=None, second=None, microsecond=None, *, errors="strict"):
    """
    Initialize a date or datetime scalar or vector from components.

    Components can be scalars or vectors, e.g. columns of a data frame. If
    any of `hour`, `minute`, `second` or `microsecond` is given, return
    datetimes, otherwise dates. Missing values in components give missing
    values.

    `errors` determines how to handle invalid components, e.g. February 30:
    ``"strict"`` to raise ``ValueError``, ``"replace"`` to replace them
    with missing values.

    >>> dt.from_components(2022, 10, 15)
    >>> dt.from_components(2022, 10, 15, 12, 30)
    >>> dt.from_components(di.Vector([2022, 2023]), 6, 30)
    """
    if errors not in ["strict", "replace"]:
        raise ValueError(f"Bad {errors=}")
    components = dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=microsecond)
    components = {k: v for k, v in components.items() if v is not None}
    if all(map(util.is_scalar, components.values())):
        components = {k: [v] for k, v in components.items()}
        return from_components(**components, errors=errors)[0]
    is_datetime = len(components) > 3
    components = dict(zip(components, np.broadcast_arrays(*[np.asarray(x, float) for x in components.values()])))
    na = np.zeros(len(components["year"]), bool)
    valid = np.ones_like(na)
    for x in components.values():
        na |= np.isnan(x)
        valid &= x == np.floor(x)
    get = lambda k: np.where(na | ~valid, 0, components.get(k, 0)).astype(np.int64)
    y, mo, d, h, mi, s, us = map(get, ["year", "month", "day", "hour", "minute", "second", "microsecond"])
    months = ((y - 1970) * 12 + mo - 1).astype("datetime64[M]")
    out = months.astype("datetime64[D]") + (d - 1).astype("timedelta64[D]")
    # Day past the end of the month overflows to the next month.
    valid &= (mo >= 1) & (mo <= 12) & (d >= 1) & (out.astype("datetime64[M]") == months)
    if is_datetime:
        valid &= (h >= 0) & (h <= 23) & (mi >= 0) & (mi <= 59) & (s >= 0) & (s <= 59) & (us >= 0) & (us <= 999999)
        out = out.astype("datetime64[us]") + (((h * 60 + mi) * 60 + s) * 1_000_000 + us).astype("timedelta64[us]")
    invalid = ~na & ~valid
    if errors == "strict" and invalid.any():
        i = np.flatnonzero(invalid)[0]
        example = ", ".join(f"{k}={v[i]:g}" for k, v in components.items())
        raise ValueError(f"Invalid components, e.g. {example}")
    out[na | invalid] = np.datetime64("NaT", "us" if is_datetime else "D")
    return out.view(Vector)

def from_string(x, format, *, errors="strict"):
    """
    Initialize a datetime scalar or vector from `x`.

    `format` uses Python ``strptime`` format codes:
    https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

    `errors` determines how to handle strings that can't be parsed:
    ``"strict"`` to raise ``ValueError``, ``"replace"`` to replace them
    with missing values.

    >>> x = di.Vector(["15.10.2022", "32.10.2022"])
    >>> dt.from_string(x, "%d.%m.%Y", errors="replace")
    """
    if errors not in ["strict", "replace"]:
        raise ValueError(f"Bad {errors=}")
    if util.is_scalar(x):
        x = Vector([x], str)
        return from_string(x, format, errors=errors)[0]
    if not (isinstance(x, np.ndarray) and isinstance(x.dtype, StringDType)):
        raise TypeError(f"Expected string, got {_type_label(x)}")
    out = np.full_like(x, None, object)
    out = out.view(Vector).as_datetime()
    na = x == dtypes.string.na_object
    if na.all(): return out
    def parse(x):
        try:
            return datetime.datetime.strptime(x, format)
        except ValueError:
            if errors == "strict": raise
            return None
    unique, _, codes = util.unique_codes(x[~na])
    f = np.vectorize(parse, otypes=[object])
    out[~na] = f(unique.astype(object)).astype(out.dtype)[codes]
    valid = ~out.is_na()
    if valid.any() and (out[valid] == out[valid].as_date()).all():
        out = out.as_date()
    return out

def hour(x):
    """
    Extract hour from datetime `x`.

    >>> x = dt.new(["2022-10-15T12:34:56"])
    >>> dt.hour(x)
    """
    return _pull_int(x, lambda y: (
        y.astype("datetime64[h]") -
        y.astype("datetime64[D]")
    ).astype(int))

@_vectorized_datetime
def isoformat(x):
    """
    Return datetime `x` as a string in ISO 8601 format.

    For dates, return `YYYY-MM-DD`, for datetimes `YYYY-MM-DDTHH:MM:SS`.
    If any element has fractional seconds, all elements get them with as
    many digits as needed to show all exactly, e.g. `.500` for milliseconds.

    >>> x = dt.new([dt.today()])
    >>> dt.isoformat(x)
    >>> y = dt.new([dt.now()])
    >>> dt.isoformat(y)
    """
    na = np.isnat(x)
    unit = "D"
    if np.datetime_data(x.dtype)[0] not in ["Y", "M", "W", "D"]:
        values = x[~na]
        unit = next(u for u in ["s", "ms", "us", "ns"] if np.all(values.astype(f"datetime64[{u}]") == values))
    if util.is_repetitive(x):
        # Formatting only unique values is faster if values repeat.
        unique, _, codes = util.unique_codes(x)
        out = Vector.fast(np.datetime_as_string(unique, unit=unit), dtypes.string)[codes]
    else:
        out = Vector.fast(np.datetime_as_string(x, unit=unit), dtypes.string)
    out[na] = dtypes.string.na_object
    return out

def isoweek(x):
    """
    Extract ISO 8601 week from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.isoweek(x)
    """
    def f(y):
        # ISO week is the week of the Thursday of the same Monday-start week.
        y = y.astype("datetime64[D]")
        thursday = y - ((y.astype(int) + 3) % 7).astype("timedelta64[D]") + np.timedelta64(3, "D")
        return (thursday - thursday.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) // 7 + 1
    return _pull_int(x, f)

def isoweekday(x):
    """
    Extract day of the week from datetime `x`.

    Day of the week is an integer between 1 and 7, where 1 is Monday and 7 is
    Sunday.

    See also: :func:`weekday`

    >>> x = dt.new(["2022-10-15"])
    >>> dt.isoweekday(x)
    """
    return weekday(x) + 1

def microsecond(x):
    """
    Extract microsecond from datetime `x`.

    >>> x = dt.new(["2022-10-15T12:34:56.789"])
    >>> dt.microsecond(x)
    """
    return _pull_int(x, lambda y: (
        y.astype("datetime64[us]") -
        y.astype("datetime64[s]")
    ).astype(int))

def minute(x):
    """
    Extract minute from datetime `x`.

    >>> x = dt.new(["2022-10-15T12:34:56"])
    >>> dt.minute(x)
    """
    return _pull_int(x, lambda y: (
        y.astype("datetime64[m]") -
        y.astype("datetime64[h]")
    ).astype(int))

def month(x):
    """
    Extract month from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.month(x)
    """
    return _pull_int(x, lambda y: y.astype("datetime64[M]").astype(int) % 12 + 1)

def new(x):
    """
    Initialize a datetime scalar or vector from `x`.

    >>> dt.new("2022-10-15")
    >>> dt.new("2022-10-15T12:00:00")
    >>> dt.new(["2022-10-15"])
    >>> dt.new(["2022-10-15T12:00:00"])
    """
    if util.is_scalar(x):
        return np.datetime64(x)
    return Vector.fast(map(np.datetime64, x), np.datetime64)

def now():
    """
    Return the current local datetime.

    >>> dt.now()
    """
    return np.datetime64(datetime.datetime.now())

def _parse_unit(unit):
    units = ["year", "quarter", "month", "week", "day", "hour", "minute", "second"]
    found = [x for x in units if f"{x}s".startswith(unit)] if unit else []
    if len(found) > 1:
        raise ValueError(f"Ambiguous {unit=}, could be {' or '.join(found)}")
    if not found:
        raise ValueError(f"Bad {unit=}")
    return found[0]

@_vectorized_datetime
def _pull_int(x, function):
    out = np.full_like(x, np.nan, float)
    out = out.view(Vector)
    na = np.isnat(x)
    if na.all(): return out
    out[~na] = function(np.asarray(x[~na]))
    return out if na.any() else out.as_integer()

@_vectorized_datetime
def _pull_str(x, function):
    out = np.full_like(x, dtypes.string.na_object, dtypes.string)
    out = out.view(Vector)
    na = np.isnat(x)
    if na.all(): return out
    # Hashing is faster than np.unique, which sorts.
    y = x[~na]
    unique, _, codes = util.unique_codes(y)
    f = np.vectorize(function, otypes=[dtypes.string])
    out[~na] = f(_to_object(unique))[codes]
    return out

def quarter(x):
    """
    Extract quarter from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.quarter(x)
    """
    y = np.ceil(month(x) / 3)
    return y if np.isnan(y).any() else y.astype(int)

@_vectorized_datetime
def replace(x, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None):
    """
    Return datetime `x` with given components replaced.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.replace(x, month=1, day=1)
    """
    given = dict(year=year, month=month, day=day, hour=hour, minute=minute, second=second, microsecond=microsecond)
    given = {k: v for k, v in given.items() if v is not None}
    keys = ["year", "month", "day"]
    is_date = np.datetime_data(x.dtype)[0] in ["Y", "M", "W", "D"]
    if not is_date:
        keys += ["hour", "minute", "second", "microsecond"]
    # Arguments shadow the component functions of the same names.
    components = {k: globals()[k](x) for k in keys} | given
    out = from_components(**components)
    return out if is_date and len(components) > 3 else out.astype(x.dtype)

def second(x):
    """
    Extract second from datetime `x`.

    >>> x = dt.new(["2022-10-15T12:34:56"])
    >>> dt.second(x)
    """
    return _pull_int(x, lambda y: (
        y.astype("datetime64[s]") -
        y.astype("datetime64[m]")
    ).astype(int))

def sequence(start, stop, unit):
    """
    Return a vector of dates or datetimes from `start` to `stop`, inclusive.

    `unit` is the step: "year", "quarter", "month", "week", "day", "hour",
    "minute" or "second", in singular or plural or abbreviated to any
    unambiguous prefix, e.g. "mo" for month and "min" for minute. Steps
    of years, quarters and months keep the day of month of `start`, limited
    to the last day of shorter months. The data type is that of `start`, or
    finer if needed for `unit`.

    >>> dt.sequence(dt.new("2026-10-19"), dt.new("2026-10-25"), "day")
    """
    steps = dict(year=(12, "M"), quarter=(3, "M"), month=(1, "M"), week=(7, "D"), day=(1, "D"), hour=(1, "h"), minute=(1, "m"), second=(1, "s"))
    n, code = steps[_parse_unit(unit)]
    start, stop = np.datetime64(start), np.datetime64(stop)
    dtype = np.promote_types(start.dtype, np.dtype(f"datetime64[{code}]"))
    if code == "M":
        x = start.astype(np.promote_types(dtype, np.dtype("datetime64[D]")))
        day = x.astype("datetime64[D]")
        first = day.astype("datetime64[M]")
        months = np.arange(first, stop.astype("datetime64[M]") + np.timedelta64(1, "M"), np.timedelta64(n, "M"))
        last = (months + np.timedelta64(1, "M")).astype("datetime64[D]") - np.timedelta64(1, "D")
        out = np.minimum(months.astype("datetime64[D]") + (day - first.astype("datetime64[D]")), last) + (x - day)
    else:
        step = np.timedelta64(n, code)
        out = np.arange(start.astype(dtype), stop.astype(dtype) + step, step)
    out = out.astype(dtype)
    return out[out <= stop].view(Vector)

def to_string(x, format):
    """
    Format datetime `x` as string.

    `format` uses Python ``strftime`` format codes:
    https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes

    >>> x = dt.new(["2022-10-15"])
    >>> dt.to_string(x, "%d.%m.%Y")
    """
    # Floor to the finest unit used in format to reduce the amount
    # of unique values to format, e.g. to months for "%Y-%m".
    units = dict(year="CYy", month="Bbhm", day="ADFGUVWadegjuwx", hour="HIklp", minute="MR", second="STXcrs")
    codes = re.findall(r"%[-_0^#]?(.)", format.replace("%%", ""))
    if all(any(code in chars for chars in units.values()) for code in codes):
        found = [unit for unit, chars in units.items() if any(code in chars for code in codes)]
        x = floor(x, found[-1] if found else "year")
    return _pull_str(x, lambda x: x.strftime(format))

def _to_object(x):
    # Python's datetime doesn't support nanoseconds,
    # for which astype(object) would give ints.
    if np.datetime_data(x.dtype)[0] == "ns":
        x = x.astype("datetime64[us]")
    return x.astype(object)

def today():
    """
    Return the current local date.

    >>> dt.today()
    """
    return np.datetime64(datetime.date.today())

def total_seconds(x):
    """
    Return timedelta `x` as a float amount of seconds.

    The result doesn't depend on the unit of `x`. Divide by e.g. 3600 or
    86400 to get hours or days. Missing values are returned as NaN.

    >>> x = dt.new(["2022-10-15T12:00:00"]) - dt.new(["2022-10-14T00:00:00"])
    >>> dt.total_seconds(x)
    >>> dt.total_seconds(x) / 86400
    """
    if util.is_scalar(x):
        x = Vector([x], np.timedelta64)
        return total_seconds(x)[0]
    if not (isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.timedelta64)):
        raise TypeError(f"Expected timedelta, got {_type_label(x)}")
    return (x / np.timedelta64(1, "s")).view(Vector)

def _type_label(x):
    return x.view(Vector).dtype_label if isinstance(x, np.ndarray) else type(x).__name__

def weekday(x):
    """
    Extract day of the week from datetime `x`.

    Day of the week is an integer between 0 and 6, where 0 is Monday and 6 is
    Sunday.

    See also: :func:`isoweekday`

    >>> x = dt.new(["2022-10-15"])
    >>> dt.weekday(x)
    """
    # 1970-01-01 was a Thursday.
    return _pull_int(x, lambda y: (y.astype("datetime64[D]").astype(int) + 3) % 7)

def year(x):
    """
    Extract year from datetime `x`.

    >>> x = dt.new(["2022-10-15"])
    >>> dt.year(x)
    """
    return _pull_int(x, lambda y: y.astype("datetime64[Y]").astype(int) + 1970)
