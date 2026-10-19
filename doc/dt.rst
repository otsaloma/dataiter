dataiter.dt
===========

.. meta::
   :description: Vectorized functions for dates and datetimes

The ``dt`` module contains vectorized functions for dealing with dates and
datetimes, similar to ``numpy.strings`` for strings. Most functions use NumPy
datetime arithmetic and are fast. :func:`~dataiter.dt.from_string`,
:func:`~dataiter.dt.to_string` and :func:`~dataiter.dt.replace` use Python's
standard library ``datetime`` module, calling it once per unique value where
possible, which is fast when values repeat, but slow for many unique values.

:func:`~dataiter.dt.day`
:func:`~dataiter.dt.floor`
:func:`~dataiter.dt.from_components`
:func:`~dataiter.dt.from_string`
:func:`~dataiter.dt.hour`
:func:`~dataiter.dt.isoformat`
:func:`~dataiter.dt.isoweek`
:func:`~dataiter.dt.isoweekday`
:func:`~dataiter.dt.microsecond`
:func:`~dataiter.dt.minute`
:func:`~dataiter.dt.month`
:func:`~dataiter.dt.new`
:func:`~dataiter.dt.now`
:func:`~dataiter.dt.quarter`
:func:`~dataiter.dt.replace`
:func:`~dataiter.dt.second`
:func:`~dataiter.dt.to_string`
:func:`~dataiter.dt.today`
:func:`~dataiter.dt.total_seconds`
:func:`~dataiter.dt.weekday`
:func:`~dataiter.dt.year`

.. automodule:: dataiter.dt
   :members:
