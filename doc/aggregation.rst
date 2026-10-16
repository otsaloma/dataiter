Aggregation
===========

.. meta::
   :description: Guide to grouping and aggregating data frames

.. note:: The following applies currently only to the
          :class:`.DataFrame` class. Aggregation with a
          :class:`.ListOfDicts` is simpler and covered by the
          API-documentation on :meth:`.ListOfDicts.aggregate`.

By aggregation, we refer to splitting a data frame into groups based on
the values of one or more columns and then calculating group-wise
summaries, such total count or mean of a column. The first step is
called ``group_by`` and the second ``aggregate``, usually written via
method chaining as ``data.group_by(...).aggregate(...)``.

A simple example below of how to calculate the total count and mean
price of AirBnb listings in New York grouped by neighbourhood. The
``aggregate`` method takes keyword arguments of the function to be used
to calculate the summary and the name of the column for that summary in
the output. The return value is a regular data frame. See the following
sections for what kinds of aggregation functions you can use.

>>> import dataiter as di
>>> data = di.read_csv("data/listings.csv")
>>> data.group_by("hood").aggregate(n=di.count(), price=di.mean("price"))
.
           hood     n   price
         string int64 float64
  ───────────── ───── ───────
0         Bronx  1198  90.176
1      Brooklyn 19931 125.056
2     Manhattan 21963 218.855
3        Queens  6068  99.745
4 Staten Island   370 116.908

Common Aggregation Functions
----------------------------

Dataiter includes ready functions for the most common summaries that you
might want to calculate. These are technically function factories, i.e.
they are functions that return functions, that will then be called
group-wise within the ``aggregate`` method. For example,
``di.mean("price")`` returns a function, that given a data frame,
returns the group-wise mean of the "price" column. The supported
functions are listed below.

* :func:`~dataiter.all`
* :func:`~dataiter.any`
* :func:`~dataiter.count`
* :func:`~dataiter.count_unique`
* :func:`~dataiter.first`
* :func:`~dataiter.last`
* :func:`~dataiter.max`
* :func:`~dataiter.mean`
* :func:`~dataiter.median`
* :func:`~dataiter.min`
* :func:`~dataiter.mode`
* :func:`~dataiter.nth`
* :func:`~dataiter.quantile`
* :func:`~dataiter.std`
* :func:`~dataiter.sum`
* :func:`~dataiter.var`

These common aggregation functions are provided for two reasons: (1)
they provide shorter, more convenient syntax than typing out lambda
functions and (2) they allow a huge conditional speed up under the hood.
The relevant caveat here is that they work only for single column
calculations. If you need to use multiple columns, such as for
calculating a weighted mean, see the next section on using arbitrary
lambda functions. And see the last section on when and how you can
benefit from the huge speed ups that these functions provide.

Arbitrary Aggregation
---------------------

If you need to access multiple columns in aggregation or you need to
calculate some more esoteric summaries than what you can accomplish with
the above, then you'll need to use custom lambda functions. These
functions should take a data frame as an argument and return a scalar
value. The ``aggregate`` method will then apply your lambda functions
group-wise.

Repeating the example up top, below is how you'd do the same with lambda
functions. Notice that the code needed is a bit more verbose and if you
try this with a data frame that has a large amount of groups (around
100,000 or more), you'll notice that it gets a bit slow, but for more
common sizes of input, it should be well usable.

>>> import dataiter as di
>>> data = di.read_csv("data/listings.csv")
>>> data.group_by("hood").aggregate(n=lambda x: x.nrow, price=lambda x: x.price.mean())
.
           hood     n   price
         string int64 float64
  ───────────── ───── ───────
0         Bronx  1198  90.176
1      Brooklyn 19931 125.056
2     Manhattan 21963 218.855
3        Queens  6068  99.745
4 Staten Island   370 116.908

Conditional Aggregation
-----------------------

A common reason to reach for lambda functions is calculating summaries
of rows that match a condition, such as the count or share of listings
priced over 200 dollars.

>>> import dataiter as di
>>> data = di.read_csv("data/listings.csv")
>>> data.group_by("hood").aggregate(n=di.count(),
...                                 n_expensive=lambda x: (x.price > 200).sum(),
...                                 p_expensive=lambda x: (x.price > 200).mean())
.
           hood     n n_expensive p_expensive
         string int64       int64     float64
  ───────────── ───── ─────────── ───────────
0         Bronx  1198          49    0.040902
1      Brooklyn 19931        2047    0.102704
2     Manhattan 21963        5494    0.250148
3        Queens  6068         326    0.053724
4 Staten Island   370          20    0.054054

You can usually do the same faster by first calculating the condition
as a boolean column and then using the common aggregation functions:
:func:`~dataiter.sum` for the count, :func:`~dataiter.mean` for the
share or :func:`~dataiter.any` for whether there are any matches. With
a large amount of groups, this can be over ten times faster.

>>> (data
...  .modify(expensive=lambda x: x.price > 200)
...  .group_by("hood")
...  .aggregate(n=di.count(),
...             n_expensive=di.sum("expensive"),
...             p_expensive=di.mean("expensive")))
.
           hood     n n_expensive p_expensive
         string int64       int64     float64
  ───────────── ───── ─────────── ───────────
0         Bronx  1198          49    0.040902
1      Brooklyn 19931        2047    0.102704
2     Manhattan 21963        5494    0.250148
3        Queens  6068         326    0.053724
4 Staten Island   370          20    0.054054

Going Fast with Numba
---------------------

The common aggregation functions listed above are implemented in
Dataiter as both pure Python code (slow) and JIT-compiled `Numba
<https://numba.pydata.org/>`_ code (fast). If you have Numba installed,
then Dataiter will **automatically** use it for aggregation involving
**boolean**, **integer**, **float**, **date**, and **datetime** columns.
If Numba is not available, Dataiter will automatically fall back on the
slower pure Python implementations. The result should be the same,
whether Numba is used or not, excluding some minor rounding or float
precision differences.

Numba is currently not a hard dependency of Dataiter, so you'll need to
install it separately::

   pip install -U numba

When, for a particular version of Dataiter, you first use a
Numba-accelerated aggregation function, the code will be compiled, which
might take a couple seconds. The compiled code is saved in `cache
<https://numba.readthedocs.io/en/stable/developer/caching.html>`_.
After that, using the function from cache will be really fast. In case
you're benchmarking something, note also that on the first use of such a
function in a Python session, the compiled code is loaded from cache on
disk, which takes something like 10–100 ms and further calls will be
faster as there's no more need to load anything.

.. note:: If you have trouble with Numba, please check the value of
          ``di.USE_NUMBA`` to see if Numba has been found. You can also
          set ``di.USE_NUMBA = False`` if you have Numba installed, but
          it's not working right, or via the environment variable
          ``DATAITER_USE_NUMBA=false``. Sometimes it's just the
          `caching
          <https://numba.readthedocs.io/en/stable/developer/caching.html>`_
          part of Numba that's causing issues. When upgrading you might
          sometimes need to delete old caches. If that doesn't help, you
          can also turn caching off with ``di.USE_NUMBA_CACHE = False``
          or the environment variable
          ``DATAITER_USE_NUMBA_CACHE=false``.
