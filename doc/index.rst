Dataiter Documentation
======================

.. meta::
   :description: Overview of Dataiter, a light-weight data frame library built on NumPy

Dataiter's :class:`.DataFrame` is a class for tabular data similar to R's
``data.frame``, implementing all common operations to manipulate data. It is
under the hood a dictionary of NumPy arrays and thus capable of fast vectorized
operations. You can consider it to be a light-weight alternative to Pandas with
a simple and consistent API. Performance-wise Dataiter relies on NumPy and
PyArrow and is likely to be at best comparable to Pandas.

Additionally Dataiter includes :class:`.ListOfDicts`, a class for manipulating
hierarchical data, such as from JSON APIs or document databases, and
:class:`.GeoJSON`, a class for manipulating data from GeoJSON files in a data
frame.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   quick-start
   comparison
   aggregation

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   dataiter
   data-frame
   data-frame-column
   geojson
   list-of-dicts
   vector
   dt
   dtypes
   regex

For Coding Agents
-----------------

* `llms.txt <llms.txt>`_
* `llms-full.txt <llms-full.txt>`_

Without a terminal, as is usually the case for coding agents, print output
is wrapped to :attr:`dataiter.PRINT_MAX_WIDTH` columns, which splits wide
data frames into several blocks of columns. To get one line per row, you
might want to disable wrapping with ``di.PRINT_MAX_WIDTH = math.inf``. To
save context, you might also want to print fewer rows, e.g.
``di.PRINT_MAX_ROWS = 10``. See :doc:`dataiter` for other print settings.
