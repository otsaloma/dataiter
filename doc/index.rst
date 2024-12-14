Dataiter Documentation
======================

Dataiter's :class:`.DataFrame` is a class for tabular data similar to R's
``data.frame``, implementing all common operations to manipulate data. It is
under the hood a dictionary of NumPy arrays and thus capable of fast vectorized
operations. You can consider it to be a light-weight alternative to Pandas with
a simple and consistent API. Performance-wise Dataiter relies on NumPy and Numba
and is likely to be at best comparable to Pandas.

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
   dtypes
   dt
