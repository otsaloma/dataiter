Dataiter Documentation
======================

Dataiter is a Python package of classes for data manipulation. Dataiter
is intended for practical data science and data engineering work with a
focus on providing a simple and consistent API for common operations.
Currently included are the following classes.

:class:`.DataFrame`
   A class for tabular data similar to R's ``data.frame`` or
   ``pandas.DataFrame``. It is under the hood a dictionary of NumPy
   arrays and thus capable of fast vectorized operations. You can
   consider this to be a light-weight alternative to Pandas with a
   simple and consistent API. Performance-wise Dataiter relies on NumPy
   and Numba and is likely to be at best comparable to Pandas.

:class:`.ListOfDicts`
   A class useful for manipulating data from JSON APIs. It provides
   functionality similar to libraries such as Underscore.js, with
   manipulation functions that iterate over the data and return a
   shallow modified copy of the original. ``attd.AttributeDict`` is used
   to provide convenient access to dictionary keys.

:class:`.GeoJSON`
   A simple wrapper class that allows reading a GeoJSON file into a
   :class:`.DataFrame` and writing a data frame to a GeoJSON file. Any
   operations on the data are thus done with methods provided by the
   data frame class. Geometry is read as-is into the "geometry" column,
   but no special geometric operations are currently supported.

.. warning:: Dataiter is still evolving and the API is subject to
             breaking changes.

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
