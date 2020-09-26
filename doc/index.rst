dataiter Documentation
======================

dataiter is a Python package of classes for data manipulation. dataiter
is intended for practical data science and data engineering work with a
focus on providing a simple and consistent API for common operations.
Currently included are the following classes.

:class:`dataiter.DataFrame`
   A class for tabular data similar to R's ``data.frame`` or
   ``pandas.DataFrame``. It is under the hood a dictionary of NumPy
   arrays and thus capable of fast vectorized operations. You can
   consider this to be a very experimental, very light-weight
   alternative to Pandas with a simple and consistent API.
   Performance-wise dataiter relies on NumPy and is likely to be at
   best comparable to Pandas.

:class:`dataiter.ListOfDicts`
   A class useful for manipulating data from JSON APIs. It provides
   functionality similar to libraries such as Underscore.js, with
   manipulation functions that iterate over the data and return a
   shallow modified copy of the original. ``attd.AttributeDict`` is used
   to provide convenient access to dictionary keys.

:class:`dataiter.GeoJSON`
   A simple wrapper class that allows reading a GeoJSON file into a
   :class:`dataiter.DataFrame` and writing a data frame to a GeoJSON
   file. Any operations on the data are thus done with methods provided
   by the data frame class. Geometry is read as-is into the "geometry"
   column, but no special geometric operations are currently supported.

.. warning:: dataiter is experimental and the API subject to change.

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   quick-start

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   dataiter
   data-frame
   data-frame-column
   geojson
   list-of-dicts
   vector
