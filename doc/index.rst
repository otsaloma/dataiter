dataiter Documentation
======================

dataiter is a Python package of classes for data manipulation. dataiter
is intended for practical data science and data engineering work with a
focus on providing a simple and consistent API for common operations.
The relevant goals are convenience, clarity, brevity and lack of
surprises. Performace-wise the goal is to be "adequate" for practical
work with real-world datasets, but limited to pure Python code,
in-memory computation etc. regularities. Currently included classes are
``DataFrame`` and ``ListOfDicts``.

* :class:`dataiter.DataFrame` is a class for tabular data similar to R's
  ``data.frame`` or ``pandas.DataFrame``. It is under the hood a
  dictionary of NumPy arrays and thus capable of fast vectorized
  operations. You can consider this to be a very experimental, very
  light-weight alternative to Pandas with a simple and consistent API.

* :class:`dataiter.ListOfDicts` is a class useful for manipulating data
  from JSON APIs. It provides functionality similar to libraries such as
  Underscore.js, with manipulation functions that iterate over the data
  and return a shallow modified copy of the original.
  ``attd.AttributeDict`` is used to provide convenient access to
  dictionary keys.

.. warning:: dataiter is experimental and the API subject to change.

.. note:: This documentation is a work in progress.

.. toctree::
   :maxdepth: 1
   :caption: General

   quick-start

.. toctree::
   :maxdepth: 1
   :caption: API Documentation

   dataiter
   data-frame
   data-frame-column
   list-of-dicts
   vector
