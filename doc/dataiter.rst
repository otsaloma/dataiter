dataiter
========

The following functions are shorthand helpers for use in conjunction
with :meth:`.DataFrame.aggregate`, see the guide on :doc:`aggregation
</aggregation>` for details.

:func:`~dataiter.all`
:func:`~dataiter.any`
:func:`~dataiter.count`
:func:`~dataiter.count_unique`
:func:`~dataiter.first`
:func:`~dataiter.last`
:func:`~dataiter.max`
:func:`~dataiter.mean`
:func:`~dataiter.median`
:func:`~dataiter.min`
:func:`~dataiter.mode`
:func:`~dataiter.nrow`
:func:`~dataiter.nth`
:func:`~dataiter.quantile`
:func:`~dataiter.std`
:func:`~dataiter.sum`
:func:`~dataiter.var`

The following read functions are convenience aliases to the correspoding
methods of the classes most suitable for the particular file type, i.e.
:class:`.DataFrame` for CSV and NPZ, :class:`.GeoJSON` for GeoJSON and
:class:`.ListOfDicts` for JSON.

:func:`~dataiter.read_csv`
:func:`~dataiter.read_geojson`
:func:`~dataiter.read_json`
:func:`~dataiter.read_npz`

.. automodule:: dataiter
   :members:
