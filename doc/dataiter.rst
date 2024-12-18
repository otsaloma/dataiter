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
:func:`~dataiter.nth`
:func:`~dataiter.quantile`
:func:`~dataiter.std`
:func:`~dataiter.sum`
:func:`~dataiter.var`

The following read functions are convenience aliases to the correspoding
methods of the classes generally most suitable for the particular file
type, i.e. :class:`.DataFrame` for CSV, NPZ and Parquet,
:class:`.GeoJSON` for GeoJSON and :class:`.ListOfDicts` for JSON.

:func:`~dataiter.read_csv`
:func:`~dataiter.read_geojson`
:func:`~dataiter.read_json`
:func:`~dataiter.read_npz`
:func:`~dataiter.read_parquet`

The following constants can be used to customize certain defaults, such as
formatting and limits for printing.

:attr:`dataiter.PRINT_MAX_WIDTH`
:attr:`dataiter.PRINT_THOUSAND_SEPARATOR`
:attr:`dataiter.PRINT_TRUNCATE_WIDTH`
:attr:`dataiter.USE_NUMBA`
:attr:`dataiter.USE_NUMBA_CACHE`

.. automodule:: dataiter
   :members: all,
             any,
             count,
             count_unique,
             first,
             last,
             max,
             mean,
             median,
             min,
             mode,
             nth,
             quantile,
             read_csv,
             read_geojson,
             read_json,
             read_npz,
             read_parquet,
             std,
             sum,
             var,
             PRINT_MAX_WIDTH,
             PRINT_THOUSAND_SEPARATOR,
             PRINT_TRUNCATE_WIDTH,
             USE_NUMBA,
             USE_NUMBA_CACHE
