dataiter
========

.. meta::
   :description: Miscellaneous functions and constants

Aggregation functions
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
    :func:`~dataiter.prod`
    :func:`~dataiter.quantile`
    :func:`~dataiter.std`
    :func:`~dataiter.sum`
    :func:`~dataiter.var`

    See also :meth:`.DataFrame.aggregate` and the guide on :doc:`aggregation
    </aggregation>`.

Convenience aliases for reading data
    :func:`~dataiter.read_csv`
    :func:`~dataiter.read_geojson`
    :func:`~dataiter.read_json`
    :func:`~dataiter.read_npz`
    :func:`~dataiter.read_parquet`

Miscellaneous functions
    :func:`~dataiter.of_type`

    See :meth:`.DataFrame.map_columns` for a use case.

Sentinels
    :attr:`~dataiter.NEXT`
    :attr:`~dataiter.PREVIOUS`

    See e.g. :meth:`.DataFrame.replace_na` for a use case.

Customization for printing and formatting
    :attr:`~dataiter.DEFAULT_PEEK_ELEMENTS`
    :attr:`~dataiter.DEFAULT_PEEK_ITEMS`
    :attr:`~dataiter.DEFAULT_PEEK_ROWS`
    :attr:`~dataiter.PRINT_FLOAT_PRECISION`
    :attr:`~dataiter.PRINT_MAX_ELEMENTS`
    :attr:`~dataiter.PRINT_MAX_ITEMS`
    :attr:`~dataiter.PRINT_MAX_ROWS`
    :attr:`~dataiter.PRINT_MAX_WIDTH`
    :attr:`~dataiter.PRINT_THOUSAND_SEPARATOR`
    :attr:`~dataiter.PRINT_TRUNCATE_WIDTH`

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
             of_type,
             prod,
             quantile,
             read_csv,
             read_geojson,
             read_json,
             read_npz,
             read_parquet,
             std,
             sum,
             var,
             NEXT,
             PREVIOUS,
             DEFAULT_PEEK_ELEMENTS,
             DEFAULT_PEEK_ITEMS,
             DEFAULT_PEEK_ROWS,
             PRINT_FLOAT_PRECISION,
             PRINT_MAX_ELEMENTS,
             PRINT_MAX_ITEMS,
             PRINT_MAX_ROWS,
             PRINT_MAX_WIDTH,
             PRINT_THOUSAND_SEPARATOR,
             PRINT_TRUNCATE_WIDTH
