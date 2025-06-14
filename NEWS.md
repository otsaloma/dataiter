2025-06-10: Dataiter 1.1
========================

* `DataFrame.pivot_longer`: New method
* `DataFrame.pivot_wider`: New method

2025-02-07: Dataiter 1.0
========================

* Silence warnings about writing NPZ files with StringDType:
  "UserWarning: Custom dtypes are saved as python objects using the
  pickle protocol. Loading this file requires allow_pickle=True to be
  set."

Dataiter can now be considered stable. If upgrading from <= 0.51,
please read the release notes for 0.99–0.9999.

2025-01-12: Dataiter 0.9999
===========================

* New module `dataiter.regex` for vectorized regular expressions
* Add proxy object `Vector.dt` for `dataiter.dt`
* Add proxy object `Vector.re` for `dataiter.regex`
* Add proxy object `Vector.str` for `numpy.strings`
* Use PyArrow instead of Pandas to read and write CSV files
* Replace Pandas dependency with PyArrow

This is likely to be a breaking change in some rare weirdly formatted
CSV files that Pandas and PyArrow might parse differently, resulting in
something like diffently guessed data types or differently detected
missing value markers. The note about stability below release 0.99 still
applies.

2024-12-15: Dataiter 0.999
==========================

* `DataFrame.fom_arrow`: Remove `strings_as_object` argument
* `DataFrame.from_pandas`: Remove `strings_as_object` argument
* `DataFrame.read_csv`: Remove `strings_as_object` argument
* `DataFrame.read_parquet`: Remove `strings_as_object` argument
* `GeoJSON.read`: Remove `strings_as_object` argument
* `ListOfDicts.to_data_frame`: Remove `strings_as_object` argument
* `read_csv`: Remove `strings_as_object` argument
* `read_geojson`: Remove `strings_as_object` argument
* `read_parquet`: Remove `strings_as_object` argument
* `Vector.as_string`: Remove `length` argument
* `Vector.is_na`: Fix to work in multidimensional cases where the
  elements of an object vector are arrays/vectors
* `Vector.rank`: Change default `method` to "min"
* `Vector.rank`: Remove `method` "average"

This is a breaking change to switch the string data type from the
fixed-width `str_` a.k.a. `<U#` to the variable-width `StringDType`
introduced in NumPy 2.0. The main benefit is greatly reduced memory use,
making strings usable without needing to be careful or falling back to
`object`. The note about stability below release 0.99 still applies.

Note that as `StringDType` is only in NumPy >= 2.0, any NPZ or Pickle
files saved cannot be opened using Dataiter < 0.99 and NumPy < 2.0. If
you need that kind of interoperability, consider using the Parquet file
format.

2024-08-17: Dataiter 0.99
=========================

* Adapt to changes in NumPy 2.0
* Bump NumPy dependency to >= 2.0

This is a minimal change to be NumPy 2.0 compatible. In the 0.99+
releases, we plan to adopt the new NumPy string dtype and fix any
regressions that come up, leading to a 1.0 release when everything looks
to be working reliably (#26). Anyone looking for extreme stability
should consider avoiding the 0.99+ releases and waiting for 1.0.

2024-06-24: Dataiter 0.51
=========================

* Mark NumPy dependency as < 2.0

2024-04-06: Dataiter 0.50
=========================

* `ListOfDicts.drop_na`: New method
* `ListOfDicts.keys`: New method
* `ListOfDicts.print_memory_use`: New method
* Fix tabular display of Unicode characters with width != 1
* Add dependency on wcwidth: https://pypi.org/project/wcwidth

2023-11-08: Dataiter 0.49
=========================

* `dt`: Handle all NaT input
* Migrate from `setup.py` to `hatch` and `pyproject.toml`

2023-10-08: Dataiter 0.48
=========================

* `Vector.as_datetime`: Add `precision` argument
* `Vector.concat`: New method
* `Vector.sort`: Fix sorting object vectors

2023-09-09: Dataiter 0.47
=========================

* `DataFrame`: Fix column and method name clash errors in certain operations
* `dt.replace`: Allow vector arguments the same length as `x`

2023-09-05: Dataiter 0.46
=========================

* `DataFrame.count`: New method, shorthand for
  `data.group_by(...).aggregate(n=di.count())`
* `Vector.rank`: Handle empty and all-NA vectors

2023-06-14: Dataiter 0.45
=========================

* `USE_NUMBA_CACHE`: New option, read from environment variable
  `DATAITER_USE_NUMBA_CACHE` if exists, defauls to `True`
* Fix a possible issue with Numba caching

2023-06-13: Dataiter 0.44
=========================

* Use `numba.extending.overload` instead of the deprecated
  `numba.generated_jit`

2023-06-08: Dataiter 0.43
=========================

* `DataFrame`: Don't try to do joins on NA values in `by` columns
* `DataFrame.drop_na`: New method

2023-05-30: Dataiter 0.42
=========================

* `DataFrame`: Truncate multiline strings when printing
* `DataFrame.from_arrow`: New method
* `DataFrame.read_parquet`: New method
* `DataFrame.to_arrow`: New method
* `DataFrame.write_parquet`: New method
* `read_parquet`: New function
* `Vector.__init__`: Fix type guessing when mixing Python and NumPy
  floats or integers and missing values
* Allow using a thousand separator when printing numbers,
  off by default, can be set with `dataiter.PRINT_THOUSAND_SEPARATOR`

2023-03-11: Dataiter 0.41
=========================

* Fix printing really small numbers

2023-02-21: Dataiter 0.40.1
===========================

* `DataFrame.modify`: Fix grouped modify on unsorted data frame

2023-02-20: Dataiter 0.40
=========================

* `Vector.map`: Add `dtype` argument

2023-02-06: Dataiter 0.39.1
===========================

* `ListOfDicts.to_data_frame`: Add `strings_as_object` argument

2023-01-21: Dataiter 0.39
=========================

* `read_csv`, `read_geojson`, `DataFrame.from_pandas`,
  `DataFrame.read_csv`, `GeoJSON.read`: Add `strings_as_object` argument

2022-12-15: Dataiter 0.38
=========================

* `DataFrame.slice_off`: New method
* `GeoJSON.to_data_frame`: New method
* Fix error with new column placeholder attributes in conjunction with
  pop, popitem and clear

2022-11-17: Dataiter 0.37
=========================

* `DataFrame`: Add placeholder attributes for columns so that
  tab completion of columns as attributes at a shell works
* `dt.from_string`: New function
* `dt.to_string`: New function
* `nrow`: Remove deprecated aggregation function
* Don't use Numba for aggregation involving strings due to bad performance

2022-10-16: Dataiter 0.36
=========================

* `dt`: New module for dealing with dates and datetimes

2022-10-03: Dataiter 0.35
=========================

* `DataFrame.from_pandas`: Speed up by avoiding unnecessary conversions
* `DataFrame.full_join`: Fix join and output when `by` is a tuple
* `GeoJSON`: Fix printing object

2022-09-17: Dataiter 0.34
=========================

* `Vector`: Handle timedeltas correctly for NA checks and printing
* `Vector.is_timedelta`: New method

2022-09-03: Dataiter 0.33
=========================

* `DataFrame.sort`: Convert object to string for sorting
* `Vector.sort`: Convert object to string for sorting
* Fix conditional Numba use when importing the numba package works,
  but caching doesn't
* Add `di-open` cli command (currently not part of the default install,
  but can be installed from source using `make install-cli`)

2022-04-02: Dataiter 0.32
=========================

* `DataFrame.modify`: Add support for grouped modification (#19)
* `DataFrame.split`: New method
* `ListOfDicts.split`: New method

2022-02-26: Dataiter 0.31
=========================

* `DataFrame.compare`: New experimental method
* `Vector.as_string`: Add `length` argument
* Change the documentation to default to the latest release ("stable")
  instead of the development version ("latest")

2022-02-19: Dataiter 0.30
=========================

* Use keyword-only arguments where appropriate – the general principle
  is that mandatory arguments are allowed as positional, but optional
  modifiers are keyword only
* Rename all instances of "missing" to "na", such as `Vector.is_missing`
  to `Vector.is_na`, the only exception being
  `ListOfDicts.fill_missing`, which becomes
  `ListOfDicts.fill_missing_keys`
* Truncate data frame object and string columns at
  `PRINT_TRUNCATE_WIDTH` (default 32) for printing

2022-02-09: Dataiter 0.29.2
===========================

* Fix aggregation functions to work with all main data types:
  boolean, integer, float, date, datetime and string
* Fix aggregation functions to handle all missing values (NaN, NaT,
  blank string) correctly, the same as implemented in Vector
* Rename aggregation functions' `dropna` arguments to `drop_missing`
* `first`, `last`, `nth`: Add `drop_missing` argument
* `Vector.drop_missing`: New method

2022-01-30: Dataiter 0.29.1
===========================

* `mode`: Fix to return first in case of ties (requires Python >= 3.8)
* `std`, `var`: Add `ddof` argument (defaults to 0 on account of Numba limitations)
* Don't try to dropna for non-float vectors in aggregation functions

2022-01-29: Dataiter 0.29
=========================

* Add shorthand helper functions for use with `DataFrame.aggregate`,
  optionally using Numba JIT-compiled code for speed
    - https://dataiter.readthedocs.io/en/latest/aggregation.html
    - https://dataiter.readthedocs.io/en/latest/data-frame.html#dataiter.DataFrame.aggregate
    - https://dataiter.readthedocs.io/en/latest/dataiter.html
* `DataFrame.map`: New method
* `ncol`: Removed
* `nrow`: Deprecated in favor of `dataiter.count`
* `read_csv`: New alias for `DataFrame.read_csv`
* `read_geojson`: New alias for `GeoJSON.read`
* `read_json`: New alias for `ListOfDicts.read_json`
* `read_npz`: New alias for `DataFrame.read_npz`

2022-01-09: Dataiter 0.28
=========================

* `DataFrame`: Make object columns work in various operations
* `DataFrame.from_json`: Add arguments `columns` and `dtypes`
* `DataFrame.from_pandas`: Add argument `dtypes`
* `DataFrame.full_join`: Speed up
* `DataFrame.read_csv`: Add argument `dtypes`
* `DataFrame.read_json`: Add arguments `columns` and `dtypes`
* `GeoJSON.read`: Add arguments `columns` and `dtypes`
* `ListOfDicts.fill_missing`: New method
* `ListOfDicts.from_json`: Add arguments `keys` and `types`
* `ListOfDicts.full_join`: Speed up
* `ListOfDicts.read_csv`: Add argument `types`, rename `columns` to `keys`
* `ListOfDicts.read_json`: Add arguments `keys` and `types`

2022-01-01: Dataiter 0.27
=========================

* `DataFrame`: Fix error message when column not found
* `DataFrame.aggregate`: Speed up
* `DataFrame.full_join`: Fix to join all possible columns
* `DataFrame.read_csv`: Try to avoid mixed types
* `ListOfDicts.full_join`: Fix to join all possible keys
* `ListOfDicts.write_csv`: Use minimal quoting
* `Vector.get_memory_use`: New method
* `Vector.rank`: Rewrite, add `method` argument
* `*.read_*`: Rename `fname` argument `path`
* `*.write_*`: Rename `fname` argument `path`
* Add comparison table dplyr vs. Dataiter vs. Pandas to documentation:
  <https://dataiter.readthedocs.io/en/latest/comparison.html>

2021-12-02: Dataiter 0.26
=========================

* `DataFrame.read_npz`: New method to read NumPy npz format
* `DataFrame.write_npz`: New method to write NumPy npz format
* `*.read_*`: Decompress `.bz2|.gz|.xz` automatically
* `*.write_*`: Compress `.bz2|.gz|.xz` automatically

2021-11-13: Dataiter 0.25
=========================

* `DataFrame.print_missing_counts`: Fix when nothing missing
* `Vector.replace_missing`: New method

2021-10-27: Dataiter 0.24
=========================

* `DataFrame.print_memory_use`: New method
* `ListOfDicts.write_csv`: Use less memory

2021-07-08: Dataiter 0.23
=========================

* `Vector.is_*`: Change to be methods instead of properties
* Drop deprecated use of `np.int`
* Drop deprecated comparisons against NaN

2021-05-13: Dataiter 0.22
=========================

* `ListOfDicts.map`: New method

2021-03-08: Dataiter 0.21
=========================

* `DataFrame.read_csv`: Add `columns` argument
* `ListOfDicts.read_csv`: Add `columns` argument

2021-03-06: Dataiter 0.20
=========================

* `DataFrame.*_join`: Handle differing by names via tuple argument
* `ListOfDicts.*_join`: Handle differing by names via tuple argument

2021-03-04: Dataiter 0.19
=========================

* Use terminal window width as maximum print width
* `Vector.__init__`: Handle NaN values in non-float vectors

2021-03-03: Dataiter 0.18
=========================

* `Vector.__init__`: Accept generators/iterators
* `Vector.map`: New method

2021-02-27: Dataiter 0.17
=========================

* `DataFrame.print_missing_counts`: New method
* `GeoJSON.read`: Handle properties differing between features
* `ListOfDicts.print_missing_counts`: New method
* `Vector.as_object`: New method

2020-10-03: Dataiter 0.16.1
===========================

* `GeoJSON.read`: Use warnings, not errors for ignored excess feature keys

2020-09-26: Dataiter 0.16
=========================

* `GeoJSON`: New class

2020-09-12: Dataiter 0.15
=========================

* `ListOfDicts.sort`: Handle descending sort for all types

2020-08-22: Dataiter 0.14
=========================

* `ListOfDicts`: Make obsoletion a warning instead of an error

2020-08-15: Dataiter 0.13
=========================

* `DataFrame`: Fix error printing blank strings (#8)

2020-07-25: Dataiter 0.12
=========================

* `DataFrame.filter`: Add `colname_value_pairs` argument
* `DataFrame.filter_out`: Add `colname_value_pairs` argument
* `ListOfDicts.__init__`: Remove arguments not intended for external use
* `ListOfDicts.rename`: Preserve order of keys
* Add documentation: https://dataiter.readthedocs.io/

2020-06-02: Dataiter 0.11
=========================

* `Vector.__init__`: Speed up by fixing type deduction

2020-05-28: Dataiter 0.10.1
===========================

* `ListOfDicts.select`: Fix return value (#7)

2020-05-21: Dataiter 0.10
=========================

* `DataFrame.aggregate`: Fix `UnicodeEncodeError` with string columns
* `DataFrame.unique`: Fix `UnicodeEncodeError` with string columns
* `ListOfDicts.select`: Return keys in requested order
* `Vector.__repr__`: Add custom conversion to string for display
* `Vector.__str__`: Add custom conversion to string for display
* `Vector.to_string`: Add custom conversion to string for display
* `Vector.to_strings`: Add custom conversion to string for display

2020-05-11: Dataiter 0.9
========================

* `Array`: Rename to `Vector`
* `Vector.head`: New method
* `Vector.range`: New method
* `Vector.sample`: New method
* `Vector.sort`: New method
* `Vector.tail`: New method
* `Vector.unique`: New method

2020-05-10: Dataiter 0.8
========================

* `DataFrame`: New class
* `ListOfDicts.__add__`: New method to support the `+` operator
* `ListOfDicts.__init__`: Rename, reorder arguments
* `ListOfDicts.__mul__`: New method to support the `*` operator
* `ListOfDicts.__repr__`: New method, format as JSON
* `ListOfDicts.__rmul__`: New method to support the `*` operator
* `ListOfDicts.__setitem__`: New method, coerce to `AttributeDict`
* `ListOfDicts.__str__`: New method, format as JSON
* `ListOfDicts.aggregate`: Speed up
* `ListOfDicts.anti_join`: New method
* `ListOfDicts.append`: New method
* `ListOfDicts.clear`: New method
* `ListOfDicts.extend`: New method
* `ListOfDicts.full_join`: New method
* `ListOfDicts.head`: New method
* `ListOfDicts.inner_join`: New method
* `ListOfDicts.insert`: New method
* `ListOfDicts.join`: Removed in favor of specific join types
* `ListOfDicts.left_join`: New method
* `ListOfDicts.pluck`: Add argument "default" to handle missing keys
* `ListOfDicts.print_`: New method
* `ListOfDicts.read_csv`: Add explicit arguments
* `ListOfDicts.read_json`: Relay arguments to `json.loads`
* `ListOfDicts.read_pickle`: New method
* `ListOfDicts.reverse`: New method
* `ListOfDicts.sample`: New method
* `ListOfDicts.semi_join`: New method
* `ListOfDicts.sort`: Change arguments to support sort direction better
* `ListOfDicts.tail`: New method
* `ListOfDicts.to_data_frame`: New method
* `ListOfDicts.to_pandas`: New method
* `ListOfDicts.unique`: Return unique by all keys if none given
* `ListOfDicts.write_csv`: Add explicit arguments
* `ListOfDicts.write_pickle`: New method

2019-12-03: Dataiter 0.7
========================

* Make `sort` handle `None` values, sorted last

2019-11-29: Dataiter 0.6
========================

* Fix `ObsoleteError` after multiple modifying actions

2019-11-10: Dataiter 0.5
========================

* Add `read_csv`
* Add `read_json`
* Add `write_csv`
* Add `write_json`

2019-11-01: Dataiter 0.4
========================

* Fix `ObsoleteError` with `deepcopy`
* Define `__deepcopy__` so that `copy.deepcopy` works too
* Add `copy` (and `__copy__` for `copy.copy`)

2019-11-01: Dataiter 0.3
========================

* Mark `ListOfDicts` object obsolete thus preventing (accidental) use if
  a chained successor has modified the shared dicts
* Add `modify_if`

2019-10-31: Dataiter 0.2
========================

* Speed up, mostly by avoiding copying (methods that modify dicts now do
  it in place rather than making a copy)

2019-09-29: Dataiter 0.1
========================

* Initial release
