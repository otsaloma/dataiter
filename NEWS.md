PENDING: Dataiter 0.29.2
========================

* Disable Numba for date and datetime columns (#18)
* Fix handling of missing values for non-float dtypes
* Rename `dropna` arguments `drop_missing`
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
