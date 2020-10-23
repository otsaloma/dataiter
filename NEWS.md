PENDING: dataiter 0.17
======================

* `GeoJSON.read`: Handle properties differing between features
* `Vector.as_object`: New method

2020-10-03: dataiter 0.16.1
===========================

* `GeoJSON.read`: Use warnings, not errors for ignored excess feature keys

2020-09-26: dataiter 0.16
=========================

* `GeoJSON`: New class

2020-09-12: dataiter 0.15
=========================

* `ListOfDicts.sort`: Handle descending sort for all types

2020-08-22: dataiter 0.14
=========================

* `ListOfDicts`: Make obsoletion a warning instead of an error

2020-08-15: dataiter 0.13
=========================

* `DataFrame`: Fix error printing blank strings (#8)

2020-07-25: dataiter 0.12
=========================

* `DataFrame.filter`: Add `colname_value_pairs` argument
* `DataFrame.filter_out`: Add `colname_value_pairs` argument
* `ListOfDicts.__init__`: Remove arguments not intended for external use
* `ListOfDicts.rename`: Preserve order of keys
* Add documentation: https://dataiter.readthedocs.io/

2020-06-02: dataiter 0.11
=========================

* `Vector.__init__`: Speed up by fixing type deduction

2020-05-28: dataiter 0.10.1
===========================

* `ListOfDicts.select`: Fix return value (#7)

2020-05-21: dataiter 0.10
=========================

* `DataFrame.aggregate`: Fix `UnicodeEncodeError` with string columns
* `DataFrame.unique`: Fix `UnicodeEncodeError` with string columns
* `ListOfDicts.select`: Return keys in requested order
* `Vector.__repr__`: Add custom conversion to string for display
* `Vector.__str__`: Add custom conversion to string for display
* `Vector.to_string`: Add custom conversion to string for display
* `Vector.to_strings`: Add custom conversion to string for display

2020-05-11: dataiter 0.9
========================

* `Array`: Rename to `Vector`
* `Vector.head`: New method
* `Vector.range`: New method
* `Vector.sample`: New method
* `Vector.sort`: New method
* `Vector.tail`: New method
* `Vector.unique`: New method

2020-05-10: dataiter 0.8
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

2019-12-03: dataiter 0.7
========================

* Make `sort` handle `None` values, sorted last

2019-11-29: dataiter 0.6
========================

* Fix `ObsoleteError` after multiple modifying actions

2019-11-10: dataiter 0.5
========================

* Add `read_csv`
* Add `read_json`
* Add `write_csv`
* Add `write_json`

2019-11-01: dataiter 0.4
========================

* Fix `ObsoleteError` with `deepcopy`
* Define `__deepcopy__` so that `copy.deepcopy` works too
* Add `copy` (and `__copy__` for `copy.copy`)

2019-11-01: dataiter 0.3
========================

* Mark `ListOfDicts` object obsolete thus preventing (accidental) use if
  a chained successor has modified the shared dicts
* Add `modify_if`

2019-10-31: dataiter 0.2
========================

* Speed up, mostly by avoiding copying (methods that modify dicts now do
  it in place rather than making a copy)

2019-09-29: dataiter 0.1
========================

* Initial release
