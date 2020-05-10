2019-05-xx: dataiter 0.9
========================

* `Array`: Rename to `Vector`
* `Vector.head`: New method
* `Vector.range`: New method
* `Vector.sample`: New method
* `Vector.sort`: New method
* `Vector.tail`: New method
* `Vector.unique`: New method

2019-05-10: dataiter 0.8
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
