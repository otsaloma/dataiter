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
