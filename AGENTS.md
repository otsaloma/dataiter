# AGENTS.md

## API Mirroring

The `DataFrame` and `ListOfDicts` classes have a mostly matching API.
When adding a method or an argument to one, also add it to the other if
it makes sense there as well.

## Validation, Testing

After making changes, run `direnv exec . make check` and `direnv exec .
make test` (or `direnv exec . pytest ...` to test only a subset). direnv
ensures these use the virtualenv present in `venv`.

## API Documentation

The API documentation currently requires new methods, properties and
attributes to be manually added to the index at the top of `doc/*.rst`
files. Run `direnv exec . make doc-check` to check these are up to date.

## Known Issues

Do not complain about these or spend too much time investigating.

- Sorting by `StringDType` columns is slow, which makes also e.g.
  grouping by strings slow, about 1–2 s per string column at 3M rows.
  NumPy's string sort is simply slow. `DataFrame.sort` already uses
  chained stable argsorts, which are faster than `np.lexsort` for
  strings. Rejected alternatives: integer codes via a Python dict (fast
  for categorical strings, slower for unique ones), Arrow's
  `dictionary_encode` (Arrow is to be used only for reading and writing
  files), Numba (doesn't support `StringDType`) and a categorical type
  or string interning (would need a second string type).
