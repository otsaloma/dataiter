# AGENTS.md

## API Design

We favor elementary operations, composability and method chaining. Do
not overload individual methods with arguments to cover things that can
be done in chained steps. Example:
`data.pivot_wider(...).replace_na(...)` instead of
`data.pivot_wider(..., fill_value=...)`. Hint in docstrings of
operations to any common pre- or post-processing steps.

The `DataFrame` and `ListOfDicts` classes have a mostly matching API.
When adding a method or an argument to one, also add it to the other if
it makes sense there as well.

## Validation, Testing

After making changes, run `uv run make check` and `uv run make test` (or
`uv run pytest ...` to test only a subset). uv ensures these use the
virtualenv present in `.venv`. Similarly, run any benchmark or test
scripts via `uv run python ...`. Generating documentation (`uv run make
doc`) is slow; skip that when making simple docstring changes that can
be expected to parse and generate fine.

## API Documentation

The API documentation currently requires new methods, properties and
attributes to be manually added to the index at the top of `doc/*.rst`
files. Run `uv run make doc-check` to check these are up to date.

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

- Grouped `DataFrame.modify` is slow for large and wide data frames,
  e.g. 2.7 s for 3M rows, 24 columns and 8k groups. Most of the time
  goes to reordering all columns by group, since the functions given
  receive full rows and can access any column, and the rest to calling a
  Python function per group. Significantly speeding it up would require
  Numba-accelerated helper functions like those for
  `DataFrame.aggregate`, which declare the columns they use and avoid
  per-group Python calls. That would be a big change, currently not
  considered worth doing, as grouped modify is not commonly used.
  Rejected alternative: reordering columns lazily upon first access (too
  much magic in a data frame subclass).

- `DataFrame` methods return new data frames with all columns copied,
  which makes long method chains on large and wide data frames slow,
  e.g. 0.34 s per `modify` for 3.4M rows and 101 columns. This is by
  design: it's consistent (no Pandas-style duality of copy-returning and
  in-place methods), safe (columns are mutable NumPy arrays, so sharing
  them would leak in-place edits between data frames) and what enables
  method chaining. Rejected alternative: sharing unchanged columns as
  read-only with copy-on-write (would prevent in-place edits of
  columns).

- Ungrouped `DataFrame.aggregate`, `DataFrame.count` and
  `ListOfDicts.aggregate` return no rows for empty data, unlike e.g.
  dplyr's `summarise`, which returns one row. This is by design: no
  data, no rows, and functions are never called on empty data. Many
  common functions raise for empty input, e.g. `max`, indexing the first
  element or a weighted average, and others give NumPy warnings, e.g.
  `mean`, which would break aggregating subsets of data in a loop.
  Similarly, grouped aggregation of empty data gives float64 columns, as
  data types can't be known without calling the functions;
  `DataFrame.rbind` ignores data types of empty data frames to deal with
  this.

- Grouping is supported only by some operations, listed in the
  docstrings of `DataFrame.group_by` and `ListOfDicts.group_by`, and
  other operations ignore it. This is by design. Grouped support for
  further operations should be evaluated on its own merit, not added for
  completeness or to match what other libraries do.

- Group columns (`DataFrame`) and keys (`ListOfDicts`) are kept only for
  the operation immediately following `group_by`, all operations return
  ungrouped data. This is by design: `group_by` chained immediately
  before the grouped operation, e.g.
  `data.group_by(...).aggregate(...)`, is explicit and clear code.
  Carrying grouping along longer would be concealed state that leads to
  surprises.
