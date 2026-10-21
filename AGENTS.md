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

## NumPy

Avoid using NumPy's cryptic dtype shorthands, example: instead of
`np.dtype("M8")` use `np.dtype("datetime64")`. As an exception,
something like `x.dtype.kind in "biufc"` is acceptable as writing that
out clearer would expand the code too much, into five lines of or'd
`np.issubdtype` checks.

## PyArrow

Besides reading and writing files, PyArrow can be used under the hood in
implementations of operations and in reusable helper functions, e.g. for
fast hashing or sorting. It must never show in any API, public or
undocumented, such as `util`: function arguments and return values are
always Dataiter or NumPy data structures, never PyArrow ones.

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

- Sorting by `StringDType` columns of nearly unique strings is slow,
  about 4 s for 7M unique strings. NumPy's string sort is simply slow.
  `DataFrame.sort` and grouping sort strings via sorted codes
  (`util.sorted_codes`: hashing with PyArrow, then sorting only unique
  values), which is fast for repeated strings, but for unique strings
  adds hashing on top of a full string sort. Rejected alternatives:
  PyArrow's `sort_indices` (fast only for already sorted data, ~3 s for
  7M random rows, slower than NumPy for a single numeric key), Numba
  (doesn't support `StringDType`) and a categorical type or string
  interning (would need a second string type).

- Grouped `DataFrame.modify` is slow for large data frames, e.g. 0.75 s
  for 3M rows and 8k groups, as it calls a Python function per group and
  reorders the columns used and the result between group and original
  order. Functions get a `LazyDataFrame`, which reorders only the
  columns they access. Significantly speeding it up further would
  require vectorized helper functions like those for
  `DataFrame.aggregate`, which avoid per-group Python calls. That would
  be a big change, currently not considered worth doing, as grouped
  modify is not commonly used.

- `DataFrame` methods return new data frames with all columns copied,
  which makes long method chains on large and wide data frames slow,
  e.g. 0.2 s per `modify` for 3.4M rows and 101 columns. This is by
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

- NumPy integers can't represent missing values, so operations that
  introduce missing values upcast integer columns to float64, e.g.
  `pivot_wider` for missing combinations, `rbind` for missing columns
  and joins for unmatched rows. Replacing the missing values afterwards,
  e.g. with `replace_na(x=0)`, keeps the column float64; users can
  convert back with e.g. `Vector.as_integer`. This is a NumPy
  limitation, not something we can work around: integers can't always
  have an integer data type.
