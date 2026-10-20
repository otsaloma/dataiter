# PENDING: Dataiter 1.5

New:
- `all`, `any`: Add `drop_na` argument, default true
- `DataFrame.compare`: No longer experimental
- `DataFrame.expand_grid`: New classmethod
- `DataFrame.explode`: New method to expand sequences or delimited strings to rows
- `DataFrame.filter`, `DataFrame.filter_out`: Match any element if value is a sequence
- `DataFrame.filter`, `DataFrame.filter_out`: Apply `rows` function group-wise if grouped
- `DataFrame.from_arrow`, `DataFrame.read_parquet`, `read_parquet`: Add `timezone` argument, by default keeping timestamps in the timezone they're stored in instead of converting to UTC
- `DataFrame.full_join`, `DataFrame.inner_join`, `DataFrame.left_join`: Add `multiple` argument to choose the first, last or all matches
- `DataFrame.head`, `DataFrame.tail`: Return first or last rows of each group if grouped
- `DataFrame.map`: Deprecated in favor of a list comprehension or `to_list_of_dicts().map`
- `DataFrame.map_columns`: New method to apply a function to columns matching a predicate, e.g. `of_type`
- `DataFrame.modify`: Evaluate arguments in order, allowing use of columns created by earlier ones
- `DataFrame.pivot_longer`, `DataFrame.pivot_wider`: Default `names` and `values` to "name" and "value"
- `DataFrame.pivot_wider`: Order new columns by first appearance instead of sorting, as tidyr does
- `DataFrame.read_csv`, `ListOfDicts.read_csv`, `read_csv`: Add `na` argument for additional missing value markers and `quote` argument to change or disable quoting
- `DataFrame.read_csv`, `ListOfDicts.read_csv`, `read_csv`: Add `colnames` argument to replace column names in the header row or, without one, the default "a", "b", "c", etc.
- `DataFrame.rename`, `ListOfDicts.rename`: Accept a function to apply to all column names or keys, e.g. `rename(str.lower)`
- `DataFrame.replace_na`: New method
- `DataFrame.select`, `DataFrame.unselect`, `ListOfDicts.select`, `ListOfDicts.unselect`: Accept glob patterns, e.g. `select("id", "price_*")`
- `DataFrame.slice`, `DataFrame.slice_off`: Accept boolean vectors, previously misread as indices 0 and 1
- `DataFrame.to_string`, `Vector.to_string`, `Vector.to_strings`: Print datetimes without fractional seconds when all are zero
- `DataFrame.to_string`, `Vector.to_string`, `Vector.to_strings`: Print timedeltas as e.g. "1 day 04:44:39" instead of a count of the unit
- `dt.floor`, `Vector.dt.floor`: New function
- `dt.from_components`: New function to create dates or datetimes from components, e.g. year, month and day columns
- `dt.from_string`: Add `errors` argument for unparseable values
- `dt.total_seconds`, `Vector.dt.total_seconds`: New function to convert timedeltas to seconds
- `GeoJSON`: Read and write feature identifiers as column "id"
- `ListOfDicts.count`: New method, replacing `list.count`, to count items grouped by keys, as `DataFrame.count` does
- `ListOfDicts.expand_grid`: New classmethod
- `ListOfDicts.filter`, `ListOfDicts.filter_out`: Match any element if value is a sequence
- `ListOfDicts.head`, `ListOfDicts.tail`: Return first or last items of each group if grouped
- `ListOfDicts.replace_na`: New method
- `mean`: Add `weights` argument for a weighted mean
- `mean`, `median`, `quantile`: Support dates and datetimes
- `of_type`: New function to check if data type is one of given, for use with `DataFrame.map_columns`
- `PRINT_MAX_ROWS`: Lower default from 100 to 20
- `prod`: New aggregation function
- `regex.extract`: New function to extract a group of the first match
- `Vector`: Convert strings when comparing to datetimes, e.g. `x >= "2025-01-01"`, previously raising an error or, for equality, silently not matching
- `Vector`: Create an object vector from lists or tuples, previously raising an error if all were of equal length
- `Vector`: Use `b""` as missing value for bytes, as `""` for strings, instead of upcasting to object for `None`
- `Vector.as_float`: Add `errors` argument for values that can't be converted, e.g. non-numeric strings
- `Vector.is_in`: New method, fast also for strings unlike `np.isin`, converting strings when comparing to datetimes
- `Vector.rank`: Add `dir` argument for descending order
- `Vector.shift`: New method to shift elements
- `Vector.str.partition`, `Vector.str.rpartition`: New functions, wrapping `np.strings.partition` and `np.strings.rpartition`
- `Vector.str.slice`: New function, wrapping `np.strings.slice` with support for missing values

Breaking changes:
- `*.from_json`, `*.read_*`, `read_*`: Rename arguments `columns` (`DataFrame`, `GeoJSON`) and `keys` (`ListOfDicts`) to `select`
- Require PyArrow >= 26
- Drop optional use of Numba in favor of vectorized aggregation helper functions in plain NumPy, remove `USE_NUMBA` and `USE_NUMBA_CACHE`

Speed-ups:
- `DataFrame.*`: Speed up copying and reordering columns for large data by processing columns in parallel threads, e.g. `filter`, `modify`, `sort` and joins, using as many threads as PyArrow, limited by e.g. `OMP_THREAD_LIMIT`
- `DataFrame.*_join`: Speed up when joining by a single numeric or datetime column
- `DataFrame.*_join`: Speed up when joining by string columns
- `DataFrame.*_join`: Speed up when joining by multiple columns, especially numeric and datetime
- `DataFrame.full_join`: Speed up by matching rows only once
- `DataFrame.aggregate`: Speed up by sorting only columns needed
- `DataFrame.aggregate`: Speed up arbitrary functions, especially with many groups and columns
- `DataFrame.aggregate`: Speed up helper functions for columns previously not handled by Numba, e.g. strings and timedeltas
- `DataFrame.aggregate`, `DataFrame.count`, `DataFrame.filter`, `DataFrame.head`, `DataFrame.modify`, `DataFrame.split`, `DataFrame.tail`: Speed up grouping, especially by strings
- `DataFrame.modify`: Speed up when grouped
- `DataFrame.aggregate`, `DataFrame.filter`, `DataFrame.filter_out`, `DataFrame.modify`: Speed up functions when grouped by reordering only columns they use
- `DataFrame.pivot_longer`: Speed up, especially with many columns
- `DataFrame.pivot_wider`: Speed up, especially with many ids or names
- `DataFrame.rbind`: Speed up by not copying columns twice and combining them in parallel threads
- `DataFrame.sort`: Speed up, especially for strings
- `DataFrame.unique`: Speed up with string columns when keeping many rows of wide data
- `DataFrame.unique`: Speed up, especially with string columns
- `DataFrame.to_arrow`, `DataFrame.to_pandas`, `DataFrame.write_parquet`: Speed up and keep data types, e.g. float32 and datetime precision
- `DataFrame.to_arrow`, `DataFrame.to_pandas`, `DataFrame.write_csv`, `DataFrame.write_parquet`: Speed up strings
- `dt.day`, `dt.hour`, `dt.isoweek`, `dt.isoweekday`, `dt.microsecond`, `dt.minute`, `dt.month`, `dt.quarter`, `dt.second`, `dt.weekday`, `dt.year`: Speed up
- `dt.from_string`: Speed up, especially when values repeat
- `dt.isoformat`: Speed up, especially when values repeat
- `dt.replace`: Speed up, especially for vector arguments
- `dt.to_string`: Speed up when values repeat
- `dt.to_string`: Speed up when format needs only part of the values, e.g. "%Y-%m"
- `count`, `first`, `last`, `max`, `min`, `nth`: Speed up when grouped, especially with many groups
- `median`, `quantile`: Speed up when grouped with many small groups
- `count_unique`, `mode`: Speed up when grouped with very many groups
- `mode`: Speed up for large groups
- `regex.*`: Speed up, especially when values repeat
- `std`, `var`: Speed up when grouped with `ddof` other than zero
- `Vector.rank`: Speed up for strings, especially when values repeat
- `Vector.sort`: Speed up for strings when values repeat
- `Vector.unique`: Speed up

Fixes:
- `DataFrame`: Name the column and lengths in the error for columns of wrong length
- `DataFrame.*`: Handle empty data frames gracefully
- `DataFrame.aggregate`, `DataFrame.count`: Return single row when ungrouped, or none for empty data
- `DataFrame.compare`: Return empty data frames instead of `None` when no differences found
- `DataFrame.compare`: Match rows with missing values in `by`, previously reported as both added and removed
- `DataFrame.compare`: Rename `by` columns named "column", "xvalue" or "yvalue" in changed values by appending underscores, previously an error
- `DataFrame.drop_na`: Use all columns if none given instead of dropping nothing
- `DataFrame.from_arrow`, `DataFrame.from_pandas`, `DataFrame.read_csv`, `DataFrame.read_parquet`: Convert nanosecond datetimes and timedeltas to microseconds, also if `dtypes` gives unitless `np.datetime64` or `np.timedelta64`, fixing errors in `dt` functions and wrong values from `tolist` and others
- `DataFrame.from_arrow`, `DataFrame.read_csv`, `DataFrame.read_parquet`: Read columns with no values as float instead of an object column of `None`
- `DataFrame.from_arrow`, `DataFrame.read_csv`, `DataFrame.read_parquet`: Read times of day as strings instead of an object column of `datetime.time`
- `DataFrame.from_arrow`, `DataFrame.read_parquet`: Fix error reading list columns, now read as object columns of lists
- `DataFrame.from_arrow`, `DataFrame.read_parquet`: Read decimals as float instead of an object column of `decimal.Decimal`
- `DataFrame.from_arrow`, `DataFrame.read_parquet`: Fix string columns becoming object if the first value is missing
- `DataFrame.from_arrow`, `DataFrame.from_pandas`, `DataFrame.read_csv`, `DataFrame.read_parquet`, `ListOfDicts.read_csv`, `read_csv`, `read_parquet`: Raise `KeyError` for `dtypes` or `types` referring to missing columns instead of silently ignoring them
- `DataFrame.from_pandas`: Fix dates becoming object, e.g. in a `to_pandas` roundtrip
- `DataFrame.*_join`: Raise `TypeError` for incompatible types of `by` columns, e.g. string and number, instead of silently matching nothing
- `DataFrame.full_join`, `ListOfDicts.full_join`: Add rows of `other` not used due to multiple matches as such, after rows of `self`, as dplyr does, instead of joining them to the matching rows of `self`, which effectively gave all matches
- `DataFrame.full_join`, `DataFrame.left_join`, `DataFrame.pivot_wider`: Keep data type of integer and boolean columns if no missing values are added
- `DataFrame.group_by`: Fix to group only returned copy
- `DataFrame.head`, `DataFrame.tail`: Return all rows except the last or first `-n` for negative `n` instead of none
- `DataFrame.pivot_longer`: Make `ids` a required argument, previously defaulting to `None`, which gave a confusing error
- `DataFrame.pivot_longer`: Keep input row order instead of sorting by `ids`, which mixed rows with duplicate `ids`
- `DataFrame.pivot_longer`: Use common data type of values if all pivoted columns are of the same kind, e.g. numbers, instead of object
- `DataFrame.pivot_wider`: Fix error with non-string names
- `DataFrame.pivot_wider`: Fix values of rows with missing `ids` becoming missing
- `DataFrame.pivot_wider`: Raise `ValueError` for duplicate combinations of `ids` and `names` instead of silently keeping the first
- `DataFrame.pivot_wider`: Raise `ValueError` for missing values in `names` instead of creating a column named e.g. "" or "nan"
- `DataFrame.print_memory_use`, `DataFrame.print_na_counts`, `ListOfDicts.print_memory_use`: Print all rows regardless of `PRINT_MAX_ROWS`
- `DataFrame.print_memory_use`: Print data type of strings as "string" instead of "StringDType(na_object='')", as `print` does
- `DataFrame.print_na_counts`, `ListOfDicts.print_na_counts`: Print "No missing values" instead of nothing or a blank list
- `DataFrame.rbind`: Ignore data types of empty data frames, fixing e.g. upcasting integers to float
- `DataFrame.rbind`: Name the column and its data types in the error for incompatible types
- `DataFrame.read_csv`: Fix `select` and string `dtypes` with `header=False`
- `DataFrame.read_csv`: Fix string `dtypes` losing e.g. leading zeros to type guessing
- `DataFrame.read_csv`: Fix reading `.xz` compressed files
- `DataFrame.print_memory_use`, `Vector.get_memory_use`: Include contents of long strings, previously left out
- `DataFrame.rename`: Raise `ValueError` for duplicate column names after renaming instead of silently dropping columns
- `DataFrame.rename`, `DataFrame.unselect`: Raise `KeyError` for missing columns instead of silently ignoring them, as `DataFrame.select` already did
- `DataFrame.sort`: Fix descending sort of unsigned integers
- `DataFrame.sort`: Fix mutating fixed-width string columns with missing values
- `DataFrame.sort`: Fix to sort missing values always last
- `DataFrame.to_string`: Fix header of empty data frames shifted one character left
- `DataFrame.to_string`, `Vector.to_string`, `Vector.to_strings`: Fix negative floats showing fewer significant digits
- `DataFrame.to_string`, `Vector.to_string`, `Vector.to_strings`: Fix infinite values switching floats to scientific notation
- `DataFrame.write_csv`: Fix to compress if path ends in `.bz2|.gz|.xz`, as documented
- `DataFrame.write_csv`: Write timedeltas as printed, e.g. "1 day 04:44:39", instead of integers with no indication of unit
- `count_unique`: Count all missing values as one instead of each NaN and NaT separately in some cases, also fixing an error for objects with missing values
- `dt.*`, `regex.*`: Raise `TypeError` for wrong input type instead of `AssertionError`
- `dt.from_string`, `Vector.rank`: Fix segmentation fault for strings in some orders, e.g. two sorted runs, due to a NumPy bug
- `dt.replace`, `dt.to_string`: Fix error for nanosecond datetimes
- `dt.isoformat`: Use the same format for all elements, with fractional seconds as precise as needed, instead of omitting them for elements with zero microseconds
- `first`, `last`, `nth`: Fix wrong results for dates and datetimes when grouped
- `ListOfDicts.*`: Return ungrouped lists, as `DataFrame` does, previously staying grouped and making e.g. `head` after `aggregate` operate group-wise
- `ListOfDicts.*`: Treat missing keys as `None` instead of raising `KeyError`
- `ListOfDicts.*_join`: Fix matching on missing values in `by` keys
- `ListOfDicts.*_join`: Keep values of `self` for keys found in both
- `ListOfDicts.full_join`: Rename keys of items found only in `other` to their names in `self` when `by` has tuples, as `DataFrame.full_join` does
- `ListOfDicts.aggregate`: Return single item when ungrouped, or none for empty data
- `ListOfDicts.drop_na`: Use all keys if none given instead of dropping nothing
- `ListOfDicts.group_by`: Fix to group only returned copy
- `ListOfDicts.read_csv`: Fix values ending up under wrong keys if `select` isn't in file order and raise `KeyError` for keys not found
- `ListOfDicts.rename`, `ListOfDicts.select`: Fix false warning about shared dicts being modified
- `ListOfDicts.tail`: Fix returning all items for `n=0`
- `ListOfDicts.to_data_frame`, `ListOfDicts.to_pandas`: Fix dropping keys missing from the first item
- `first`, `last`, `max`, `min`, `nth`: Fix error for string vectors
- `max`, `mean`, `median`, `min`, `quantile`, `sum`: Fix errors and wrong results for timedeltas
- `first`, `last`, `max`, `min`, `mode`, `nth`: Keep data type when grouped, e.g. float32 or object, instead of converting to float64 or int64
- `first`, `last`, `max`, `median`, `min`, `mode`, `nth`, `quantile`: Return `None` instead of NaT or NaN for datetime and timedelta vectors with no values, as `mean` does
- `mean`, `median`, `quantile`: Fix ignoring `drop_na=False` for datetime vectors
- `median`, `quantile`: Return missing values for groups with missing values if `drop_na=False`, as NumPy does, previously ignored for some data types
- `mean`: Round timedelta vectors to the nearest unit instead of truncating toward zero, as when grouped, and return `None` instead of NaN for no values
- `mean`, `std`, `sum`, `var`: Calculate in float64 also for float32 for better accuracy
- `mode`: Return Python builtin types for vectors, as other aggregation functions do, instead of NumPy scalars
- `mode`: Count all missing values as one if `drop_na=False`, as `count_unique` does, instead of each NaN and NaT separately
- `std`, `sum`, `var`: Raise `TypeError` for dates and datetimes
- `std`, `var`: Return zero instead of NaN for a single value with the default `ddof=0`, as NumPy does
- `Vector`: Convert fixed-width string arrays, e.g. from `np.where`, to StringDType
- `Vector`: Create a float vector from all NaN values instead of an object vector of `None`
- `Vector.as_boolean`: Raise `ValueError` for missing values
- `Vector.as_bytes`: Keep missing values instead of converting e.g. NaN and `None` to "nan" and "None"
- `Vector.as_float`: Convert missing values to NaN, fixing an error for strings and garbage values for datetimes and timedeltas
- `Vector.as_integer`: Raise `ValueError` for missing values
- `Vector.as_string`: Keep missing values instead of converting e.g. NaN and `None` to "nan" and "None"
- `Vector.dt`: Add missing `isoformat`
- `Vector.range`: Fix to ignore missing values of datetimes and strings
- `Vector.rank`: Fix error when all values are missing
- `Vector.rank`: Fix missing values getting the same rank as tied values ranked last when `method="min"`
- `Vector.replace_na`: Fix error for value of a different type if there are no missing values, e.g. `""` for integers
- `Vector.sort`: Raise `TypeError` for `np.sort(vector)` with a hint to use `vector.sort()` instead, previously a confusing error about unexpected argument "axis"

# 2026-10-05: Dataiter 1.4

- `DataFrame.aggregate`, `DataFrame.split`: Speed up
- `DataFrame.semi_join`: Fix matching on missing values in `by` columns
- `DataFrame.unique`: Speed up
- `DataFrame.unique`: Fix missing timedelta values not being considered equal
- `read_parquet`: Fix ignoring `columns` and `dtypes` arguments
- `Vector.is_na`: Speed up for integer and boolean vectors
- Speed up iterating over vectors and data frame columns, e.g. in joins
- Fix NumPy deprecation warnings about `NaT` values missing unit
- Fix deleting a data frame column leaving behind a bogus attribute
- Add Markdown versions of documentation pages, `llms.txt` and
  `llms-full.txt` for coding agents and LLMs

# 2026-04-12: Dataiter 1.3.1

- Fix NumPy version dependency in `pyproject.toml` for PyPI
- Fix Python version dependency in `pyproject.toml` for PyPI

# 2026-04-12: Dataiter 1.3

- `Vector.str.contains`: New proxy method (equivalent to find(x) > -1)
- Remove StringDType sort and unique performance workarounds which are
  no longer needed since NumPy >= 2.4.0 as the sorting performance
  issues have been fixed in NumPy
- Speed up data frame string column sorting by removing workaround for
  StringDType lexsort segfault, which has been fixed in NumPy >= 2.2.1

# 2025-10-05: Dataiter 1.2

- `DataFrame.count`: Avoid name clash if column `n` already exists
- `dt.isoformat`: New function
- `mode`: Fix Numba variant on some recent versions of Numba

# 2025-06-10: Dataiter 1.1

- `DataFrame.pivot_longer`: New method
- `DataFrame.pivot_wider`: New method

# 2025-02-07: Dataiter 1.0

- Silence warnings about writing NPZ files with StringDType:
  "UserWarning: Custom dtypes are saved as python objects using the
  pickle protocol. Loading this file requires allow_pickle=True to be
  set."

Dataiter can now be considered stable. If upgrading from <= 0.51,
please read the release notes for 0.99–0.9999.

# 2025-01-12: Dataiter 0.9999

- New module `dataiter.regex` for vectorized regular expressions
- Add proxy object `Vector.dt` for `dataiter.dt`
- Add proxy object `Vector.re` for `dataiter.regex`
- Add proxy object `Vector.str` for `numpy.strings`
- Use PyArrow instead of Pandas to read and write CSV files
- Replace Pandas dependency with PyArrow

This is likely to be a breaking change in some rare weirdly formatted
CSV files that Pandas and PyArrow might parse differently, resulting in
something like diffently guessed data types or differently detected
missing value markers. The note about stability below release 0.99 still
applies.

# 2024-12-15: Dataiter 0.999

- `DataFrame.fom_arrow`: Remove `strings_as_object` argument
- `DataFrame.from_pandas`: Remove `strings_as_object` argument
- `DataFrame.read_csv`: Remove `strings_as_object` argument
- `DataFrame.read_parquet`: Remove `strings_as_object` argument
- `GeoJSON.read`: Remove `strings_as_object` argument
- `ListOfDicts.to_data_frame`: Remove `strings_as_object` argument
- `read_csv`: Remove `strings_as_object` argument
- `read_geojson`: Remove `strings_as_object` argument
- `read_parquet`: Remove `strings_as_object` argument
- `Vector.as_string`: Remove `length` argument
- `Vector.is_na`: Fix to work in multidimensional cases where the
  elements of an object vector are arrays/vectors
- `Vector.rank`: Change default `method` to "min"
- `Vector.rank`: Remove `method` "average"

This is a breaking change to switch the string data type from the
fixed-width `str_` a.k.a. `<U#` to the variable-width `StringDType`
introduced in NumPy 2.0. The main benefit is greatly reduced memory use,
making strings usable without needing to be careful or falling back to
`object`. The note about stability below release 0.99 still applies.

Note that as `StringDType` is only in NumPy >= 2.0, any NPZ or Pickle
files saved cannot be opened using Dataiter < 0.99 and NumPy < 2.0. If
you need that kind of interoperability, consider using the Parquet file
format.

# 2024-08-17: Dataiter 0.99

- Adapt to changes in NumPy 2.0
- Bump NumPy dependency to >= 2.0

This is a minimal change to be NumPy 2.0 compatible. In the 0.99+
releases, we plan to adopt the new NumPy string dtype and fix any
regressions that come up, leading to a 1.0 release when everything looks
to be working reliably (#26). Anyone looking for extreme stability
should consider avoiding the 0.99+ releases and waiting for 1.0.

# 2024-06-24: Dataiter 0.51

- Mark NumPy dependency as < 2.0

# 2024-04-06: Dataiter 0.50

- `ListOfDicts.drop_na`: New method
- `ListOfDicts.keys`: New method
- `ListOfDicts.print_memory_use`: New method
- Fix tabular display of Unicode characters with width != 1
- Add dependency on wcwidth: https://pypi.org/project/wcwidth

# 2023-11-08: Dataiter 0.49

- `dt`: Handle all NaT input
- Migrate from `setup.py` to `hatch` and `pyproject.toml`

# 2023-10-08: Dataiter 0.48

- `Vector.as_datetime`: Add `precision` argument
- `Vector.concat`: New method
- `Vector.sort`: Fix sorting object vectors

# 2023-09-09: Dataiter 0.47

- `DataFrame`: Fix column and method name clash errors in certain operations
- `dt.replace`: Allow vector arguments the same length as `x`

# 2023-09-05: Dataiter 0.46

- `DataFrame.count`: New method, shorthand for
  `data.group_by(...).aggregate(n=di.count())`
- `Vector.rank`: Handle empty and all-NA vectors

# 2023-06-14: Dataiter 0.45

- `USE_NUMBA_CACHE`: New option, read from environment variable
  `DATAITER_USE_NUMBA_CACHE` if exists, defauls to `True`
- Fix a possible issue with Numba caching

# 2023-06-13: Dataiter 0.44

- Use `numba.extending.overload` instead of the deprecated
  `numba.generated_jit`

# 2023-06-08: Dataiter 0.43

- `DataFrame`: Don't try to do joins on NA values in `by` columns
- `DataFrame.drop_na`: New method

# 2023-05-30: Dataiter 0.42

- `DataFrame`: Truncate multiline strings when printing
- `DataFrame.from_arrow`: New method
- `DataFrame.read_parquet`: New method
- `DataFrame.to_arrow`: New method
- `DataFrame.write_parquet`: New method
- `read_parquet`: New function
- `Vector.__init__`: Fix type guessing when mixing Python and NumPy
  floats or integers and missing values
- Allow using a thousand separator when printing numbers,
  off by default, can be set with `dataiter.PRINT_THOUSAND_SEPARATOR`

# 2023-03-11: Dataiter 0.41

- Fix printing really small numbers

# 2023-02-21: Dataiter 0.40.1

- `DataFrame.modify`: Fix grouped modify on unsorted data frame

# 2023-02-20: Dataiter 0.40

- `Vector.map`: Add `dtype` argument

# 2023-02-06: Dataiter 0.39.1

- `ListOfDicts.to_data_frame`: Add `strings_as_object` argument

# 2023-01-21: Dataiter 0.39

- `read_csv`, `read_geojson`, `DataFrame.from_pandas`,
  `DataFrame.read_csv`, `GeoJSON.read`: Add `strings_as_object` argument

# 2022-12-15: Dataiter 0.38

- `DataFrame.slice_off`: New method
- `GeoJSON.to_data_frame`: New method
- Fix error with new column placeholder attributes in conjunction with
  pop, popitem and clear

# 2022-11-17: Dataiter 0.37

- `DataFrame`: Add placeholder attributes for columns so that
  tab completion of columns as attributes at a shell works
- `dt.from_string`: New function
- `dt.to_string`: New function
- `nrow`: Remove deprecated aggregation function
- Don't use Numba for aggregation involving strings due to bad performance

# 2022-10-16: Dataiter 0.36

- `dt`: New module for dealing with dates and datetimes

# 2022-10-03: Dataiter 0.35

- `DataFrame.from_pandas`: Speed up by avoiding unnecessary conversions
- `DataFrame.full_join`: Fix join and output when `by` is a tuple
- `GeoJSON`: Fix printing object

# 2022-09-17: Dataiter 0.34

- `Vector`: Handle timedeltas correctly for NA checks and printing
- `Vector.is_timedelta`: New method

# 2022-09-03: Dataiter 0.33

- `DataFrame.sort`: Convert object to string for sorting
- `Vector.sort`: Convert object to string for sorting
- Fix conditional Numba use when importing the numba package works,
  but caching doesn't
- Add `di-open` cli command (currently not part of the default install,
  but can be installed from source using `make install-cli`)

# 2022-04-02: Dataiter 0.32

- `DataFrame.modify`: Add support for grouped modification (#19)
- `DataFrame.split`: New method
- `ListOfDicts.split`: New method

# 2022-02-26: Dataiter 0.31

- `DataFrame.compare`: New experimental method
- `Vector.as_string`: Add `length` argument
- Change the documentation to default to the latest release ("stable")
  instead of the development version ("latest")

# 2022-02-19: Dataiter 0.30

- Use keyword-only arguments where appropriate – the general principle
  is that mandatory arguments are allowed as positional, but optional
  modifiers are keyword only
- Rename all instances of "missing" to "na", such as `Vector.is_missing`
  to `Vector.is_na`, the only exception being
  `ListOfDicts.fill_missing`, which becomes
  `ListOfDicts.fill_missing_keys`
- Truncate data frame object and string columns at
  `PRINT_TRUNCATE_WIDTH` (default 32) for printing

# 2022-02-09: Dataiter 0.29.2

- Fix aggregation functions to work with all main data types:
  boolean, integer, float, date, datetime and string
- Fix aggregation functions to handle all missing values (NaN, NaT,
  blank string) correctly, the same as implemented in Vector
- Rename aggregation functions' `dropna` arguments to `drop_missing`
- `first`, `last`, `nth`: Add `drop_missing` argument
- `Vector.drop_missing`: New method

# 2022-01-30: Dataiter 0.29.1

- `mode`: Fix to return first in case of ties (requires Python >= 3.8)
- `std`, `var`: Add `ddof` argument (defaults to 0 on account of Numba limitations)
- Don't try to dropna for non-float vectors in aggregation functions

# 2022-01-29: Dataiter 0.29

- Add shorthand helper functions for use with `DataFrame.aggregate`,
  optionally using Numba JIT-compiled code for speed
    - https://dataiter.readthedocs.io/en/latest/aggregation.html
    - https://dataiter.readthedocs.io/en/latest/data-frame.html#dataiter.DataFrame.aggregate
    - https://dataiter.readthedocs.io/en/latest/dataiter.html
- `DataFrame.map`: New method
- `ncol`: Removed
- `nrow`: Deprecated in favor of `dataiter.count`
- `read_csv`: New alias for `DataFrame.read_csv`
- `read_geojson`: New alias for `GeoJSON.read`
- `read_json`: New alias for `ListOfDicts.read_json`
- `read_npz`: New alias for `DataFrame.read_npz`

# 2022-01-09: Dataiter 0.28

- `DataFrame`: Make object columns work in various operations
- `DataFrame.from_json`: Add arguments `columns` and `dtypes`
- `DataFrame.from_pandas`: Add argument `dtypes`
- `DataFrame.full_join`: Speed up
- `DataFrame.read_csv`: Add argument `dtypes`
- `DataFrame.read_json`: Add arguments `columns` and `dtypes`
- `GeoJSON.read`: Add arguments `columns` and `dtypes`
- `ListOfDicts.fill_missing`: New method
- `ListOfDicts.from_json`: Add arguments `keys` and `types`
- `ListOfDicts.full_join`: Speed up
- `ListOfDicts.read_csv`: Add argument `types`, rename `columns` to `keys`
- `ListOfDicts.read_json`: Add arguments `keys` and `types`

# 2022-01-01: Dataiter 0.27

- `DataFrame`: Fix error message when column not found
- `DataFrame.aggregate`: Speed up
- `DataFrame.full_join`: Fix to join all possible columns
- `DataFrame.read_csv`: Try to avoid mixed types
- `ListOfDicts.full_join`: Fix to join all possible keys
- `ListOfDicts.write_csv`: Use minimal quoting
- `Vector.get_memory_use`: New method
- `Vector.rank`: Rewrite, add `method` argument
- `*.read_*`: Rename `fname` argument `path`
- `*.write_*`: Rename `fname` argument `path`
- Add comparison table dplyr vs. Dataiter vs. Pandas to documentation:
  <https://dataiter.readthedocs.io/en/latest/comparison.html>

# 2021-12-02: Dataiter 0.26

- `DataFrame.read_npz`: New method to read NumPy npz format
- `DataFrame.write_npz`: New method to write NumPy npz format
- `*.read_*`: Decompress `.bz2|.gz|.xz` automatically
- `*.write_*`: Compress `.bz2|.gz|.xz` automatically

# 2021-11-13: Dataiter 0.25

- `DataFrame.print_missing_counts`: Fix when nothing missing
- `Vector.replace_missing`: New method

# 2021-10-27: Dataiter 0.24

- `DataFrame.print_memory_use`: New method
- `ListOfDicts.write_csv`: Use less memory

# 2021-07-08: Dataiter 0.23

- `Vector.is_*`: Change to be methods instead of properties
- Drop deprecated use of `np.int`
- Drop deprecated comparisons against NaN

# 2021-05-13: Dataiter 0.22

- `ListOfDicts.map`: New method

# 2021-03-08: Dataiter 0.21

- `DataFrame.read_csv`: Add `columns` argument
- `ListOfDicts.read_csv`: Add `columns` argument

# 2021-03-06: Dataiter 0.20

- `DataFrame.*_join`: Handle differing by names via tuple argument
- `ListOfDicts.*_join`: Handle differing by names via tuple argument

# 2021-03-04: Dataiter 0.19

- Use terminal window width as maximum print width
- `Vector.__init__`: Handle NaN values in non-float vectors

# 2021-03-03: Dataiter 0.18

- `Vector.__init__`: Accept generators/iterators
- `Vector.map`: New method

# 2021-02-27: Dataiter 0.17

- `DataFrame.print_missing_counts`: New method
- `GeoJSON.read`: Handle properties differing between features
- `ListOfDicts.print_missing_counts`: New method
- `Vector.as_object`: New method

# 2020-10-03: Dataiter 0.16.1

- `GeoJSON.read`: Use warnings, not errors for ignored excess feature keys

# 2020-09-26: Dataiter 0.16

- `GeoJSON`: New class

# 2020-09-12: Dataiter 0.15

- `ListOfDicts.sort`: Handle descending sort for all types

# 2020-08-22: Dataiter 0.14

- `ListOfDicts`: Make obsoletion a warning instead of an error

# 2020-08-15: Dataiter 0.13

- `DataFrame`: Fix error printing blank strings (#8)

# 2020-07-25: Dataiter 0.12

- `DataFrame.filter`: Add `colname_value_pairs` argument
- `DataFrame.filter_out`: Add `colname_value_pairs` argument
- `ListOfDicts.__init__`: Remove arguments not intended for external use
- `ListOfDicts.rename`: Preserve order of keys
- Add documentation: https://dataiter.readthedocs.io/

# 2020-06-02: Dataiter 0.11

- `Vector.__init__`: Speed up by fixing type deduction

# 2020-05-28: Dataiter 0.10.1

- `ListOfDicts.select`: Fix return value (#7)

# 2020-05-21: Dataiter 0.10

- `DataFrame.aggregate`: Fix `UnicodeEncodeError` with string columns
- `DataFrame.unique`: Fix `UnicodeEncodeError` with string columns
- `ListOfDicts.select`: Return keys in requested order
- `Vector.__repr__`: Add custom conversion to string for display
- `Vector.__str__`: Add custom conversion to string for display
- `Vector.to_string`: Add custom conversion to string for display
- `Vector.to_strings`: Add custom conversion to string for display

# 2020-05-11: Dataiter 0.9

- `Array`: Rename to `Vector`
- `Vector.head`: New method
- `Vector.range`: New method
- `Vector.sample`: New method
- `Vector.sort`: New method
- `Vector.tail`: New method
- `Vector.unique`: New method

# 2020-05-10: Dataiter 0.8

- `DataFrame`: New class
- `ListOfDicts.__add__`: New method to support the `+` operator
- `ListOfDicts.__init__`: Rename, reorder arguments
- `ListOfDicts.__mul__`: New method to support the `*` operator
- `ListOfDicts.__repr__`: New method, format as JSON
- `ListOfDicts.__rmul__`: New method to support the `*` operator
- `ListOfDicts.__setitem__`: New method, coerce to `AttributeDict`
- `ListOfDicts.__str__`: New method, format as JSON
- `ListOfDicts.aggregate`: Speed up
- `ListOfDicts.anti_join`: New method
- `ListOfDicts.append`: New method
- `ListOfDicts.clear`: New method
- `ListOfDicts.extend`: New method
- `ListOfDicts.full_join`: New method
- `ListOfDicts.head`: New method
- `ListOfDicts.inner_join`: New method
- `ListOfDicts.insert`: New method
- `ListOfDicts.join`: Removed in favor of specific join types
- `ListOfDicts.left_join`: New method
- `ListOfDicts.pluck`: Add argument "default" to handle missing keys
- `ListOfDicts.print_`: New method
- `ListOfDicts.read_csv`: Add explicit arguments
- `ListOfDicts.read_json`: Relay arguments to `json.loads`
- `ListOfDicts.read_pickle`: New method
- `ListOfDicts.reverse`: New method
- `ListOfDicts.sample`: New method
- `ListOfDicts.semi_join`: New method
- `ListOfDicts.sort`: Change arguments to support sort direction better
- `ListOfDicts.tail`: New method
- `ListOfDicts.to_data_frame`: New method
- `ListOfDicts.to_pandas`: New method
- `ListOfDicts.unique`: Return unique by all keys if none given
- `ListOfDicts.write_csv`: Add explicit arguments
- `ListOfDicts.write_pickle`: New method

# 2019-12-03: Dataiter 0.7

- Make `sort` handle `None` values, sorted last

# 2019-11-29: Dataiter 0.6

- Fix `ObsoleteError` after multiple modifying actions

# 2019-11-10: Dataiter 0.5

- Add `read_csv`
- Add `read_json`
- Add `write_csv`
- Add `write_json`

# 2019-11-01: Dataiter 0.4

- Fix `ObsoleteError` with `deepcopy`
- Define `__deepcopy__` so that `copy.deepcopy` works too
- Add `copy` (and `__copy__` for `copy.copy`)

# 2019-11-01: Dataiter 0.3

- Mark `ListOfDicts` object obsolete thus preventing (accidental) use if
  a chained successor has modified the shared dicts
- Add `modify_if`

# 2019-10-31: Dataiter 0.2

- Speed up, mostly by avoiding copying (methods that modify dicts now do
  it in place rather than making a copy)

# 2019-09-29: Dataiter 0.1

- Initial release
