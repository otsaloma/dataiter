Out-of-Memory Data
==================

.. meta::
   :description: Guide to working with data too large to fit in memory

Dataiter works strictly in-memory: a :class:`.DataFrame` is a dictionary of
NumPy arrays and operations return new data frames with all columns copied. You
need memory for at least a few times the size of the data you work with. You can
check the size of your data frame with :meth:`.DataFrame.print_memory_use`.

If your data doesn't fit in memory, the following sections describe how you can
try to work around the problem. If none of that helps or if you consistently
need to process out-of-memory data, consider using `DuckDB
<https://duckdb.org/>`_ or `Polars <https://pola.rs/>`_ instead.

Read Only What You Need
-----------------------

Data files often have more columns than you need. The ``read_*`` functions take
a ``select`` argument to read only the given columns, also as glob patterns.
With Parquet, which stores data by column, the columns not selected are not read
at all, which makes reading also faster. This alone is often enough to fit data
in memory.

.. code-block:: python

    data = di.read_parquet("sales.parquet", select=["customer_id", "ts", "price"])

Processing in Chunks
--------------------

:meth:`.DataFrame.read_parquet_chunked` reads a Parquet file, yielding it one
chunk (given amount of rows) at a time. You can then reduce each chunk to a
result and combine the chunked results, so that you never need memory for the
whole data, but only for one chunk + the accumulating result.

As an example, consider finding the latest order of each customer from 150M
orders of 15M customers. For each chunk, we combine the chunk with the latest
orders found so far and keep only the latest order per customer. This works
because the latest order of combined data is the latest of the latest orders of
its parts.

.. code-block:: python

    latest = di.DataFrame()
    select = ["customer_id", "order_id", "ts"]
    for chunk in di.read_parquet_chunked("sales.parquet", 10_000_000, select=select):
        latest = latest.rbind(chunk).sort(ts=-1, order_id=-1).unique("customer_id")

Example simplified from `DuckDB 2.0 vs Polars vs Spark on One Machine: 150M Rows
Under a 4.5 GiB Cap <https://dev.to/firfircelik/duckdb-20-vs-polars-vs-spark-on-one-machine-150m-rows-under-a-45-gib-cap-h38>`_.
Dataiter's speed here is not far from Polars and memory use stays well under
control, but does require you to write more complicated code to loop over
chunks.

Another common case is grouped aggregation. Aggregate each chunk, collect the
results and aggregate those again, e.g. counts by summing them. Means you can
calculate from sums and counts at the end. Generally, processing in chunks works
when the result can be calculated from the results of chunks and the combined
result fits in memory.

There's no chunked reading for other file formats, as they can't be read in
chunks efficiently. If you have a huge CSV file, convert it to Parquet first,
e.g. with DuckDB: ``duckdb -c "COPY (FROM 'data.csv') TO 'data.parquet'"``.

Combining DuckDB and Dataiter
-----------------------------

DuckDB can query Parquet and CSV files directly with SQL, reading only what the
query needs, in parallel and within a memory limit. You can use DuckDB's Python
package to reduce data to a size that fits in memory, convert the result to a
Dataiter data frame via Arrow and continue from there with Dataiter.

.. code-block:: python

    data = di.DataFrame.from_arrow(duckdb.sql("""
        SELECT customer_id, ts, qty, price
        FROM 'sales.parquet'
        WHERE ts >= '2026-09-01' AND status = 'completed'
    """).to_arrow_table())

Use ``to_arrow_table``, as ``arrow`` returns a reader of record batches, which
:meth:`.DataFrame.from_arrow` doesn't handle.
