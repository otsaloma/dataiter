#!/usr/bin/env python3
# Benchmark Polars equivalents of the data frame benchmarks in tools/benchmark.py.
# Usage: uv run --with polars python tools/benchmark-polars.py [-o OUTPUT] [PATTERN...]

import click
import functools
import numpy as np
import polars as pl
import tempfile
import time
import uuid

from dataiter import test

rng = np.random.default_rng(0)

@functools.cache
def data_frame(path, nrow=1_000_000):
    data = pl.read_csv(test.get_data_path(path), null_values=["NA"], infer_schema_length=None)
    n = nrow // len(data)
    data = pl.concat([data] * (n + 1)).head(nrow)
    return data[rng.permutation(len(data))]

@functools.cache
def data_frame_random(nrows, ngroups):
    return pl.DataFrame(dict(g=rng.integers(ngroups, size=nrows),
                             a=rng.normal(10, 2, nrows)))

def data_frame_aggregate_128():
    data = data_frame("vehicles.csv")
    start = time.time()
    (data
     .group_by("make", maintain_order=True)
     .agg(pl.len().alias("n"),
          pl.col("hwy").mean(),
          pl.col("cty").mean()))
    return time.time() - start

def data_frame_aggregate_3264():
    data = data_frame("vehicles.csv")
    start = time.time()
    (data
     .group_by("make", "model", maintain_order=True)
     .agg(pl.len().alias("n"),
          pl.col("hwy").mean(),
          pl.col("cty").mean()))
    return time.time() - start

def data_frame_aggregate_14668():
    data = data_frame("vehicles.csv")
    start = time.time()
    (data
     .group_by("make", "model", "year", maintain_order=True)
     .agg(pl.len().alias("n"),
          pl.col("hwy").mean(),
          pl.col("cty").mean()))
    return time.time() - start

def data_frame_aggregate_100000_lambda():
    data = data_frame_random(1_000_000, 100_000)
    start = time.time()
    (data
     .group_by("g", maintain_order=True)
     .agg(pl.col("a").map_elements(lambda x: np.mean(x), return_dtype=pl.Float64).alias("a_mean"),
          pl.col("a").map_elements(lambda x: np.std(x), return_dtype=pl.Float64).alias("a_std")))
    return time.time() - start

def data_frame_aggregate_100000_short():
    data = data_frame_random(1_000_000, 100_000)
    start = time.time()
    (data
     .group_by("g", maintain_order=True)
     .agg(pl.col("a").mean().alias("a_mean"),
          pl.col("a").std(ddof=0).alias("a_std")))
    return time.time() - start

def data_frame_filter_is_in_1000():
    data = data_frame("vehicles.csv")
    models = data["model"].unique(maintain_order=True)[:1000].to_list()
    start = time.time()
    data.filter(pl.col("model").is_in(models))
    return time.time() - start

def data_frame_full_join():
    data = data_frame("vehicles.csv")
    meta = data.select("make", "model").unique(maintain_order=True)
    meta = pl.concat([meta, meta.with_columns(model=pl.lit("X"))])
    meta = meta.with_columns(random=rng.random(len(meta)))
    start = time.time()
    data.join(meta, on=["make", "model"], how="full", coalesce=True, maintain_order="left_right")
    return time.time() - start

def data_frame_left_join():
    data = data_frame("vehicles.csv")
    meta = data.select("make", "model").unique(maintain_order=True)
    meta = meta.with_columns(random=rng.random(len(meta)))
    start = time.time()
    data.join(meta, on=["make", "model"], how="left", maintain_order="left")
    return time.time() - start

def data_frame_left_join_int():
    data = data_frame("vehicles.csv")
    meta = data.select("id").unique(maintain_order=True)
    meta = meta.with_columns(random=rng.random(len(meta)))
    start = time.time()
    data.join(meta, on="id", how="left", maintain_order="left")
    return time.time() - start

def data_frame_left_join_mixed():
    data = data_frame("vehicles.csv")
    meta = data.select("make", "year").unique(maintain_order=True)
    meta = meta.with_columns(random=rng.random(len(meta)))
    start = time.time()
    data.join(meta, on=["make", "year"], how="left", maintain_order="left")
    return time.time() - start

def data_frame_left_join_to_larger():
    meta = data_frame("vehicles.csv")
    meta = meta.with_columns(uuid=pl.Series([str(uuid.uuid4()) for i in range(len(meta))]))
    data = meta.select("uuid").head(300)
    start = time.time()
    data.join(meta, on="uuid", how="left", maintain_order="left")
    return time.time() - start

def data_frame_modify_grouped():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.with_columns(hwy_share=pl.col("hwy") / pl.col("hwy").sum().over("make", "model"))
    return time.time() - start

def data_frame_pivot_longer():
    data = data_frame("vehicles.csv").unique(["make", "model", "year"], maintain_order=True)
    data = data.pivot(on="make", index=["model", "year"], values="hwy")
    start = time.time()
    data.unpivot(index=["model", "year"], variable_name="make", value_name="hwy")
    return time.time() - start

def data_frame_pivot_wider():
    data = data_frame("vehicles.csv").unique(["make", "model", "year"], maintain_order=True)
    start = time.time()
    data.pivot(on="make", index=["model", "year"], values="hwy")
    return time.time() - start

# Polars concatenates without copying by default (rechunk=False),
# making rbind_2 and rbind_100 nearly free.

def data_frame_rbind_2():
    data = data_frame("vehicles.csv", 500_000)
    start = time.time()
    pl.concat([data] * 2)
    return time.time() - start

def data_frame_rbind_100():
    data = data_frame("vehicles.csv", 10_000)
    start = time.time()
    pl.concat([data] * 100)
    return time.time() - start

def data_frame_rbind_100000():
    data = data_frame("vehicles.csv", 10)
    start = time.time()
    pl.concat([data] * 100_000)
    return time.time() - start

def data_frame_read_csv():
    path = test.get_data_path("vehicles.csv")
    start = time.time()
    pl.read_csv(path, null_values=["NA"], infer_schema_length=None)
    return time.time() - start

def data_frame_read_parquet():
    data = data_frame("vehicles.csv")
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        data.write_parquet(f.name)
        start = time.time()
        pl.read_parquet(f.name)
        return time.time() - start

def data_frame_sort():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.sort("make", "model", "year", maintain_order=True)
    return time.time() - start

def data_frame_sort_categorical():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.sort("make", maintain_order=True)
    return time.time() - start

def data_frame_sort_unique():
    data = data_frame("vehicles.csv")
    data = data.with_columns(uuid=pl.Series([str(uuid.uuid4()) for i in range(len(data))]))
    start = time.time()
    data.sort("uuid", maintain_order=True)
    return time.time() - start

def data_frame_unique():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.unique(["make", "model", "year"], keep="first", maintain_order=True)
    return time.time() - start

def data_frame_write_csv():
    data = data_frame("vehicles.csv")
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        start = time.time()
        data.write_csv(f.name)
        return time.time() - start

def data_frame_write_parquet():
    data = data_frame("vehicles.csv")
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        start = time.time()
        data.write_parquet(f.name)
        return time.time() - start

def is_benchmark(name):
    return name.startswith("data_frame_") and name != "data_frame_random"

BENCHMARKS = sorted(filter(is_benchmark, dir()), key=lambda x: (
    [x.zfill(9) if x.isdigit() else x for x in x.split("_")]))

def run_benchmarks(benchmarks, output, rounds):
    width = max(map(len, benchmarks)) + 2
    for i, benchmark in enumerate(benchmarks):
        print(f"{i+1:2d}/{len(benchmarks)}. ", end="", flush=True)
        print(f"{benchmark+' ':.<{width}} ", end="", flush=True)
        try:
            f = globals()[benchmark]
            elapsed = 1000 * min(f() for i in range(rounds))
            print(f"{elapsed:5.0f} ms", flush=True)
        except Exception as error:
            elapsed = -1
            print(error.__class__.__name__)
            if not output: raise
        yield {"name": benchmark, "elapsed": round(elapsed)}

@click.command()
@click.option("-o", "--output", help="Filename for optional CSV output")
@click.option("-r", "--rounds", default=5, help="Number of rounds per benchmark")
@click.option("--version", default=f"polars {pl.__version__}", help="Version for CSV output")
@click.argument("pattern", nargs=-1)
def main(output, rounds, version, pattern):
    pattern = pattern or "_"
    f = lambda x: any(y in x for y in pattern)
    benchmarks = list(filter(f, BENCHMARKS))
    results = pl.DataFrame(list(run_benchmarks(benchmarks, output, rounds)))
    results = results.with_columns(version=pl.lit(version))
    if output:
        assert output.endswith(".csv")
        print(f"Writing {output}...")
        results.write_csv(output)

if __name__ == "__main__":
    main()
