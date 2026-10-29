#!/usr/bin/env python3
# Benchmark Pandas equivalents of the data frame benchmarks in tools/benchmark.py.
# Usage: uv run python tools/benchmark-pandas.py [-o OUTPUT] [PATTERN...]

import click
import functools
import numpy as np
import pandas as pd
import tempfile
import time
import uuid

from dataiter import test

rng = np.random.default_rng(0)

@functools.cache
def _data_frame(path, nrow):
    data = pd.read_csv(test.get_data_path(path))
    n = nrow // len(data)
    data = pd.concat([data] * (n + 1), ignore_index=True).head(nrow)
    return data.iloc[rng.permutation(len(data))].reset_index(drop=True)

def data_frame(path, nrow=1_000_000):
    return _data_frame(path, nrow).copy()

@functools.cache
def _data_frame_random(nrows, ngroups):
    return pd.DataFrame(dict(g=rng.integers(ngroups, size=nrows),
                             a=rng.normal(10, 2, nrows)))

def data_frame_random(nrows, ngroups):
    return _data_frame_random(nrows, ngroups).copy()

def data_frame_aggregate_128():
    data = data_frame("vehicles.csv")
    start = time.time()
    (data
     .groupby("make")
     .agg(n=("hwy", "size"),
          hwy=("hwy", "mean"),
          cty=("cty", "mean"))
     .reset_index())
    return time.time() - start

def data_frame_aggregate_3264():
    data = data_frame("vehicles.csv")
    start = time.time()
    (data
     .groupby(["make", "model"])
     .agg(n=("hwy", "size"),
          hwy=("hwy", "mean"),
          cty=("cty", "mean"))
     .reset_index())
    return time.time() - start

def data_frame_aggregate_14668():
    data = data_frame("vehicles.csv")
    start = time.time()
    (data
     .groupby(["make", "model", "year"])
     .agg(n=("hwy", "size"),
          hwy=("hwy", "mean"),
          cty=("cty", "mean"))
     .reset_index())
    return time.time() - start

def data_frame_aggregate_100000_lambda():
    data = data_frame_random(1_000_000, 100_000)
    start = time.time()
    (data
     .groupby("g")
     .agg(a_mean=("a", lambda x: np.mean(x)),
          a_std=("a", lambda x: np.std(x)))
     .reset_index())
    return time.time() - start

def data_frame_aggregate_100000_short():
    data = data_frame_random(1_000_000, 100_000)
    start = time.time()
    (data
     .groupby("g")
     .agg(a_mean=("a", "mean"),
          a_std=("a", "std"))
     .reset_index())
    return time.time() - start

def data_frame_filter_is_in_1000():
    data = data_frame("vehicles.csv")
    models = data.model.unique()[:1000].tolist()
    start = time.time()
    data[data.model.isin(models)]
    return time.time() - start

def data_frame_full_join():
    data = data_frame("vehicles.csv")
    meta = data[["make", "model"]].drop_duplicates()
    meta = pd.concat([meta, meta.assign(model="X")], ignore_index=True)
    meta["random"] = rng.random(len(meta))
    start = time.time()
    data.merge(meta, on=["make", "model"], how="outer")
    return time.time() - start

def data_frame_left_join():
    data = data_frame("vehicles.csv")
    meta = data[["make", "model"]].drop_duplicates()
    meta["random"] = rng.random(len(meta))
    start = time.time()
    data.merge(meta, on=["make", "model"], how="left")
    return time.time() - start

def data_frame_left_join_int():
    data = data_frame("vehicles.csv")
    meta = data[["id"]].drop_duplicates()
    meta["random"] = rng.random(len(meta))
    start = time.time()
    data.merge(meta, on="id", how="left")
    return time.time() - start

def data_frame_left_join_mixed():
    data = data_frame("vehicles.csv")
    meta = data[["make", "year"]].drop_duplicates()
    meta["random"] = rng.random(len(meta))
    start = time.time()
    data.merge(meta, on=["make", "year"], how="left")
    return time.time() - start

def data_frame_left_join_to_larger():
    meta = data_frame("vehicles.csv")
    meta["uuid"] = [str(uuid.uuid4()) for i in range(len(meta))]
    data = meta[["uuid"]].head(300)
    start = time.time()
    data.merge(meta, on="uuid", how="left")
    return time.time() - start

def data_frame_modify_grouped():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.assign(hwy_share=data.hwy / data.groupby(["make", "model"]).hwy.transform("sum"))
    return time.time() - start

def data_frame_pivot_longer():
    data = data_frame("vehicles.csv").drop_duplicates(["make", "model", "year"])
    data = data.pivot(index=["model", "year"], columns="make", values="hwy").reset_index()
    data.columns.name = None
    start = time.time()
    data.melt(id_vars=["model", "year"], var_name="make", value_name="hwy")
    return time.time() - start

def data_frame_pivot_wider():
    data = data_frame("vehicles.csv").drop_duplicates(["make", "model", "year"])
    start = time.time()
    data.pivot(index=["model", "year"], columns="make", values="hwy").reset_index()
    return time.time() - start

def data_frame_rbind_2():
    data = data_frame("vehicles.csv", 500_000)
    start = time.time()
    pd.concat([data] * 2, ignore_index=True)
    return time.time() - start

def data_frame_rbind_100():
    data = data_frame("vehicles.csv", 10_000)
    start = time.time()
    pd.concat([data] * 100, ignore_index=True)
    return time.time() - start

def data_frame_rbind_100000():
    data = data_frame("vehicles.csv", 10)
    start = time.time()
    pd.concat([data] * 100_000, ignore_index=True)
    return time.time() - start

def data_frame_read_csv():
    path = test.get_data_path("vehicles.csv")
    start = time.time()
    pd.read_csv(path)
    return time.time() - start

def data_frame_read_parquet():
    data = data_frame("vehicles.csv")
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        data.to_parquet(f.name, index=False)
        start = time.time()
        pd.read_parquet(f.name)
        return time.time() - start

def data_frame_sort():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.sort_values(["make", "model", "year"])
    return time.time() - start

def data_frame_sort_categorical():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.sort_values("make")
    return time.time() - start

def data_frame_sort_unique():
    data = data_frame("vehicles.csv")
    data["uuid"] = [str(uuid.uuid4()) for i in range(len(data))]
    start = time.time()
    data.sort_values("uuid")
    return time.time() - start

def data_frame_unique():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.drop_duplicates(["make", "model", "year"])
    return time.time() - start

def data_frame_write_csv():
    data = data_frame("vehicles.csv")
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        start = time.time()
        data.to_csv(f.name, index=False)
        return time.time() - start

def data_frame_write_parquet():
    data = data_frame("vehicles.csv")
    with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
        start = time.time()
        data.to_parquet(f.name, index=False)
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
@click.option("--version", default=f"pandas {pd.__version__}", help="Version for CSV output")
@click.argument("pattern", nargs=-1)
def main(output, rounds, version, pattern):
    pattern = pattern or "_"
    f = lambda x: any(y in x for y in pattern)
    benchmarks = list(filter(f, BENCHMARKS))
    results = pd.DataFrame(run_benchmarks(benchmarks, output, rounds))
    results["version"] = version
    if output:
        assert output.endswith(".csv")
        print(f"Writing {output}...")
        results.to_csv(output, index=False)

if __name__ == "__main__":
    main()
