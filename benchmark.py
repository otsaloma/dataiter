#!/usr/bin/env python3

import click
import dataiter as di
import numpy as np
import random
import time

from dataiter import test
from statistics import mean


def data_frame(path, nrow=1_000_000):
    data = test.data_frame(path)
    n = nrow // data.nrow
    data = data.rbind(*([data] * n))
    return data.head(nrow)

def data_frame_random(nrows, ngroups):
    return di.DataFrame(g=np.random.choice(ngroups, nrows, replace=True),
                        a=np.random.normal(10, 2, nrows))

def data_frame_full_join():
    data = data_frame("vehicles.csv")
    meta = data.select("make", "model").unique()
    meta = meta.rbind(meta.modify(model="X"))
    meta.random = np.random.random(meta.nrow)
    assert meta.anti_join(data, "make", "model").nrow > 0
    start = time.time()
    data.full_join(meta, "make", "model")
    return time.time() - start

def data_frame_group_by_aggregate_128():
    data = data_frame("vehicles.csv")
    start = time.time()
    (data
     .group_by("make")
     .aggregate(
         n=di.count(),
         hwy=di.mean("hwy"),
         cty=di.mean("cty")))
    return time.time() - start

def data_frame_group_by_aggregate_3264():
    data = data_frame("vehicles.csv")
    start = time.time()
    (data
     .group_by("make", "model")
     .aggregate(
         n=di.count(),
         hwy=di.mean("hwy"),
         cty=di.mean("cty")))
    return time.time() - start

def data_frame_group_by_aggregate_14668():
    data = data_frame("vehicles.csv")
    start = time.time()
    (data
     .group_by("make", "model", "year")
     .aggregate(
         n=di.count(),
         hwy=di.mean("hwy"),
         cty=di.mean("cty")))
    return time.time() - start

def data_frame_group_by_aggregate_100000():
    data = data_frame_random(1_000_000, 100_000)
    start = time.time()
    (data
     .group_by("g")
     .aggregate(
         a_mean=di.mean("a"),
         a_std=di.std("a")))
    return time.time() - start

def data_frame_group_by_aggregate_100000_lambda():
    data = data_frame_random(1_000_000, 100_000)
    start = time.time()
    (data
     .group_by("g")
     .aggregate(
         a_mean=lambda x: np.mean(x.a),
         a_std=lambda x: np.std(x.a)))
    return time.time() - start

def data_frame_left_join():
    data = data_frame("vehicles.csv")
    meta = data.select("make", "model").unique()
    meta.random = np.random.random(meta.nrow)
    start = time.time()
    data.left_join(meta, "make", "model")
    return time.time() - start

def data_frame_read_csv():
    start = time.time()
    test.data_frame("vehicles.csv")
    return time.time() - start

def data_frame_read_json():
    start = time.time()
    test.data_frame("vehicles.json")
    return time.time() - start

def data_frame_rbind_2():
    # 2 * 500,000 = 1,000,000
    data = data_frame("vehicles.csv", 500_000)
    start = time.time()
    data.rbind(data)
    return time.time() - start

def data_frame_rbind_100():
    # 100 * 10,000 = 1,000,000
    data = data_frame("vehicles.csv", 10_000)
    start = time.time()
    data.rbind(*([data] * (100 - 1)))
    return time.time() - start

def data_frame_rbind_100000():
    # 100,000 * 10 = 1,000,000
    data = data_frame("vehicles.csv", 10)
    start = time.time()
    data.rbind(*([data] * (100_000 - 1)))
    return time.time() - start

def data_frame_sort():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.sort(make=1, model=1, year=-1)
    return time.time() - start

def list_of_dicts(path, length=100_000):
    data = test.list_of_dicts(path)
    n = length // len(data) + 1
    data = data * n
    return data.head(length)

def list_of_dicts_full_join():
    data = list_of_dicts("vehicles.json")
    meta = data.deepcopy().select("make", "model").unique()
    meta = meta + meta.deepcopy().modify(model=lambda x: "X")
    meta = meta.modify(random=lambda x: random.random())
    assert len(meta.anti_join(data, "make", "model")) > 0
    start = time.time()
    data.full_join(meta, "make", "model")
    return time.time() - start

def list_of_dicts_group_by_aggregate_128():
    data = list_of_dicts("vehicles.json")
    start = time.time()
    (data
     .group_by("make")
     .aggregate(
         n=len,
         hwy=lambda x: mean(x.pluck("hwy")),
         cty=lambda x: mean(x.pluck("cty"))))
    return time.time() - start

def list_of_dicts_group_by_aggregate_3264():
    data = list_of_dicts("vehicles.json")
    start = time.time()
    (data
     .group_by("make", "model")
     .aggregate(
         n=len,
         hwy=lambda x: mean(x.pluck("hwy")),
         cty=lambda x: mean(x.pluck("cty"))))
    return time.time() - start

def list_of_dicts_group_by_aggregate_14668():
    data = list_of_dicts("vehicles.json")
    start = time.time()
    (data
     .group_by("make", "model", "year")
     .aggregate(
         n=len,
         hwy=lambda x: mean(x.pluck("hwy")),
         cty=lambda x: mean(x.pluck("cty"))))
    return time.time() - start

def list_of_dicts_left_join():
    data = list_of_dicts("vehicles.json")
    meta = data.deepcopy().select("make", "model").unique()
    meta = meta.modify(random=lambda x: random.random())
    start = time.time()
    data.left_join(meta, "make", "model")
    return time.time() - start

def list_of_dicts_read_csv():
    start = time.time()
    test.list_of_dicts("vehicles.csv")
    return time.time() - start

def list_of_dicts_read_json():
    start = time.time()
    test.list_of_dicts("vehicles.json")
    return time.time() - start

def list_of_dicts_sort():
    data = list_of_dicts("vehicles.csv")
    start = time.time()
    data.sort(make=1, model=1, year=-1)
    return time.time() - start

def vector_fast_list():
    seq = list(range(1_000_000))
    start = time.time()
    di.Vector.fast(seq, int)
    return time.time() - start

def vector_fast_np():
    seq = list(range(1_000_000))
    seq = np.array(seq)
    start = time.time()
    di.Vector.fast(seq, int)
    return time.time() - start

def vector_new_list():
    seq = list(range(1_000_000))
    start = time.time()
    di.Vector(seq)
    return time.time() - start

def vector_new_np():
    seq = list(range(1_000_000))
    seq = np.array(seq)
    start = time.time()
    di.Vector(seq)
    return time.time() - start


BENCHMARKS = sorted([
    x for x in dir() if
    x.startswith(("data_frame_", "list_of_dicts_", "vector_")) and
    x not in ["data_frame_random"]
], key=lambda x: (
    [x.zfill(9) if x.isdigit() else x for x in x.split("_")]
))

@click.command()
@click.option("-o", "--output", help="Filename for optional CSV output")
@click.option("-r", "--rounds", default=5, help="Number of rounds per benchmark")
@click.argument("pattern", nargs=-1)
def main(output, rounds, pattern):
    """Benchmark dataiter functions."""
    benchmarks = BENCHMARKS.copy()
    if pattern:
        f = lambda x: any(y in x for y in pattern)
        benchmarks = list(filter(f, benchmarks))
    results = di.ListOfDicts()
    for i, benchmark in enumerate(benchmarks):
        width = max(map(len, benchmarks))
        padding = "." * (width + 1 - len(benchmark))
        print(f"{i+1:2d}/{len(benchmarks)}. {benchmark} {padding} ", end="", flush=True)
        try:
            f = globals()[benchmark]
            elapsed = 1000 * min(f() for i in range(rounds))
            print("{:5.0f} ms".format(elapsed), flush=True)
        except Exception as e:
            elapsed = -1
            print(e.__class__.__name__)
            if not output: raise
        list.append(results, {
            "name":    benchmark,
            "version": di.__version__,
            "elapsed": round(elapsed),
        })
    if output:
        assert output.endswith(".csv")
        print(f"Writing {output}...")
        results.write_csv(output)


if __name__ == "__main__":
    main()
