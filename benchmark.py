#!/usr/bin/env python3

import dataiter as di
import numpy as np
import random
import sys
import time

from dataiter import test

def data_frame(fname, nrow=1000000):
    data = test.data_frame(fname)
    n = nrow // data.nrow
    data = data.rbind(*([data] * n))
    return data.head(nrow)

def data_frame_full_join():
    data = data_frame("vehicles.csv")
    meta = data.select("make", "model").unique()
    meta.random = np.random.random(meta.nrow)
    start = time.time()
    data.full_join(meta, "make", "model")
    return time.time() - start

def data_frame_group_by_aggregate_00128():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.group_by("make").aggregate(n=di.nrow)
    return time.time() - start

def data_frame_group_by_aggregate_03264():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.group_by("make", "model").aggregate(n=di.nrow)
    return time.time() - start

def data_frame_group_by_aggregate_14668():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.group_by("make", "model", "year").aggregate(n=di.nrow)
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

def data_frame_rbind_000002():
    # 2 * 500,000 = 1,000,000
    data = data_frame("vehicles.csv", 500000)
    start = time.time()
    data.rbind(data)
    return time.time() - start

def data_frame_rbind_000100():
    # 100 * 10,000 = 1,000,000
    data = data_frame("vehicles.csv", 10000)
    start = time.time()
    data.rbind(*([data] * (100 - 1)))
    return time.time() - start

def data_frame_rbind_100000():
    # 100,000 * 10 = 1,000,000
    data = data_frame("vehicles.csv", 10)
    start = time.time()
    data.rbind(*([data] * (100000 - 1)))
    return time.time() - start

def data_frame_sort():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.sort(year=-1, make=1, model=1)
    return time.time() - start

def list_of_dicts(fname, length=100000):
    data = test.list_of_dicts(fname)
    n = length // len(data) + 1
    data = data * n
    return data.head(length)

def list_of_dicts_full_join():
    data = list_of_dicts("vehicles.json")
    meta = data.deepcopy().select("make", "model").unique()
    meta = meta.modify(random=lambda x: random.random())
    start = time.time()
    data.full_join(meta, "make", "model")
    return time.time() - start

def list_of_dicts_group_by_aggregate_00128():
    data = list_of_dicts("vehicles.json")
    start = time.time()
    data.group_by("make").aggregate(n=len)
    return time.time() - start

def list_of_dicts_group_by_aggregate_03264():
    data = list_of_dicts("vehicles.json")
    start = time.time()
    data.group_by("make", "model").aggregate(n=len)
    return time.time() - start

def list_of_dicts_group_by_aggregate_14668():
    data = list_of_dicts("vehicles.json")
    start = time.time()
    data.group_by("make", "model", "year").aggregate(n=len)
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
    data.sort(year=-1, make=1, model=1)
    return time.time() - start

def vector_fast_list():
    seq = list(range(1000000))
    start = time.time()
    di.Vector.fast(seq, int)
    return time.time() - start

def vector_fast_np():
    seq = list(range(1000000))
    seq = np.array(seq)
    start = time.time()
    di.Vector.fast(seq, int)
    return time.time() - start

def vector_new_list():
    seq = list(range(1000000))
    start = time.time()
    di.Vector(seq)
    return time.time() - start

def vector_new_np():
    seq = list(range(1000000))
    seq = np.array(seq)
    start = time.time()
    di.Vector(seq)
    return time.time() - start

is_benchmark = lambda x: x.startswith(("data_frame_", "list_of_dicts_", "vector_"))
benchmarks = list(filter(is_benchmark, dir()))
if sys.argv[1:]:
    # If arguments given, limit to matching benchmarks.
    f = lambda x: any(y in x for y in sys.argv[1:])
    benchmarks = list(filter(f, benchmarks))
for i, benchmark in enumerate(benchmarks):
    width = max(map(len, benchmarks))
    padding = "." * (width + 1 - len(benchmark))
    label = f"{benchmark} {padding}"
    print(f"{i+1:2d}/{len(benchmarks)}. {label} ", end="", flush=True)
    elapsed = globals()[benchmark]()
    print("{:5.0f} ms".format(elapsed * 1000), flush=True)
