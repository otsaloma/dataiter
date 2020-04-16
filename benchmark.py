#!/usr/bin/env python3

import random
import sys
import time

from dataiter import DataFrame
from dataiter import ListOfDicts
from dataiter import test

def data_frame(fname):
    fname = test.get_data_filename(fname)
    extension = fname.split(".")[-1]
    read = getattr(DataFrame, f"read_{extension}")
    return read(fname)

def data_frame_full_join():
    return -0.001

def data_frame_group_by_aggregate():
    return -0.001

def data_frame_left_join():
    return -0.001

def data_frame_rbind_001():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.rbind(data)
    return time.time() - start

def data_frame_rbind_100():
    data = data_frame("vehicles.csv")
    start = time.time()
    data.rbind(*([data] * 100))
    return time.time() - start

def list_of_dicts(fname):
    fname = test.get_data_filename(fname)
    extension = fname.split(".")[-1]
    read = getattr(ListOfDicts, f"read_{extension}")
    return read(fname)

def list_of_dicts_full_join():
    data = list_of_dicts("vehicles.json")
    meta = (list_of_dicts("vehicles.json")
            .select("make", "model")
            .unique("make", "model")
            .modify(random=lambda x: random.random()))

    start = time.time()
    data.full_join(meta, "make", "model")
    return time.time() - start

def list_of_dicts_group_by_aggregate():
    data = list_of_dicts("vehicles.json")
    start = time.time()
    data.group_by("make").aggregate(n=len)
    return time.time() - start

def list_of_dicts_left_join():
    data = list_of_dicts("vehicles.json")
    meta = (list_of_dicts("vehicles.json")
            .select("make", "model")
            .unique("make", "model")
            .modify(x=lambda x: random.random()))

    start = time.time()
    data.left_join(meta, "make", "model")
    return time.time() - start

is_benchmark = lambda x: x.startswith(("data_frame_", "list_of_dicts_"))
benchmarks = sys.argv[1:] or list(filter(is_benchmark, dir()))
for i, benchmark in enumerate(benchmarks):
    width = max(map(len, benchmarks))
    padding = "." * (width + 1 - len(benchmark))
    label = f"{benchmark} {padding}"
    print(f"{i+1:2d}/{len(benchmarks)}. {label} ", end="")
    elapsed = globals()[benchmark]()
    print("{:5.0f} ms".format(elapsed * 1000))
