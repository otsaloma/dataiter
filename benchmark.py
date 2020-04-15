#!/usr/bin/env python3

import sys
import time

from dataiter import DataFrame
from dataiter import test

def data_frame(fname):
    fname = test.get_data_filename(fname)
    extension = fname.split(".")[-1]
    read = getattr(DataFrame, f"read_{extension}")
    return read(fname)

def data_frame_rbind_001():
    data = data_frame("vehicles.csv")
    start = time.time()
    data = data.rbind(data)
    return time.time() - start

def data_frame_rbind_100():
    data = data_frame("vehicles.csv")
    start = time.time()
    data = data.rbind(*([data] * 100))
    return time.time() - start

BENCHMARKS = [
    "data_frame_rbind_001",
    "data_frame_rbind_100",
]

benchmarks = sys.argv[1:] or BENCHMARKS
for i, benchmark in enumerate(benchmarks):
    print(f"{i+1}/{len(benchmarks)}. {benchmark}: ", end="")
    elapsed = globals()[benchmark]()
    print("{:4.0f} ms".format(elapsed * 1000))
