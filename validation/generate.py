#!/usr/bin/env python3

import sys

from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import dataiter as di
import numpy as np

def read_csv(path):
    data = di.DataFrame.read_csv(path)
    for name in data.colnames:
        # Drop all rows with NAs to avoid upcasting to float
        # and differing NA representation in output.
        data = data.filter_out(data[name].is_missing())
        if data[name].is_string():
            # Use all lower case for strings to avoid differing
            # sorting lower vs. upper case characters.
            data[name] = np.char.lower(data[name])
    return data

# 1. Aggregate
(read_csv("../data/vehicles.csv")
 .group_by("make", "model")
 .aggregate(
     n=di.nrow,
     cyl_min=lambda x: x.cyl.min(),
     cyl_max=lambda x: x.cyl.max())
 .write_csv("1.py.csv"))

# 2. Sort
(read_csv("../data/vehicles.csv")
 .sort(make=1, model=1, year=1)
 .write_csv("2.py.csv"))

# 3. Unique
(read_csv("../data/vehicles.csv")
 .unique("make", "model", "year")
 .write_csv("3.py.csv"))
