#!/usr/bin/env python3

import atexit
import cProfile
import dataiter as di
import os
import pstats
import tempfile

from dataiter import test

def data_frame(fname, nrow=1000000):
    data = test.data_frame(fname)
    n = nrow // data.nrow
    data = data.rbind(*([data] * n))
    return data.head(nrow)

def prof(data):
    data.group_by("make", "model").aggregate(n=di.nrow)

handle, fname = tempfile.mkstemp()
atexit.register(os.remove, fname)
data = data_frame("vehicles.csv")
cProfile.run("prof(data)", fname)
stats = pstats.Stats(fname)
stats.sort_stats(pstats.SortKey.CUMULATIVE)
stats.print_stats(25)
