#!/usr/bin/env python3

import cProfile
import dataiter as di
import pstats

from dataiter import test

def data_frame(path, nrow=1000000):
    data = test.data_frame(path)
    n = nrow // data.nrow
    data = data.rbind(*([data] * n))
    return data.head(nrow)

data = data_frame("vehicles.csv")
pr = cProfile.Profile()
pr.enable()
data.group_by("make", "model").aggregate(n=di.nrow)
pr.disable()
ps = pstats.Stats(pr)
ps.sort_stats("cumtime")
ps.print_stats(20, "/dataiter/")
