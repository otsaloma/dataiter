#!/usr/bin/env python3

import sys
sys.path.insert(0, "..")

import dataiter as di
import statistics

from statistics import mean
from statistics import median
from statistics import mode

def read_json(path):
    data = di.read_json(path)
    for name in list(data[0].keys()):
        # Drop all rows with NAs to avoid upcasting to float
        # and differing NA representation in output.
        data = data.filter_out(lambda x: x[name] is None)
        for item in data:
            if isinstance(item[name], str):
                # Use all lower case for strings to avoid differing
                # sorting of lower vs. upper case characters.
                item[name] = item[name].lower()
    return data

round2 = lambda x: round(x, 2) if x is not None else None
stdev = lambda x: statistics.stdev(x) if len(x) > 1 else None
variance = lambda x: statistics.variance(x) if len(x) > 1 else None

# AGGREGATE
(read_json("../data/vehicles.json")
 .modify(fuel_regular=lambda x: x.fuel == "regular")
 .group_by("make", "model")
 .aggregate(
     all_fuel_regular=lambda x: all(x.pluck("fuel_regular")),
     any_fuel_regular=lambda x: any(x.pluck("fuel_regular")),
     count=len,
     count_unique_cyl=lambda x: len(set(x.pluck("cyl"))),
     first_hwy=lambda x: x[0].hwy,
     last_hwy=lambda x: x[-1].hwy,
     max_hwy=lambda x: max(x.pluck("hwy")),
     mean_hwy=lambda x: mean(x.pluck("hwy")),
     median_hwy=lambda x: median(x.pluck("hwy")),
     min_hwy=lambda x: min(x.pluck("hwy")),
     mode_year=lambda x: mode(x.pluck("year")),
     nth_id=lambda x: x[0].id,
     quantile_hwy=lambda x: di.quantile(di.Vector(x.pluck("hwy")), 0.75),
     std_hwy=lambda x: stdev(x.pluck("hwy")),
     sum_hwy=lambda x: sum(x.pluck("hwy")),
     var_hwy=lambda x: variance(x.pluck("hwy")))
 .modify(mean_hwy=lambda x: round2(x.mean_hwy))
 .modify(std_hwy =lambda x: round2(x.std_hwy))
 .modify(var_hwy =lambda x: round2(x.var_hwy))
 .write_csv("aggregate.ld.csv"))

# ANTI JOIN
reviews = read_json("../data/listings-reviews.json")
(read_json("../data/listings.json")
 .anti_join(reviews, "id")
 .write_csv("anti_join.ld.csv"))

# FILTER
(read_json("../data/vehicles.json")
 .filter(lambda x: x.year < 2000)
 .filter(lambda x: x.cyl < 10)
 .write_csv("filter.ld.csv"))

# FILTER OUT
(read_json("../data/vehicles.json")
 .filter_out(lambda x: x.year < 2000)
 .filter_out(lambda x: x.cyl < 10)
 .write_csv("filter_out.ld.csv"))

# FULL JOIN
reviews = read_json("../data/listings-reviews.json")
reviews = reviews + reviews
(read_json("../data/listings.json")
 .full_join(reviews, "id")
 .write_csv("full_join.ld.csv"))

# INNER JOIN
reviews = read_json("../data/listings-reviews.json")
(read_json("../data/listings.json")
 .inner_join(reviews, "id")
 .write_csv("inner_join.ld.csv"))

# LEFT JOIN
reviews = read_json("../data/listings-reviews.json")
(read_json("../data/listings.json")
 .left_join(reviews, "id")
 .write_csv("left_join.ld.csv"))

# SEMI JOIN
reviews = read_json("../data/listings-reviews.json")
(read_json("../data/listings.json")
 .semi_join(reviews, "id")
 .write_csv("semi_join.ld.csv"))

# SORT
(read_json("../data/vehicles.json")
 .sort(make=1, model=1, year=-1)
 .write_csv("sort.ld.csv"))

# UNIQUE
(read_json("../data/vehicles.json")
 .unique("make", "model", "year")
 .write_csv("unique.ld.csv"))
