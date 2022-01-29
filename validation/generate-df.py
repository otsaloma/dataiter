#!/usr/bin/env python3

import sys
sys.path.insert(0, "..")

import dataiter as di
import numpy as np

def read_csv(path):
    data = di.read_csv(path)
    for name in data.colnames:
        # Drop all rows with NAs to avoid upcasting to float
        # and differing NA representation in output.
        data = data.filter_out(data[name].is_missing())
        if data[name].is_string():
            # Use all lower case for strings to avoid differing
            # sorting of lower vs. upper case characters.
            data[name] = np.char.lower(data[name])
    return data

# AGGREGATE
(read_csv("../data/vehicles.csv")
 .group_by("make", "model")
 .aggregate(
     n=di.count(),
     cyl_min=di.min("cyl"),
     cyl_max=di.max("cyl"))
 .write_csv("aggregate.df.csv"))

# ANTI JOIN
reviews = read_csv("../data/listings-reviews.csv")
(read_csv("../data/listings.csv")
 .anti_join(reviews, "id")
 .write_csv("anti_join.df.csv"))

# FILTER
(read_csv("../data/vehicles.csv")
 .filter(lambda x: x.year < 2000)
 .filter(lambda x: x.cyl < 10)
 .write_csv("filter.df.csv"))

# FILTER OUT
(read_csv("../data/vehicles.csv")
 .filter_out(lambda x: x.year < 2000)
 .filter_out(lambda x: x.cyl < 10)
 .write_csv("filter_out.df.csv"))

# FULL JOIN
reviews = read_csv("../data/listings-reviews.csv")
reviews = reviews.rbind(reviews)
(read_csv("../data/listings.csv")
 .full_join(reviews, "id")
 .write_csv("full_join.df.csv"))

# INNER JOIN
reviews = read_csv("../data/listings-reviews.csv")
(read_csv("../data/listings.csv")
 .inner_join(reviews, "id")
 .write_csv("inner_join.df.csv"))

# LEFT JOIN
reviews = read_csv("../data/listings-reviews.csv")
(read_csv("../data/listings.csv")
 .left_join(reviews, "id")
 .write_csv("left_join.df.csv"))

# SEMI JOIN
reviews = read_csv("../data/listings-reviews.csv")
(read_csv("../data/listings.csv")
 .semi_join(reviews, "id")
 .write_csv("semi_join.df.csv"))

# SORT
(read_csv("../data/vehicles.csv")
 .sort(make=1, model=1, year=-1)
 .write_csv("sort.df.csv"))

# UNIQUE
(read_csv("../data/vehicles.csv")
 .unique("make", "model", "year")
 .write_csv("unique.df.csv"))
