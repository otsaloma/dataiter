#!/usr/bin/env python3

import sys
sys.path.insert(0, "..")

import dataiter as di

def read_json(path):
    data = di.ListOfDicts.read_json(path)
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

# AGGREGATE
(read_json("../data/vehicles.json")
 .group_by("make", "model")
 .aggregate(
     n=len,
     cyl_min=lambda x: min(x.pluck("cyl")),
     cyl_max=lambda x: max(x.pluck("cyl")))
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
