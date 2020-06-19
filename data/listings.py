#!/usr/bin/env python3

import dataiter as di

# Same as for documentation.
di.PRINT_MAX_WIDTH = 64

data = (
    di.DataFrame.read_csv("orig/listings.csv")
    .select("id",
            "neighbourhood_group_cleansed",
            "zipcode",
            "accommodates",
            "square_feet",
            "price")

    .rename(hood="neighbourhood_group_cleansed")
    .rename(guests="accommodates")
    .rename(sqft="square_feet")
)

data.price = [int(float(x.lstrip("$").replace(",", ""))) for x in data.price]
print(data.head())
data.write_csv("listings.csv")
data.write_json("listings.json")
