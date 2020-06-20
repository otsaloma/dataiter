#!/usr/bin/env python3

import dataiter as di

# Same as for documentation.
di.PRINT_MAX_WIDTH = 64

def parse_price(price):
    return int(float(price.lstrip("$").replace(",", "")))

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
    .modify(price=lambda data: [parse_price(x) for x in data.price])
)

print(data.head())
data.write_csv("listings.csv")
data.write_json("listings.json")
