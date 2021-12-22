# -*- coding: utf-8-unix -*-

library(tidyverse)

read_csv = function(path) {
    data = readr::read_csv(path, show_col_types=FALSE, lazy=FALSE)
    for (name in colnames(data)) {
        # Drop all rows with NAs to avoid upcasting to float
        # and differing NA representation in output.
        data = data[!is.na(data[[name]]),]
        if (is.character(data[[name]]))
            # Use all lower case for strings to avoid differing
            # sorting of lower vs. upper case characters.
            data[[name]] = str_to_lower(data[[name]])
    }
    return(data)
}

write_csv = function(data, path) {
    readr::write_csv(data, path, na="")
}

# AGGREGATE
read_csv("../data/vehicles.csv") %>%
    group_by(make, model) %>%
    summarise(
        n=n(),
        cyl_min=min(cyl),
        cyl_max=max(cyl)) %>%
    write_csv("aggregate.R.csv")

# FILTER
read_csv("../data/vehicles.csv") %>%
    filter(year < 2000) %>%
    filter(cyl < 20) %>%
    write_csv("filter.R.csv")

# INNER JOIN
reviews = read_csv("../data/listings-reviews.csv")
read_csv("../data/listings.csv") %>%
    inner_join(reviews, by="id") %>%
    write_csv("inner_join.R.csv")

# LEFT JOIN
reviews = read_csv("../data/listings-reviews.csv")
read_csv("../data/listings.csv") %>%
    left_join(reviews, by="id") %>%
    write_csv("left_join.R.csv")

# SORT
read_csv("../data/vehicles.csv") %>%
    arrange(make, model, desc(year)) %>%
    write_csv("sort.R.csv")

# UNIQUE
read_csv("../data/vehicles.csv") %>%
    distinct(make, model, year, .keep_all=TRUE) %>%
    write_csv("unique.R.csv")
