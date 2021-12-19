# -*- coding: utf-8-unix -*-

library(tidyverse)

read_csv_ = function(path) {
    data = read_csv(path, show_col_types=FALSE, lazy=FALSE)
    for (name in colnames(data)) {
        # Drop all rows with NAs to avoid upcasting to float
        # and differing NA representation in output.
        data = data[!is.na(data[[name]]),]
        if (is.character(data[[name]]))
            # Use all lower case for strings to avoid differing
            # sorting lower vs. upper case characters.
            data[[name]] = str_to_lower(data[[name]])
    }
    return(data)
}

# 1. Aggregate
read_csv_("../data/vehicles.csv") %>%
    group_by(make, model) %>%
    summarise(
        n=n(),
        cyl_min=min(cyl),
        cyl_max=max(cyl)) %>%
    write_csv("1.R.csv")

# 2. Sort
read_csv_("../data/vehicles.csv") %>%
    arrange(make, model, year) %>%
    write_csv("2.R.csv")

# 3. Unique
read_csv_("../data/vehicles.csv") %>%
    distinct(make, model, year, .keep_all=TRUE) %>%
    write_csv("3.R.csv")
