# -*- coding: utf-8-unix -*-

suppressPackageStartupMessages({
    library(dplyr)
    library(readr)
})

options(dplyr.summarise.inform=FALSE)

Mode = function(x) {
    # https://stackoverflow.com/q/2547402
    ux = unique(x)
    return(ux[which.max(tabulate(match(x, ux)))])
}

read_csv = function(path) {
    data = readr::read_csv(path, show_col_types=FALSE, lazy=FALSE)
    for (name in colnames(data)) {
        # Drop all rows with NAs to avoid upcasting to float
        # and differing NA representation in output.
        data = data[!is.na(data[[name]]),]
        if (is.character(data[[name]]))
            # Use all lower case for strings to avoid differing
            # sorting of lower vs. upper case characters.
            data[[name]] = tolower(data[[name]])
    }
    return(data)
}

write_csv = function(data, path) {
    readr::write_csv(data, path, na="")
}

# AGGREGATE
read_csv("../data/vehicles.csv") %>%
    mutate(fuel_regular=(fuel == "regular")) %>%
    group_by(make, model) %>%
    summarise(
        all_fuel_regular=all(fuel_regular),
        any_fuel_regular=any(fuel_regular),
        count=n(),
        count_unique_cyl=n_distinct(cyl),
        first_hwy=first(hwy),
        last_hwy=last(hwy),
        max_hwy=max(hwy),
        mean_hwy=mean(hwy),
        median_hwy=median(hwy),
        min_hwy=min(hwy),
        mode_year=Mode(year),
        nth_id=nth(id, 1),
        quantile_hwy=quantile(hwy, 0.75, type=7),
        sum_hwy=sum(hwy)) %>%
    write_csv("aggregate.R.csv")

# ANTI JOIN
reviews = read_csv("../data/listings-reviews.csv")
read_csv("../data/listings.csv") %>%
    anti_join(reviews, by="id") %>%
    write_csv("anti_join.R.csv")

# FILTER
read_csv("../data/vehicles.csv") %>%
    filter(year < 2000) %>%
    filter(cyl < 10) %>%
    write_csv("filter.R.csv")

# FILTER OUT
read_csv("../data/vehicles.csv") %>%
    filter(!(year < 2000)) %>%
    filter(!(cyl < 10)) %>%
    write_csv("filter_out.R.csv")

# FULL JOIN
reviews = read_csv("../data/listings-reviews.csv")
reviews = bind_rows(reviews, reviews)
read_csv("../data/listings.csv") %>%
    full_join(reviews, by="id") %>%
    write_csv("full_join.R.csv")

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

# SEMI JOIN
reviews = read_csv("../data/listings-reviews.csv")
read_csv("../data/listings.csv") %>%
    semi_join(reviews, by="id") %>%
    write_csv("semi_join.R.csv")

# SORT
read_csv("../data/vehicles.csv") %>%
    arrange(make, model, desc(year)) %>%
    write_csv("sort.R.csv")

# UNIQUE
read_csv("../data/vehicles.csv") %>%
    distinct(make, model, year, .keep_all=TRUE) %>%
    write_csv("unique.R.csv")
