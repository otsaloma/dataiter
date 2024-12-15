data |>
    filter(year == 2021) |>
    arrange(desc(sales)) |>
    head(10)
