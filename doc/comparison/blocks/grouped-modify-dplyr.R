data %>%
    group_by(year, month) %>%
    mutate(fraction=sales/sum(sales))
