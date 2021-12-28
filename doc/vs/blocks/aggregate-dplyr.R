data %>%
    group_by(year, month) %>%
    summarise(
        sales_total=sum(sales),
        sales_per_day=mean(sales))
