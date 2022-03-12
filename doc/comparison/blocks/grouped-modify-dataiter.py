(data
 .group_by("year", "month")
 .modify(fraction=lambda x: (
     x.sales / x.sales.sum())))
