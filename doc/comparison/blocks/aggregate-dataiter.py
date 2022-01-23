(data
 .group_by("year", "month")
 .aggregate(
     sales_total=di.sum("sales"),
     sales_per_day=di.mean("sales"),
 ))

(data
 .group_by("year", "month")
 .aggregate(
     sales_total=lambda x: x.sales.sum(),
     sales_per_day=lambda x: x.sales.mean(),
 ))
