(data
 .groupby(["year", "month"], as_index=False)
 .agg(
     sales_total=("sales", "sum"),
     sales_per_day=("sales", "mean")))

(data
 .groupby(["year", "month"], as_index=False)
 .apply(lambda x: pd.Series({
     "sales_total": x["sales"].sum(),
     "sales_per_day": x["sales"].mean()})))
