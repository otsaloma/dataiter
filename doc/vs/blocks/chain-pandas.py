(data
 .loc[lambda x: x["year"] == 2021]
 .sort_values("sales", ascending=False)
 .head(10))
