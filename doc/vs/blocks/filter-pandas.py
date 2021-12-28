data[data["year"] == 2021]
data.loc[data["year"] == 2021]
data[lambda x: x["year"] == 2021]
data.loc[lambda x: x["year"] == 2021]
data.query("year == 2021")
