data1.merge(data2, how="left",  on="id")
data1.merge(data2, how="inner", on="id")
data1.merge(data2, how="outer", on="id")
