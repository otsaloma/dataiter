# No singular operation for semi join
data1.merge(data2[["id"]], how="left_anti", on="id")
