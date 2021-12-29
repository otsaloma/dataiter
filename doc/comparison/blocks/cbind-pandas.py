data1 = data1.reset_index(drop=True)
data2 = data2.reset_index(drop=True)
pd.concat([data1, data2], axis=1)
