data.assign(c=(data["a"] + data["b"]))
data.assign(c=lambda x: x["a"] + x["b"])
