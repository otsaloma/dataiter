#!/usr/bin/env python3

import dataiter as di
import inspect

df = di.DataFrame()
ld = di.ListOfDicts()

base_df = {}
base_ld = []

print("")
print("Methods missing from DataFrame:")
for name in sorted(dir(ld)):
    if name in dir(df): continue
    if name.startswith("_"): continue
    if name in dir(base_ld) and name not in dir(base_df): continue
    if not inspect.ismethod(getattr(ld, name)): continue
    print(f"... {name}")

print("")
print("Methods missing from ListOfDicts:")
for name in sorted(dir(df)):
    if name in dir(ld): continue
    if name.startswith("_"): continue
    if name in dir(base_df) and name not in dir(base_ld): continue
    if not inspect.ismethod(getattr(df, name)): continue
    print(f"... {name}")
