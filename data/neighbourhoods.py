#!/usr/bin/env python3

import dataiter as di

# Rewrite to have one line per feature.
data = di.read_geojson("orig/neighbourhoods.geojson")
data.write("neighbourhoods.geojson")
