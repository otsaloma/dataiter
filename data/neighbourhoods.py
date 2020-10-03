#!/usr/bin/env python3

import dataiter as di

# Rewrite to have one line per feature.
data = di.GeoJSON.read("input/neighbourhoods.geojson")
data.write("neighbourhoods.geojson")
