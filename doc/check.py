#!/usr/bin/env python3

import dataiter as di
import inspect

from pathlib import Path

DOCUMENTED_SPECIALS = [
    "__init__",
]

PAGES = {
    "dataiter.rst": di,
    "data-frame.rst": di.DataFrame,
    "data-frame-column.rst": di.DataFrameColumn,
    "dt.rst": di.dt,
    "geojson.rst": di.GeoJSON,
    "list-of-dicts.rst": di.ListOfDicts,
    "vector.rst": di.Vector,
}

SKIP = [
    "DataFrame.COLUMN_PLACEHOLDER",
    "GeoJSON.to_string",
    "Vector.to_strings",
]

DIRECTORY = Path(__file__).parent
for page, obj in PAGES.items():
    text = (DIRECTORY / page).read_text("utf-8")
    source = inspect.getsourcefile(obj)
    print(f"Checking {source}...")
    for name, value in inspect.getmembers(obj):
        if (name.startswith("_") and
            name not in DOCUMENTED_SPECIALS):
            continue
        if not inspect.getmodule(value): continue
        module = inspect.getmodule(value)
        if inspect.ismodule(obj):
            # Skip objects documented separately.
            if inspect.ismodule(value): continue
            if inspect.isclass(value): continue
        if inspect.isclass(obj):
            # Skip base class methods from NumPy etc.
            if inspect.getsourcefile(module) != source: continue
        full_name = f"{obj.__name__}.{name}"
        if full_name in SKIP: continue
        print(f"... {full_name}")
        if full_name not in text:
            raise Exception("Not found")
