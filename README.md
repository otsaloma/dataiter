Python Classes for Data Manipulation
====================================

[![Build Status](https://travis-ci.org/otsaloma/dataiter.svg)](https://travis-ci.org/otsaloma/dataiter)
[![PyPI](https://img.shields.io/pypi/v/dataiter.svg)](https://pypi.org/project/dataiter/)

dataiter currently includes classes `DataFrame` and `ListOfDicts`.

**`DataFrame`** is a class for tabular data similar to R's `data.frame`
or `pandas.DataFrame`. It is under the hood a dictionary of NumPy arrays
and thus capable of fast vectorized operations. You can consider this to
be a very experimental, very light-weight alternative to Pandas with a
simple and consistent API.

**`ListOfDicts`** is a class useful for manipulating data from JSON
APIs. It provides functionality similar to libraries such as
Underscore.js, with manipulation functions that iterate over the data
and return a shallow modified copy of the original. `attd.AttributeDict`
is used to provide convenient access to dictionary keys.

## Installation

```bash
pip install dataiter
```

## Documentation

dataiter is experimental, undocumented and the API subject to change. If
you nevertheless want to try it, take a look at the below examples and
see the source code for available functions, and the tests for more
usage examples.

### Quick Start Example: `DataFrame`

```
>>> import dataiter as di
>>> data = di.DataFrame.read_csv("data/vehicles.csv")
>>> data.filter(data.make == "Saab").sort(year=1).head(3)

     id make model  year        class           trans             drive
  int64 <U34  <U39 int64         <U34            <U32              <U26
  ----- ---- ----- ----- ------------ --------------- -----------------
0   380 Saab   900  1985 Compact Cars Automatic 3-spd Front-Wheel Drive
1   381 Saab   900  1985 Compact Cars Automatic 3-spd Front-Wheel Drive
2   382 Saab   900  1985 Compact Cars    Manual 5-spd Front-Wheel Drive

      cyl   displ    fuel   hwy   cty
  float64 float64    <U27 int64 int64
  ------- ------- ------- ----- -----
0       4       2 Regular    19    16
1       4       2 Regular    21    16
2       4       2 Regular    23    17

```

### Quick Start Example: `ListOfDicts`

```
>>> import dataiter as di
>>> data = di.ListOfDicts.read_json("data/vehicles.json")
>>> data.filter(make="Saab").sort(year=1).head(1)
[
  {
    "id": 380,
    "make": "Saab",
    "model": "900",
    "year": 1985,
    "class": "Compact Cars",
    "trans": "Automatic 3-spd",
    "drive": "Front-Wheel Drive",
    "cyl": 4,
    "displ": 2,
    "fuel": "Regular",
    "hwy": 19,
    "cty": 16
  }
]
```
