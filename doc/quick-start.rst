Quick Start
===========

Installation
------------

::

   pip install dataiter

You'll need Python 3.6 or later.

Usage
-----

Similar to NumPy, Pandas etc. we'll typically import the module under a
shorter alias and then access everything via that alias.

::

   import dataiter as di

DataFrame
---------

>>> import dataiter as di
>>> data = di.DataFrame.read_csv("data/vehicles.csv")
>>> data.filter(data.make == "Saab").sort(year=1).head(3)
.
     id make model  year        class           trans             drive
  int64 <U34  <U39 int64         <U34            <U32              <U26
  ----- ---- ----- ----- ------------ --------------- -----------------
0   380 Saab   900  1985 Compact Cars Automatic 3-spd Front-Wheel Drive
1   381 Saab   900  1985 Compact Cars Automatic 3-spd Front-Wheel Drive
2   382 Saab   900  1985 Compact Cars    Manual 5-spd Front-Wheel Drive
.
      cyl   displ    fuel   hwy   cty
  float64 float64    <U27 int64 int64
  ------- ------- ------- ----- -----
0       4       2 Regular    19    16
1       4       2 Regular    21    16
2       4       2 Regular    23    17
.

ListOfDicts
-----------

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
