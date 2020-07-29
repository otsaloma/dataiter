ListOfDicts
===========

Introduction
------------

ListOfDicts is what the name implies and can be initialized from a
regular list of regular dicts.

>>> data = di.ListOfDicts([dict(x=1, y=2), dict(x=3, y=4)])
>>> data
[
  {
    "x": 1,
    "y": 2
  },
  {
    "x": 3,
    "y": 4
  }
]

Reading Data
------------

Class methods ``ListOfDicts.read_*`` can be used to read data from a file.

>>> data = di.ListOfDicts.read_json("data/listings.json")
>>> data.head()
[
  {
    "id": 2060,
    "hood": "Manhattan",
    "zipcode": "10040",
    "guests": 2,
    "sqft": null,
    "price": 100
  },
  {
    "id": 2595,
    "hood": "Manhattan",
    "zipcode": "10018",
    "guests": 2,
    "sqft": null,
    "price": 225
  },
  {
    "id": 3831,
    "hood": "Brooklyn",
    "zipcode": "11238",
    "guests": 3,
    "sqft": 500.0,
    "price": 89
  }
]

Method Chaining
---------------

Most of the data-modifying methods return copies, which means that you
can use method chaining to carry out any multi-step data processing.

>>> # The cheapest two-guest Airbnbs in Manhattan
>>> data = di.ListOfDicts.read_json("data/listings.json")
>>> data.filter(hood="Manhattan").filter(guests=2).sort(price=1).head()
[
  {
    "id": 42279170,
    "hood": "Manhattan",
    "zipcode": "10013",
    "guests": 2,
    "sqft": null,
    "price": 0
  },
  {
    "id": 42384530,
    "hood": "Manhattan",
    "zipcode": "10036",
    "guests": 2,
    "sqft": null,
    "price": 0
  },
  {
    "id": 18835820,
    "hood": "Manhattan",
    "zipcode": "10021",
    "guests": 2,
    "sqft": null,
    "price": 10
  }
]
