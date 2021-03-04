Quick Start
===========

DataFrame
---------

>>> import dataiter as di
>>> data = di.DataFrame.read_csv("data/listings.csv")
>>> data.price_per_guest = data.price / data.guests
>>> data.head()
.
     id      hood zipcode guests    sqft price price_per_guest
  int64      <U13    <U11  int64 float64 int64         float64
  ----- --------- ------- ------ ------- ----- ---------------
0  2060 Manhattan   10040      2     nan   100          50.000
1  2595 Manhattan   10018      2     nan   225         112.500
2  3831  Brooklyn   11238      3     500    89          29.667
3  5099 Manhattan   10016      2     nan   200         100.000
4  5121  Brooklyn   11216      2     nan    60          30.000
5  5136  Brooklyn   11232      4     nan   253          63.250
6  5178 Manhattan   10019      2     nan    79          39.500
7  5203 Manhattan   10025      1     nan    79          79.000
8  5238 Manhattan   10002      2     nan   150          75.000
9  5441 Manhattan   10036      2     nan    99          49.500
.
>>> data.filter(hood="Manhattan").filter(guests=2).sort(price=1).head()
.
        id      hood zipcode guests    sqft price price_per_guest
     int64      <U13    <U11  int64 float64 int64         float64
  -------- --------- ------- ------ ------- ----- ---------------
0 42279170 Manhattan   10013      2     nan     0             0.0
1 42384530 Manhattan   10036      2     nan     0             0.0
2 18835820 Manhattan   10021      2     nan    10             5.0
3 20171179 Manhattan   10027      2     nan    10             5.0
4 14858544 Manhattan     nan      2     nan    15             7.5
5 31397084 Manhattan   10002      2     nan    19             9.5
6 22289683 Manhattan   10031      2     nan    20            10.0
7  7760204 Manhattan   10040      2     nan    22            11.0
8 43292527 Manhattan   10033      2     nan    22            11.0
9 43268040 Manhattan   10033      2     nan    23            11.5
.

GeoJSON
-------

>>> import dataiter as di
>>> data = di.GeoJSON.read("data/neighbourhoods.geojson")
>>> data.head()
.
     neighbourhood neighbourhood_group       geometry
              <U26                <U13         object
  ---------------- ------------------- --------------
0        Bayswater              Queens <MultiPolygon>
1         Allerton               Bronx <MultiPolygon>
2      City Island               Bronx <MultiPolygon>
3 Ditmars Steinway              Queens <MultiPolygon>
4       Ozone Park              Queens <MultiPolygon>
5          Fordham               Bronx <MultiPolygon>
6       Whitestone              Queens <MultiPolygon>
7    Arden Heights       Staten Island <MultiPolygon>
8         Arrochar       Staten Island <MultiPolygon>
9          Arverne              Queens <MultiPolygon>
.

ListOfDicts
-----------

>>> import dataiter as di
>>> data = di.ListOfDicts.read_json("data/listings.json")
>>> data = data.modify(price_per_guest=lambda x: x.price / x.guests)
>>> data.head()
[
  {
    "id": 2060,
    "hood": "Manhattan",
    "zipcode": "10040",
    "guests": 2,
    "sqft": null,
    "price": 100,
    "price_per_guest": 50.0
  },
  {
    "id": 2595,
    "hood": "Manhattan",
    "zipcode": "10018",
    "guests": 2,
    "sqft": null,
    "price": 225,
    "price_per_guest": 112.5
  },
  {
    "id": 3831,
    "hood": "Brooklyn",
    "zipcode": "11238",
    "guests": 3,
    "sqft": 500.0,
    "price": 89,
    "price_per_guest": 29.666666666666668
  }
]
>>> data.filter(hood="Manhattan").filter(guests=2).sort(price=1).head()
[
  {
    "id": 42279170,
    "hood": "Manhattan",
    "zipcode": "10013",
    "guests": 2,
    "sqft": null,
    "price": 0,
    "price_per_guest": 0.0
  },
  {
    "id": 42384530,
    "hood": "Manhattan",
    "zipcode": "10036",
    "guests": 2,
    "sqft": null,
    "price": 0,
    "price_per_guest": 0.0
  },
  {
    "id": 18835820,
    "hood": "Manhattan",
    "zipcode": "10021",
    "guests": 2,
    "sqft": null,
    "price": 10,
    "price_per_guest": 5.0
  }
]
