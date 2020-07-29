DataFrame
=========

Introduction
------------

DataFrame is a dict of NumPy arrays. You can create a new data frame the
various ways you would create a regular dict. Keys here are column names
and values are column vectors. Columns can be given as any kind of a
sequence, they will be converted to the proper type automatically. Any
scalar values will be broadcast so that the dimensions are consistent.

>>> data = di.DataFrame(x=[1, 2, 3], y=True)
>>> data
.
      x    y
  int64 bool
  ----- ----
0     1 True
1     2 True
2     3 True
.

Reading Data
------------

Class methods ``DataFrame.read_*`` can be used to read data from a file.

>>> data = di.DataFrame.read_csv("data/listings.csv")
>>> data.head()
.
     id      hood zipcode guests    sqft price
  int64      <U13    <U11  int64 float64 int64
  ----- --------- ------- ------ ------- -----
0  2060 Manhattan   10040      2     nan   100
1  2595 Manhattan   10018      2     nan   225
2  3831  Brooklyn   11238      3     500    89
3  5099 Manhattan   10016      2     nan   200
4  5121  Brooklyn   11216      2     nan    60
5  5136  Brooklyn   11232      4     nan   253
6  5178 Manhattan   10019      2     nan    79
7  5203 Manhattan   10025      1     nan    79
8  5238 Manhattan   10002      2     nan   150
9  5441 Manhattan   10036      2     nan    99
.

Columns
-------

Individual columns can be accessed by attribute notation in addition to
the usual dict bracket notation. New columns can be created simply by
assignment. Columns are NumPy arrays and support vectorized operations.

>>> data = di.DataFrame.read_csv("data/listings.csv")
>>> data.price_per_guest = data.price / data.guests
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

Method Chaining
---------------

Most of the data-modifying methods return copies, which means that you
can use method chaining to carry out any multi-step data processing.

>>> # The cheapest two-guest Airbnbs in Manhattan
>>> data = di.DataFrame.read_csv("data/listings.csv")
>>> data.filter(hood="Manhattan").filter(guests=2).sort(price=1).head()
.
        id      hood zipcode guests    sqft price
     int64      <U13    <U11  int64 float64 int64
  -------- --------- ------- ------ ------- -----
0 42279170 Manhattan   10013      2     nan     0
1 42384530 Manhattan   10036      2     nan     0
2 18835820 Manhattan   10021      2     nan    10
3 20171179 Manhattan   10027      2     nan    10
4 14858544 Manhattan     nan      2     nan    15
5 31397084 Manhattan   10002      2     nan    19
6 22289683 Manhattan   10031      2     nan    20
7  7760204 Manhattan   10040      2     nan    22
8 43292527 Manhattan   10033      2     nan    22
9 43268040 Manhattan   10033      2     nan    23
.
