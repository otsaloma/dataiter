Simple, Light-Weight Data Frames for Python
===========================================

[![PyPI](https://img.shields.io/pypi/v/dataiter.svg)](https://pypi.org/project/dataiter)
[![Downloads](https://pepy.tech/badge/dataiter/month)](https://pepy.tech/project/dataiter)

Dataiter's **`DataFrame`** is a class for tabular data similar to R's
`data.frame`, implementing all common operations to manipulate data. It
is under the hood a dictionary of NumPy arrays and thus capable of fast
vectorized operations. You can consider it to be a light-weight
alternative to Pandas with a simple and consistent API. Performance-wise
Dataiter relies on NumPy and Numba and is likely to be at best
comparable to Pandas.

## Installation

```bash
# Latest stable version
pip install -U dataiter

# Latest development version
pip install -U git+https://github.com/otsaloma/dataiter

# Numba (optional)
pip install -U numba
```

Dataiter optionally uses **Numba** to speed up certain operations. If
you have Numba installed, Dataiter will use it automatically. It's
currently not a hard dependency, so you need to install it separately.

## Quick Start

```python
import dataiter as di
data = di.read_csv("data/listings.csv")
data.filter(hood="Manhattan", guests=2).sort(price=1).head()
.
        id      hood zipcode guests    sqft price
     int64    string  string  int64 float64 int64
  ──────── ───────── ─────── ────── ─────── ─────
0 42279170 Manhattan   10013      2     nan     0
1 42384530 Manhattan   10036      2     nan     0
2 18835820 Manhattan   10021      2     nan    10
3 20171179 Manhattan   10027      2     nan    10
4 14858544 Manhattan              2     nan    15
5 31397084 Manhattan   10002      2     nan    19
6 22289683 Manhattan   10031      2     nan    20
7  7760204 Manhattan   10040      2     nan    22
8 43292527 Manhattan   10033      2     nan    22
9 43268040 Manhattan   10033      2     nan    23
.
```

## Documentation

https://dataiter.readthedocs.io/

If you're familiar with either dplyr (R) or Pandas (Python), the
comparison table in the documentation will give you a quick overview of
the differences and similarities in common operations.

https://dataiter.readthedocs.io/en/stable/comparison.html

## Development

To install a virtualenv for development, use

    make venv

or, for a specific Python version

    make PYTHON=python3.X venv
