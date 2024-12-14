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
