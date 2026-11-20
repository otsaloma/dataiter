# TODO

- Drop the PyArrow workaround for strings in `Vector.is_in` once we can
  require a NumPy release with fast `np.isin` for `StringDType`,
  expected after 2.5.4: https://github.com/numpy/numpy/pull/32693

- Use `np.minmax` in `DataFrame._get_codes` to get both minimum and
  maximum in one pass, once we can require a NumPy release with it,
  expected after 2.5.4: https://github.com/numpy/numpy/pull/32231

- Use `descending=True` for `dir=-1` in `Vector.sort`, instead of
  reversing an ascending sort, and in `Vector.rank(method="ordinal")`,
  instead of sorting reversed codes, which for numbers is twice as fast,
  once we can require a NumPy release with descending sorts also for
  `StringDType` and object, expected after 2.5.4:
  https://github.com/numpy/numpy/pull/32286
  https://github.com/numpy/numpy/pull/31431
