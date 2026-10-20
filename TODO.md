# TODO

- Drop the PyArrow workaround for strings in `Vector.is_in` once we can
  require a NumPy release with fast `np.isin` for `StringDType`,
  expected after 2.5.4: https://github.com/numpy/numpy/pull/32693

- Use `np.minmax` in `DataFrame._get_codes` to get both minimum and
  maximum in one pass, once we can require a NumPy release with it,
  expected after 2.5.4: https://github.com/numpy/numpy/pull/32231
