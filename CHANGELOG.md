## Changelog for numpy_linalg_extras

### v0.2.0

- Requires Python 3.7
- Requires NumPy 1.16
- The `gufuncs` for matrix multiplication and division can handle vectors.
- Deprecated `gufunc.vec` module as it is no longer needed.
- Switched to numpy's version of `matmul`, as it is now a `gufunc` so my version is no longer needed.
- Deprecated mixins in `gufuncs`, as they are no longer needed.
- Simplified `__array_ufunc__` methods.
- Combinations of `pinvarray` and `invarray` in linear algebra `gufuncs` make more sense.
- C code closer to compliance with NumPy modified PEP-0007

### v0.1.0

- Unit tests. Coverage probably not great.
- gufuncs for `pinv`, `inv`, `lq`.
- Wrapped version of `numpy.random` and `numpy.fft`

### v0.0.0

Copied from repo `sl_py_tools/numpy_tricks/linalg`.