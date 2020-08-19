"""Extension module containing gufuncs related to matrix multiplication.

They use explicit for-loops in C.
"""
from numpy import ufunc

matmul: ufunc
matmul__doc__ = """Matrix-matrix product.

.. deprecated:: 0.2.0
    This `gufunc` is no longer needed as NumPy switched to a gufunc in v1.16.

Uses BLAS routine `_gemm` for acceleration.
Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
X: ndarray (...,M,N) or (N,)
    Matrix multiplying from left.
Y: ndarray (...,N,P) or (N,)
    Matrix multiplying from right.

Returns
-------
Z: ndarray (...,M,P), (...,M), (...,P) or ()
    Result of matrix multiplication.
"""

rmatmul: ufunc
rmatmul__doc__ = """Reversed matrix-matrix product.

Uses BLAS routine `_gemm` for acceleration.
Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
Y: ndarray (...,N,P) or (N,)
    Matrix multiplying from right.
X: ndarray (...,M,N) or (N,)
    Matrix multiplying from left.

Returns
-------
Z: ndarray (...,M,P), (...,M), (...,P) or ()
    Result of matrix multiplication.
"""

norm: ufunc
norm__doc__ = """Euclidean norm of a vector.

Unlike `numpy.linalg.norm`, it only computes the vector 2-norm.

Parameters
-----------
X: ndarray (...,N)
    Vector, or array of vectors.

Returns
-------
Z: float
    Euclidean norm of X.
"""

rtrue_divide: ufunc
rtrue_divide__doc__ = """Reversed division Z = X \\ Y.

Parameters
-----------
X: ndarray (...)
    Denominator.
Y: ndarray (...)
    Numerator.

Returns
-------
Z: ndarray (...)
    Result of division.");
"""
