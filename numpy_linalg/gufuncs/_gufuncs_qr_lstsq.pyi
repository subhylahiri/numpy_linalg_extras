"""Extension module containing gufuncs related to linear least-square solving,
Moore-Penrose matrix pseudo-inversion and QR/LQ decomposition.

They use the same LAPACK implementation as numpy.
"""
from numpy import ufunc

qr_m: ufunc
"""QR decomposition.

Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.
`K` = `M` or `N`, depending on which of `qr_m` or `qr_n` was called.
When M < N, `qr_n` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
Q: ndarray (...,M,K)
    Matrix with orthonormal columns.
R: ndarray (...,K,N)
    Matrix with zeros below the diagonal.
"""

qr_n: ufunc
"""QR decomposition.

Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.
`K` = `M` or `N`, depending on which of `qr_m` or `qr_n` was called.
When M < N, `qr_n` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
Q: ndarray (...,M,K)
    Matrix with orthonormal columns.
R: ndarray (...,K,N)
    Matrix with zeros below the diagonal.
"""

qr_rm: ufunc
"""QR decomposition.

Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.
This version only returns `R`.
`K` = `M` or `N`, depending on which of `qr_rm` or `qr_rn` was called.
When M < N, `qr_rn` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
R: ndarray (...,K,N)
    Matrix with zeros below the diagonal.
"""

qr_rn: ufunc
"""QR decomposition.

Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.
This version only returns `R`.
`K` = `M` or `N`, depending on which of `qr_rm` or `qr_rn` was called.
When M < N, `qr_rn` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
R: ndarray (...,K,N)
    Matrix with zeros below the diagonal.
"""

qr_rawm: ufunc
"""QR decomposition.

Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.
This version returns `R`, the Householder reflectors, `v`,
and their scaling factors, `tau`.
`K` = `M` or `N`, depending on which of `qr_rawm` or `qr_rawn` was called.
When M < N, `qr_rawn` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
H: ndarray (...,N,M)
    Transposed matrix (Fortran order). Below and on the diagonal: `R`.
    Above the diagonal: the Householder reflectors `v`.
tau: ndarray (...,K,)
    Scaling factors for Householder reflectors.
Notes
-----
Nothing will fail if you make the wrong choice of `qr_rawm` or `qr_rawn`.
You will merely waste memory from keeping unused elements of `TAU`.
"""

qr_rawn: ufunc
"""QR decomposition.

Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.
This version returns `R`, the Householder reflectors, `v`,
and their scaling factors, `tau`.
`K` = `M` or `N`, depending on which of `qr_rawm` or `qr_rawn` was called.
When M < N, `qr_rawn` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
H: ndarray (...,N,M)
    Transposed matrix (Fortran order). Below and on the diagonal: `R`.
    Above the diagonal: the Householder reflectors `v`.
tau: ndarray (...,K,)
    Scaling factors for Householder reflectors.
Notes
-----
Nothing will fail if you make the wrong choice of `qr_rawm` or `qr_rawn`.
You will merely waste memory from keeping unused elements of `TAU`.
"""

lq_m: ufunc
"""LQ decomposition.

Factor a matrix as `A = LQ` with `Q` orthogonal and `L` upper-triangular.
`K` = `M` or `N`, depending on which of `lq_m` or `lq_n` was called.
When M < N, `lq_n` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
L: ndarray (...,M,K)
    Matrix with zeros above the diagonal.
Q: ndarray (...,K,N)
    Matrix with orthonormal rows.
"""

lq_n: ufunc
"""LQ decomposition.

Factor a matrix as `A = LQ` with `Q` orthogonal and `L` upper-triangular.
`K` = `M` or `N`, depending on which of `lq_m` or `lq_n` was called.
When M < N, `lq_n` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
L: ndarray (...,M,K)
    Matrix with zeros above the diagonal.
Q: ndarray (...,K,N)
    Matrix with orthonormal rows.
"""

lq_lm: ufunc
"""LQ decomposition.

Factor a matrix as `A = LQ` with `Q` orthogonal and `L` upper-triangular.
This version only returns `L`.
`K` = `M` or `N`, depending on which of `lq_rm` or `lq_rn` was called.
When M > N, `lq_rm` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
L: ndarray (...,M,K)
    Matrix with zeros above the diagonal.
"""

lq_ln: ufunc
"""LQ decomposition.

Factor a matrix as `A = LQ` with `Q` orthogonal and `L` upper-triangular.
This version only returns `L`.
`K` = `M` or `N`, depending on which of `lq_rm` or `lq_rn` was called.
When M > N, `lq_rm` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
L: ndarray (...,M,K)
    Matrix with zeros above the diagonal.
"""

lq_rawm: ufunc
"""LQ decomposition.

Factor a matrix as `A = LQ` with `Q` orthogonal and `L` upper-triangular.
This version returns `L`, the Householder reflectors, `v`,
and their scaling factors, `tau`.
`K` = `M` or `N`, depending on which of `lq_rawm` or `lq_rawn` was called.
When M < N, `lq_rawn` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
H: ndarray (...,N,M)
    Transposed matrix (Fortran order). Above and on the diagonal: `R`.
    Below the diagonal: the Householder reflectors `v`.
tau: ndarray (...,K,)
    Scaling factors for Householder reflectors.

Notes
-----
Nothing will fail if you make the wrong choice of `lq_rawm` or `lq_rawn`.
You will merely waste memory from keeping unused elements of `TAU`.
"""

lq_rawn: ufunc
"""LQ decomposition.

Factor a matrix as `A = LQ` with `Q` orthogonal and `L` upper-triangular.
This version returns `L`, the Householder reflectors, `v`,
and their scaling factors, `tau`.
`K` = `M` or `N`, depending on which of `lq_rawm` or `lq_rawn` was called.
When M < N, `lq_rawn` cannot be called.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
H: ndarray (...,N,M)
    Transposed matrix (Fortran order). Above and on the diagonal: `R`.
    Below the diagonal: the Householder reflectors `v`.
tau: ndarray (...,K,)
    Scaling factors for Householder reflectors.

Notes
-----
Nothing will fail if you make the wrong choice of `lq_rawm` or `lq_rawn`.
You will merely waste memory from keeping unused elements of `TAU`.
"""

lstsq: ufunc
"""Least-square solution of linear system.

Find the least-square solution of the equation `AX = B` for `X`.
Unlike `numpy.linalg.lstsq`, this version does not return `residuals`,
rank or singuar values, and uses the default value for `rcond`.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,M,N) or (M,)
    Matrix of coefficients.
B: ndarray (...,M,NRHS) or (M,)
    Matrix of result vectors.

Returns
-------
X: ndarray (...,N,NRHS), (...,N), (...,NRHS) or ()
    Matrix of solution vectors.
"""

rlstsq: ufunc
"""Least-square solution of reversed linear system.

Find the least-square solution of the equation `A = XB` for `X`.
Unlike `numpy.linalg.lstsq`, this version does not return `residuals`,
rank or singuar values, and uses the default value for `rcond`.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,NRHS,M) or (M,)
    Matrix of result vectors.
B: ndarray (...,N,M) or (M,)
    Matrix of coefficients.

Returns
-------
X: ndarray (...,NRHS,N), (...,N), (...,NRHS) or ()
    Matrix of solution vectors.
"""

lstsq_qrm: ufunc
"""Least-square solution of linear system.

Find the least-square solution of the equation `AX = B` for `X`.
Unlike `numpy.linalg.lstsq`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

This version does not return `residuals`, rank or singuar values.
Instead, it returns the QR/LQ factors of `A` for future use.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,M,N) or (M,)
    Matrix of coefficients.
B: ndarray (...,M,NRHS) or (M,)
    Matrix of result vectors.

Returns
-------
X: ndarray (...,N,NRHS), (...,N), (...,NRHS) or ()
    Matrix of solution vectors.
AF: ndarray (...,N,M) or (M,)
    QR/LQ factors of `A`, with `Q` in Householder form.
TAU: ndarray[int] (...,K)
    Householder weights for `Q`, `K=min(M,N)`.

Notes
-----
Nothing will fail if you make the wrong choice of `lstsq_qrm` or `lstsq_qrn`.
You will merely waste memory from keeping unused elements of `TAU`.

`B` can cause `AF` to broadcast to have more/larger dimensions than `A`.
If `A` is a vector, this can cause core dimension-errors in `(r)qr_lstq`.
This can be avoided using the `unbroadcast_factors` function.

See Also
--------
qr in 'raw' mode.
lq in 'raw' mode.
unbroadcast_factors
"""

lstsq_qrn: ufunc
"""Least-square solution of linear system.

Find the least-square solution of the equation `AX = B` for `X`.
Unlike `numpy.linalg.lstsq`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

This version does not return `residuals`, rank or singuar values.
Instead, it returns the QR/LQ factors of `A` for future use.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,M,N) or (M,)
    Matrix of coefficients.
B: ndarray (...,M,NRHS) or (M,)
    Matrix of result vectors.

Returns
-------
X: ndarray (...,N,NRHS), (...,N), (...,NRHS) or ()
    Matrix of solution vectors.
AF: ndarray (...,N,M) or (M,)
    QR/LQ factors of `A`, with `Q` in Householder form.
TAU: ndarray[int] (...,K)
    Householder weights for `Q`, `K=min(M,N)`.

Notes
-----
Nothing will fail if you make the wrong choice of `lstsq_qrm` or `lstsq_qrn`.
You will merely waste memory from keeping unused elements of `TAU`.

`B` can cause `AF` to broadcast to have more/larger dimensions than `A`.
If `A` is a vector, this can cause core dimension-errors in `(r)qr_lstq`.
This can be avoided using the `unbroadcast_factors` function.

See Also
--------
qr in 'raw' mode.
lq in 'raw' mode.
unbroadcast_factors
"""

rlstsq_qrm: ufunc
"""Least-square solution of reversed linear system.

Find the least-square solution of the equation `A = XB` for `X`.
Unlike `numpy.linalg.lstsq`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

This version does not return `residuals`, rank or singuar values.
Instead, it returns the QR/LQ factors of `B` for future use.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,NRHS,M) or (M,)
    Matrix of result vectors.
B: ndarray (...,N,M) or (M,)
    Matrix of coefficients.

Returns
-------
X: ndarray (...,NRHS,N), (...,N), (...,NRHS) or ()
    Matrix of solution vectors.
BF: ndarray (...,M,N) or (M,)
    QR/LQ factors of `B`, with `Q` in Householder form.
TAU: ndarray[int] (...,K)
    Householder weights for `Q`, `K=min(M,N)`.

Notes
-----
Nothing will fail if you make the wrong choice of `rlstsq_qrm` or `rlstsq_qrn`.
You will merely waste memory from keeping unused elements of `TAU`.

`A` can cause `BF` to broadcast to have more/larger dimensions than `B`.
If `B` is a vector, this can cause core dimension-errors in `(r)qr_lstq`.
This can be avoided using the `unbroadcast_factors` function.

See Also
--------
qr in 'raw' mode.
lq in 'raw' mode.
unbroadcast_factors
"""

rlstsq_qrn: ufunc
"""Least-square solution of reversed linear system.

Find the least-square solution of the equation `A = XB` for `X`.
Unlike `numpy.linalg.lstsq`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

This version does not return `residuals`, rank or singuar values.
Instead, it returns the QR/LQ factors of `B` for future use.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,NRHS,M) or (M,)
    Matrix of result vectors.
B: ndarray (...,N,M) or (M,)
    Matrix of coefficients.

Returns
-------
X: ndarray (...,NRHS,N), (...,N), (...,NRHS) or ()
    Matrix of solution vectors.
BF: ndarray (...,M,N) or (M,)
    QR/LQ factors of `B`, with `Q` in Householder form.
TAU: ndarray[int] (...,K)
    Householder weights for `Q`, `K=min(M,N)`.

Notes
-----
Nothing will fail if you make the wrong choice of `rlstsq_qrm` or `rlstsq_qrn`.
You will merely waste memory from keeping unused elements of `TAU`.

`A` can cause `BF` to broadcast to have more/larger dimensions than `B`.
If `B` is a vector, this can cause core dimension-errors in `(r)qr_lstq`.
This can be avoided using the `unbroadcast_factors` function.

See Also
--------
qr in 'raw' mode.
lq in 'raw' mode.
unbroadcast_factors
"""

qr_lstsq: ufunc
"""Least-square solution of linear system.

Find the least-square solution of the equation `AX = B` for `X`.
Unlike `numpy.linalg.lstsq`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

This version does not return `residuals`, rank or singuar values.
It uses the QR/LQ factors of `A` from previous use of `(r)lstsq_qr`.
Instead, it returns the QR factors of `A` for future use.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
AF: ndarray (...,N,M) or (M,)
    QR/LQ factors of `A`, with `Q` in Householder form.
TAU: ndarray[int] (...,K)
    Householder weights for `Q`, `K=min(M,N)`.
B: ndarray (...,M,NRHS) or (M,)
    Matrix of result vectors.

Returns
-------
X: ndarray (...,N,NRHS), (...,N), (...,NRHS) or ()
    Matrix of solution vectors.

Notes
-----
Nothing will fail if you make the wrong choice of `lstsq_qrm` or `lstsq_qrn`.
You will merely waste memory from keeping unused elements of `TAU`.

See Also
--------
qr in 'raw' mode.
lq in 'raw' mode.
"""

rqr_lstsq: ufunc
"""Least-square solution of reversed linear system.

Find the least-square solution of the equation `A = XB` for `X`.
Unlike `numpy.linalg.lstsq`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

This version does not return `residuals`, rank or singuar values.
It uses the QR/LQ factors of `B` from previous use of `(r)lstsq_qr`.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,NRHS,M) or (M,)
    Matrix of result vectors.
BF: ndarray (...,M,N) or (M,)
    QR/LQ factors of `B`, with `Q` in Householder form.
TAU: ndarray[int] (...,K)
    Householder weights for `Q`, `K=min(M,N)`.

Returns
-------
X: ndarray (...,NRHS,N), (...,N), (...,NRHS) or ()
    Matrix of solution vectors.
Notes
-----
Nothing will fail if you make the wrong choice of `lstsq_qrm` or `lstsq_qrn`.
You will merely waste memory from keeping unused elements of `TAU`.

See Also
--------
qr in 'raw' mode.
lq in 'raw' mode.
"""

pinv: ufunc
"""Moore-Penrose pseudoinverse.

Find the inverse of `A` in the space orthogonal to its null-space.
Unlike `numpy.linalg.pinv`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix.

Returns
-------
AP: ndarray (...,N,M)
    Pseudoinverse of `A`.
"""

pinv_qrm: ufunc
"""Moore-Penrose pseudoinverse.

Find the inverse of `A` in the space orthogonal to its null-space.
Unlike `numpy.linalg.pinv`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

This version returns the QR/LQ factors of `A` for future use.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix.


Returns
-------
AP: ndarray (...,N,M)
    Pseudoinverse of `A`.
AF: ndarray (...,N,M)
    QR/LQ factors of `A`, with `Q` in Householder form.
TAU: ndarray[int] (...,K)
    Householder weights for `Q`, `K=min(M,N)`.
Notes
-----
Nothing will fail if you make the wrong choice of `pinv_qrm` or `pinv_qrn`.
You will merely waste memory from keeping unused elements of `TAU`.

See Also
--------
qr in 'raw' mode.
"""

pinv_qrn: ufunc
"""Moore-Penrose pseudoinverse.

Find the inverse of `A` in the space orthogonal to its null-space.
Unlike `numpy.linalg.pinv`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

This version returns the QR/LQ factors of `A` for future use.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix.


Returns
-------
AP: ndarray (...,N,M)
    Pseudoinverse of `A`.
AF: ndarray (...,N,M)
    QR/LQ factors of `A`, with `Q` in Householder form.
TAU: ndarray[int] (...,K)
    Householder weights for `Q`, `K=min(M,N)`.
Notes
-----
Nothing will fail if you make the wrong choice of `pinv_qrm` or `pinv_qrn`.
You will merely waste memory from keeping unused elements of `TAU`.

See Also
--------
qr in 'raw' mode.
"""

qr_pinv: ufunc
"""Moore-Penrose pseudoinverse.

Find the inverse of `A` in the space orthogonal to its null-space.
Unlike `numpy.linalg.pinv`, this version uses QR instead of SVD, so it does
not work with rank deficient matrices, `A`.

This version uses the QR/LQ factors of `A` from previous use of `pinv_qrm/n`
or `(r)lstsq_qrm/n`.

Parameters
-----------
AF: ndarray (...,N,M)
    QR/LQ factors of `A`, with `Q` in Householder form.
TAU: ndarray[int] (...,K)
    Householder weights for `Q`, `K=min(M,N)`.

Returns
-------
AP: ndarray (...,N,M)
    Pseudoinverse of `A`.
Notes
-----
Nothing will fail if you make the wrong choice of `pinv_qrm` or `pinv_qrn`.
You will merely waste memory from keeping unused elements of `TAU`.

See Also
--------
qr in 'raw' mode.
"""
