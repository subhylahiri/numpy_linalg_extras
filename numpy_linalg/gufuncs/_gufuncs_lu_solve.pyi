from numpy import ufunc

lu_m: ufunc
lu_m.__doc__ = """LU decomposition.

Write matrix `A` as `A = PLU`, where  `P` is a permutation matrix,
`L` is lower triangular with ones on the diagonal and
`U` is upper triangular.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
L: ndarray (...,M,K)
    Lower diagonal matrix with unit diagonals. `K=min(M,N)`.
U: ndarray (...,K,N)
    Upper triangular matrix.
IP: ndarray (...,K)
   Vector of pivot indices, where swaps `A[...,i,:] <--> A[...,IP[i],:]`
   are performed in order `i=1,...,K`.

Notes
-----
Nothing will fail if you make the wrong choice of `lu_m` or `lu_n`.
You will merely waste memory for the unused elements of `L`, `U`, and `IP`.
"""

lu_n: ufunc
lu_n.__doc__ = """LU decomposition.

Write matrix `A` as `A = PLU`, where  `P` is a permutation matrix,
`L` is lower triangular with ones on the diagonal and
`U` is upper triangular.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
L: ndarray (...,M,K)
    Lower diagonal matrix with unit diagonals. `K=min(M,N)`.
U: ndarray (...,K,N)
    Upper triangular matrix.
IP: ndarray (...,K)
   Vector of pivot indices, where swaps `A[...,i,:] <--> A[...,IP[i],:]`
   are performed in order `i=1,...,K`.

Notes
-----
Nothing will fail if you make the wrong choice of `lu_m` or `lu_n`.
You will merely waste memory for the unused elements of `L`, `U`, and `IP`.
"""

lu_rawm: ufunc
lu_rawm.__doc__ = """LU decomposition.

Write matrix `A` as `A = PLU`, where  `P` is a permutation matrix,
`L` is lower triangular with ones on the diagonal and
`U` is upper triangular.


Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
AF: ndarray (...,N,M)
    Transposed matrix (Fortran order).
   LU factors of `A`, with the nonzero elements of `U` below and on
   the diagonal, nonzero elements of `L` above the diagonal.
   The diagonal elements of `L`, which are all 1, are not stored."
IP: ndarray (...,K)
   Vector of pivot indices, where swaps `A[...,i,:] <--> A[...,IP[i],:]`
   are performed in order `i=1,...,K`, `K=min(M,N)`.

Notes
-----
Nothing will fail if you make the wrong choice of `lu_rawm` or `lu_rawn`.
You will merely waste memory from keeping unused elements of `IP`.
"""

lu_rawn: ufunc
lu_rawn.__doc__ = """LU decomposition.

Write matrix `A` as `A = PLU`, where  `P` is a permutation matrix,
`L` is lower triangular with ones on the diagonal and
`U` is upper triangular.


Parameters
-----------
A: ndarray (...,M,N)
    Matrix to be factored.

Returns
-------
AF: ndarray (...,N,M)
    Transposed matrix (Fortran order).
   LU factors of `A`, with the nonzero elements of `U` below and on
   the diagonal, nonzero elements of `L` above the diagonal.
   The diagonal elements of `L`, which are all 1, are not stored."
IP: ndarray (...,K)
   Vector of pivot indices, where swaps `A[...,i,:] <--> A[...,IP[i],:]`
   are performed in order `i=1,...,K`, `K=min(M,N)`.

Notes
-----
Nothing will fail if you make the wrong choice of `lu_rawm` or `lu_rawn`.
You will merely waste memory from keeping unused elements of `IP`.
"""

pivot: ufunc
pivot.__doc__ = """Perform row pivoting.

Permute the rows of `A` according to pivot indices, `IP`.
If `IP[I] = J`, rows `I` and `J` will be swapped,
in order of increasing `I`.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix before pivoting.
IP: ndarray[int] (...,M)
    Pivot indices from LU factoring, etc.

Returns
-------
AP: ndarray (...,M,N)
    Pivoted matrix.
"""

rpivot: ufunc
rpivot.__doc__ = """Undo row pivoting.

Permute the rows of `A` according to pivot indices, `IP`.
If `IP[I] = J`, rows `I` and `J` will be swapped,
in order of decreasing `I`.

Parameters
-----------
AP: ndarray (...,M,N)
    Pivoted matrix.
IP: ndarray[int] (...,M)
    Pivot indices from LU factoring, etc.

Returns
-------
A: ndarray (...,M,N)
    Matrix before pivoting.
"""

inv: ufunc
inv.__doc__ = """Matrix inverse.

Find the matrix multiplicative inverse of `A`.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix.

Returns
-------
AI: ndarray (...,N,M)
    Inverse of `A`.
"""

inv_lu: ufunc
inv_lu.__doc__ = """Matrix inverse.

Find the matrix multiplicative inverse of `A`.
This version returns the LU factors of `A` for future use.

Parameters
-----------
A: ndarray (...,M,N)
    Matrix.

Returns
-------
AI: ndarray (...,N,M)
    Inverse of `A`.
AF: ndarray (...,M,N)
   LU factors of `A`, with the nonzero elements of `U` above and on
   the diagonal, nonzero elements of `L` below the diagonal.
   The diagonal elements of `L`, which are all 1, are not stored."
IP: ndarray (...,K)
   Vector of pivot indices, where swaps `A[...,i,:] <--> A[...,IP[i],:]`
   are performed in order `i=1,...,K`, `K=min(M,N)`.

See Also
--------
lu in 'raw' mode.
"""

lu_inv: ufunc
lu_inv.__doc__ = """Matrix inverse.

Find the matrix multiplicative inverse of `A`.
This version uses the LU factors of `A` from previous use of `inv_lu`
or `(r)solve_lu`.

Parameters
-----------
AF: ndarray (...,M,N)
   LU factors of `A`, with the nonzero elements of `U` above and on
   the diagonal, nonzero elements of `L` below the diagonal.
   The diagonal elements of `L`, which are all 1, are not stored."
IP: ndarray (...,K)
   Vector of pivot indices, where swaps `A[...,i,:] <--> A[...,IP[i],:]`
   are performed in order `i=1,...,K`, `K=min(M,N)`.

Returns
-------
AI: ndarray (...,N,M)
    Inverse of `A`.

See Also
--------
lu in 'raw' mode."
"""

solve: ufunc
solve.__doc__ = """Solve linear system.

Solve the equation `AX = B` for `X`.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,N,N)
    Matrix of coefficients.
B: ndarray (...,N,NRHS) or (N,)
    Matrix of result vectors.

Returns
-------
X: ndarray (...,N,NRHS) or (N,)
    Matrix of solution vectors.
"""

rsolve: ufunc
rsolve.__doc__ = """Solve reversed linear system.

Solve the equation `A = XB` for `X`.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,NRHS,N) or (N,)
    Matrix of result vectors.
B: ndarray (...,N,N)
    Matrix of coefficients.

Returns
-------
X: ndarray (...,NRHS,N) or (N,)
    Matrix of solution vectors.
"""

solve_lu: ufunc
solve_lu.__doc__ = """Solve linear system.

Solve  equation `AX = B` for `X`.
This version also returns the LU factors of `A` for future use.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,N,N)
    Matrix of coefficients.
B: ndarray (...,N,NRHS) or (N,)
    Matrix of result vectors.

Returns
-------
X: ndarray (...,N,NRHS) or (N,)
    Matrix of solution vectors.
AF: ndarray (...,N,N)
    LU factors of `A`.
IP: ndarray[int] (...,N)
    Pivot indices from LU factoring.

See Also
--------
lu in 'raw' mode.
"""

rsolve_lu: ufunc
rsolve_lu.__doc__ = """Solve reversed linear system.

Solve the equation `A = XB` for `X`.
This version also returns the LU factors of `B` for future use.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,NRHS,N) or (N,)
    Matrix of result vectors.
B: ndarray (...,N,N)
    Matrix of coefficients.

Returns
-------
X: ndarray (...,NRHS,N) or (N,)
    Matrix of solution vectors.
BF: ndarray (...,N,N)
    LU factors of `B`.
IP: ndarray[int] (...,N)
    Pivot indices from LU factoring.

See Also
--------
lu in 'raw' mode.
"""

lu_solve: ufunc
lu_solve.__doc__ = """Solve linear system.

Solve  equation `AX = B` for `X`.
This version uses the LU factors of `A` from earlier use of `(r)solve_lu`.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
AF: ndarray (...,N,N)
    LU factors of `A`.
IP: ndarray[int] (...,N)
    Pivot indices from LU factoring.
B: ndarray (...,N,NRHS) or (N,)
    Matrix of result vectors.

Returns
-------
X: ndarray (...,N,NRHS) or (N,)
    Matrix of solution vectors.

See Also
--------
lu in 'raw' mode.
"""

rlu_solve: ufunc
rlu_solve.__doc__ = """Solve reversed linear system.

Solve the equation `A = XB` for `X`.
This version uses the LU factors of `B` from earlier use of `(r)solve_lu`.

Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.

Parameters
-----------
A: ndarray (...,NRHS,N) or (N,)
    Matrix of result vectors.
BF: ndarray (...,N,N)
    LU factors of `B`.
IP: ndarray[int] (...,N)
    Pivot indices from LU factoring.

Returns
-------
X: ndarray (...,NRHS,N) or (N,)
    Matrix of solution vectors.

See Also
--------
lu in 'raw' mode.
"""

