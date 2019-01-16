# -*- coding: utf-8 -*-
# =============================================================================
# Created on Thu Dec  7 17:20:43 2017
# @author: Subhy
# package: linalg_tricks
# =============================================================================
"""
Linear algebra routines.

Classes
-------
lnarray
    Subclass of `numpy.ndarray` with properties such as `pinv/inv` for matrix
    division, `t` for transposing stacks of matrices, `c`, `r` and `s` for
    dealing with stacks of vectors and scalars.
pinvarray
    Provides interface for matrix division when it is matrix multiplied (@).
    Returned by `lnarray.pinv`. It calls `np.linalg.lstsq` behind the scenes.
    Does not actually pseudoinvert the matrix unless it is explicitly called.
    I think it is best not to store these objects in variables, and call on
    `lnarray.pinv` on the rhs instead.
invarray
    Provides interface for matrix division when it is matrix multiplied (@).
    Returned by `lnarray.inv`. It calls `np.linalg.solve` behind the scenes.
    Does not actually invert the matrix unless it is explicitly called.
    I think it is best not to store these objects in variables, and call on
    `lnarray.inv` on the rhs instead.
lnmatrix
    Subclass of `lnarray` which swaps matrix/elementwise multiplication and
    division from the right. Shouldn't be necessary given `lnarray`'s syntax.
ldarray
    `lnarray` subclass which overloads bitshift operators for matrix division.
    One of several reasons why this is a bad idea is that bitshifting has lower
    operator priority than division, so you will have to use parentheses often.
    I think you're better off sticking with `lnarray`.

Functions
---------
transpose
    Transpose last two indices.
col
    Treat multi-dim array as a stack of column vectors.
row
    Treat multi-dim array as a stack of row vectors.
scal
    Treat multi-dim array as a stack of scalars.
matldiv
    Matrix division from left.
matrdiv
    Matrix division from right.
lu
    LU decomposition with broadcasting and subclass passing.
qr
    QR decomposition with broadcasting and subclass passing.
Also includes `gufuncs` for `matmul`, etc, and versions of most `numpy`array
creation and manipulation routines and `wrappers` for converting the rest.

Examples
--------
>>> import numpy as np
>>> import linalg as sp
>>> x = sp.lnarray(np.random.rand(2, 3, 4))
>>> y = sp.lnarray(np.random.rand(2, 3, 4))
>>> z = x.pinv @ y
>>> w = x @ y.pinv
>>> u = x @ y.t
>>> v = (x.r @ y[:, None, ...].t).ur
>>> a = sp.ldarray(np.random.rand(2, 3, 4))
>>> b = sp.ldarray(np.random.rand(2, 3, 4))
>>> c = (a << b)
>>> d = (a >> b)
"""

from ._lnarray import lnarray, pinvarray, invarray, lnmatrix
from ._ldarray import ldarray
from ._linalg import (transpose, dagger, col, row, scalar, qr, lu,
                      matmul, solve, rsolve, lstsq, rlstsq, matldiv, matrdiv)
from .gufuncs import norm
from . import gufuncs
from . import _ln_wrap as wrappers
from ._ln_creation import *
from ._ln_manipulation import *
# import _ld_wrap as wrappers
# from ._ld_creation import *
# from ._ld_manipulation import *
assert gufuncs
assert wrappers
assert norm
assert all((lnarray, pinvarray, invarray, lnmatrix, ldarray))
assert all((transpose, dagger, col, row, scalar, qr, lu, matmul, solve, rsolve,
            lstsq, rlstsq, matldiv, matrdiv))
