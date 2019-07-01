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

Functions
---------
flattish
    Flatten a subset of axes.
expand_dims
    Add new singleton axes.
transpose
    Transpose last two indices.
dagger
    Conjugate-transpose last two indices.
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
lq
    LQ decomposition with broadcasting and subclass passing.
lqr
    For wide matrices LQ decomposition, otherwise QR decomposition.
Also includes `gufuncs` for `linalg` etc, and versions of most `numpy`array
creation and manipulation routines and `wrappers` for converting the rest.

Examples
--------
>>> import numpy as np
>>> import linalg as la
>>> x = la.random.rand(2, 3, 4)
>>> y = la.random.rand(2, 3, 4)
>>> z = x.pinv @ y
>>> w = x @ y.pinv
>>> u = x @ y.t
>>> v = (x.r @ y[:, None, ...].t).ur
"""
from . import _lnarray, _linalg
from ._lnarray import lnarray, pinvarray, invarray
from ._linalg import (transpose, dagger, col, row, scalar, qr, lq, lqr, lu,
                      matldiv, matrdiv, flattish, expand_dims)
from .gufuncs import matmul, solve, rsolve, lstsq, rlstsq, norm
from . import gufuncs
from . import _ln_wrap as wrappers
from . import random
from ._ln_creation import (
    empty, empty_like, eye, identity, ones, ones_like, zeros, zeros_like, full,
    full_like, array, asarray, asanyarray, ascontiguousarray, copy, frombuffer,
    fromfile, fromfunction, fromiter, fromstring, loadtxt, arange, linspace,
    logspace, geomspace, meshgrid, diag, diagflat, tri, tril, triu, vander)
from ._ln_manipulation import (
    reshape, moveaxis, rollaxis, swapaxes, atleast_1d, atleast_2d, atleast_3d,
    broadcast_to, broadcast_arrays, squeeze, asfarray, asfortranarray,
    asarray_chkfinite, require, concatenate, stack, column_stack, dstack,
    hstack, vstack, block, split, array_split, dsplit, hsplit, vsplit, tile,
    repeat, delete, insert, append, resize, trim_zeros, unique, flip, fliplr,
    flipud, roll, rot90)

from . import version
__version__ = version.max_version(
    "0.2.0", gufuncs, wrappers, random, _lnarray, _linalg)

assert all((gufuncs, wrappers, random, norm))
assert all((lnarray, pinvarray, invarray))
assert all((transpose, dagger, col, row, scalar, qr, lq, lqr, lu,
            matmul, solve, rsolve, lstsq, rlstsq, matldiv, matrdiv))
assert all((
    empty, empty_like, eye, identity, ones, ones_like, zeros, zeros_like, full,
    full_like, array, asarray, asanyarray, ascontiguousarray, copy, frombuffer,
    fromfile, fromfunction, fromiter, fromstring, loadtxt, arange, linspace,
    logspace, geomspace, meshgrid, diag, diagflat, tri, tril, triu, vander))
assert all((
   reshape, moveaxis, rollaxis, swapaxes, atleast_1d, atleast_2d, atleast_3d,
   broadcast_to, broadcast_arrays, expand_dims, squeeze, asfarray, flattish,
   asfortranarray, asarray_chkfinite, require, concatenate, stack,
   column_stack, dstack, hstack, vstack, block, split, array_split, dsplit,
   hsplit, vsplit, tile, repeat, delete, insert, append, resize, trim_zeros,
   unique, flip, fliplr, flipud, reshape, roll, rot90))
