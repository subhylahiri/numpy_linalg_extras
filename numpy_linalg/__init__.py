# -*- coding: utf-8 -*-
# =============================================================================
# Created on Thu Dec  7 17:20:43 2017
# @author: Subhy
# package: linalg_tricks
# =============================================================================
"""Linear algebra routines.

This package contains classes and functions that make the syntax for linear
algebra in `numpy` cleaner, particularly with respect to broadcasting and
matrix division.

It also includes versions of most `numpy`array creation and manipulation
routines and `wrappers` for converting the rest.


Routine Listings
----------------
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
gufuncs
    Module with `gufuncs` for `linalg` etc.
convert
    Module with helpers for writing __array_ufunc__ methods.
wrappers
    Module with functions that change function return types to `lnarray`.
version
    Module with tools for dealing with version numbering.
testing
    Module with tools for writing unit tests.

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
from . import wrappers
from . import convert
from . import random
from . import fft
from ._ln_creation import (
    empty, empty_like, eye, identity, ones, ones_like, zeros, zeros_like, full,
    full_like, array, asarray, asanyarray, ascontiguousarray, copy, frombuffer,
    fromfile, fromfunction, fromiter, fromstring, loadtxt, arange, linspace,
    logspace, geomspace, meshgrid, diag, diagflat, tri, tril, triu, vander)
from ._ln_indexing import (
    ravel_multi_index, unravel_index, ix_, fill_diagonal, mask_indices,
    diag_indices, tril_indices, triu_indices, indices,
    diag_indices_from, tril_indices_from, triu_indices_from)
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

assert any((gufuncs, wrappers, random, norm))
assert any((lnarray, pinvarray, invarray))
assert any((transpose, dagger, col, row, scalar, qr, lq, lqr, lu,
            matmul, solve, rsolve, lstsq, rlstsq, matldiv, matrdiv))
assert any((
    True, empty, empty_like, eye, identity, ones, ones_like, zeros, zeros_like,
    full, full_like, array, asarray, asanyarray, ascontiguousarray, copy,
    frombuffer, fromfile, fromfunction, fromiter, fromstring, loadtxt,
    arange, linspace, logspace, geomspace, meshgrid, diag, diagflat, vander,
    tri, tril, triu))
assert any ((
    True, ravel_multi_index, unravel_index, ix_, fill_diagonal, mask_indices,
    diag_indices, tril_indices, triu_indices, indices,
    diag_indices_from, tril_indices_from, triu_indices_from))
assert any((
    True, asfarray, asfortranarray, asarray_chkfinite, require,
    reshape, moveaxis, rollaxis, swapaxes, atleast_1d, atleast_2d, atleast_3d,
    broadcast_to, broadcast_arrays, expand_dims, squeeze, flattish,
    concatenate, stack, column_stack, dstack, hstack, vstack, block,
    split, array_split, dsplit, hsplit, vsplit, tile, repeat, delete, insert,
    append, resize, trim_zeros, unique, flip, fliplr, flipud, roll, rot90))
