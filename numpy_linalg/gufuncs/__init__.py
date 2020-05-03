# -*- coding: utf-8 -*-
# =============================================================================
# Created on Sat Sep  1 20:35:47 2018
#
# @author: subhy
# =============================================================================
"""
Generalized ufunc versions of `numpy.linalg` routines:

They raise `FloatingPointError` or warnings instead of `LinAlgError`.

`numpy.linalg` broadcasting rules apply.

This module also contains utilities (for internal use) that help in defining
functions and classes that use these gufuncs:

Routine Listings
----------------
matmul
    An alias of `numpy.matmul`.
rmatmul
    Reversed matrix multiplication (only useful to implement binary operators).
solve
    Solve systems of linear equations.
rsolve
    Solve reversed systems of linear equations.
solve_lu, rsolve_lu
    Also return LU factors for use next time.
lu_solve, rlu_solve
    Use LU factors from previous time.
inv
    Moore-Penrose pseudoinverse.
inv_lu
    Also return LU factors for use next time.
lu_inv
    Use LU factors from previous time.
lstsq
    Least square solution of systems of linear equations.
rlstsq
    Least square solution of reversed systems of linear equations.
lstsq_qrm, lstsq_qrn, rlstsq_qrm, rlstsq_qrn
    Also return QR factors for use next time.
qr_lstsq, rqr_lstsq
    Use QR factors from previous time.
pinv
    Moore-Penrose pseudoinverse.
pinv_qrm, pinv_qrn
    Also return QR factors for use next time.
qr_pinv
    Use QR factors from previous time.
norm
    Vector 2-norm.
lu_*
    LU factorization in various forms - {m,n,rawm,rawn}.
        m: #(rows) < #(columns).
        n: #(rows) > #(columns).
        raw: return `AF`, which contains `L` and `U`, and `ipiv`.
pivot, rpivot
    Perform row pivots with the output of lu_*.
qr_*
    QR factorization in various forms - {m,n,rm,rn,rawm,rawn}.
        m: #(rows) < #(columns).
        n: #(rows) > #(columns).
        r: only return `R`.
        raw: return `H` and `tau`, from which `Q` and `R` can be computed.
lq_*
    LQ factorization in various forms - {m,n,lm,ln,rawm,rawn}.
        m: #(rows) < #(columns).
        n: #(rows) > #(columns).
        l: only return `L`.
        raw: return `H` and `tau`, from which `Q` and `L` can be computed.
rtrue_tivide
    Reversed division (only useful to implement binary operators).
unbroadcast_factors
    Undo broadcasting in factors returned by gufuncs.
make_errobj
    create an error handler object for use as ``extobj`` gufunc parameter.
return_shape_mat
    Shape of result of broadcasted matrix multiplication
fam
    Module with variables for determining the behaviour of 1D arrays,
    `pinvarrays` and `invarrays` in linear algebraic functions:
MatmulOperatorsMixin
    Mixin class that uses `matmul` from here to define @ operators. Deprecated.
LNArrayOperatorsMixin
    Subclass of `numpy.lib.mixins.NDArrayOperatorsMixin` that uses `matmul`
    from here to define @ operators. Deprecated.
vec
    Module of functions for dealing with vector arguments. Deprecated.
"""
from . import _gufuncs_cloop
from . import _gufuncs_blas
from . import _gufuncs_lu_solve
from . import _gufuncs_qr_lstsq
# In _families we choose which versions of matmul, rmatmul, norm to use.
from ._families import (matmul, rmatmul, norm, rtrue_divide)
from ._gufuncs_lu_solve import (lu_m, lu_n, lu_rawm, lu_rawn, pivot, rpivot,
                                solve, rsolve, solve_lu, rsolve_lu, lu_solve,
                                rlu_solve, inv, inv_lu, lu_inv)
from ._gufuncs_qr_lstsq import (
    qr_m, qr_n, qr_rm, qr_rn, qr_rawm, qr_rawn, lq_m, lq_n, lq_lm, lq_ln,
    lq_rawm, lq_rawn, lstsq, rlstsq, lstsq_qrm, lstsq_qrn, rlstsq_qrm,
    rlstsq_qrn, qr_lstsq, rqr_lstsq, pinv, pinv_qrm, pinv_qrn, qr_pinv)

from . import _families as fam
from ._util import make_errobj, unbroadcast_factors
from ._util import return_shape, array_return_shape
from ._util import MatmulOperatorsMixin, LNArrayOperatorsMixin
from . import _vectors as vec

from ..version import max_version as _version
__version__ = _version("0.2.0", _gufuncs_blas, _gufuncs_cloop,
                       _gufuncs_lu_solve, _gufuncs_qr_lstsq, vec, fam)

# fool pyflakes
assert any((norm, rtrue_divide, matmul, rmatmul))
assert any((make_errobj, unbroadcast_factors))
assert any((MatmulOperatorsMixin, LNArrayOperatorsMixin))
assert any((lu_m, lu_n, lu_rawm, lu_rawn, solve, rsolve, solve_lu, rsolve_lu,
            lu_solve, rlu_solve, pivot, rpivot, inv, inv_lu, lu_inv))
assert any((
    qr_m, qr_n, qr_rm, qr_rn, qr_rawm, qr_rawn, lq_m, lq_n, lq_lm, lq_ln,
    lq_rawm, lq_rawn, lstsq, rlstsq, lstsq_qrm, lstsq_qrn, rlstsq_qrm,
    rlstsq_qrn, qr_lstsq, rqr_lstsq, pinv, pinv_qrm, pinv_qrn, qr_pinv))
