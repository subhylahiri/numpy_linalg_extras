# -*- coding: utf-8 -*-
"""
Tools for generalized ufunc versions of `numpy.linalg` routines:

The following variables, for internal use, are useful for determining the
behaviour of 1D arrays, pinvarrays and invarrays in linear algebraic functions:

solve_family, solve_lu_family, lu_solve_family:
    2x2 tuples of gufuncs used to interpret behaviour of `invarrays` in them.
    Member[x][y] interprets first/second argument as a divisor if x/y is True.
lstsq_family, lstsq_qrm_family, lstsq_qrn_family, qr_lstsq_family:
    2x2 tuples of gufuncs used to interpret behaviour of `pinvarrays` in them.
truediv_family:
    2x2 tuples of ufuncs used to interpret behaviour of `(p)invarrays` in them.
    Member[x][y] interprets second/first argument as a divisor if x/y is False.
inverse_arguments:
    dict `{gufunc: Tuple[bool]}` describing gufunc's position in its family.
inverse_scalar_arguments:
    dict `{ufunc: Tuple[bool]}` describing ufunc's position in its family.
same_family:
    Function returning true if both its arguments are in the same super-family.
"""
__all__ = [
    'inverse_arguments',
    'inverse_scalar_arguments',
    'same_family',
    'solve_family',
    'solve_lu_family',
    'lu_solve_family',
    'lstsq_family',
    'lstsq_qrm_family',
    'lstsq_qrn_family',
    'qr_lstsq_family',
    'truediv_family',
]
import itertools as _it
import numpy as _np
from ._gufuncs_cloop import rtrue_divide, norm  # , matmul, rmatmul
from ._gufuncs_blas import matmul, rmatmul  # , norm
# from numpy import matmul
from ._gufuncs_lu_solve import (solve, rsolve, solve_lu, rsolve_lu, lu_solve,
                                rlu_solve)
from ._gufuncs_qr_lstsq import (lstsq, rlstsq, lstsq_qrm, lstsq_qrn,
                                rlstsq_qrm, rlstsq_qrn, qr_lstsq, rqr_lstsq)
assert norm
# =============================================================================
# Categories of binary operators
# =============================================================================

# maps Tuple[bool] (left, right) -> gufunc
# if left, 1st argument of gufunc is 'inverted'
# if right, 2nd argument of gufunc is 'inverted'
solve_family = ((matmul, rsolve), (solve, rmatmul))
solve_lu_family = ((matmul, rsolve_lu), (solve_lu, rmatmul))
lu_solve_family = ((matmul, rlu_solve), (lu_solve, rmatmul))
lstsq_family = ((matmul, rlstsq), (lstsq, None))
lstsq_qrm_family = ((matmul, rlstsq_qrm), (lstsq_qrm, None))
lstsq_qrn_family = ((matmul, rlstsq_qrn), (lstsq_qrn, None))
qr_lstsq_family = ((matmul, rqr_lstsq), (qr_lstsq, None))

_solve_families = [
        solve_family,
        solve_lu_family,
        lu_solve_family,
]
_lstsq_families = [
        lstsq_family,
        lstsq_qrm_family,
        lstsq_qrn_family,
        qr_lstsq_family,
]
_solve_funcs = set([z for x in _solve_families for y in x for z in y])
_lstsq_funcs = set([z for x in _lstsq_families for y in x for z in y])

_families = _solve_families + _lstsq_families
# maps gufunc -> (left, right) Tuple[bool]
# if left, 1st argument of gufunc is 'inverted'
# if right, 2nd argument of gufunc is 'inverted'
inverse_arguments = {}
_bools = (False, True)
for _family, _left_arg, _right_arg in _it.product(_families, _bools, _bools):
    _func = _family[_left_arg][_right_arg]
    if _func is not None and _func not in inverse_arguments.keys():
        inverse_arguments[_func] = (_left_arg, _right_arg)
# NOTE: rmatmul doesn't fit the pattern, needs special handling

# backwards maps Tuple[bool] (left, right) -> ufunc
# if left, *2nd* argument of ufunc is a *numerator*
# if right, *1st* argument of ufunc is a *numerator*
truediv_family = ((None, _np.true_divide), (rtrue_divide, _np.multiply))
inverse_scalar_arguments = {}
# backwards maps ufunc -> (left, right) Tuple[bool]
for _left_arg, _right_arg in _it.product(_bools, _bools):
    _func = truediv_family[_left_arg][_right_arg]
    if _func is not None and _func not in inverse_scalar_arguments.keys():
        inverse_scalar_arguments[_func] = (_left_arg, _right_arg)


def same_family(ufunc_in, ufunc_out) -> bool:
    """Are the two ufuncs from the same family?
    """
    return any([(ufunc_in in x) and (ufunc_out in x)
                for x in [_solve_funcs, _lstsq_funcs]])
