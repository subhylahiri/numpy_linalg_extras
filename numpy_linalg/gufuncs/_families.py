# -*- coding: utf-8 -*-
"""
Tools for generalized ufunc versions of `numpy.linalg` routines:

The following variables, for internal use, are useful for determining the
behaviour of pinvarrays and invarrays in linear algebraic functions:

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
from ._gufuncs_blas import rmatmul  # matmul, norm
from numpy import matmul
from . import _gufuncs_lu_solve as _gls
from . import _gufuncs_qr_lstsq as _gql

assert norm
# =============================================================================
# Categories of binary operators
# =============================================================================

# maps Tuple[bool] (left, right) -> gufunc
# if left, 1st argument of gufunc is 'inverted'
# if right, 2nd argument of gufunc is 'inverted'
solve_family = ((matmul, _gls.rsolve), (_gls.solve, rmatmul))
solve_lu_family = ((matmul, _gls.rsolve_lu), (_gls.solve_lu, rmatmul))
lu_solve_family = ((matmul, _gls.rlu_solve), (_gls.lu_solve, rmatmul))
lstsq_family = ((matmul, _gql.rlstsq), (_gql.lstsq, None))
lstsq_qrm_family = ((matmul, _gql.rlstsq_qrm), (_gql.lstsq_qrm, None))
lstsq_qrn_family = ((matmul, _gql.rlstsq_qrn), (_gql.lstsq_qrn, None))
qr_lstsq_family = ((matmul, _gql.rqr_lstsq), (_gql.qr_lstsq, None))

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

# maps gufunc -> (left, right) Tuple[bool]
# if left, 1st argument of gufunc is 'inverted'
# if right, 2nd argument of gufunc is 'inverted'
inverse_arguments = {}
super_families = {}

# backwards maps Tuple[bool] (left, right) -> ufunc
# if left, *2nd* argument of ufunc is a *numerator*
# if right, *1st* argument of ufunc is a *numerator*
truediv_family = ((None, _np.true_divide), (rtrue_divide, _np.multiply))
# backwards maps ufunc -> (left, right) Tuple[bool]
inverse_scalar_arguments = {}


def _add_family(inv_arg_dict, family):
    """Add a new family to inverse-argument dict
    """
    bools = (False, True)
    for left_arg, right_arg in _it.product(bools, bools):
        func = family[left_arg][right_arg]
        if func is not None and func not in inv_arg_dict.keys():
            inv_arg_dict[func] = (left_arg, right_arg)
    # NOTE: rmatmul doesn't fit the pattern, needs special handling


def _add_family_set(inv_arg_dict, func_set, *families):
    """create a set of families
    """
    for family in families:
        _add_family(inv_arg_dict, family)
        func_set.update(y for x in family for y in x)


def add_to_super_family(name, *families):
    """Add a gufunc family to a super-family

    For special handling by (p)invarrays
    """
    _add_family_set(inverse_arguments, super_families[name], *families)


def add_new_super_family(name, *families):
    """Create a new super-family of gufunc families
    """
    super_families[name] = set()
    add_to_super_family(name, *families)


def same_family(ufunc_in, ufunc_out) -> bool:
    """Are the two ufuncs from the same super-family?
    """
    return any({ufunc_in, ufunc_out} <= x for x in super_families.values())


def add_scalar_family(*families):
    """Add a scalar ufunc family

    For special handling by (p)invarrays
    """
    for family in families:
        _add_family(inverse_scalar_arguments, family)


# make maps gufunc -> (left, right) Tuple[bool]
add_new_super_family('solve', *_solve_families)
add_new_super_family('lstsq', *_lstsq_families)
# make backwards maps ufunc -> (left, right) Tuple[bool]
add_scalar_family(truediv_family)
