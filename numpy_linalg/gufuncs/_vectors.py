# -*- coding: utf-8 -*-
"""
Generalized ufunc versions of `numpy.linalg` routines:

This module contains utilities (for internal use) that help in defining
functions and classes that use these gufuncs:

vec2mat:
    Convert vectors to single column/row matrices for linear algebra gufuncs.
mat2vec:
    Convert column/row-matrices back to vectors from linear algebra gufuncs.
vec_wrap:
    Wrap a gufunc with special handling for vectors.
"""
import functools as _ft
import numpy as _np
from ._families import inverse_arguments
from ._util import make_errobj


def vec2mat(x, y, case=(False, False)):
    """Convert vectors to single column/row matrices for linear algebra gufuncs

    Only does anything when `x.ndim==1` or `y.ndim==1`.

    Parameters
    ----------
    x,y : ndarray
        Left/right-hand-side of a linear algebra binary operation.
    case : Tuple[bool or None, bool or None]
        convert to row or column? `case = (x_ax, y_ax)`.
            ax = False: x/y -> row/column
            ax = True: x/y -> column/row
            ax = None: do not change
        for `case=(True, True)` we also reverse `needs_squeeze`

    Returns
    -------
    x,y : ndarray
        The inputs, but with singleton axes added so that linear algebra
        gufuncs behave correctly.
    needs_squeeze : list(bool)
        Tells us if axes [-2, -1] need to be removed from the gufunc output.
    """
    needs_squeeze = [False, False]
    if x.ndim == 1 and case[0] is not None:
        x = _np.expand_dims(x, case[0] + 0)
        needs_squeeze[0] = True
    if y.ndim == 1 and case[1] is not None:
        y = _np.expand_dims(y, 1 - case[1])
        needs_squeeze[1] = True
    if all(case):
        needs_squeeze = [x for x in reversed(needs_squeeze)]
    return x, y, needs_squeeze


def mat2vec(z, squeeze):
    """Convert column/row-matrices back to vectors from linear algebra gufuncs

    Parameters
    ----------
    z : ndarray
        Output of a gufunc performing a linear algebra binary operation.
    needs_squeeze : list(bool)
        Tells us if axes [-2, -1] need to be removed from the gufunc output.

    Returns
    -------
    z : ndarray
        The input, stripped of any sigleton axes added by `vec2mat`.
    """
    axs = (-2,) * squeeze[0] + (-1,) * squeeze[1]
    z = z.squeeze(axis=axs)
    return z[()] if z.ndim == 0 else z


# these help adjust docstrings in wrapped functions
_vec_doc = """
Does matrix-matrix, matrix-vector, vector-matrix and vector-vector versions,
with vector versions used *only* when one-dimensional.
"""
_bin_doc = "It is intended for use in binary operators.\n"


def vec_wrap(gufunc, case=()):
    """Wrap a gufunc with special handling for vectors

    Parameters
    ----------
    case : int
        convert to row or column? `case = (x_ax, y_ax)`.
        ax = 0: x/y -> row/column
        ax = 1: x/y -> column/row
        ax = 2: do not change
    """
    if not case:
        case = inverse_arguments[gufunc]

    @_ft.wraps(gufunc)
    def wrapper(x, y, *args, **kwargs):
        x, y, squeeze = vec2mat(_np.asanyarray(x), _np.asanyarray(y), case)
        make_errobj("Failure in linalg routine: " + gufunc.__name__, kwargs)
        z = gufunc(x, y, *args, **kwargs)
        return mat2vec(z, squeeze)

    wrapper.__doc__ = wrapper.__doc__.replace("\nParameters",
                                              "\n" + _vec_doc + "\nParameters")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,M,N)",
                                              "(...,M,N) or (N,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,N,P)",
                                              "(...,N,P) or (N,)")
    wrapper.__doc__ = wrapper.__doc__.replace(
                            "(...,M,P)", "(...,M,P), (...,M), (...,P) or ()")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,M,NRHS)",
                                              "(...,M,NRHS) or (M,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,NRHS,M)",
                                              "(...,NRHS,M) or (M,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,N,NRHS)",
                                              "(...,N,NRHS) or (N,)")
    wrapper.__doc__ = wrapper.__doc__.replace("(...,NRHS,N)",
                                              "(...,NRHS,N) or (N,)")
    wrapper.__doc__ = wrapper.__doc__.replace(_bin_doc, "")
    return wrapper
