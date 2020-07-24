# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 12:10:02 2017

@author: Subhy

Linear algebra routines.

Functions
---------
flattish
    Flatten a subset of axes.
expand_dims
    Add new singleton axes.
transpose
    Transpose last two indices.
dagger
    Transpose-conjugate last two indices.
col
    Treat multi-dim array as a stack of column vectors.
row
    Treat multi-dim array as a stack of row vectors.
scal
    Treat multi-dim array as a stack of scalars.
matmul
    Matrix multiplication.
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
"""
import typing as ty
from warnings import warn

import numpy as np
import numpy.linalg.linalg as nla

from . import gufuncs as gf
from .gufuncs import lstsq, matmul, rlstsq, rmatmul, rsolve, solve
from .wrappers import set_module

__all__ = [
    'flattish',
    'foldaxis',
    'expand_dims',
    'transpose',
    'dagger',
    'col',
    'row',
    'scalar',
    'matmul',
    'rmatmul',
    'solve',
    'rsolve',
    'lstsq',
    'rlstsq',
    'matldiv',
    'matrdiv',
    'qr',
    'lq',
    'lqr',
    'lu',
]

# =============================================================================
# Reshaping for linear algebra
# =============================================================================


@set_module('numpy_linalg')
def flattish(arr: np.ndarray, start: int = 0, stop: ty.Optional[int] = None
             ) -> np.ndarray:
    """Partial flattening.

    Flattens those axes in the range [start:stop).

    Parameters
    ----------
    arr : np.ndarray (...,L,M,N,...,P,Q,R,...)
        Array to be partially flattened.
    start : int, optional, default: 0
        First axis of group to be flattened.
    stop : int or None, optional, default: None
        First axis *after* group to be flattened. Goes to end if it is None.

    Returns
    -------
    new_arr : np.ndarray (...,L,M*N*...*P*Q,R,...)
        Partially flattened array.

    Raises
    ------
    ValueError
        If `start > stop`.
    """
    if stop is None:
        stop = arr.ndim
    newshape = arr.shape[:start] + (-1,) + arr.shape[stop:]
    if len(newshape) > arr.ndim + 1:
        raise ValueError(f"start={start} > stop={stop}")
    return np.reshape(arr, newshape)


@set_module('numpy_linalg')
def foldaxis(arr: np.ndarray, axis: int, shape: ty.Tuple[int, ...]
             ) -> np.ndarray:
    """Partial unflattening.

    Folds an `axis` into `shape`.

    Parameters
    ----------
    arr : np.ndarray (...,L,M*N*...*P*Q,R,...)
        Array to be partially folded.
    axis : int
        Axis to be folded.
    shape : Tuple[int, ...]
        Shape to fold `axis` into. One element can be -1, like `numpy.reshape`.

    Returns
    -------
    new_arr : np.ndarray (...,L,M,N,...,P,Q,R,...)
        Partially unflattened array.

    Raises
    ------
    ValueError
        If multiple elements of `shape` are -1.
        If `arr.shape[axis] != prod(shape)` (unless one element is -1).
    """
    minus_one = np.count_nonzero([siz == -1 for siz in shape])
    if minus_one > 1:
        raise ValueError(f"Axis size {arr.shape[axis]} cannot fold to {shape}")
    if minus_one == 0 and np.prod(shape) != arr.shape[axis]:
        raise ValueError(f"Axis size {arr.shape[axis]} cannot fold to {shape}")
    axis %= arr.ndim
    newshape = arr.shape[:axis] + shape + arr.shape[axis+1:]
    return np.reshape(arr, newshape)


@set_module('numpy_linalg')
def expand_dims(arr: np.ndarray, *axis) -> np.ndarray:
    """Expand the shape of the array with length one axes.

    Alias of `numpy.expand_dims` when `*axis` is a single `int`. If `axis`
    is a sequence of `int`, axis numbers are relative to the *final* shape.

    Parameters
    ----------
    arr : np.ndarray (...,L,M,N,...,P,Q,...)
        Array to be expanded.
    *axis : int, Tuple[int]
        Positions where new axes are inserted. Negative numbers are allowed.
        Numbers are with respect to the final shape.
        .. deprecated:: 0.3.0
            Passing multiple axis arguments rather than a tuple of them is
            deprecated to match the behaviour in NumPy 1.18. This will raise
            an error in version 0.4.0.

    Returns
    -------
    new_arr : np.ndarray (...,L,1,M,1,N,...,P,1,Q,...)
        Expanded array.

    Raises
    ------
    ValueError
        If any axes are out of range or repeated.

    See Also
    --------
    numpy.expand_dims
    """
    if len(axis) == 0:
        return arr
    if len(axis) == 1:
        return np.expand_dims(arr, axis[0])
    warn("Pass a tuple of ints to expand_dims rather than multiple arguments."
         + " This will aise an error in version 0.4.0.", DeprecationWarning)
    new_dim = arr.ndim + len(axis)
    if any(axs >= new_dim or axs < -new_dim for axs in axis):
        raise ValueError(f'Axes out of range for {new_dim}-array: {axis}')
    axes_sort = tuple(np.sort(np.mod(axis, new_dim)))
    if len(axes_sort) > len(set(axes_sort)):
        raise ValueError(f'repeated axis, arguments: {axes_sort}')
    return expand_dims(expand_dims(arr, axes_sort[0]), *axes_sort[1:])


@set_module('numpy_linalg')
def transpose(arr: np.ndarray) -> np.ndarray:
    """Transpose last two axes.

    Transposing last two axes fits better with `numpy.linalg`'s broadcasting,
    which treats multi-dim arrays as stacks of matrices.

    Parameters
    ----------
    arr : np.ndarray, (..., M, N)

    Returns
    -------
    transposed : np.ndarray, (..., N, M)
    """
    if arr.ndim < 2:
        return arr
    return nla.transpose(arr)


@set_module('numpy_linalg')
def dagger(arr: np.ndarray) -> np.ndarray:
    """Hermitian conjugate over last two indices.

    Transposing last two indices fits better with `np.linalg`'s broadcasting,
    which treats multi-dim arrays as stacks of matrices.

    Parameters
    ----------
    arr : np.ndarray, (..., M, N)

    Returns
    -------
    conjugated : np.ndarray, (..., N, M)
    """
    return transpose(arr.conj())


@set_module('numpy_linalg')
def col(arr: np.ndarray) -> np.ndarray:
    """Treat multi-dim array as a stack of column vectors.

    Achieves this by inserting a singleton dimension in last slot.
    You'll have an extra singleton after any linear algebra operation from the
    left.

    Parameters
    ----------
    arr : np.ndarray, (..., N)

    Returns
    -------
    expanded : np.ndarray, (..., N, 1)
    """
    return expand_dims(arr, -1)


@set_module('numpy_linalg')
def row(arr: np.ndarray) -> np.ndarray:
    """Treat multi-dim array as a stack of row vectors.

    Achieves this by inserting a singleton dimension in second-to-last slot.
    You'll have an extra singleton after any linear algebra operation from the
    right.

    Parameters
    ----------
    arr : np.ndarray, (..., N)

    Returns
    -------
    expanded : np.ndarray, (..., 1, N)
    """
    return expand_dims(arr, -2)


@set_module('numpy_linalg')
def scalar(arr: np.ndarray) -> np.ndarray:
    """Treat multi-dim array as a stack of scalars.

    Achieves this by inserting singleton dimensions in last two slots.

    Parameters
    ----------
    arr : np.ndarray, (...,)

    Returns
    -------
    expanded : np.ndarray, (..., 1, 1)
    """
    return expand_dims(arr, (-2, -1))


# =============================================================================
# Division & Multiplication
# =============================================================================


@set_module('numpy_linalg')
def matldiv(lft: np.ndarray, rgt: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Matrix division from left.

    Computes :math:`result = lft \\ rgt = lft^{-1} rgt` for invertible `lft`,
    or :math:`lft^+ rgt` for non-invertible `lft`.
    Pseudo-inverse version uses `lstsq`.
    Full inverse version uses `solve`.
    Both versions broadcast using gufunc machinery.

    Parameters
    ----------
    lft : (..., M, N) array_like
        Divisor or Denominator.
    rgt : {(M,), (..., M, K)} array_like
        Dividend or Numerator.
    out : {(N,), (..., N, K)} np.ndarray
        array to store the output in.

    Returns
    -------
    result : {(N,), (..., N, K)} np.ndarray
        Quotient. It has the same type as `lft` or `rgt`.

    Raises
    ------
    LinAlgError
        If `lft` is not invertible and `lstsq` doesn't converge.

    See also
    --------
    `np.linalg.solve` : performs exact matrix division.
    `np.linalg.lstsq` : performs least-square matrix division.
    """
    if lft.ndim > 1 and lft.shape[-1] == lft.shape[-2]:
        try:
            return solve(lft, rgt, *args, **kwargs)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return lstsq(lft, rgt, *args, **kwargs)


@set_module('numpy_linalg')
def matrdiv(lft: np.ndarray, rgt: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Matrix division from right.

    Computes :math:`result = lft / rgt = lft rgt^{-1}` for invertible `rgt`,
    or :math:`lft rgt^+` for non-invertible `rgt`.
    Pseudo-inverse version uses `rlstsq`.
    Full inverse version uses `rsolve`.
    Both versions broadcast using gufunc machinery.

    Parameters
    ----------
    lft : {(M,), (..., K, M)} array_like
        Dividend or Numerator.
    rgt : (..., N, M) array_like
        Divisor or Denominator.
    out : {(N,), (..., K, N)} np.ndarray
        array to store the output in.

    Returns
    -------
    result : {(N,), (..., K, N)} np.ndarray
        Quotient. It has the same type as `lft` or `rgt`.

    Raises
    ------
    LinAlgError
        If `lft` is not invertible and `lstsq` doesn't converge.

    See also
    --------
    `np.linalg.solve` : performs exact matrix division.
    `np.linalg.lstsq` : performs least-square matrix division.
    """
    if rgt.ndim > 1 and rgt.shape[-1] == rgt.shape[-2]:
        try:
            return rsolve(lft, rgt, *args, **kwargs)
        except (np.linalg.LinAlgError, ValueError):
            pass
    return rlstsq(lft, rgt, *args, **kwargs)


# =============================================================================
# Matrix decomposition
# =============================================================================
# pylint: disable=invalid-name


QR_MODES = {'reduced': (gf.qr_m, gf.qr_n),
            'complete': (gf.qr_m, gf.qr_m),
            'r': (gf.qr_rm, gf.qr_rn),
            'raw': (gf.qr_rawm, gf.qr_rawn)}


@set_module('numpy_linalg')
def qr(arr: np.ndarray, mode: str = 'reduced',
       out1: ty.Optional[np.ndarray] = None,
       out2: ty.Optional[np.ndarray] = None,
       **kwds) -> ty.Tuple[np.ndarray, ...]:
    """QR decomposition.

    Factor a matrix as `A = QR` with `Q` orthogonal and `R` upper-triangular.
    `K = min(M,N)`, except for mode `complete`, where `K = M`.

    Parameters
    -----------
    A: ndarray (...,M,N)
        Matrix to be factored.
    mode: str
        Case insensitive, chosen from:
        **reduced** - default, use inner dimensionality ``K=min(M,N)``,
        **complete** - use inner dimensionality ``K=M``, for square `Q`,
        **r** or **R** - return `R` only,
        **raw** - return `H,tau`, which determine `Q` and `R` (see below).

    Returns
    -------
    Q: ndarray (...,M,K). Modes: `reduced, complete`.
        Matrix with orthonormal columns.
    R: ndarray (...,K,N). Modes: `reduced, complete, r`.
        Matrix with zeros below the diagonal.
    H: ndarray (...,N,M). Modes: `raw`.
        Transposed matrix (Fortran order). It contains the following:
            On & below diagonal: non-zero part of `R`.
            Above diagonal: Householder reflectors `v` in rows.
    tau: ndarray (...,K,). Modes: `raw`.
        Scaling factors for Householder reflectors. The unit normal to the
        reflection plane is ``V = sqrt(tau/2) [0 ... 0 1 v^T]^T``.
    """
    if mode.lower() not in QR_MODES.keys():
        raise ValueError('Modes known to qr: reduced, complete, r, raw.\n'
                         + 'Unknown mode: ' + mode)
    ufunc = QR_MODES[mode.lower()][arr.shape[-2] > arr.shape[-1]]
    out = kwds.pop('out', (out1, out2)[:ufunc.nout])
    gf.make_errobj("QR failed: rank deficient?", kwds)
    return ufunc(arr, out=out, **kwds)


LQ_MODES = {'reduced': (gf.lq_m, gf.lq_n),
            'complete': (gf.lq_n, gf.lq_n),
            'l': (gf.lq_lm, gf.lq_ln),
            'raw': (gf.lq_rawm, gf.lq_rawn)}


@set_module('numpy_linalg')
def lq(arr: np.ndarray, mode: str = 'reduced',
       out1: ty.Optional[np.ndarray] = None,
       out2: ty.Optional[np.ndarray] = None,
       **kwds) -> ty.Tuple[np.ndarray, ...]:
    """LQ decomposition.

    Factor a matrix as `A = LQ` with `Q` orthogonal and `L` lower-triangular.
    `K = min(M,N)`, except for mode `complete`, where `K = N`.

    Parameters
    -----------
    A: ndarray (...,M,N)
        Matrix to be factored.
    mode: str
        Case insensitive, chosen from:
        **reduced** - default, use inner dimensionality ``K=min(M,N)``,
        **complete** - use inner dimensionality ``K=N``, for square `Q`,
        **l** or **L** - return `L` only,
        **raw** - return `H,tau`, which determine `Q` and `L` (see below).

    Returns
    -------
    L: ndarray (...,M,K). Modes: `reduced, complete, l`.
        Matrix with zeros above the diagonal.
    Q: ndarray (...,K,N). Modes: `reduced, complete`.
        Matrix with orthonormal rows.
    H: ndarray (...,N,M). Modes: `raw`.
        Transposed matrix (Fortran order). It contains the following:
            On & above diagonal: non-zero part of `L`.
            Below diagonal: Householder reflectors `v*` in columns (conjugate).
    tau: ndarray (...,K,). Modes: `raw`.
        Scaling factors for Householder reflectors. The unit normal to the
        reflection plane is ``V = sqrt(tau/2) [0 ... 0 1 v^T]^T``.
    """
    if mode.lower() not in LQ_MODES.keys():
        raise ValueError('Modes known to lq: reduced, complete, l, raw.\n'
                         + 'Unknown mode: ' + mode)
    ufunc = LQ_MODES[mode.lower()][arr.shape[-2] > arr.shape[-1]]
    out = kwds.pop('out', (out1, out2)[:ufunc.nout])
    gf.make_errobj("LQ failed: rank deficient?", kwds)
    return ufunc(arr, out=out, **kwds)


@set_module('numpy_linalg')
def lqr(arr: np.ndarray, mode: str = 'reduced',
        out1: ty.Optional[np.ndarray] = None,
        out2: ty.Optional[np.ndarray] = None,
        **kwds) -> ty.Tuple[np.ndarray, ...]:
    """LQ/QR decomposition.

    Factor a matrix as `A = LQ` or `A = QR` with `Q` orthogonal,
    `L` lower-triangular or `R` upper-triangular. Uses LQ decomposition for
    wide matrices or QR decomposition for tall and square matrices.
    `K = min(M,N)`, except for mode `complete`, where `K = max(M,N)`.

    Parameters
    -----------
    A: ndarray (...,M,N)
        Matrix to be factored.
    mode: str (default = `reduced`)
        Case insensitive, chosen from:
        **reduced** - use inner dimensionality ``K=min(M,N)`` for square `L/R`,
        **complete** - use inner dimensionality ``K=max(M,N)`` for square `Q`,
        **l**, **L**, **r** or **R** - return `L/R` only (square),
        **raw** - return `H,tau`, which determine `Q` and `L/R` (see below).

    Returns
    -------
    L: ndarray (...,M,K). Modes: `reduced, complete, l, r`. M < N.
        Matrix with zeros above the diagonal.
    Q: ndarray, `reduced` mode: (...,M,N), `complete` mode:  (...,K,K).
        Matrix with orthonormal rows.
    R: ndarray (...,K,N). Modes: `reduced, complete, l, r`. M >= N.
        Matrix with zeros below the diagonal.
    H: ndarray (...,N,M). Modes: `raw`.
        Transposed matrix (Fortran order). It contains the following:
            On & above/below diagonal: non-zero part of `L/R`.
            Below/above diagonal: Householder reflectors `v*` or `v`.
    tau: ndarray (...,K,). Modes: `raw`.
        Scaling factors for Householder reflectors. The unit normal to the
        reflection plane is ``V = sqrt(tau/2) [0 ... 0 1 v^T]^T``.
    """
    if arr.shape[-2] < arr.shape[-1]:
        mode = 'l' if mode.lower() == 'r' else mode
        return lq(arr, mode, out1, out2, **kwds)
    mode = 'r' if mode.lower() == 'l' else mode
    return qr(arr, mode, out1, out2, **kwds)


LU_MODES = {'separate': (gf.lu_m, gf.lu_n),
            'raw': (gf.lu_rawm, gf.lu_rawn)}


@set_module('numpy_linalg')
def lu(arr: np.ndarray, mode: str = 'separate',
       out1: ty.Optional[np.ndarray] = None,
       out2: ty.Optional[np.ndarray] = None,
       out3: ty.Optional[np.ndarray] = None,
       **kwds) -> ty.Tuple[np.ndarray, np.ndarray]:
    """LU decomposition.

    Factor a matrix as `A = PLU` with `P` a permutation matrix,
    `L` lower- and `U` upper-triangular.
    `K = min(M,N)`. The diagonal elements of `L` are 1.

    Parameters
    -----------
    A: ndarray (...,M,N)
        Matrix to be factored.
    mode: str
        chosen from:
        **separate** - default, return `L` and `U` separately,
        **raw** - return `AF`, which contains `L` and `U` (see below).

    Returns
    -------
    L: ndarray (...,M,K). Modes: `separate`.
        Matrix with zeros above the diagonal and ones on the diagonal.
    U: ndarray (...,K,N). Modes: `separate`.
        Matrix with zeros below the diagonal.
    AF: ndarray (...,N,M). Modes: `raw`.
        Raw matrix output from Lapack, transposed (Fortran order).
            On & above diagonal: non-zero part of `U`.
            Below diagonal: non-zero part of `L`, excluding diagonal.
    ipiv: ndarray (...,K,). Modes: `separate, raw`.
        Pivot indices
    """
    if mode not in LU_MODES.keys():
        raise ValueError('Modes known to lu: separate, raw.\n'
                         + 'Unknown mode: ' + mode)
    ufunc = LU_MODES[mode][arr.shape[-2] > arr.shape[-1]]
    out = kwds.pop('out', (out1, out2, out3)[:ufunc.nout])
    gf.make_errobj("LU failed: rank deficient?", kwds)
    return ufunc(arr, out=out, **kwds)
