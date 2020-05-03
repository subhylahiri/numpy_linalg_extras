# -*- coding: utf-8 -*-
# =============================================================================
# Created on Sat Sep  1 20:35:47 2018
#
# @author: subhy
# =============================================================================
"""
Generalized ufunc versions of `numpy.linalg` routines:

This module also contains utilities (for internal use) that help in defining
functions and classes that use these gufuncs:

Routine Listings
----------------
unbroadcast_factors
    Undo broadcasting in factors returned by gufuncs.
make_errobj
    create an error handler object for use as ``extobj`` gufunc parameter.
return_shape_mat
    Shape of result of broadcasted matrix multiplication.
MatmulOperatorsMixin
    Mixin class that uses `matmul` from here to define @ operators. Deprecated.
LNArrayOperatorsMixin
    Subclass of `numpy.lib.mixins.NDArrayOperatorsMixin` that uses `matmul`
    from here to define @ operators. Deprecated.
"""
from typing import Tuple
import numpy as np
import numpy.lib.mixins as mix
from numpy.lib.mixins import _numeric_methods
from ._families import matmul

# =============================================================================
# Error handling
# =============================================================================
with np.errstate(invalid='call'):
    _ERROBJ = np.geterrobj()[:2]


def make_errobj(msg, kwdict=None):
    """Create an error handler list

    Parameters
    ----------
    msg: str
        Message for ``LinAlgError``
    kwdict: Optional[Dict[str, Any]]
        Dictionary of nameword arguments to which we add ``extobj``.
        If it already has an extobj, it is left unchanged.

    Returns
    -------
    extobj
        List of [buffer size, error mask, error handler].
        Error mask corresponds to ``invalid='raise'``.
        Error handler raises ``LinAlgError`` with message ``msg``.
    """
    extobj = list(_ERROBJ)  # make a copy

    def callback(err, flag):
        """Raise LinAlgError"""
        raise np.linalg.LinAlgError(msg)

    extobj.append(callback)
    if kwdict is not None and not kwdict.get('extobj'):
        kwdict['extobj'] = extobj
    return extobj


# =============================================================================
# Undo overbroadcast factors
# =============================================================================


def unbroadcast_factors(original, *factors):
    """Undo broadcasting in factors returned by gufuncs.

    Parameters
    ----------
    original : np.ndarray or Tuple[int]
        Matrix whose factors we received, or its shape. Assumes that last two
        axes are the cores dimensions and any earlier axes are broadcast
        dimensions (unless it is one-dimensional, in which case it has no
        broadcast dimensions).
    *factors : np.ndarray
        Matrix factors returned by gufunc. Assumes that last two axes of
        `factors[0]` are cores dimensions (unless `original` is one-dimensional,
        in which case only the last axis is assumed cores) and any earlier axes
        are broadcast dimensions. All subsequent factors are assumed to have
        the same broadcast dimensions.
    Returns
    -------
    *unbc_factors : np.ndarray
        Matrix factors with spurious broadcast dimensions removed, so that they
        broadcast with `original`.
    """
    if isinstance(original, np.ndarray):
        original = original.shape
    bc_dim = factors[0].ndim - 2
    if len(original) == 1:
        bc_dim += 1
    original = original[:-2]
    squeeze = tuple(np.arange(bc_dim - len(original)))
    original = (1,)*len(squeeze) + original
    shrink = [orig == 1 for orig in original]
    slc = tuple(np.where(shrink, slice(1), slice(None)))
    return tuple(fac[slc].squeeze(squeeze) for fac in factors)


# =============================================================================
# Mixin for linear algebra operators
# =============================================================================


class MatmulOperatorsMixin():
    """Mixin for defining __matmul__ special methods via gufuncs

    .. deprecated:: 0.2.0
          `MatmulOperatorsMixin` will be removed in numpy_linalg 0.3.0, it is
          replaced by `numpy.lib.mixins.NDArrayOperatorsMixin` because the
          latter now uses the `matmul` gufunc rendering this mixin obsolete.
    """
    __matmul__, __rmatmul__, __imatmul__ = _numeric_methods(matmul, 'matmul')


class LNArrayOperatorsMixin(mix.NDArrayOperatorsMixin):
    """Mixin for defining operator special methods via __array_ufunc__

    .. deprecated:: 0.2.0
          `LNArrayOperatorsMixin` will be removed in numpy_linalg 0.3.0, it
          is replaced by `numpy.lib.mixins.NDArrayOperatorsMixin` because the
          latter now uses the `matmul` gufunc rendering this mixin obsolete.

    See also
    --------
    `numpy.lib.mixins.NDArrayOperatorsMixin` : base class.
    """


# =============================================================================
# Shape for linear algebra
# =============================================================================


def _split_signature(signature: str) -> Tuple[Tuple[str, ...], ...]:
    """Convert text signature into tuples of axes size names

    Parameters
    ----------
    signature : str
        Text signature of matrix operation, without optional axes or spaces,

    Returns
    -------
    axes_sizes : Tuple[Tuple[str, ...], ...]
        Tuples of core axes sizes as string variablr names,
        e.g. `(('a','b'),('b','c'),('a','c'))`
    """
    signature = signature.lstrip('(').rstrip(')').replace('->', ',')
    arrays = []
    for array in signature.split('),('):
        arrays.append(tuple(array.split(',')))
    return tuple(arrays)


def return_shape(signature: str, *shapes: Tuple[int, ...]) -> Tuple[int, ...]:
    """Shape of result of broadcasted matrix operation

    Parameters
    ----------
    signature : Tuple[str]
        Signature of the operation, without optional axes or spaces,
        e.g. `'(a,b),(b,c)->(a,c)'`
    shapes : Tuple[int, ...]
        Shapes of arguments of matrix operation.

    Returns
    -------
    output_shape : Tuple[int]
        Shape of result of broadcasted matrix operation.

    Raises
    ------
    ValueError
        If `arrays.shape`s do not match signatures.
    """
    msg = (f'Shape: {shapes}. Signature: {signature}.')
    sigs = _split_signature(signature)
    dims = [len(sig) for sig in sigs]
    broads, cores, sizes = [], [], {}
    if any(len(shape) < dim for shape, dim in zip(shapes, dims)):
        raise ValueError('Not enough cores dimensions. ' + msg)
    for shape, dim in zip(shapes, dims):
        broads.append(shape[:-dim])
        cores.append(shape[-dim:])
    for sig, core in zip(sigs, cores):
        for name, siz in zip(sig, core):
            sizes.setdefault(name, siz)
            if sizes[name] != siz:
                raise ValueError('Inner matrix dimensions mismatch: ' + msg)
    broad = np.broadcast(*(np.empty(broad) for broad in broads)).shape
    core = []
    for name in sigs[-1]:
        core.append(sizes[name])
    return broad + tuple(core)


def array_return_shape(signature: str, *arrays: np.ndarray) -> Tuple[int, ...]:
    """Shape of result of broadcasted matrix operation

    Parameters
    ----------
    signature : Tuple[str]
        Signature of the operation, without optional axes or spaces,
        e.g. `'(a,b),(b,c)->(a,c)'`
    arrays : np.ndarray
        Arguments of matrix operation.

    Returns
    -------
    output_shape : Tuple[int]
        Shape of result of broadcasted matrix operation.

    Raises
    ------
    ValueError
        If `arrays.shape`s do not match signatures.
    """
    return return_shape(signature, *(array.shape for array in arrays))
