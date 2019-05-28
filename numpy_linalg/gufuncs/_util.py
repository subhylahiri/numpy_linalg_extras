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

MatmulOperatorsMixin
    Mixin class that uses `matmul` from here to define @ operators.
LNArrayOperatorsMixin
    Subclass of `numpy.lib.mixins.NDArrayOperatorsMixin` that uses `matmul`
    from here to define @ operators.
make_errobj:
    create an error handler object for use as ``extobj`` gufunc parameter.
return_shape_mat:
    Shape of result of broadcasted matrix multiplication
"""
import numpy as _np
import numpy.lib.mixins as _mix
from ._families import matmul


# =============================================================================
# Error handling
# =============================================================================
with _np.errstate(invalid='call'):
    _errobj = _np.geterrobj()[:2]


def make_errobj(msg, kwdict=None):
    """Create an error handler list

    Parameters
    ----------
    msg: str
        Message for ``LinAlgError``
    kwdict: Optional[Dict[str, Any]]
        Dictionary of keyword arguments to which we add ``extobj``.
        If it already has an extobj, it is left unchanged.

    Returns
    -------
    extobj
        List of [buffer size, error mask, error handler].
        Error mask corresponds to ``invalid='raise'``.
        Error handler raises ``LinAlgError`` with message ``msg``.
    """
    extobj = list(_errobj)  # make a copy

    def callback(err, flag):
        """Raise LinAlgError"""
        raise _np.linalg.LinAlgError(msg)

    extobj.append(callback)
    if kwdict is not None and not kwdict.get('extobj'):
        kwdict['extobj'] = extobj
    return extobj


# =============================================================================
# Mixin for linear algebra operators
# =============================================================================


class MatmulOperatorsMixin():
    """Mixin for defining __matmul__ special methods via gufuncs
    """
    __matmul__, __rmatmul__, __imatmul__ = _mix._numeric_methods(matmul,
                                                                 'matmul')


class LNArrayOperatorsMixin(_mix.NDArrayOperatorsMixin, MatmulOperatorsMixin):
    """Mixin for defining operator special methods via __array_ufunc__

    See also
    --------
    `numpy.lib.mixins.NDArrayOperatorsMixin` : base class.
    """
    pass


# =============================================================================
# Shape for linear algebra
# =============================================================================


def return_shape_mat(x, y):
    """Shape of result of broadcasted matrix multiplication
    """
    if x.ndim == 0 or y.ndim == 0:
        raise ValueError('Scalar operations not supported. Use mul.')
    if y.ndim == 1:
        return x.shape[:-1]
    if x.ndim == 1:
        return y.shape[:-2] + y.shape[-1:]
    if x.shape[-1] != y.shape[-2]:
        msg = 'Inner matrix dimensions mismatch: {0} and {1}.'
        raise ValueError(msg.format(x.shape, y.shape))
    return _np.broadcast(x[..., :1], y[..., :1, :]).shape
