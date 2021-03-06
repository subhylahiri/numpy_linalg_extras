# -*- coding: utf-8 -*-
# =============================================================================
"""Numpy's array creation routines, adapted to produce `lnarray`s instead of
`numpy.ndarray`s.

See `numpy` documentation for array creation routines.
This module doesn't include any record arrays or char/string stuff.

To use some other array class, change the arguments of `Wrappers`.
"""
from typing import Union

import numpy as np
import numpy.lib.npyio as npio

from . import wrappers as wr
from ._lnarray import lnarray
# =============================================================================
__all__ = [
    'empty', 'eye', 'identity', 'ones', 'zeros', 'full', 'array', 'asarray',
    'asanyarray', 'ascontiguousarray', 'asfortranarray', 'asarray_chkfinite',
    'asfarray', 'copy', 'require', 'load', 'loadtxt', 'genfromtxt', 'fromfile',
    'fromregex', 'frombuffer', 'fromstring', 'fromfunction', 'fromiter',
    'arange', 'linspace', 'logspace', 'geomspace', 'meshgrid',
    'ravel_multi_index', 'unravel_index', 'diag_indices', 'mask_indices',
    'tril_indices', 'triu_indices', 'indices', 'mgrid', 'ogrid', 'r_', 'c_',
]
wrap = wr.Wrappers(lnarray, "numpy_linalg",
                   {'ndarray': "lnarray", 'NpzFile': "LnNpzFile"})

# =========================================================================
# Classes
# =========================================================================


@wrap.decorate(np.lib.index_tricks.nd_grid)
class LnNdGrid(wr.WrappedSubscriptable, wrappers=wrap, method="several"):
    """Wrapped version of numpy.lib.index_tricks.nd_grid adapted to produce
    `lnarray`s instead of `numpy.ndarray`s.

    Parameters
    ----------
    obj : numpy.lib.index_tricks.nd_grid
        The intance being wrapped.

    See Also
    --------
    numpy.mgrid
    numpy.ogrid
    """
    _obj: np.lib.index_tricks.nd_grid


mgrid = LnNdGrid(np.mgrid)
ogrid = LnNdGrid(np.ogrid)


class LnAxisConcatenator(wr.WrappedSubscriptable, wrappers=wrap, method="sub"):
    """Wrapped version of numpy.lib.index_tricks.AxisConcatenator adapted to
    produce `lnarray`s instead of `numpy.ndarray`s.

    Parameters
    ----------
    obj : numpy.lib.index_tricks.AxisConcatenator
        The intance being wrapped.

    See Also
    --------
    numpy.r_
    numpy.c_
    """
    _obj: np.lib.index_tricks.AxisConcatenator


r_ = LnAxisConcatenator(np.r_)
c_ = LnAxisConcatenator(np.c_)


@wrap.decorate(npio.NpzFile)
class LnNpzFile(wr.WrappedSubscriptable, wrappers=wrap, method="check"):
    """Wrapped version of numpy.lib.npyio.NpzFile adapted to produce
    `lnarray`s instead of `numpy.ndarray`s.

    Parameters
    ----------
    obj : numpy.lib.npyio.NpzFile
        The intance being wrapped.

    See Also
    --------
    numpy.load
    numpy.lib.npyio.NpzFile
    """
    _obj: npio.NpzFile

    def __enter__(self):
        self._obj = self._obj.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._obj.__exit__(exc_type, exc_val, exc_tb)


wr_load = wrap.check(np.load)


@wrap.decorate(np.load)
def load(*args, **kwargs) -> Union[lnarray, LnNpzFile]:
    """Wrapped version of numpy.load adapted to produce `lnarray`s instead of
    `numpy.ndarray`s.

    See Also
    --------
    numpy.load
    numpy.lib.npyio.NpzFile
    """
    result = wr_load(*args, **kwargs)
    if isinstance(result, npio.NpzFile):
        return LnNpzFile(result)
    return result


# =========================================================================
# Ones and zeros
# =========================================================================
empty = wrap.one(np.empty)
eye = wrap.one(np.eye)
identity = wrap.one(np.identity)
ones = wrap.one(np.ones)
zeros = wrap.one(np.zeros)
full = wrap.one(np.full)

# =========================================================================
# From existing data
# =========================================================================
array = wrap.sub(np.array)
asarray = wrap.sub(np.asarray)
asanyarray = wrap.sub(np.asanyarray)
ascontiguousarray = wrap.one(np.ascontiguousarray)
asfortranarray = wrap.one(np.asfortranarray)
asarray_chkfinite = wrap.one(np.asarray_chkfinite)
asfarray = wrap.one(np.asfarray)
copy = wrap.one(np.copy)
require = wrap.sub(np.require)
fromfunction = wrap.one(np.fromfunction)
fromiter = wrap.one(np.fromiter)
frombuffer = wrap.one(np.frombuffer)
fromstring = wrap.one(np.fromstring)
# -----------------------------------------------------------------------------
# input/output
# -----------------------------------------------------------------------------
fromregex = wrap.one(np.fromregex)
fromfile = wrap.one(np.fromfile)
loadtxt = wrap.one(np.loadtxt)
genfromtxt = wrap.one(np.genfromtxt)

# =========================================================================
# Numerical ranges
# =========================================================================
arange = wrap.one(np.arange)
linspace = wrap.one(np.linspace)
logspace = wrap.one(np.logspace)
geomspace = wrap.one(np.geomspace)
meshgrid = wrap.several(np.meshgrid)

# =========================================================================
# Generating index arrays
# =========================================================================
ravel_multi_index = wrap.several(np.ravel_multi_index)
unravel_index = wrap.several(np.unravel_index)
diag_indices = wrap.several(np.diag_indices)
mask_indices = wrap.several(np.mask_indices)
tril_indices = wrap.several(np.tril_indices)
triu_indices = wrap.several(np.triu_indices)
indices = wrap.several(np.indices)
