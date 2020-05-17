# -*- coding: utf-8 -*-
# =============================================================================
# Created on Tue Dec  5 02:46:29 2017
# @author: subhy
# module:_creation_ln
# =============================================================================
"""
Numpy's array creation routines, adapted to produce `lnarray`s instead of
`numpy.ndarray`s.

See `numpy` documentation for array creation routines.
This module doesn't include any record arrays or char/string stuff.
To use some other array class, change the arguments of `Wrappers`.
"""

import numpy as np
from . import wrappers as wr
from ._lnarray import lnarray

__all__ = [
    'empty', 'eye', 'identity', 'ones', 'zeros', 'full',
    'empty_like', 'ones_like', 'zeros_like', 'full_like',
    'array', 'asarray', 'asanyarray', 'ascontiguousarray', 'copy', 'loadtxt',
    'asfortranarray', 'asarray_chkfinite', 'require',
    'frombuffer', 'fromfile', 'fromfunction', 'fromiter', 'fromstring',
    'arange', 'linspace', 'logspace', 'geomspace', 'meshgrid',
    'ravel_multi_index', 'unravel_index', 'diag_indices', 'mask_indices',
    'tril_indices', 'triu_indices', 'indices',
]
wrap = wr.Wrappers(lnarray, "numpy_linalg")
wrapd = wr.DeprecatedWrappers(lnarray, "numpy_linalg")
# =========================================================================
# Ones and zeros
# =========================================================================

# existing arrrays
empty_like = wrapd.sub(np.empty_like)
ones_like = wrapd.sub(np.ones_like)
zeros_like = wrapd.sub(np.zeros_like)
full_like = wrapd.sub(np.full_like)
# new arrays
empty = wrap.one(np.empty)
eye = wrap.one(np.eye)
identity = wrap.one(np.identity)
ones = wrap.one(np.ones)
zeros = wrap.one(np.zeros)
full = wrap.one(np.full)


# =========================================================================
# From existing data
# =========================================================================


# new(ish) arrays
array = wrap.sub(np.array)
asarray = wrap.sub(np.asarray)
asanyarray = wrap.sub(np.asanyarray)
ascontiguousarray = wrap.one(np.ascontiguousarray)
copy = wrap.one(np.copy)
frombuffer = wrap.one(np.frombuffer)
fromfile = wrap.one(np.fromfile)
fromfunction = wrap.one(np.fromfunction)
fromiter = wrap.one(np.fromiter)
fromstring = wrap.one(np.fromstring)
loadtxt = wrap.one(np.loadtxt)
asfortranarray = wrap.one(np.asfortranarray)
asarray_chkfinite = wrap.one(np.asarray_chkfinite)
require = wrap.sub(np.require)


# =========================================================================
# Numerical ranges
# =========================================================================


# new arrays
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


# existing arrrays
