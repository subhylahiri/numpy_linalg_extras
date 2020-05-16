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
To use some other array class, change the second import statement.
"""

import numpy as np
from . import wrappers as wr
from ._lnarray import lnarray

__all__ = [
    'empty', 'empty_like', 'eye', 'identity', 'ones', 'ones_like',
    'zeros', 'zeros_like', 'full', 'full_like',
    'array', 'asarray', 'asanyarray', 'ascontiguousarray', 'copy', 'loadtxt',
    'frombuffer', 'fromfile', 'fromfunction', 'fromiter', 'fromstring',
    'arange', 'linspace', 'logspace', 'geomspace', 'meshgrid',
    'diag', 'diagflat', 'tri', 'tril', 'triu', 'vander'
]



wrap_one = wr.make_wrap_one(lnarray, "numpy_linalg")
wrap_sub = wr.make_wrap_sub(lnarray, "numpy_linalg")
wrap_several = wr.make_wrap_several(lnarray, "numpy_linalg")
# =========================================================================
# Ones and zeros
# =========================================================================

# new arrays
empty = wrap_one(np.empty)
eye = wrap_one(np.eye)
identity = wrap_one(np.identity)
ones = wrap_one(np.ones)
zeros = wrap_one(np.zeros)
full = wrap_one(np.full)
# existing arrrays
empty_like = wrap_sub(np.empty_like)
ones_like = wrap_sub(np.ones_like)
zeros_like = wrap_sub(np.zeros_like)
full_like = wrap_sub(np.full_like)


# =========================================================================
# From existing data
# =========================================================================


# new(ish) arrays
array = wrap_sub(np.array)
asarray = wrap_one(np.asarray)
asanyarray = wrap_sub(np.asanyarray)
ascontiguousarray = wrap_one(np.ascontiguousarray)
copy = wrap_one(np.copy)
frombuffer = wrap_one(np.frombuffer)
fromfile = wrap_one(np.fromfile)
fromfunction = wrap_one(np.fromfunction)
fromiter = wrap_one(np.fromiter)
fromstring = wrap_one(np.fromstring)
loadtxt = wrap_one(np.loadtxt)


# =========================================================================
# Numerical ranges
# =========================================================================


# new arrays
arange = wrap_one(np.arange)
linspace = wrap_one(np.linspace)
logspace = wrap_one(np.logspace)
geomspace = wrap_one(np.geomspace)
meshgrid = wrap_several(np.meshgrid)


# =========================================================================
# Building matrices
# =========================================================================


# existing arrrays
diag = wrap_one(np.diag)
diagflat = wrap_one(np.diagflat)
tri = wrap_one(np.tri)
tril = wrap_one(np.tril)
triu = wrap_one(np.triu)
vander = wrap_one(np.vander)
