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
from . import _ln_wrap as _wr

__all__ = ['empty', 'empty_like', 'eye', 'identity', 'ones', 'ones_like',
           'zeros', 'zeros_like', 'full', 'full_like',
           'array', 'asarray', 'asanyarray', 'ascontiguousarray', 'copy',
           'frombuffer', 'fromfile', 'fromfunction', 'fromiter', 'fromstring',
           'loadtxt',
           'arange', 'linspace', 'logspace', 'geomspace', 'meshgrid',
           'diag', 'diagflat', 'tri', 'tril', 'triu', 'vander'
           ]


# =============================================================================
# Ones and zeros
# =============================================================================


empty = _wr.wrap_one(np.empty)
empty_like = _wr.wrap_sub(np.empty_like)
eye = _wr.wrap_one(np.eye)
identity = _wr.wrap_one(np.identity)
ones = _wr.wrap_one(np.ones)
ones_like = _wr.wrap_sub(np.ones_like)
zeros = _wr.wrap_one(np.zeros)
zeros_like = _wr.wrap_sub(np.zeros_like)
full = _wr.wrap_one(np.full)
full_like = _wr.wrap_sub(np.full_like)


# =============================================================================
# From existing data
# =============================================================================


array = _wr.wrap_sub(np.array)
asarray = _wr.wrap_one(np.asarray)
asanyarray = _wr.wrap_sub(np.asanyarray)
ascontiguousarray = _wr.wrap_one(np.ascontiguousarray)
copy = _wr.wrap_one(np.copy)
frombuffer = _wr.wrap_one(np.frombuffer)
fromfile = _wr.wrap_one(np.fromfile)
fromfunction = _wr.wrap_one(np.fromfunction)
fromiter = _wr.wrap_one(np.fromiter)
fromstring = _wr.wrap_one(np.fromstring)
loadtxt = _wr.wrap_one(np.loadtxt)


# =============================================================================
# Numerical ranges
# =============================================================================


arange = _wr.wrap_one(np.arange)
linspace = _wr.wrap_one(np.linspace)
logspace = _wr.wrap_one(np.logspace)
geomspace = _wr.wrap_one(np.geomspace)
meshgrid = _wr.wrap_one(np.meshgrid)


# =============================================================================
# Building matrices
# =============================================================================


diag = _wr.wrap_one(np.diag)
diagflat = _wr.wrap_one(np.diagflat)
tri = _wr.wrap_one(np.tri)
tril = _wr.wrap_one(np.tril)
triu = _wr.wrap_one(np.triu)
vander = _wr.wrap_one(np.vander)
