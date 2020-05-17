# -*- coding: utf-8 -*-
# =========================================================================
# Created on Mon Sep 17 23:42:28 2018
#
# @author: subhy
# =========================================================================
"""
Numpy's array manipulation routines, adapted to produce `lnarray`s instead of
`numpy.ndarray`s.

See `numpy` documentation for array manipulation routines.
To use some other array class, change the arguments of `Wrappers`t.
"""
import numpy as np
from . import wrappers as wr
from ._lnarray import lnarray

__all__ = [
    'reshape', 'moveaxis', 'rollaxis', 'swapaxes',
    'atleast_1d', 'atleast_2d', 'atleast_3d',
    'broadcast_to', 'broadcast_arrays', 'expand_dims', 'squeeze',
    'concatenate', 'stack', 'column_stack', 'dstack', 'hstack', 'vstack',
    'block', 'split', 'array_split', 'dsplit', 'hsplit', 'vsplit', 'tile',
    'repeat', 'delete', 'insert', 'append', 'resize', 'trim_zeros', 'unique',
    'flip', 'fliplr', 'flipud', 'roll', 'rot90', 'ix_', 'fill_diagonal',
    'diag_indices_from', 'tril_indices_from', 'triu_indices_from',
    'asfarray', 'diag', 'diagflat', 'tri', 'tril', 'triu', 'vander',
]
wrap = wr.DeprecatedWrappers(lnarray, "numpy_linalg")
# =========================================================================
# Array manipulation routines
# =========================================================================
# =========================================================================
# Basic operations
# =========================================================================
# copyto = wrap.one(np.copyto)
# =========================================================================
# Changing array shape
# =========================================================================
reshape = wrap.one(np.reshape)
# =========================================================================
# Transpose-like operations
# =========================================================================
moveaxis = wrap.one(np.moveaxis)
rollaxis = wrap.one(np.rollaxis)
swapaxes = wrap.one(np.swapaxes)
# transpose = wrap.one(np.transpose)z
# =========================================================================
# Changing number of dimensions
# =========================================================================
atleast_1d = wrap.one(np.atleast_1d)
atleast_2d = wrap.one(np.atleast_2d)
atleast_3d = wrap.one(np.atleast_3d)
broadcast_to = wrap.sub(np.broadcast_to)
broadcast_arrays = wrap.subseveral(np.broadcast_arrays)
expand_dims = wrap.one(np.expand_dims)
squeeze = wrap.one(np.squeeze)
# =========================================================================
# Joining arrays
# =========================================================================
concatenate = wrap.one(np.concatenate)
stack = wrap.one(np.stack)
column_stack = wrap.one(np.column_stack)
dstack = wrap.one(np.dstack)
hstack = wrap.one(np.hstack)
vstack = wrap.one(np.vstack)
block = wrap.one(np.block)
# =========================================================================
# Splitting arrays
# =========================================================================
split = wrap.several(np.split)
array_split = wrap.several(np.array_split)
dsplit = wrap.several(np.dsplit)
hsplit = wrap.several(np.hsplit)
vsplit = wrap.several(np.vsplit)
# =========================================================================
# Tiling arrays
# =========================================================================
tile = wrap.one(np.tile)
repeat = wrap.one(np.repeat)
# =========================================================================
# Adding and removing elements
# =========================================================================
delete = wrap.one(np.delete)
insert = wrap.one(np.insert)
append = wrap.one(np.append)
resize = wrap.one(np.resize)
trim_zeros = wrap.one(np.trim_zeros)
unique = wrap.one(np.unique)
# =========================================================================
# Rearranging elements
# =========================================================================
flip = wrap.one(np.flip)
fliplr = wrap.one(np.fliplr)
flipud = wrap.one(np.flipud)
reshape = wrap.one(np.reshape)
roll = wrap.one(np.roll)
rot90 = wrap.one(np.rot90)
# =========================================================================
# Indexing-like operations
# =========================================================================
ix_ = wrap.several(np.ix_)
fill_diagonal = wrap.several(np.fill_diagonal)
diag_indices_from = wrap.several(np.diag_indices_from)
tril_indices_from = wrap.several(np.tril_indices_from)
triu_indices_from = wrap.several(np.triu_indices_from)
# =========================================================================
# Building matrices
# =========================================================================
asfarray = wrap.one(np.asfarray)
diag = wrap.one(np.diag)
diagflat = wrap.one(np.diagflat)
tri = wrap.one(np.tri)
tril = wrap.one(np.tril)
triu = wrap.one(np.triu)
vander = wrap.one(np.vander)
