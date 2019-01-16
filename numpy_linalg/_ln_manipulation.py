# -*- coding: utf-8 -*-
# =============================================================================
# Created on Mon Sep 17 23:42:28 2018
#
# @author: subhy
# =============================================================================
"""
Numpy's array manipulation routines, adapted to produce `lnarray`s instead of
`numpy.ndarray`s.

See `numpy` documentation for array manipulation routines.
To use some other array class, change the second import statement.
"""
import numpy as np
from . import _ln_wrap as _wr

__all__ = [
           'reshape', 'moveaxis', 'rollaxis', 'swapaxes',
           'atleast_1d', 'atleast_2d', 'atleast_3d',
           'broadcast_to', 'broadcast_arrays', 'expand_dims', 'squeeze',
           'asfarray', 'asfortranarray', 'asarray_chkfinite', 'require',
           'concatenate', 'stack', 'column_stack', 'dstack', 'hstack',
           'vstack', 'block', 'split', 'array_split', 'dsplit', 'hsplit',
           'vsplit', 'tile', 'repeat', 'delete', 'insert', 'append', 'resize',
           'trim_zeros', 'unique', 'flip', 'fliplr', 'flipud', 'reshape',
           'roll', 'rot90'
           ]

# =============================================================================
# Array manipulation routines
# =============================================================================
# =============================================================================
# Basic operations
# =============================================================================
# copyto = _wr.wrap_one(np.copyto)
# =============================================================================
# Changing array shape
# =============================================================================
reshape = _wr.wrap_one(np.reshape)
# =============================================================================
# Transpose-like operations
# =============================================================================
moveaxis = _wr.wrap_one(np.moveaxis)
rollaxis = _wr.wrap_one(np.rollaxis)
swapaxes = _wr.wrap_one(np.swapaxes)
# transpose = _wr.wrap_one(np.transpose)
# =============================================================================
# Changing number of dimensions
# =============================================================================
atleast_1d = _wr.wrap_one(np.atleast_1d)
atleast_2d = _wr.wrap_one(np.atleast_2d)
atleast_3d = _wr.wrap_one(np.atleast_3d)
broadcast_to = _wr.wrap_sub(np.broadcast_to)
broadcast_arrays = _wr.wrap_subseveral(np.broadcast_arrays)
expand_dims = _wr.wrap_one(np.expand_dims)
squeeze = _wr.wrap_one(np.squeeze)
# =============================================================================
# Changing kind of array
# =============================================================================
asfarray = _wr.wrap_one(np.asfarray)
asfortranarray = _wr.wrap_one(np.asfortranarray)
asarray_chkfinite = _wr.wrap_one(np.asarray_chkfinite)
require = _wr.wrap_one(np.require)
# =============================================================================
# Joining arrays
# =============================================================================
concatenate = _wr.wrap_one(np.concatenate)
stack = _wr.wrap_one(np.stack)
column_stack = _wr.wrap_one(np.column_stack)
dstack = _wr.wrap_one(np.dstack)
hstack = _wr.wrap_one(np.hstack)
vstack = _wr.wrap_one(np.vstack)
block = _wr.wrap_one(np.block)
# =============================================================================
# Splitting arrays
# =============================================================================
split = _wr.wrap_several(np.split)
array_split = _wr.wrap_several(np.array_split)
dsplit = _wr.wrap_several(np.dsplit)
hsplit = _wr.wrap_several(np.hsplit)
vsplit = _wr.wrap_several(np.vsplit)
# =============================================================================
# Tiling arrays
# =============================================================================
tile = _wr.wrap_one(np.tile)
repeat = _wr.wrap_one(np.repeat)
# =============================================================================
# Adding and removing elements
# =============================================================================
delete = _wr.wrap_one(np.delete)
insert = _wr.wrap_one(np.insert)
append = _wr.wrap_one(np.append)
resize = _wr.wrap_one(np.resize)
trim_zeros = _wr.wrap_one(np.trim_zeros)
unique = _wr.wrap_one(np.unique)
# =============================================================================
# Rearranging elements
# =============================================================================
flip = _wr.wrap_one(np.flip)
fliplr = _wr.wrap_one(np.fliplr)
flipud = _wr.wrap_one(np.flipud)
reshape = _wr.wrap_one(np.reshape)
roll = _wr.wrap_one(np.roll)
rot90 = _wr.wrap_one(np.rot90)
