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
To use some other array class, change the second import statement.
"""
import numpy as np
from . import _ln_wrap as wr
from ._lnarray import lnarray

__all__ = [
           'reshape', 'moveaxis', 'rollaxis', 'swapaxes',
           'atleast_1d', 'atleast_2d', 'atleast_3d',
           'broadcast_to', 'broadcast_arrays', 'expand_dims', 'squeeze',
           'asfarray', 'asfortranarray', 'asarray_chkfinite', 'require',
           'concatenate', 'stack', 'column_stack', 'dstack', 'hstack',
           'vstack', 'block', 'split', 'array_split', 'dsplit', 'hsplit',
           'vsplit', 'tile', 'repeat', 'delete', 'insert', 'append', 'resize',
           'trim_zeros', 'unique', 'flip', 'fliplr', 'flipud',
           'roll', 'rot90'
           ]

wrap_one = wr.make_wrap_one(lnarray, "numpy_linalg")
wrap_sub = wr.make_wrap_sub(lnarray, "numpy_linalg")
wrap_several = wr.make_wrap_several(lnarray, "numpy_linalg")
wrap_subseveral = wr.make_wrap_subseveral(lnarray, "numpy_linalg")
# =========================================================================
# Array manipulation routines
# =========================================================================
# =========================================================================
# Basic operations
# =========================================================================
# copyto = wrap_one(np.copyto)
# =========================================================================
# Changing array shape
# =========================================================================
reshape = wrap_one(np.reshape)
# =========================================================================
# Transpose-like operations
# =========================================================================
moveaxis = wrap_one(np.moveaxis)
rollaxis = wrap_one(np.rollaxis)
swapaxes = wrap_one(np.swapaxes)
# transpose = wrap_one(np.transpose)
# =========================================================================
# Changing number of dimensions
# =========================================================================
atleast_1d = wrap_one(np.atleast_1d)
atleast_2d = wrap_one(np.atleast_2d)
atleast_3d = wrap_one(np.atleast_3d)
broadcast_to = wrap_sub(np.broadcast_to)
broadcast_arrays = wrap_subseveral(np.broadcast_arrays)
expand_dims = wrap_one(np.expand_dims)
squeeze = wrap_one(np.squeeze)
# =========================================================================
# Changing kind of array
# =========================================================================
asfarray = wrap_one(np.asfarray)
asfortranarray = wrap_one(np.asfortranarray)
asarray_chkfinite = wrap_one(np.asarray_chkfinite)
require = wrap_sub(np.require)
# =========================================================================
# Joining arrays
# =========================================================================
concatenate = wrap_one(np.concatenate)
stack = wrap_one(np.stack)
column_stack = wrap_one(np.column_stack)
dstack = wrap_one(np.dstack)
hstack = wrap_one(np.hstack)
vstack = wrap_one(np.vstack)
block = wrap_one(np.block)
# =========================================================================
# Splitting arrays
# =========================================================================
split = wrap_several(np.split)
array_split = wrap_several(np.array_split)
dsplit = wrap_several(np.dsplit)
hsplit = wrap_several(np.hsplit)
vsplit = wrap_several(np.vsplit)
# =========================================================================
# Tiling arrays
# =========================================================================
tile = wrap_one(np.tile)
repeat = wrap_one(np.repeat)
# =========================================================================
# Adding and removing elements
# =========================================================================
delete = wrap_one(np.delete)
insert = wrap_one(np.insert)
append = wrap_one(np.append)
resize = wrap_one(np.resize)
trim_zeros = wrap_one(np.trim_zeros)
unique = wrap_one(np.unique)
# =========================================================================
# Rearranging elements
# =========================================================================
flip = wrap_one(np.flip)
fliplr = wrap_one(np.fliplr)
flipud = wrap_one(np.flipud)
reshape = wrap_one(np.reshape)
roll = wrap_one(np.roll)
rot90 = wrap_one(np.rot90)
