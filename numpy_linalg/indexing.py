"""Wrapped version of module numpy
"""

import numpy as _pr
from . import _ln_wrap as _wr
from ._lnarray import lnarray as _lnarray

__all__ = [
    'ravel_multi_index',
    'unravel_index',
    'ix_',
    'fill_diagonal',
    'diag_indices',
    'diag_indices_from',
    'mask_indices',
    'tril_indices',
    'tril_indices_from',
    'triu_indices',
    'triu_indices_from',
    'indices',
]

_wrap_several = _wr.make_wrap_several(_lnarray, "numpy_linalg")
# new arrays
ravel_multi_index = _wrap_several(_pr.ravel_multi_index)
unravel_index = _wrap_several(_pr.unravel_index)
ix_ = _wrap_several(_pr.ix_)
diag_indices = _wrap_several(_pr.diag_indices)
mask_indices = _wrap_several(_pr.mask_indices)
tril_indices = _wrap_several(_pr.tril_indices)
triu_indices = _wrap_several(_pr.triu_indices)
indices = _wrap_several(_pr.indices)
# existing arrays:
fill_diagonal = _wrap_several(_pr.fill_diagonal)
diag_indices_from = _wrap_several(_pr.diag_indices_from)
tril_indices_from = _wrap_several(_pr.tril_indices_from)
triu_indices_from = _wrap_several(_pr.triu_indices_from)
