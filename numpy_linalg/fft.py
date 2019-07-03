# -*- coding: utf-8 -*-
"""Wrapped numpy.fft routines
"""
import numpy as np
from . import _ln_wrap as _wr

__all__ = [
    'fft',
    'ifft',
    'rfft',
    'irfft',
    'hfft',
    'ihfft',
    'rfftn',
    'irfftn',
    'rfft2',
    'irfft2',
    'fft2',
    'ifft2',
    'fftn',
    'ifftn',
]

fft = _wr.wrap_one(np.fft.fft)
ifft = _wr.wrap_one(np.fft.ifft)
rfft = _wr.wrap_one(np.fft.rfft)
irfft = _wr.wrap_one(np.fft.irfft)
hfft = _wr.wrap_one(np.fft.hfft)
ihfft = _wr.wrap_one(np.fft.ihfft)
rfftn = _wr.wrap_one(np.fft.rfftn)
irfftn = _wr.wrap_one(np.fft.irfftn)
rfft2 = _wr.wrap_one(np.fft.rfft2)
irfft2 = _wr.wrap_one(np.fft.irfft2)
fft2 = _wr.wrap_one(np.fft.fft2)
ifft2 = _wr.wrap_one(np.fft.ifft2)
fftn = _wr.wrap_one(np.fft.fftn)
ifftn = _wr.wrap_one(np.fft.ifftn)
