# -*- coding: utf-8 -*-
"""Wrapped numpy.fft routines
"""
import numpy.fft as nf
from . import wrappers as _wr
from ._lnarray import lnarray as _lnarray

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
    'fftshift',
    'ifftshift',
    'fftfreq',
    'rfftfreq',
]
_wrap_one = _wr.make_wrap_one(_lnarray, "numpy_linalg.fft")
_wrap_oned = _wr.deprecated(_wrap_one)
# existing arrays
fft = _wrap_oned(nf.fft)
ifft = _wrap_one(nf.ifft)
rfft = _wrap_one(nf.rfft)
irfft = _wrap_one(nf.irfft)
hfft = _wrap_one(nf.hfft)
ihfft = _wrap_one(nf.ihfft)
rfftn = _wrap_one(nf.rfftn)
irfftn = _wrap_one(nf.irfftn)
rfft2 = _wrap_one(nf.rfft2)
irfft2 = _wrap_one(nf.irfft2)
fft2 = _wrap_one(nf.fft2)
ifft2 = _wrap_one(nf.ifft2)
fftn = _wrap_one(nf.fftn)
ifftn = _wrap_one(nf.ifftn)
fftshift = _wrap_one(nf.fftshift)
ifftshift = _wrap_one(nf.ifftshift)
# new arrays
fftfreq = _wrap_one(nf.fftfreq)
rfftfreq = _wrap_one(nf.rfftfreq)
