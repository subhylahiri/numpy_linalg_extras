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
ifft = _wrap_oned(nf.ifft)
rfft = _wrap_oned(nf.rfft)
irfft = _wrap_oned(nf.irfft)
hfft = _wrap_oned(nf.hfft)
ihfft = _wrap_oned(nf.ihfft)
rfftn = _wrap_oned(nf.rfftn)
irfftn = _wrap_oned(nf.irfftn)
rfft2 = _wrap_oned(nf.rfft2)
irfft2 = _wrap_oned(nf.irfft2)
fft2 = _wrap_oned(nf.fft2)
ifft2 = _wrap_oned(nf.ifft2)
fftn = _wrap_oned(nf.fftn)
ifftn = _wrap_oned(nf.ifftn)
fftshift = _wrap_oned(nf.fftshift)
ifftshift = _wrap_oned(nf.ifftshift)
# new arrays
fftfreq = _wrap_one(nf.fftfreq)
rfftfreq = _wrap_one(nf.rfftfreq)
