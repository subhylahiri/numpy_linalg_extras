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
_wrap = _wr.Wrappers(_lnarray, "numpy_linalg.fft")
_wrapd = _wr.DeprecatedWrappers(_lnarray, "numpy_linalg.fft")

# existing arrays
fft = _wrapd.one(nf.fft)
ifft = _wrapd.one(nf.ifft)
rfft = _wrapd.one(nf.rfft)
irfft = _wrapd.one(nf.irfft)
hfft = _wrapd.one(nf.hfft)
ihfft = _wrapd.one(nf.ihfft)
rfftn = _wrapd.one(nf.rfftn)
irfftn = _wrapd.one(nf.irfftn)
rfft2 = _wrapd.one(nf.rfft2)
irfft2 = _wrapd.one(nf.irfft2)
fft2 = _wrapd.one(nf.fft2)
ifft2 = _wrapd.one(nf.ifft2)
fftn = _wrapd.one(nf.fftn)
ifftn = _wrapd.one(nf.ifftn)
fftshift = _wrapd.one(nf.fftshift)
ifftshift = _wrapd.one(nf.ifftshift)
# new arrays
fftfreq = _wrap.one(nf.fftfreq)
rfftfreq = _wrap.one(nf.rfftfreq)
