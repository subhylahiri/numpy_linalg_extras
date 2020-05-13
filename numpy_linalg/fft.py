# -*- coding: utf-8 -*-
"""Wrapped numpy.fft routines
"""
import numpy.fft as nf
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
    'fftshift',
    'ifftshift',
    'fftfreq',
    'rfftfreq',
]

fft = _wr.wrap_one(nf.fft)
ifft = _wr.wrap_one(nf.ifft)
rfft = _wr.wrap_one(nf.rfft)
irfft = _wr.wrap_one(nf.irfft)
hfft = _wr.wrap_one(nf.hfft)
ihfft = _wr.wrap_one(nf.ihfft)
rfftn = _wr.wrap_one(nf.rfftn)
irfftn = _wr.wrap_one(nf.irfftn)
rfft2 = _wr.wrap_one(nf.rfft2)
irfft2 = _wr.wrap_one(nf.irfft2)
fft2 = _wr.wrap_one(nf.fft2)
ifft2 = _wr.wrap_one(nf.ifft2)
fftn = _wr.wrap_one(nf.fftn)
ifftn = _wr.wrap_one(nf.ifftn)

fftshift = _wr.wrap_one(nf.fftshift)
ifftshift = _wr.wrap_one(nf.ifftshift)
# array creation
fftfreq = _wr.wrap_one(nf.fftfreq)
rfftfreq = _wr.wrap_one(nf.rfftfreq)
