# -*- coding: utf-8 -*-
# =============================================================================
# Created on Tue Dec  5 02:46:29 2017
# @author: subhy
# module: _ln_wrap
# =============================================================================
"""
Functions that change the return type of functions from `ndarray` to `lnarray`.
They can be used as function decorators.
To use some other array class, change the first import statement and the
docstrings.

Functions
---------
wrap_one
    Create version of `numpy` function with single `lnarray` output.
wrap_several
    Create version of `numpy` function with multiple `lnarray` outputs.
wrap_some
    Create version of `numpy` function with some `lnarray` outputs, some
    non-array outputs.
wrap_sub
    Create version of `numpy` function with single `lnarray` output, passing
    through subclasses.
wrap_subseveral
    Create version of `numpy` function with multiple `lnarray` outputs, passing
    through subclasses.
wrap_subsome
    Create version of `numpy` function with some `lnarray` outputs, some
    non-array outputs, passing through subclasses.
"""
from functools import wraps as _wraps
from typing import List as _List

import numpy as _np

from ._lnarray import lnarray as _array

# =============================================================================
# Wrapping functionals
# =============================================================================


def wrap_one(np_func):
    """Create version of numpy function with single lnarray output.

    Does not pass through subclasses of `lnarray`

    Parameters
    ----------
    np_func : function
        A function that returns a single `ndarray`.

    Returns
    -------
    my_func : function
        A function that returns a single `lnarray`.
    """
    @_wraps(np_func)
    def wrapped(*args, **kwargs):
        return _converter(np_func(*args, **kwargs))
    return wrapped


def wrap_several(np_func):
    """Create version of numpy function with multiple lnarray outputs.

    Does not pass through subclasses of `lnarray`

    Parameters
    ----------
    np_func : function
        A function that returns a tuple of `ndarray`s.

    Returns
    -------
    my_func : function
        A function that returns a tuple of `lnarray`s.
    """
    @_wraps(np_func)
    def wrapped(*args, **kwargs):
        output = np_func(*args, **kwargs)
        return (_converter(x) for x in output)
    return wrapped


def wrap_some(np_func):
    """Create version of numpy function with some lnarray outputs, some
    non-array outputs.

    Does not pass through subclasses of `lnarray`

    Parameters
    ----------
    np_func : function
        A function that returns a mixed tuple of `ndarray`s and others.

    Returns
    -------
    my_func : function
        A function that returns a mixed tuple of `lnarray`s and others.
    """
    @_wraps(np_func)
    def wrapped(*args, **kwargs):
        output = np_func(*args, **kwargs)
        return (_converter_check(x) for x in output)
    return wrapped


def wrap_sub(np_func):
    """Create version of numpy function with single lnarray output.

    Does pass through subclasses of `lnarray`

    Parameters
    ----------
    np_func : function
        A function that returns a single `ndarray`.

    Returns
    -------
    my_func : function
        A function that returns a single `lnarray`.
    """
    @_wraps(np_func)
    def wrapped(*args, **kwargs):
        return _converter_sub(np_func(*args, **kwargs))
    return wrapped


def wrap_subseveral(np_func):
    """Create version of numpy function with multiple lnarray outputs.

    Does pass through subclasses of `lnarray`

    Parameters
    ----------
    np_func : function
        A function that returns a tuple of `ndarray`s.

    Returns
    -------
    my_func : function
        A function that returns a tuple of `lnarray`s.
    """
    @_wraps(np_func)
    def wrapped(*args, **kwargs):
        output = np_func(*args, **kwargs)
        return (_converter_sub(x) for x in output)
    return wrapped


def wrap_subsome(np_func):
    """Create version of numpy function with some lnarray outputs, some
    non-array outputs.

    Does pass through subclasses of `lnarray`

    Parameters
    ----------
    np_func : function
        A function that returns a mixed tuple of `ndarray`s and others.

    Returns
    -------
    my_func : function
        A function that returns a mixed tuple of `lnarray`s and others.
    """
    @_wraps(np_func)
    def wrapped(*args, **kwargs):
        output = np_func(*args, **kwargs)
        return (_converter_subcheck(x) for x in output)
    return wrapped


# =============================================================================
# Script to create modules
# =============================================================================


def wrap_module(file_name: str, funcs: _List[str], wrapper: str = 'wrap_one',
                parent: str = 'numpy', imps: str = '', internal: bool = False):
    """Create a wrapped version of a numpy module

    Parameters
    ----------
    file_name: str
        Name of file for new module (without extension).
    funcs: List[str]
        List of names of functions to wrap (from parent.__all__?).
    wrapper: str = 'wrap_one'
        Name of function from this mudule used to wrap.
    parent: str = 'numpy'
        Name of module containing unwrapped functions.
    imps: str = ''
        To be written after docstring, before imports & functions.
    """
    with open(file_name + '.py', 'w') as f:
        f.write('"""Wrapped version of module ' + parent + '\n')
        f.write('"""\n')
        f.write(imps + '\n')
        f.write('import ' + parent + ' as _pr\n')
        if internal:
            f.write('from . ')
        f.write('import _ln_wrap as _wr\n\n')
        f.write('__all__ = [\n')
        for fn in funcs:
            f.write("    '" + fn + "',\n")
        f.write('          ]\n\n')
        for fn in funcs:
            f.write(fn + " = _wr." + wrapper + "(_pr." + fn + ")\n")
        f.write('\n')


# =============================================================================
# Private stuff
# =============================================================================


def _converter(a: _np.ndarray) -> _array:
    return a.view(_array)


def _converter_check(a):
    if isinstance(a, _np.ndarray):
        return _converter(a)
    else:
        return a


def _converter_sub(a):
    if isinstance(a, _array):
        return a
    else:
        return _converter(a)


def _converter_subcheck(a):
    if isinstance(a, _array):
        return a
    elif isinstance(a, _np.ndarray):
        return _converter(a)
    else:
        return a
