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
from warnings import warn

import numpy as _np

# from ._lnarray import lnarray as _ARRAY

# =============================================================================
# Wrapping functionals
# =============================================================================


def set_module(module: str = 'numpy_linalg'):
    """Create a decorator to set the __module__ attribute"""
    def decorator(thing):
        thing.__module__ = module
        return thing
    return decorator


def deprecated(old_wrapper):
    def new_wrapper(np_func):
        old_wrapped = old_wrapper(np_func)
        module = getattr(old_wrapped, '__module__', "numpy_linalg")
        msg = f"Use {np_func.__module__}.{np_func.__name__} "
        msg += f"instead of {module}.{np_func.__name__}"
        @_wraps(np_func)
        def new_wrapped(*args, **kwargs):
            warn(msg, DeprecationWarning, 2)
            return old_wrapped(*args, **kwargs)
        new_wrapped.__module__ = old_wrapped.__module__
        return new_wrapped
    return new_wrapper

def make_wrap_one(array_type, module_name: str=None):
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
            return _converter(np_func(*args, **kwargs), array_type)
        if module_name is not None:
            wrapped.__module__ = module_name
        return wrapped
    return wrap_one


def make_wrap_several(array_type, module_name: str=None):
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
            return (_converter(x, array_type) for x in output)
        if module_name is not None:
            wrapped.__module__ = module_name
        return wrapped
    return wrap_several


def make_wrap_some(array_type, module_name: str=None):
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
        if module_name is not None:
            wrapped.__module__ = module_name
        return wrapped
    return wrap_some


def make_wrap_sub(array_type, module_name: str=None):
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
        if module_name is not None:
            wrapped.__module__ = module_name
        return wrapped
    return wrap_sub


def make_wrap_subseveral(array_type, module_name: str=None):
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
        if module_name is not None:
            wrapped.__module__ = module_name
        return wrapped
    return wrap_subseveral


def make_wrap_some(array_type, module_name: str=None):
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
        if module_name is not None:
            wrapped.__module__ = module_name
        return wrapped
    return wrap_subsome


# =============================================================================
# Script to create modules
# =============================================================================


def wrap_module(file_name: str, funcs: _List[str], wrapper: str = 'wrap_one',
                parent: str = 'numpy', imps: str = '', internal: bool = True,
                module: str = "numpy_linalg"):
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
        f.write(f'"""Wrapped version of module {parent}\n')
        f.write('"""\n')
        f.write(imps + '\n')
        f.write(f'import {parent} as _pr\n')
        if internal:
            f.write('from . ')
        f.write('import _ln_wrap as _wr\n\n')
        if internal:
            f.write('from ._lnarray ')
        f.write('import lnarray as _lnarray\n\n')
        f.write('__all__ = [\n')
        for fn in funcs:
            f.write(f"    '{fn}',\n")
        f.write(']\n\n')
        f.write(f'_{wrapper} = _wr.make_{wrapper}(_lnarray)\n')
        for fn in funcs:
            f.write(f"{fn} = _{wrapper}(_pr.{fn}, {module})\n")


# =============================================================================
# Private stuff
# =============================================================================


def _converter(a, array_type):
    return a.view(array_type)


def _converter_check(a, array_type):
    if isinstance(a, _np.ndarray):
        return _converter(a, array_type)
    return a


def _converter_sub(a, array_type):
    if isinstance(a, array_type):
        return a
    return _converter(a, array_type)


def _converter_subcheck(a, array_type):
    if isinstance(a, _np.ndarray) and not isinstance(a, array_type):
        return _converter(a, array_type)
    return a
