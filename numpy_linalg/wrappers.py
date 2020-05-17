# -*- coding: utf-8 -*-
# =============================================================================
# Created on Tue Dec  5 02:46:29 2017
# @author: subhy
# module: wrappers
# =============================================================================
"""
Functions that change the return type of functions from `ndarray` to `lnarray`.
They can be used as function decorators.
To use some other array class, change the first import statement and the
docstrings.argument of `make_...`.

Functions
---------
The following are created by `make_<function name>`, which takes arguments
`array_type` (the array class to use) and `module_name` (the module where these
functions will be exposed).

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
import typing as _ty
from typing import List as _List, Optional as _Optional
from warnings import warn as _warn

import numpy as _np

_Arr = _ty.TypeVar("MyArray")
_NpFn = _ty.Callable[..., _np.ndarray]
_MyFn = _ty.Callable[..., _Arr]
# =============================================================================
# Wrapping functionals
# =============================================================================


def set_module(module: str = 'numpy_linalg'):
    """Create a decorator to set the __module__ attribute"""
    def decorator(thing):
        thing.__module__ = module
        return thing
    return decorator


class Wrappers:
    """Wrappers for array functions
    """
    _arr_cls: _ty.Type[_Arr]
    _mod_name: _Optional[str]

    def __init__(self, array_type: _ty.Type[_Arr],
                 module_name: _Optional[str] = None):
        self._arr_cls = array_type
        self._mod_name = module_name

    def func_hook(self, np_func: _NpFn, wrapped: _MyFn[_Arr]) -> _MyFn[_Arr]:
        if self._mod_name is not None:
            wrapped.__module__ = self._mod_name
        return wrapped

    def one(self, np_func: _NpFn) -> _MyFn[_Arr]:
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
            return self._converter(np_func(*args, **kwargs))
        return self.func_hook(np_func, wrapped)

    def several(self, np_func: _NpFn) -> _MyFn[_Arr]:
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
            return (self._converter(x) for x in output)
        return self.func_hook(np_func, wrapped)

    def some(self, np_func: _NpFn) -> _MyFn[_Arr]:
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
            return (self._converter_check(x) for x in output)
        return self.func_hook(np_func, wrapped)

    def sub(self, np_func: _NpFn) -> _MyFn[_Arr]:
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
            return self._converter_sub(np_func(*args, **kwargs))
        return self.func_hook(np_func, wrapped)

    def subseveral(self, np_func: _NpFn) -> _MyFn[_Arr]:
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
            return (self._converter_sub(x) for x in output)
        return self.func_hook(np_func, wrapped)

    def subsome(self, np_func: _NpFn) -> _MyFn[_Arr]:
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
            return (self._converter_subcheck(x) for x in output)
        return self.func_hook(np_func, wrapped)

    def _converter(self, arr: _np.ndarray) -> _Arr:
        return arr.view(self._arr_cls)

    def _converter_check(self, arr: _np.ndarray) -> _Arr:
        if isinstance(arr, _np.ndarray):
            return self._converter(arr)
        return arr

    def _converter_sub(self, arr: _np.ndarray) -> _Arr:
        if isinstance(arr, self._arr_cls):
            return arr
        return self._converter(arr)

    def _converter_subcheck(self, arr: _np.ndarray) -> _Arr:
        if isinstance(arr, _np.ndarray) and not isinstance(arr, self._arr_cls):
            return self._converter(arr)
        return arr


class DeprecatedWrappers(Wrappers):
    """Wrappers for deprecated functions
    """
    def func_hook(self, np_func: _NpFn, wrapped: _MyFn[_Arr]) -> _MyFn[_Arr]:
        msg = f"Use {np_func.__module__}.{np_func.__name__} "
        msg += f"instead of {self._mod_name}.{np_func.__name__}"
        @_wraps(np_func)
        def new_wrapped(*args, **kwargs):
            _warn(msg, DeprecationWarning, 2)
            return wrapped(*args, **kwargs)
        return super().func_hook(np_func, new_wrapped)

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
        f.write('import wrappers as _wr\n\n')
        if internal:
            f.write('from ._lnarray ')
        f.write('import lnarray as _lnarray\n\n')
        f.write('__all__ = [\n')
        for fn in funcs:
            f.write(f"    '{fn}',\n")
        f.write(']\n\n')
        f.write(f'_wrap = _wr.Wrappers(_lnarray, "{module}")\n')
        for fn in funcs:
            f.write(f"{fn} = _wrap.{wrapper}(_pr.{fn})\n")


# =============================================================================
# Wrap a class
# =============================================================================

class WrappedClass:
    """Clas to wrap the metthods of another class
    """
    wrap: _ty.ClassVar[Wrappers]

    def __init_subclass__(cls, array_type, module_name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.wrap = Wrappers(array_type, module_name)

    def __getattr__(self, attr):
        return self.wrap.one(getattr(self.obj, attr))
