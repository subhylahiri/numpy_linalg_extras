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

Classes
-------
Wrapper
    Class with methods to wrap `numpy` functions to return `lnarray`s instead
    of `ndarray`s:

    one
        Create version of `numpy` function with single `lnarray` output.
    several
        Create version of `numpy` function with multiple `lnarray` outputs.
    some
        Create version of `numpy` function with some `lnarray` outputs, some
        non-array outputs.
    sub
        Create version of `numpy` function with single `lnarray` output,
        passing through subclasses.
    subseveral
        Create version of `numpy` function with multiple `lnarray` outputs,
        passing through subclasses.
    subsome
        Create version of `numpy` function with some `lnarray` outputs, some
        non-array outputs, passing through subclasses.
DeprecatedWrappers
    A version of `Wrappers` that raises a deprecation warning when the wrapped
    functions are used.
WrappedClass
    When this class is subclassed, the resulting class has the same methods as
    the object passed in the constructor, except they return `lnarray`s instead
    of `ndarray`s.
WrappedSubscriptable
    When this class is subclassed, the resulting class's instances can be
    subscripted in the same manner as the object passed in the constructor,
    except it will return `lnarray`s instead of `ndarray`s.

Example
-------
```
import numpy as np
from . import wrappers as wr
from ._lnarray import lnarray

wrap = wr.Wrappers(lnarray, "numpy_linalg")

@wr.set_module("numpy_linalg")
class LnNdGrid(wr.WrappedSubscriptable, wrappers=wrap, method="several"):
    \"\"\"
    See Also
    --------
    numpy.mgrid
    numpy.ogrid
    \"\"\"
    obj: np.lib.index_tricks.nd_grid

mgrid = LnNdGrid(np.mgrid)
ogrid = LnNdGrid(np.ogrid)
empty = wrap.one(np.empty)
ones = wrap.one(np.ones)
zeros = wrap.one(np.zeros)
full = wrap.one(np.full)
```
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
# Wrapping functions
# =============================================================================


def set_module(module: str = 'numpy_linalg'):
    """Create a decorator to set the __module__ attribute

    Parameters
    ----------
    module : str, None, optional
        The module from which the function will be imported, by default `None`.
    """
    def decorator(thing):
        thing.__module__ = module
        return thing
    return decorator


class Wrappers:
    """Wrappers for array functions

    This class's methods can be used to wrap `numpy` functions to return
    `lnarray`s instead of `ndarray`s. Any other class can be substituted for
    `lnarray` in the previouas sentence.

    Parameters
    ----------
    array_type : Type[Array], optional
        The array class to which outputs are converted, by default `ndarray`.
    module : str, None, optional
        The module from which the functions will be imported, by default `None`
    """
    _arr_cls: _ty.Type[_Arr]
    _mod_name: _Optional[str]

    def __init__(self, array_type: _ty.Type[_Arr] = _np.ndarray,
                 module: _Optional[str] = None):
        self._arr_cls = array_type
        self._mod_name = module

    def func_hook(self, np_func: _NpFn, wrapped: _MyFn[_Arr]) -> _MyFn[_Arr]:
        """This method is called on the wrapped function before returning

        Parameters
        ----------
        np_func : Callable[...->ndarray]
            The `numpy` function being wrapped.
        wrapped : Callable[...->Array]
            The wrapped function.

        Returns
        -------
        wrapped : Callable[...->Array]
            The wrapped function.
        """
        doc = getattr(np_func, '__doc__', None)
        if doc is not None:
            wrapped.__doc__ = doc.replace('ndarray', self._arr_cls.__name__)
        if self._mod_name is not None:
            wrapped.__module__ = self._mod_name
        return wrapped

    def one(self, np_func: _NpFn) -> _MyFn[_Arr]:
        """Create version of numpy function with single lnarray output.

        Does not pass through subclasses of `lnarray`

        Parameters
        ----------
        np_func : Callable[...->ndarray]
            A function that returns a single `ndarray`.

        Returns
        -------
        my_func : Callable[...->Array]
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
        np_func : Callable[...->ndarray]
            A function that returns a tuple of `ndarray`s.

        Returns
        -------
        my_func : Callable[...->Array]
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
        np_func : Callable[...->ndarray]
            A function that returns a mixed tuple of `ndarray`s and others.

        Returns
        -------
        my_func : Callable[...->Array]
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
        np_func : Callable[...->ndarray]
            A function that returns a single `ndarray`.

        Returns
        -------
        my_func : Callable[...->Array]
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
        np_func : Callable[...->ndarray]
            A function that returns a tuple of `ndarray`s.

        Returns
        -------
        my_func : Callable[...->Array]
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
        np_func : Callable[...->ndarray]
            A function that returns a mixed tuple of `ndarray`s and others.

        Returns
        -------
        my_func : Callable[...->Array]
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

    See Also
    --------
    Wrappers
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


def wrap_module(file_name: str, funcs: _List[str], wrapper: str = 'one',
                parent: str = 'numpy', imps: str = '', internal: bool = True,
                module: str = "numpy_linalg"):
    """Create a wrapped version of a numpy module

    This function can be used to create (the first draft of) a module file that
    has versions of functions that return `lnarray`s instead of `ndarray`s.

    Parameters
    ----------
    file_name : str
        Name of file for new module (without extension).
    funcs : List[str]
        List of names of functions to wrap (from parent.__all__?).
    wrapper : str = 'one'
        Name of `Wrappers` method used to wrap.
    parent : str = 'numpy'
        Name of module containing unwrapped functions.
    imps : str = ''
        To be written after docstring, before imports & functions.
    internal : bool = True
        Is the wrapped module in the same package as this module?
    module : str = 'numpy_linalg'
        Name of module from which these functions will be imported.
    """
    with open(file_name + '.py', 'w') as f:
        f.write(f'"""Wrapped version of module {parent}\n')
        f.write('"""\n')
        f.write(imps + '\n')
        f.write(f'import {parent} as _pr\n')
        package = '.' if internal else 'numpy_linalg'
        f.write(f'from {package} import wrappers as _wr\n')
        package = '._lnarray' if internal else 'numpy_linalg'
        f.write(f'from {package} import lnarray as _lnarray\n\n')
        f.write('__all__ = [\n')
        for fun in funcs:
            f.write(f"    '{fun}',\n")
        f.write(']\n\n')
        f.write(f'_wrap = _wr.Wrappers(_lnarray, "{module}")\n\n')
        for fun in funcs:
            f.write(f"{fun} = _wrap.{wrapper}(_pr.{fun})\n")


# =============================================================================
# Wrap a class
# =============================================================================

class WrappedClass:
    """Class to wrap the methods of another class

    When this class is subclassed, the resulting class has the same methods as
    the object passed in the constructor, except they return `lnarray`s instead
    of `ndarray`s.

    The array class you wish to use must be passed as the `array_type`
    parameter when subclassing. You can also pass the name of module from
    which these functions will be imported as the `module` parameter and
    the name of the `Wrappers` method to use as the `method` parameter.
    You cap ass a `Wrappers` instance as the `wrappers` parameter instead of
    the `array_type` and `module` parameters.

    Parameters
    ----------
    object : Any
        An instance of the class being wrapped.

    Notes
    -----
    Does not work for `__getitem__`. For that, use `WrappedSubscriptable`.

    The methods are created when called, so your IDE's autocomplete, etc, will
    most likely not see them.
    """
    wrappers: _ty.ClassVar[Wrappers]
    wrap: _ty.ClassVar[_ty.Callable[[_NpFn], _MyFn[_Arr]]]

    def __init_subclass__(cls,
                          wrappers: _ty.Optional[Wrappers] = None,
                          array_type: _ty.Type[_Arr] = _np.ndarray,
                          module: _ty.Optional[str] = None,
                          method: str = 'subsome',
                          **kwargs):
        super().__init_subclass__(**kwargs)
        cls.wrappers = wrappers
        if wrappers is None:
            cls.wrappers = Wrappers(array_type, module)
        cls.wrap = getattr(cls.wrappers, method)

    def __init__(self, obj):
        self.obj = obj
        self.wrappers.func_hook(obj, self)

    def __getattr__(self, attr: str):
        return self.wrap(getattr(self.obj, attr))

    def __dir__(self):
        return dir(self.obj)


class WrappedSubscriptable(WrappedClass):
    """Class to wrap the __getitem__ method of another class

    When this class is subclassed, the resulting class can be subscripted in
    the same manner as the object passed in the constructor, except it will
    return `lnarray`s instead of `ndarray`s.

    The array class you wish to use must be passed as the `array_type`
    parameter when subclassing. You can also pass the name of module from
    which these functions will be imported as the `module` parameter and
    the name of the `Wrappers` method to use as the `method` parameter.
    You cap ass a `Wrappers` instance as the `wrappers` parameter instead of
    the `array_type` and `module` parameters.

    Parameters
    ----------
    object : Any
        An instance of the subscriptable class being wrapped.
    """
    _get: _MyFn[_Arr]

    def __init__(self, obj):
        super().__init__(obj)
        self._get = self.wrap(self.obj.__getitem__)

    def __getitem__(self, key):
        return self._get(key)
