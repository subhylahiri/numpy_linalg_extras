# -*- coding: utf-8 -*-
"""
Helper for writing __array_ufunc__
"""
import itertools
from typing import Tuple, Any, List, Sequence, Callable
import numpy as np
ArgTuple = Tuple[Any]
# ======================================================================
# %% Inputs
# ======================================================================


def conv_loop_in(converter: Callable,
                 obj_typ, tup: ArgTuple) -> (ArgTuple, List[bool]):
    """Process inputs in an __array_ufunc__ method

    Parameters
    ----------
    obj_typ
        The type of object that needs converting.
    tup: Tuple[Any]
        Tuple of inputs to ufunc (or ``out`` argument)

    Returns
    -------
    out: Tuple[Any]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv: List[bool]
        List of bools telling us if each input was converted.
    """
    out = []
    conv = []
    for obj in tup:
        if isinstance(obj, obj_typ):
            out.append(converter(obj))
            conv.append(True)
        else:
            out.append(obj)
            conv.append(False)
    return tuple(out), conv


def prepare_via_view() -> Callable:
    """Create function to convert object to an array using view method
    """
    def converter(thing):
        """convert to array using view method
        """
        return thing.view(np.ndarray)
    return converter


def prepare_via_attr(attr: str) -> Callable:
    """Create function to convert object to an array using an attribute

    Parameters
    ----------
    attr: str, None
        The name of the ``obj_typ`` attribute to use in place of class.
    """
    def converter(thing):
        """convert to array using an attribute
        """
        return getattr(thing, attr)
    return converter


def conv_loop_in_view(obj_typ, tup: ArgTuple) -> (ArgTuple, List[bool]):
    """Process inputs in an __array_ufunc__ method using view method

    Parameters
    ----------
    obj_typ
        The type of object that needs converting via ``view`` method.
    tup: Tuple[Any]
        Tuple of inputs to ufunc (or ``out`` argument)

    Returns
    -------
    out: Tuple[Any]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv: List[bool]
        List of bools telling us if each input was converted.
    """
    return conv_loop_in(prepare_via_view(), obj_typ, tup)


def conv_loop_in_attr(attr: str, obj_typ, tup: ArgTuple) -> (ArgTuple,
                                                             List[bool]):
    """Process inputs in an __array_ufunc__ method using an attribute

    Parameters
    ----------
    attr: str, None
        The name of the ``obj_typ`` attribute to use in place of class.
    obj_typ
        The type of object that needs converting with its ``attr`` attribute.
    tup: Tuple[Any]
        Tuple of inputs to ufunc (or ``out`` argument)

    Returns
    -------
    out: Tuple[Any]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv: List[bool]
        List of bools telling us if each input was converted.
    """
    return conv_loop_in(prepare_via_attr(attr), obj_typ, tup)


# ======================================================================
# %% Outputs
# ======================================================================


def conv_loop_out(converter: Callable, results: ArgTuple, outputs: ArgTuple,
                  conv: Sequence[bool] = ()) -> ArgTuple:
    """Process outputs in an __array_ufunc__ method

    Parameters
    ----------
    converter
        Function to perform conversions.
    results: Tuple[Any]
        Tuple of outputs from ufunc
    outputs: Tuple[Any]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv: Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results: Tuple[Any]
        New tuple of results from ufunc with conversions.
    """
    if not conv:
        conv = itertools.repeat(True)
    results_out = []
    for result, output, cout in zip(results, outputs, conv):
        if output is None:
            if cout:
                results_out.append(converter(result))
            else:
                results_out.append(result)
        else:
            results_out.append(output)
    return tuple(results_out)


def restore_via_attr(obj, attr: str) -> Callable:
    """Create function to convert arrays by setting obj.attr

    Parameters
    ----------
    obj
        The template object for conversions.
    attr: str, None
        The name of the ``type(obj)`` attribute returned in place of class.
        It will try to use ``obj.copy(attr=result)``. If that fails, it will
        use ``obj.copy()`` followed by ``setattr(newobj, attr, result)``.
    """
    def converter(thing):
        """convert arrays by setting obj.attr
        """
        try:
            return obj.copy(**{attr: thing})
        except TypeError:
            pass
        thing_out = obj.copy()
        setattr(thing_out, attr, thing)
        return thing_out
    return converter


def restore_via_init(obj) -> Callable:
    """Create function to convert arrays  using obj.__init__

    Parameters
    ----------
    obj
        The template object for conversions.
    """
    def converter(thing):
        """convert arrays using obj.__init__
        """
        return type(obj)(thing)
    return converter


def restore_via_view(obj) -> Callable:
    """Create function to convert arrays using array.view

    Parameters
    ----------
    obj
        The template object for conversions.
    """
    def converter(thing):
        """convert arrays using array.view
        """
        return thing.view(type(obj))
    return converter


def conv_loop_out_attr(obj, attr: str, results: ArgTuple, outputs: ArgTuple,
                       conv: Sequence[bool] = ()) -> ArgTuple:
    """Process outputs in an __array_ufunc__ method

    Makes a copy of ``obj`` with ``obj.attr = result``.

    Parameters
    ----------
    obj
        The template object for conversions.
    attr: str, None
        The name of the ``type(obj)`` attribute returned in place of class.
    results: Tuple[Any]
        Tuple of outputs from ufunc
    outputs: Tuple[Any]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv: Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results: Tuple[Any]
        New tuple of results from ufunc with conversions.

    Notes
    -----
    It will try to use ``obj.copy(attr=result)``. If that fails, it will use
    ``obj.copy()`` followed by ``setattr(newobj, attr, result)``.
    """
    return conv_loop_out(restore_via_attr(obj, attr), results, outputs, conv)


def conv_loop_out_init(obj, results: ArgTuple, outputs: ArgTuple,
                       conv: Sequence[bool] = ()) -> ArgTuple:
    """Process outputs in an __array_ufunc__ method

    Creates an instance of ``type(obj)`` with ``result`` as its argument.

    Parameters
    ----------
    obj
        The template object for conversions.
    results: Tuple[Any]
        Tuple of outputs from ufunc
    outputs: Tuple[Any]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv: Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results: Tuple[Any]
        New tuple of results from ufunc with conversions.
    """
    return conv_loop_out(restore_via_init(obj), results, outputs, conv)


def conv_loop_out_view(obj, results: ArgTuple, outputs: ArgTuple,
                       conv: Sequence[bool] = ()) -> ArgTuple:
    """Process outputs in an __array_ufunc__ method

    Calls ``result.view`` with ``type(obj)`` with as its argument.

    Parameters
    ----------
    obj
        The template object for conversions.
    results: Tuple[Any]
        Tuple of outputs from ufunc
    outputs: Tuple[Any]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv: Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results: Tuple[Any]
        New tuple of results from ufunc with conversions.
    """
    return conv_loop_out(restore_via_view(obj), results, outputs, conv)
