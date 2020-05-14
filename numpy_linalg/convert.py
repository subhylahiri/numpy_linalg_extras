# -*- coding: utf-8 -*-
"""Helpers for writing __array_ufunc__ methods.

Routine Listings
----------------
conv_in_attr
    Process inputs in an `__array_ufunc__` method using an attribute.
conv_in_view
    Process inputs in an `__array_ufunc__` method using a view method.
conv_out_attr
    Process outputs in an `__array_ufunc__` method using an attribute.
conv_out_init
    Process outputs in an `__array_ufunc__` method using a constructor.
conv_out_view
    Process outputs in an `__array_ufunc__` method using a view method.

Converters that take a callback function as a parameter:

conv_input
    Process inputs to a `ufunc`.
conv_in_out
    Process `out` argument of a `ufunc`.
conv_out
    Process outputs from a `ufunc`.

Creators of allback functions to convert a single iput/output:

prepare_via_attr
    Create a callback to process inputs using an attribute.
prepare_via_view
    Create a callback to process inputs using a view method.
restore_via_attr
    Create a callback to process an output using an attribute.
restore_via_init
    Create a callback to process an output using a constructor.
restore_via_view
    Create a callback to process an output using a view method.

Example
-------
```
import numpy_linalg.convert as cv

    class MyClass():

        def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
            '''Handling ufunce with MyClass
            '''
            args, _ = cv.conv_in_attr('attr', MyClass, inputs)
            conv = [True] + [False] * (ufunc.nout-1)
            outputs, conv = cv.conv_in_attr('attr', MyClass, kwargs, conv)
            results = self.attr.__array_ufunc__(ufunc, method, *args, **kwargs)
            return cv.conv_out_attr(self, 'attr', results, outputs, conv)
```
"""
import itertools as _itertools
import typing as _ty

import numpy as _np

MyArray = _ty.TypeVar('MyArray')
BaseArray = _ty.TypeVar('BaseArray')
OutTuple = _ty.Tuple[_ty.Optional[BaseArray], ...]
ArrayTuple = _ty.Tuple[BaseArray, ...]
ArgTuple = _ty.Tuple[_ty.Union[_np.ndarray, MyArray], ...]
ArgDict = _ty.Dict[str, _ty.Any]
ArgsIn = _ty.Union[ArgTuple[MyArray], ArgDict]
BoolList = _ty.List[bool]
Preparer = _ty.Callable[[MyArray], BaseArray]
Restorer = _ty.Callable[[BaseArray], MyArray]
# ==============================================================================
# Inputs
# ==============================================================================


def conv_input(converter: Preparer[MyArray, BaseArray],
               obj_typ: _ty.Type[MyArray],
               args: ArgTuple[MyArray]) -> (ArrayTuple[BaseArray], BoolList):
    """Process inputs in an __array_ufunc__ method.

    Parameters
    ----------
    converter : Callable[MyArray -> ndarray]
        Function to convert specified type to `ndarray`s.
    obj_typ : Type[MyArray]
        The type of object that needs converting.
    args : Tuple[ndarray|MyArray, ...]
        Tuple of inputs to ufunc (or ``out`` argument)

    Returns
    -------
    out : Tuple[ndarray|MyArray, ...]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv : List[bool]
        List of bools telling us if each input was converted.
    """
    out = []
    conv = []
    for obj in args:
        if isinstance(obj, obj_typ):
            out.append(converter(obj))
            conv.append(True)
        else:
            out.append(obj)
            conv.append(False)
    return out, conv


def conv_in_out(converter: Preparer[MyArray, BaseArray],
                obj_typ: _ty.Type[MyArray],
                kwds: ArgDict,
                conv_out: BoolList) -> (OutTuple[BaseArray], BoolList):
    """Process the out keyword in an __array_ufunc__ method.

    Parameters
    ----------
    converter : Callable[MyArray -> ndarray]
        Function to convert specified type to `ndarray`s.
    obj_typ : Type[MyArray]
        The type of object that needs converting.
    kwds : Dict[str, Any]
        Dict of key word inputs to ufunc
    conv_out : List[bool]
        List of bools for converting outputs. Should have the correct length.

    Returns
    -------
    out : Tuple[ndarray|None, ...]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv : List[bool]
        List of bools telling us if each input was converted.
    """
    outputs = kwds.pop('out', None)
    if outputs:
        out_args, conv_out = conv_input(converter, obj_typ, outputs)
        kwds['out'] = tuple(out_args)
    else:
        outputs = (None,) * len(conv_out)
    return outputs, conv_out


def _conv_in(converter: Preparer[MyArray, BaseArray],
             obj_typ: _ty.Type[MyArray],
             tup: ArgsIn,
             *conv_out) -> (OutTuple[BaseArray], BoolList):
    """Call one of conv_input or conv_in_out"""
    if isinstance(tup, tuple):
        return conv_input(converter, obj_typ, tup)
    return conv_in_out(converter, obj_typ, tup, *conv_out)


def prepare_via_view(std_array: _ty.Type[BaseArray] = _np.ndarray
                     ) -> Preparer[MyArray, BaseArray]:
    """Create function to convert object to an array using view method.

    Returns
    -------
    converter : Callable[MyArray -> ndarray]
        Function to convert specified type to `ndarray`s.
    """
    def converter(thing: MyArray) -> BaseArray:
        """convert to array using view method
        """
        return thing.view(std_array)
    return converter


def prepare_via_attr(attr: str) -> Preparer[MyArray, BaseArray]:
    """Create function to convert object to an array using an attribute.

    Parameters
    ----------
    attr: str
        The name of the ``obj_typ`` attribute to use in place of class.

    Returns
    -------
    converter : Callable[MyArray -> ndarray]
        Function to convert specified type to `ndarray`s.
    """
    def converter(thing: MyArray) -> BaseArray:
        """convert to array using an attribute
        """
        return getattr(thing, attr)
    return converter


def conv_in_view(obj_typ: _ty.Type[MyArray],
                 tup: ArgsIn[MyArray],
                 *conv_out) -> (OutTuple[_np.ndarray], BoolList):
    """Process inputs in an __array_ufunc__ method using view method.

    If the second argument is a tuple, we assume we are processing the ufunc's
    inputs. If it is a dict, we are processing the `out` keyword.

    Parameters
    ----------
    obj_typ : Type[MyArray]
        The type of object that needs converting via ``view`` method.
    tup : Tuple[ndarray|MyArray, ...], Dict[str, Any]
        Tuple of inputs to ufunc (or ``out`` argument)
        Dict of key word inputs to ufunc
    conv_out : Sequence[bool], optional
        List of bools for converting outputs. Should have the correct length.

    Returns
    -------
    out : Tuple[ndarray|None, ...]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv : List[bool]
        List of bools telling us if each input was converted.
    """
    return _conv_in(prepare_via_view(), obj_typ, tup, *conv_out)


def conv_in_attr(attr: str,
                 obj_typ: _ty.Type[MyArray],
                 tup: ArgsIn[MyArray],
                 *conv_out: bool) -> (OutTuple[BaseArray], BoolList):
    """Process inputs in an __array_ufunc__ method using an attribute.

    If the third argument is a tuple, we assume we are processing the ufunc's
    inputs. If it is a dict, we are processing the `out` keyword.

    Parameters
    ----------
    attr : str, None
        The name of the ``obj_typ`` attribute to use in place of class.
    obj_typ : Type[MyArray]
        The type of object that needs converting with its ``attr`` attribute.
    tup : Tuple[ndarray|MyArray, ...], Dict[str, Any]
        Tuple of inputs to ufunc (or ``out`` argument)
        Dict of key word inputs to ufunc
    conv_out : Sequence[bool], optional
        List of bools for converting outputs. Should have the correct length.

    Returns
    -------
    out : Tuple[ndarray|None]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv : List[bool]
        List of bools telling us if each input was converted.
    """
    return _conv_in(prepare_via_attr(attr), obj_typ, tup, *conv_out)


# ======================================================================
# Outputs
# ======================================================================


def conv_out(converter: Restorer[BaseArray, MyArray],
             results: ArrayTuple[BaseArray],
             outputs: OutTuple[BaseArray],
             conv: _ty.Sequence[bool] = ()) -> ArgTuple[MyArray]:
    """Process outputs in an __array_ufunc__ method.

    Parameters
    ----------
    converter : Callable[ndarray -> MyArray]
        Function to perform conversions back from `ndarray` to specified type.
    results : Tuple[ndarray]
        Tuple of outputs from ufunc
    outputs : Tuple[ndarray|None]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv : Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Replaced with itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results : Tuple[ndarray|MyArray, ...]
        New tuple of results from ufunc with conversions.
    """
    if results is NotImplemented:
        return NotImplemented
    if not isinstance(results, tuple):
        results = (results,)
    if not conv:
        conv = _itertools.repeat(True)
    results_out = []
    for result, output, cout in zip(results, outputs, conv):
        if output is not None:
            results_out.append(output)
        elif cout:
            results_out.append(converter(result))
        else:
            results_out.append(result)
    if len(results_out) == 1:
        return results_out[0]
    return tuple(results_out)


def restore_via_attr(obj: MyArray, attr: str) -> Restorer[BaseArray, MyArray]:
    """Create function to convert arrays by setting obj.attr.

    Parameters
    ----------
    obj : MyArray
        The template object for conversions.
    attr: str, None
        The name of the ``type(obj)`` attribute returned in place of class.
        It will try to use ``obj.copy(attr=result)``. If that fails, it will
        use ``obj.copy()`` followed by ``setattr(newobj, attr, result)``.

    Returns
    -------
    converter : Callable[ndarray -> MyArray]
        Function to perform conversions back from `ndarray` to specified type.
    """
    def converter(thing: BaseArray) -> MyArray:
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


def restore_via_init(objt: _ty.Type[MyArray]) -> Restorer[BaseArray, MyArray]:
    """Create function to convert arrays  using objt.__init__.

    Parameters
    ----------
    objt : Type[MyArray]
        The type of object to converst to.

    Returns
    -------
    converter : Callable[ndarray -> MyArray]
        Function to perform conversions back from `ndarray` to specified type.
    """
    def converter(thing: BaseArray) -> MyArray:
        """convert arrays using objt.__init__
        """
        return objt(thing)
    return converter


def restore_via_view(objt: _ty.Type[MyArray]) -> Restorer[BaseArray, MyArray]:
    """Create function to convert arrays using array.view.

    Parameters
    ----------
    objt : Type[MyArray]
        The type of object to converst to.

    Returns
    -------
    converter : Callable[ndarray -> MyArray]
        Function to perform conversions back from `ndarray` to specified type.
    """
    def converter(thing: BaseArray) -> MyArray:
        """convert arrays using array.view
        """
        return thing.view(objt)
    return converter


def conv_out_attr(obj: MyArray,
                  attr: str,
                  results: ArrayTuple[BaseArray],
                  outputs: OutTuple[BaseArray],
                  conv: _ty.Sequence[bool] = ()) -> ArgTuple[MyArray]:
    """Process outputs in an __array_ufunc__ method using an attribute.

    Makes a copy of ``obj`` with ``obj.attr = result``.

    Parameters
    ----------
    obj : MyArray
        The template object for conversions.
    attr : str
        The name of the ``type(obj)`` attribute returned in place of class.
    results : Tuple[ndarray]
        Tuple of outputs from ufunc
    outputs : Tuple[ndarray|None]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv : Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results : Tuple[ndarray|MyArray]
        New tuple of results from ufunc with conversions.

    Notes
    -----
    It will try to use ``obj.copy(attr=result)``. If that fails, it will use
    ``obj.copy()`` followed by ``setattr(newobj, attr, result)``.
    """
    return conv_out(restore_via_attr(obj, attr), results, outputs, conv)


def conv_out_init(obj: MyArray,
                  results: ArgTuple[BaseArray],
                  outputs: OutTuple[BaseArray],
                  conv: _ty.Sequence[bool] = ()) -> ArgTuple[MyArray]:
    """Process outputs in an __array_ufunc__ method using a constructor.

    Creates an instance of ``type(obj)`` with ``result`` as its argument.

    Parameters
    ----------
    obj : MyArray
        The template object for conversions.
    results : Tuple[ndarray]
        Tuple of outputs from ufunc
    outputs : Tuple[ndarray|None]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv : Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results : Tuple[ndarray|MyArray]
        New tuple of results from ufunc with conversions.
    """
    return conv_out(restore_via_init(type(obj)), results, outputs, conv)


def conv_out_view(obj: MyArray,
                  results: ArrayTuple[BaseArray],
                  outputs: OutTuple[BaseArray],
                  conv: _ty.Sequence[bool] = ()) -> ArgTuple[MyArray]:
    """Process outputs in an __array_ufunc__ method using a view method.

    Calls ``result.view`` with ``type(obj)`` with as its argument.

    Parameters
    ----------
    obj : MyArray
        The template object for conversions.
    results : Tuple[ndarray]
        Tuple of outputs from ufunc
    outputs : Tuple[ndarray|None]
        ``out`` argument of ufunc, or tuple of ``None``.
    conv : Sequence[bool], default: ()
        Sequence of bools telling us if each output should be converted.
        Converted to itertools.repeat(True) if bool(conv) == False (default)

    Returns
    -------
    results : Tuple[ndarray|MyArray]
        New tuple of results from ufunc with conversions.
    """
    return conv_out(restore_via_view(type(obj)), results, outputs, conv)
