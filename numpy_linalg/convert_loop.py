# -*- coding: utf-8 -*-
"""
Helpers for writing __array_ufunc__

Example
-------
    ```
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        '''Handling ufunce with MyClass
        '''
        args, _ = cvl.conv_loop_in_attr('attr', MyClass, inputs)
        conv = [True] + [False] * (ufunc.nout-1)
        outputs, conv = cvl.conv_loop_in_attr('attr', MyClass, kwargs, conv)
        results = self.attr.__array_ufunc__(ufunc, method, *args, **kwargs)
        return cvl.conv_loop_out_attr(self, 'attr', results, outputs, conv)
    ```
"""
import itertools as _itertools
import typing as _ty
import numpy as _np
ArgTuple = _ty.Tuple[_ty.Any]
ArgDict = _ty.Dict[str, _ty.Any]
Args = _ty.Union[ArgTuple, ArgDict]
BoolList = _ty.List[bool]
# ======================================================================
# %% Inputs
# ======================================================================


def conv_loop_input(converter: _ty.Callable,
                    obj_typ,
                    args: ArgTuple) -> (ArgTuple, BoolList):
    """Process inputs in an __array_ufunc__ method

    Parameters
    ----------
    obj_typ
        The type of object that needs converting.
    args: Tuple[Any]
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
    for obj in args:
        if isinstance(obj, obj_typ):
            out.append(converter(obj))
            conv.append(True)
        else:
            out.append(obj)
            conv.append(False)
    return out, conv


def conv_loop_in_out(converter: _ty.Callable, obj_typ, kwargs: ArgDict,
                     conv_out: BoolList) -> (ArgTuple, BoolList):
    """Process the out keyword in an __array_ufunc__ method

    Parameters
    ----------
    obj_typ
        The type of object that needs converting.
    kwargs: Dict[str, Any]
        Dict of key word inputs to ufunc
    conv_out: List[bool]
        List of bools for converting outputs. Should have the correct length.

    Returns
    -------
    out: Tuple[Any]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv: List[bool]
        List of bools telling us if each input was converted.
    """
    outputs = kwargs.pop('out', None)
    if outputs:
        out_args, conv_out = conv_loop_input(converter, obj_typ, outputs)
        kwargs['out'] = tuple(out_args)
    else:
        outputs = (None,) * len(conv_out)
    return outputs, conv_out


def _conv_loop_in(converter, obj_typ, tup, *conv_out):
    """Call one of conv_loop_input or conv_loop_in_out"""
    if isinstance(tup, tuple):
        return conv_loop_input(converter, obj_typ, tup)
    return conv_loop_in_out(converter, obj_typ, tup, *conv_out)


def prepare_via_view() -> _ty.Callable:
    """Create function to convert object to an array using view method
    """
    def converter(thing):
        """convert to array using view method
        """
        return thing.view(_np.ndarray)
    return converter


def prepare_via_attr(attr: str) -> _ty.Callable:
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


def conv_loop_in_view(obj_typ, tup: Args, *conv_out) -> (ArgTuple, BoolList):
    """Process inputs in an __array_ufunc__ method using view method

    Parameters
    ----------
    obj_typ
        The type of object that needs converting via ``view`` method.
    tup: Tuple[Any], Dict[str, Any]
        Tuple of inputs to ufunc (or ``out`` argument)
        Dict of key word inputs to ufunc
    conv_out: List[bool], optional
        List of bools for converting outputs. Should have the correct length.

    Returns
    -------
    out: Tuple[Any]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv: List[bool]
        List of bools telling us if each input was converted.
    """
    return _conv_loop_in(prepare_via_view(), obj_typ, tup, *conv_out)


def conv_loop_in_attr(attr: str, obj_typ, tup: Args,
                      *conv_out) -> (ArgTuple, BoolList):
    """Process inputs in an __array_ufunc__ method using an attribute

    Parameters
    ----------
    attr: str, None
        The name of the ``obj_typ`` attribute to use in place of class.
    obj_typ
        The type of object that needs converting with its ``attr`` attribute.
    tup: Tuple[Any], Dict[str, Any]
        Tuple of inputs to ufunc (or ``out`` argument)
        Dict of key word inputs to ufunc
    conv_out: List[bool], optional
        List of bools for converting outputs. Should have the correct length.

    Returns
    -------
    out: Tuple[Any]
        New tuple of inputs to ufunc (or ``out`` argument) with conversions.
    conv: List[bool]
        List of bools telling us if each input was converted.
    """
    return _conv_loop_in(prepare_via_attr(attr), obj_typ, tup, *conv_out)


# ======================================================================
# %% Outputs
# ======================================================================


def conv_loop_out(converter: _ty.Callable,
                  results: ArgTuple,
                  outputs: ArgTuple,
                  conv: _ty.Sequence[bool] = ()) -> ArgTuple:
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
    if results is NotImplemented:
        return NotImplemented
    if not isinstance(results, tuple):
        results = (results,)
    if not conv:
        conv = _itertools.repeat(True)
    results_out = []
    for result, output, cout in zip(results, outputs, conv):
        if output is None:
            if cout:
                results_out.append(converter(result))
            else:
                results_out.append(result)
        else:
            results_out.append(output)
    if len(results_out) == 1:
        return results_out[0]
    return tuple(results_out)


def restore_via_attr(obj, attr: str) -> _ty.Callable:
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


def restore_via_init(obj) -> _ty.Callable:
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


def restore_via_view(obj) -> _ty.Callable:
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


def conv_loop_out_attr(obj,
                       attr: str,
                       results: ArgTuple,
                       outputs: ArgTuple,
                       conv: _ty.Sequence[bool] = ()) -> ArgTuple:
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


def conv_loop_out_init(obj,
                       results: ArgTuple,
                       outputs: ArgTuple,
                       conv: _ty.Sequence[bool] = ()) -> ArgTuple:
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


def conv_loop_out_view(obj,
                       results: ArgTuple,
                       outputs: ArgTuple,
                       conv: _ty.Sequence[bool] = ()) -> ArgTuple:
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
