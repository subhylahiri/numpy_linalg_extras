# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:37:24 2017

@author: subhy

Classes that provide nicer syntax for matrix division and tools for working
with `numpy.linalg`'s broadcasting.

Routine Listings
----------------
lnarray
    Subclass of `numpy.ndarray` with properties such as `inv` for matrix
    division, `t` for transposing stacks of matrices, `c`, `r` and `s` for
    dealing with stacks of vectors and scalars.
pinvarray
    Provides interface for matrix division when it is matrix multiplied (@).
    Does not actually invert the matrix. It will raise if you try to do
    anything other than matrix multiplication or multiplication by scalars.
invarray
    Provides interface for matrix division when it is matrix multiplied (@).
    Does not actually invert the matrix. It will raise if you try to do
    anything other than matrix multiplication or multiplication by scalars.

Examples
--------
>>> import numpy as np
>>> import linalg as sp
>>> x = sp.lnarray(np.random.rand(2, 3, 4))
>>> y = sp.lnarray(np.random.rand(2, 3, 4))
>>> z = x.inv @ y
>>> w = x @ y.inv
>>> u = x @ y.t
>>> v = (x.r @ y.t).ur
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple, Union, ClassVar

import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin

from . import _linalg as la
from . import gufuncs as gf
from .gufuncs import fam
from .convert import conv_in_attr, conv_out_init, conv_in_view, conv_out_view

# pylint: disable=invalid-name
# =============================================================================
# Export functions
# =============================================================================


__all__ = [
        'lnarray',
        'invarray',
        'pinvarray',
        ]


# =============================================================================
# Class: lnarray
# =============================================================================
_TUPLE_FUNCS = {np.concatenate, np.stack, np.column_stack, np.dstack,
                np.hstack, np.vstack}


class lnarray(np.ndarray):
    """Array object with linear algebra customisation.

    This is a subclass of `np.ndarray` with some added properties.
    The most important is matrix division via a lazy inverse.
    It also has some properties to work with broadcasting rules of `np.linalg`.

    Parameters
    ----------
    input_array : array_like
        The constructed array gets its data from `input_array`.
        Data is copied if necessary, as per `np.asarray`.

    Attributes
    ----------
    pinv : pinvarray
        Lazy pseudoinverse. When matrix multiplying, performs matrix division.
        Note: call as a property. If you call it as a function, you'll get the
        actual pseudoinverse.
    inv : invarray
        Lazy inverse. When matrix multiplying, performs matrix division.
        Note: call as a property. If you call it as a function, you'll get the
        actual inverse.
    t
        Transpose last two axes.
    c
        Insert singleton in last slot -> stack of column vectors.
    r
        Insert singleton in second last slot -> stack of row vectors.
    s
        Insert singleton in last two slots -> stack of scalars.
    uc, ur, us
        Undo effect of `r`, `c`, `s`.

    Examples
    --------
    >>> import numpy as np
    >>> import numpy_linalg as la
    >>> x = la.lnarray(np.random.rand(2, 3, 4))
    >>> y = la.lnarray(np.random.rand(2, 3, 4))
    >>> z = x.pinv @ y
    >>> w = x @ y.pinv
    >>> u = x @ y.t
    >>> v = (x.r @ y.t).ur

    See also
    --------
    np.ndarray : the super class.
    pinvarray : class that provides an interface for matrix division.
    invarray : class that provides an interface for matrix division.
    """

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    # Last thing not implemented by ndarray:
    def __imatmul__(self, other: np.ndarray) -> lnarray:
        """In-place matrix multiplication

        Parameters
        ----------
        other : np.ndarray
            Matrix to multiply by. Must be square, `other.ndim <= self.ndim`,
            broadcasting must not expand shape of `self`.

        See Also
        --------
        `np.matmul` : implements this operator.
        """
        return gf.matmul(self, other, out=(self,))

    def __array_function__(self, func, types, args, kwargs):
        if not all(issubclass(t, lnarray) for t in types):
            return NotImplemented
        if func in _TUPLE_FUNCS:
            args = list(args)
            args[0], _ = conv_in_view(lnarray, args[0])
        elif func == np.block:
            args = list(args)
            args[0] = _nested_func(args[0],
                                   lambda x: conv_in_view(lnarray, (x,))[0])
        else:
            args, _ = conv_in_view(lnarray, args)
        outs, conv_out = conv_in_view(lnarray, kwargs, ())
        result = super().__array_function__(func, (), tuple(args), kwargs)
        return conv_out_view(self, result, outs, conv_out)

    def flattish(self, start: int, stop: int) -> lnarray:
        """Partial flattening.

        Flattens those axes in the range [start:stop)

        See Also
        --------
        numpy_linalg.flattish
        """
        return la.flattish(self, start, stop)

    def expand_dims(self, *axis) -> lnarray:
        """Expand the shape of the array with length one axes

        Alias of `numpy.expand_dims` when `*axis` is a single `int`. If `axis`
        is a sequence of `int`, axis numbers are relative to the *final* shape.

        See Also
        --------
        expand_dims
        """
        return la.expand_dims(self, *axis)

    @property
    def t(self) -> lnarray:
        """Transpose last two indices.

        Transposing last two axes fits better with `np.linalg`'s
        broadcasting, which treats multi-dim arrays as arrays of matrices.

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> transposed : lnarray, (..., N, M)

        See Also
        --------
        transpose
        """
        return la.transpose(self)

    @property
    def h(self) -> lnarray:
        """Hermitian-conjugate over last two indices.

        Transposing last two axes fits better with `np.linalg`'s
        broadcasting, which treats multi-dim arrays as arrays of matrices.

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> transposed : lnarray, (..., N, M)

        See Also
        --------
        dagger
        """
        return la.dagger(self)

    @property
    def r(self) -> lnarray:
        """Treat multi-dim array as a stack of row vectors.

        Inserts a singleton axis in second-last slot.

        Parameters/Results
        ------------------
        a : lnarray, (..., N) --> expanded : lnarray, (..., 1, N)

        See Also
        --------
        row
        """
        return la.row(self)

    @property
    def c(self) -> lnarray:
        """Treat multi-dim array as a stack of column vectors.

        Inserts a singleton axis in last slot.

        Parameters/Results
        ------------------
        a : lnarray, (..., N) --> expanded : lnarray, (..., N, 1)

        See Also
        --------
        col
        """
        return la.col(self)

    @property
    def s(self) -> lnarray:
        """Treat multi-dim array as a stack of scalars.

        Inserts singleton axes in last two slots.

        Parameters/Results
        ------------------
        a : lnarray, (...,) --> expanded : lnarray, (..., 1, 1)

        See Also
        --------
        scalar
        """
        return la.scalar(self)

    @property
    def ur(self) -> lnarray:
        """Undo effect of `r`.

        Parameters/Results
        ------------------
        a : lnarray, (..., 1, N) --> squeezed : lnarray, (..., N)

        Raises
        ------
        ValueError
            If a.shape[-2] != 1
        """
        return self.squeeze(axis=-2)

    @property
    def uc(self) -> lnarray:
        """Undo effect of `c`.

        Parameters/Results
        ------------------
        a : lnarray, (..., N, 1) --> squeezed : lnarray, (..., N)

        Raises
        ------
        ValueError
            If a.shape[-1] != 1
        """
        return self.squeeze(axis=-1)

    @property
    def us(self) -> lnarray:
        """Undo effect of `s`.

        Parameters/Results
        ------------------
        a : lnarray, (..., 1, 1) --> squeezed : lnarray, (...,)

        Raises
        ------
        ValueError
            If a.shape[-2] != 1 or a.shape[-1] != 1
        """
        return self.squeeze(axis=(-2, -1))

    @property
    def pinv(self) -> pinvarray:
        """Lazy matrix pseudoinverse

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> a_inv : pinvarray, (..., N, M)
            When matrix multiplying, performs matrix division.

        See also
        --------
        pinvarray
        """
        return pinvarray(self)

    @property
    def inv(self) -> invarray:
        """Lazy matrix inverse

        Parameters/Results
        ------------------
        a : lnarray, (..., M, M) --> a_inv : invarray, (..., M, M)
            When matrix multiplying, performs matrix division.

        Raises
        ------
        ValueError
            If `a.shape[-2] != a.shape[-1]`

        See also
        --------
        invarray
        """
        return invarray(self)


# =============================================================================
# Helpers for pinvarray
# =============================================================================


def _nested_func(nested, func):
    if isinstance(nested, list):
        return [_nested_func(x, func) for x in nested]
    return func(nested)


def _inv_input(ufunc: np.ufunc, pinv_in: Sequence[bool]) -> Tuple[bool, ...]:
    """Is each gufunc input an array to be lazily (pseudo)inverted?

    Parameters
    ----------
    ufunc : ufunc
        The original ufunc that was called
    pinv_in: Sequence[bool]
        Tells us if each original argument was a (p)invarray

    Returns
    -------
    left, right
        `gufuncs.fam.inverse_arguments` of ufunc to call
    swap
        should inputs be swapped?
    """
    # NOTE: rmatmul doesn't fit the pattern, needs special handling
    swap = ufunc == gf.rmatmul
    if swap and all(pinv_in):
        return (True, True, True)
    # inverse_arguments tells us if each argument is a 'denominator'.
    func_in = fam.inverse_arguments[ufunc]
    # `(p)invarray` in 'numerator' slot -> 'denominator' & vice versa => `xor`.
    return tuple(x ^ y for x, y in zip(pinv_in, func_in)) + (swap,)


def _inv_input_scalar(ufunc: np.ufunc,
                      pinv_in: Sequence[bool]) -> Tuple[bool, ...]:
    """Is the other ufunc input a numerator (after swapping for inverse, etc)?

    Parameters
    ----------
    ufunc
        The original ufunc that was called
    pinv_in: Sequence[bool]
        Tells us if each original argument was a (p)invarray

    Returns
    -------
    left, right : bool
        `gufuncs.fam.inverse_scalar_arguments` of ufunc to call
    """
    # if both arguments are (p)invarrays, return indices of `None` in ufunc map
    if all(pinv_in):
        return (False, False)
    # inverse_scalar_arguments tells us if the other argument is a numerator.
    func_in = fam.inverse_scalar_arguments[ufunc]
    # `(p)invarray` in 'numerator' slot -> 'denominator' & vice versa => `xor`.
    return tuple(x ^ y for x, y in zip(pinv_in, func_in))


def _disallow_solve_pinv(ufunc: np.ufunc, is_pinv: Sequence[bool]) -> bool:
    """Check that pinvarray is not in denominator of solve"""
    if not fam.same_family(ufunc, gf.solve) or (ufunc == gf.rmatmul):
        return False
    denom = fam.inverse_arguments[ufunc]
    return any(x and y for x, y in zip(denom, is_pinv))


def _who_chooses(obj: pinvarray,
                 ufunc: np.ufunc,
                 inputs: Sequence[Arrayish],
                 pinv_in: Sequence[bool]) -> pinvarray:
    """Which input's gufunc_map should we use"""
    if ufunc not in fam.inverse_arguments.keys():
        return obj
    denom = _inv_input(ufunc, pinv_in)[:2]
    is_inv = [isinstance(x, invarray) for x in inputs]
    is_pinv = [x and not y for x, y in zip(pinv_in, is_inv)]
    choosers = [x and y for x, y in zip(pinv_in, denom)]

    if _disallow_solve_pinv(ufunc, is_pinv):
        return None
    if all(choosers) and any(is_pinv):
        return None
    if any(choosers):
        # only change the chooser if it is a denominator and a (p)invarray
        return inputs[choosers.index(True)]
    return obj


def _implicit_ufunc(inputs: Sequence[Arrayish]) -> List[Arrayish]:
    """Convert (p)invarray to explicit (pseudo)inverse for other ufuncs.

    Alternative to __array_ufunc__ returns `NotImplemented`: other ufuncs use
    implicit inversion.
    Not used on the basis that explicit > implicit.
    Use __call__ if you want an actual (pseudo)inverse matrix.
    """
    return [x() if isinstance(x, pinvarray) else x for x in inputs]


def _implicit_getattr(obj: pinvarray, attr: str):
    """Convert (p)invarray to explicit (pseudo)inverse for other attributes.

    Alternative to returning `AttributeError()`: get it from explicit inverse.
    This would allow other operations to work with implicit inversion.
    Not used on the basis that explicit > implicit.
    Use __call__ if you want an actual (pseudo)inverse matrix.
    """
    if hasattr(obj._to_invert, attr):
        return getattr(obj(), attr)
    raise AttributeError(f"Attribute {attr} not found in pinvarray or inverse")


# =============================================================================
# Class: pinvarray
# =============================================================================


class pinvarray(NDArrayOperatorsMixin):
    """Lazy matrix pseudoinverse of `lnarray`.

    Does not actually perform the matrix pseudoinversion.
    It will use least-square matrix division for matmul (@) with an `lnarray`.

    It is intended to be ephemeral, appearing in larger expressions rather than
    being stored in a variable.
    Use `pinvarray()` to get the actual pseudoinverse.

    This object contains a reference to the original array, so in place
    modifications of a `pinvarray` object will affect the original `lnarray`
    object and vice versa.

    Parameters
    ----------
    to_invert : lnarray
        The array whose pseudoinverse is being taken.

    Returns
    -------
    concrete : lnarray
        Returns the actual, concrete pseudoinverse, calculating it if it has
        not already been done.

    Attributes
    ----------
    pinv : lnarray
        Returns the original array that needed inverting.

    Notes
    -----
    It can also be multiplied and divided by a nonzero scalar or stack of
    scalars, i.e. `ndarray` with last two dimensions singletons.
    Actually divides/multiplies the pre-inversion object.
    It will also behave appropriately in the `(r)lstsq` functions from *this*
    package (not the `numpy` versions).

    *Any* other operation or attribute access would require actually
    performing the pseudoinversion and using that instead (except for `len`,
    `shape`, `size`, 'ndim`, `repr`, `str`, `t`, `h` and `inv`).

    It uses `gufuncs.lstsq`,`gufuncs.rlstsq` for @ and `gufuncs.pinv` for ().

    Examples
    --------
    >>> import numpy as np
    >>> import linalg as sp
    >>> x = sp.lnarray(np.random.rand(2, 3, 4))
    >>> y = sp.lnarray(np.random.rand(2, 3, 4))
    >>> z = x.inv @ y
    >>> w = x @ y.inv

    Raises
    ------
    FloatingPointError
        If computation does not converge.

    See also
    --------
    lnarray : the array class used.
    invarray : class that provides an interface for matrix division.
    """
    _to_invert: lnarray
    _inverted: Optional[lnarray]
    _factored: Tuple[lnarray, ...]

    # _ufunc_map[arg1][arg2] -> ufunc_out, where:
    # ar1/arg2 = is the second/first argument the numerator?
    # ufunc_out = ufunc to use instead of original for scalar operator
    _ufunc_map: ClassVar[fam.UFMap] = fam.truediv_family
    # _gufunc_map[arg1][arg2] -> gufunc_out, where:
    # ar1/arg2 = is the first/second argument an array to be lazily inverted?
    # ufunc_out = gufunc to use instead of original in matrix operation
    _gufunc_map: ClassVar[fam.UFMap] = fam.lstsq_family

    # these ufuncs are passed on to self._to_invert
    _unary_ufuncs: ClassVar[Set[np.ufunc]] = {np.positive, np.negative}

    def __init__(self, to_invert: lnarray):
        if isinstance(to_invert, lnarray):
            # don't want to mess up subclasses, so that `pinv` returns input
            self._to_invert = to_invert
        else:
            # in case input is `ndarray` or `array_like`
            self._to_invert = np.asarray(to_invert).view(lnarray)
        self._inverted = None
        self._factored = ()

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwds):
        """Handling ufuncs with pinvarrays
        """
        # which inputs are we converting?
        # For most inputs, we swap multiplication & division instead of inverse
        args, pinv_in = conv_in_attr('_to_invert', pinvarray, inputs)
        obj = _who_chooses(self, ufunc, inputs, pinv_in)
        if obj is None:
            return NotImplemented
        if obj is not self:
            return obj.__array_ufunc__(ufunc, method, *inputs, **kwds)
        ufunc, args, pinv_out = self._choose_ufunc(ufunc, args, pinv_in)
        if ufunc is None:
            return NotImplemented
        outs, pinv_out = conv_in_attr('_to_invert', pinvarray, kwds, pinv_out)
        results = self._to_invert.__array_ufunc__(ufunc, method, *args, **kwds)
        return conv_out_init(self, results, outs, pinv_out)

    def _choose_ufunc(self, ufunc: np.ufunc, args: Tuple[np.ndarray],
                      pinv_in: Sequence[bool]) -> (np.ufunc, Tuple[np.ndarray],
                                                   List[bool]):
        """Choose which ufunc to use, swap args if needed, convert result?"""
        pinv_out = [False] * ufunc.nout  # which outputs need converting back?
        if ufunc in fam.inverse_arguments.keys():
            # _who_chooses -> correct choice across families
            left_arg, right_arg, swap = _inv_input(ufunc, pinv_in)
            ufunc = self._gufunc_map[left_arg][right_arg]
            # NOTE: rmatmul doesn't fit the pattern, needs special handling
            if swap:
                args = args[1::-1] + args[2:]
            # only operation that returns `invarray` is `invarray @ invarray`
            pinv_out[0] = left_arg and right_arg
        elif ufunc in fam.inverse_scalar_arguments.keys():
            left_arg, right_arg = _inv_input_scalar(ufunc, pinv_in)
            ufunc = self._ufunc_map[left_arg][right_arg]
            pinv_out[0] = True  # one of left_arg/right_arg must be True
        elif ufunc in self._unary_ufuncs:
            # Apply ufunc to self._to_invert.
            # Already converted input; just need to convert output back
            pinv_out[0] = True
        else:
            # Not a linalg operator ufunc. Not what this class is for.
            ufunc = None
            # Alternative: other ufuncs use implicit inversion.
            # args = _implicit_ufunc(inputs)
            # Not used because explicit > implicit.
        return ufunc, args, pinv_out

    # This would allow other operations to work with implicit inversion.
    # Not used because explicit > implicit.
    # def __getattr__(self, attr):
    #     """Get `np.ndarray proerties from actual (pseudo)inverse.
    #     """
    #    return _implicit_getattr(self, attr)

    # make this __call__? __array__? @property? get()? do()?
    def __call__(self) -> lnarray:
        """Get the actual (pseudo)inverse

        Returns
        -------
        inverted
            The (pseudo)inverse of the `lnarray` whose `(p)inv` this object is.
        """
        # Can't know if self._to_invert has been modified, so:
        return self._invert()
        # If self._to_invert has not been (pseudo)inverted, it will compute the
        # (pseudo)inverse first. Otherwise, it will use the stored value.
        # if self._inverted is None:
        #     self._inverted = self._invert()
        # return self._inverted

    def _invert(self) -> lnarray:
        """Actually perform (pseudo)inverse
        """
        if self.ndim < 2:
            # scalar or vector
            return self._to_invert / gf.norm(self._to_invert)**2
        if self.ndim >= 2:
            # pinv broadcasts
            return gf.pinv(self._to_invert)
        raise ValueError('Nothing to invert? ' + str(self._to_invert))

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self) -> str:
        return str(self._to_invert) + '**(+)'

    def __repr__(self) -> str:
        selfname = type(self).__name__
        extra = len(type(self._to_invert).__name__) - len(selfname)
        rep = repr(self._to_invert).replace("\n" + " " * extra,
                                            "\n" + " " * -extra)
        return selfname + rep[(len(selfname) + extra):]

    def swapaxes(self, axis1: int, axis2: int) -> pinvarray:
        """Interchange two axes in a copy

        Parameters
        ----------
        axis1/axis2 : int
            First/Second axis.

        Returns
        -------
        a_swapped : pinvarray
            For NumPy >= 1.10.0, if `a.pinv` is an ndarray, then new pinvarray
            containing a view of `a.pinv` is returned; otherwise a new array is
            created.
        """
        if axis1 % self.ndim in {self.ndim - 1, self.ndim - 2}:
            axis1 = (-3 - axis1) % self.ndim
        if axis2 % self.ndim in {self.ndim - 1, self.ndim - 2}:
            axis2 = (-3 - axis2) % self.ndim
        return type(self)(self._to_invert.swapaxes(axis1, axis2))

    @property
    def shape(self) -> Tuple[int, ...]:
        """Effective shape of pinvarray in matmul etc."""
        # Matrix operations are allowed with x.inv when allowed for x.t
        return self._to_invert.t.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._to_invert.ndim

    @property
    def size(self) -> int:
        """Number of elements."""
        return self._to_invert.size

    @property
    def dtype(self) -> np.dtype:
        """Data type of array elements."""
        return self._to_invert.dtype

    @property
    def pinv(self) -> lnarray:
        """Uninverted matrix."""
        return self._to_invert

    @property
    def t(self) -> pinvarray:
        """Transpose over last two indices.

        A copy of object, but view of data
        """
        return type(self)(self._to_invert.t)

    @property
    def h(self) -> pinvarray:
        """Hermitian conjugate over last two indices.

        A copy of object, but view of data
        """
        return type(self)(self._to_invert.h)

    def copy(self, order: str = 'C', **kwds) -> pinvarray:
        """Copy data

        Parameters
        ----------
        to_invert : lnarray, optional
            Replacement for array whose pseudoinverse is being taken.
            Default: a copy of the original array.
        order : str, optional
            Storage order for copy of the original array.
            Default: 'C'.
        """
        _to_invert = kwds.pop('_to_invert', None)
        if _to_invert is None:
            _to_invert = self._to_invert.copy(order=order)
        return type(self)(_to_invert, **kwds)


# =============================================================================
# Class: invarray
# =============================================================================


class invarray(pinvarray):
    """Lazy matrix inverse of `lnarray`.

    Does not actually perform the matrix inversion.
    It will use matrix division for matmul (@) with an `lnarray`.

    It is intended to be ephemeral, appearing in larger expressions rather than
    being stored in a variable.
    Use `invarray()` to get the actual inverse.

    This object contains a reference to the original array, so in place
    modifications of a `invarray` object will affect the original `lnarray`
    object and vice versa.

    Parameters
    ----------
    to_invert : lnarray
        The array whose inverse is being taken.

    Returns
    -------
    concrete : lnarray
        Returns the actual, concrete inverse, calculating it if it has
        not already been done.

    Attributes
    ----------
    inv : lnarray
        Returns the original array that needed inverting.

    Notes
    -----
    It can also be multiplied and divided by a nonzero scalar or stack of
    scalars, i.e. `ndarray` with last two dimensions singletons.
    Actually divides/multiplies the pre-inversion object.
    It will also behave appropriately in the `(r)solve` functions from *this*
    package (not the `numpy` versions).

    *Any* other operation or attribute access would require actually
    performing the inversion and using that instead (except for `len`,
    `shape`, `size`, 'ndim`, `repr`, `str`, `t` and `inv`).

    It uses `gufuncs.solve`, `gufuncs.rsolve` for @ and `gufuncs.inv` for ().

    Examples
    --------
    >>> import numpy as np
    >>> import linalg as sp
    >>> x = sp.lnarray(np.random.rand(2, 3, 3))
    >>> y = sp.lnarray(np.random.rand(2, 3, 3))
    >>> z = x.inv @ y
    >>> w = x @ y.inv

    Raises
    ------
    FloatingPointError
        If original matrix is not full rank.
    ValueError
        if original matrix is not square.

    See also
    --------
    lnarray : the array class used.
    pinvarray : class that provides an interface for matrix pseudo-division.
    """
    # _gufunc_map[arg1][arg2] -> gufunc_out, where:
    # ar1/arg2 = is the first/second argument an array to be lazily inverted?
    # ufunc_out = gufunc to use instead of original in matrix operation
    _gufunc_map: ClassVar[fam.UFMap] = fam.solve_family

    def __init__(self, to_invert: lnarray):
        super().__init__(to_invert)
        if to_invert.ndim < 2 or (to_invert.shape[-1] != to_invert.shape[-2]):
            raise ValueError("Array to invert is not square: shape = "
                             + f"{to_invert.shape}.")

    def __str__(self) -> str:
        return str(self._to_invert) + '**(-1)'

    @property
    def pinv(self) -> lnarray:
        """Uninverted matrix."""
        raise TypeError('This is an invarray, not a pinvarray!')

    @property
    def inv(self) -> lnarray:
        """Uninverted matrix."""
        return self._to_invert

    def _invert(self) -> lnarray:
        """Actually perform inverse."""
        if self.ndim >= 2 and self.shape[-2] == self.shape[-1]:
            # square
            return gf.inv(self._to_invert)
        raise ValueError('Cannot be inverted? ' + str(self._to_invert))


Arrayish = Union[np.ndarray, lnarray, pinvarray, invarray]
