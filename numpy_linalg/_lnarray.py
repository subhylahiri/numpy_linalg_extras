# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 21:37:24 2017

@author: subhy

Classes that provide nicer syntax for matrix division and tools for working
with `numpy.linalg`'s broadcasting.

Classes
-------
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
lnmatrix
    Subclass of `lnarray` with swapped elementwise/matrix multiplication and
    division.

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
from typing import Optional, Tuple, Sequence
import numpy as np
import numpy.lib.mixins as _mix
from . import _linalg as la
from . import gufuncs as gf
from . import convert_loop as cv


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

    Properties
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
    `np.ndarray` : the super class.
    `pinvarray` : class that provides an interface for matrix division.
    `invarray` : class that provides an interface for matrix division.
    """

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

#    def __array_finalize__(self, obj):
#        # We are not adding any attributes
#        pass
    (__matmul__,
     __rmatmul__,
     __imatmul__) = _mix._numeric_methods(gf.matmul, 'matmul')

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Customise ufunc behaviour
        """
        args = list(cv.conv_loop_in_view(lnarray, inputs)[0])
        # Set of ufuncs that need special handling for vectors
        if ufunc in gf.inverse_arguments.keys():
            args[0], args[1], squeeze = gf.vec2mat(
                    args[0], args[1], gf.inverse_arguments[ufunc]
            )
        args = tuple(args)

        outputs = kwargs.pop('out', None)
        if outputs:
            out_args = cv.conv_loop_in_view(lnarray, outputs)[0]
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        if ufunc in gf.inverse_arguments.keys():
            gf.make_errobj("Failure in routine: " + ufunc.__name__, kwargs)
        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)
        if ufunc in gf.inverse_arguments.keys() and any(squeeze):
            results = (gf.mat2vec(results[0], squeeze),) + results[1:]
        results = cv.conv_loop_out_view(self, results, outputs)

        return results[0] if len(results) == 1 else results

    def flattish(self, start: int, stop: int) -> 'lnarray':
        """Partial flattening.

        Flattens those axes in the range [start:stop)

        See Also
        --------
        linalg.flattish
        """
        return la.flattish(self, start, stop)

    def expand_dims(self, *axis) -> 'lnarray':
        """Expand the shape of the array with length one axes

        Alias of `numpy.expand_dims` when `*axis` is a single `int`. If `axis`
        is a sequence of `int`, axis numbers are relative to the *final* shape.

        See Also
        --------
        linalg.expand_dims
        """
        return la.expand_dims(self, *axis)

    @property
    def t(self) -> 'lnarray':
        """Transpose last two indices.

        Transposing last two axes fits better with `np.linalg`'s
        broadcasting, which treats multi-dim arrays as arrays of matrices.

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> transposed : lnarray, (..., N, M)

        See Also
        --------
        linalg.transpose
        """
        return la.transpose(self)

    @property
    def h(self) -> 'lnarray':
        """Hermitian-conjugate over last two indices.

        Transposing last two axes fits better with `np.linalg`'s
        broadcasting, which treats multi-dim arrays as arrays of matrices.

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> transposed : lnarray, (..., N, M)

        See Also
        --------
        linalg.dagger
        """
        return la.dagger(self)

    @property
    def r(self) -> 'lnarray':
        """Treat multi-dim array as a stack of row vectors.

        Inserts a singleton axis in second-last slot.

        Parameters/Results
        ------------------
        a : lnarray, (..., N) --> expanded : lnarray, (..., 1, N)

        See Also
        --------
        linalg.row
        """
        return la.row(self)

    @property
    def c(self) -> 'lnarray':
        """Treat multi-dim array as a stack of column vectors.

        Inserts a singleton axis in last slot.

        Parameters/Results
        ------------------
        a : lnarray, (..., N) --> expanded : lnarray, (..., N, 1)

        See Also
        --------
        linalg.col
        """
        return la.col(self)

    @property
    def s(self) -> 'lnarray':
        """Treat multi-dim array as a stack of scalars.

        Inserts singleton axes in last two slots.

        Parameters/Results
        ------------------
        a : lnarray, (...,) --> expanded : lnarray, (..., 1, 1)

        See Also
        --------
        linalg.scalar
        """
        return la.scalar(self)

    @property
    def ur(self) -> 'lnarray':
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
    def uc(self) -> 'lnarray':
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
    def us(self) -> 'lnarray':
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
    def pinv(self) -> 'pinvarray':
        """Lazy matrix pseudoinverse

        Parameters/Results
        ------------------
        a : lnarray, (..., M, N) --> a_inv : pinvarray, (..., N, M)
            When matrix multiplying, performs matrix division.

        See also
        --------
        `pinvarray`
        """
        return pinvarray(self)

    @property
    def inv(self) -> 'invarray':
        """Lazy matrix inverse

        Parameters/Results
        ------------------
        a : lnarray, (..., M, M) --> a_inv : pinvarray, (..., M, M)
            When matrix multiplying, performs matrix division.

        Raises
        ------
        ValueError
            If a.shape[-2] != a.shape[-1]

        See also
        --------
        `invarray`
        """
        return invarray(self)


# =============================================================================
# Helpers for pinvarray
# =============================================================================


def _inv_input(ufunc, pinv_in: Sequence[bool]) -> Tuple[bool, ...]:
    """Is each gufunc input an array to be lazily (pseudo)inverted?

    Parameters
    ----------
    ufunc
        The original ufunc that was called
    pinv_in: Sequence[bool]
        Tells us if each original argument was a (p)invarray
    Returns
    -------
    left, right
        `gufuncs.inverse_arguments` of ufunc to call
    swap
        should imputs be swapped?
    """
    # inverse_arguments tells us if each argument is a 'denominator'.
    # A `(p)invarray` in a 'numerator' slot -> 'denominator' & vice versa.
    # Hence `xor`.
    func_in = gf.inverse_arguments[ufunc]
    # NOTE: rmatmul doesn't fit the pattern, needs special handling
    if all(func_in) and all(pinv_in):
        return (True, True, True)
    swap = all(func_in)
    return tuple(x ^ y for x, y in zip(pinv_in, func_in)) + (swap,)


def _inv_input_scalar(ufunc, pinv_in: Sequence[bool]) -> bool:
    """Is the other ufunc input a numerator (after swapping for inverse, etc)?

    Parameters
    ----------
    ufunc
        The original ufunc that was called
    pinv_in: Sequence[bool]
        Tells us if each original argument was a (p)invarray
    """
    # inverse_scalar_arguments tells us if the other argument is a 'numerator'.
    # A `(p)invarray` in a 'numerator' slot -> 'denominator' & vice versa.
    # Hence `xor`.
    # if both arguments are (p)invarrays, return (indices of) None
    if all(pinv_in):
        return (False, False)
    func_in = gf.inverse_scalar_arguments[ufunc]
    return tuple(x ^ y for x, y in zip(pinv_in, func_in))


# =============================================================================
# Class: pinvarray
# =============================================================================


class pinvarray(gf.LNArrayOperatorsMixin):
    """Lazy matrix pseudoinverse of `lnarray`.

    Does not actually perform the matrix pseudoinversion.
    It will use least-square matrix division for matmul (@) with an `lnarray`.

    It is intended to be ephemeral, appearing in larger expressions rather than
    being stored in a variable.
    Use `pinvarray()` to get the actual pseudoinverse.

    This object contains a reference to the original array, so in place
    modifications of a `pinvarray` object will affect the original `lnarray`
    object.

    Methods
    -------
    self() -> lnarray
        Returns the actual, concrete pseudoinverse, calculating it if it has
        not already been done.
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

    It uses `gufuncs.lstsq`,`gufuncs.rlstsq` for @ and `np.linalg.pinv` for ().

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
    LinAlgError
        If computation does not converge.

    See also
    --------
    `lnarray` : the array class used.
    `invarray` : class that provides an interface for matrix division.
    """
    _to_invert: lnarray
    _inverted: Optional[lnarray]

    # _ufunc_map[arg1][arg2] -> ufunc_out, where:
    # ar1/arg2 = is the second/first argument the numerator?
    # ufunc_out = ufunc to use instead of original for scalar operator
    _ufunc_map = gf.truediv_family
    # _gufunc_map[arg1][arg2] -> gufunc_out, where:
    # ar1/arg2 = is the first/second argument an array to be lazily inverted?
    # ufunc_out = gufunc to use instead of original in matrix operation
    _gufunc_map = gf.lstsq_family

    # these ufuncs are passed on to self._to_invert
    _unary_ufuncs = {np.positive, np.negative}

    def __init__(self, to_invert: lnarray):
        if isinstance(to_invert, lnarray) and not isinstance(to_invert,
                                                             lnmatrix):
            # don't want to mess up subclasses, so that `pinv` returns input
            self._to_invert = to_invert
        else:
            # in case input is `lnmatrix`, `ndarray` or `array_like`
            self._to_invert = np.asarray(to_invert).view(lnarray)
        self._inverted = None

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Handling ufuncs with pinvarrays
        """
        # which inputs are we converting?
        # For most inputs, we swap multiplication & division instead of inverse
        args, pinv_in = cv.conv_loop_in_attr('_to_invert', pinvarray, inputs)

        pinv_out = [False] * ufunc.nout  # which outputs need converting back?
        if ufunc in gf.inverse_arguments.keys():
            if not gf.same_family(ufunc, self._gufunc_map[0][1]):
                return NotImplemented
            left_arg, right_arg, swap = _inv_input(ufunc, pinv_in)
            ufunc = self._gufunc_map[left_arg][right_arg]
            # NOTE: rmatmul doesn't fit the pattern, needs special handling
            if swap:
                args = (args[1], args[0]) + args[2:]
            # only operation that returns `invarray` is `invarray @ invarray`
            pinv_out[0] = left_arg and right_arg
        elif ufunc in gf.inverse_scalar_arguments.keys():
            left_arg, right_arg = _inv_input_scalar(ufunc, pinv_in)
            ufunc = self._ufunc_map[left_arg][right_arg]
            pinv_out[0] = True  # one of left_arg/right_arg must be True
        elif ufunc in self._unary_ufuncs:
            # Apply ufunc to self._to_invert.
            # Already converted input; just need to convert output back
            pinv_out[0] = True
        else:
            return NotImplemented
        if ufunc is None:
            return NotImplemented
        # Alternative: other ufuncs use implicit inversion.
        # Not used on the basis that explicit > implicit. Use __call__ instead.
        # args = []
        # for input_ in inputs:
        #     if isinstance(input_, pinvarray):
        #         args.append(input_())
        #     else:
        #         args.append(input_)
        # args = tuple(args)
        # return self._to_invert.__array_ufunc__(ufunc, method, *args,
        #                                        **kwargs)

        outputs = kwargs.pop('out', None)
        if outputs:
            out_args, pinv_out = cv.conv_loop_in_attr('_to_invert',
                                                      pinvarray, outputs)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = self._to_invert.__array_ufunc__(ufunc, method, *args,
                                                  **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if ufunc.nout == 1:
            results = (results,)

        results = cv.conv_loop_out_init(self, results, outputs, pinv_out)

        return results[0] if len(results) == 1 else results

    # This would allow other operations to work with implicit inversion.
    # Not used on the basis that explicit > implicit. Use __call__ instead.
    # def __getattr__(self, name):
    #     """Get `np.ndarray proerties from actual (pseudo)inverse.
    #
    #     Get unknown attributes from `lnarray` stored as `self._inverted`.
    #
    #     Notes
    #     -----
    #     If self._to_invert has not been (pseudo)inverted, it will compute the
    #     (pseudo)inverse first.
    #     """
    #     if hasattr(self._to_invert, name):
    #         return getattr(self(), name)
    #     else:
    #         raise AttributeError

    # make this __call__? @property? get()? do()? __array__?
    def __call__(self) -> lnarray:
        """Get actual (pseudo)inverse

        Returns
        -------
        inverted
            The (pseudo)inverse of the `lnarray` whose `inv` this object is,
            stored as `self._inverted`.

        Notes
        -----
        If self._to_invert has not been (pseudo)inverted, it will compute the
        (pseudo)inverse first.
        Otherwise, it will use the stored value.
        """
        if self._inverted is None:
            self._invert()
        return self._inverted

    def __len__(self):
        return self.shape[0]

    def __str__(self) -> str:
        return str(self._to_invert) + '**(+)'

    def __repr__(self) -> str:
        selfname = type(self).__name__
        extra = len(type(self._to_invert).__name__) - len(selfname)
        rep = repr(self._to_invert).replace("\n" + " " * extra,
                                            "\n" + " " * -extra)
        return selfname + rep[(len(selfname) + extra):]

    def swapaxes(self, axis1, axis2) -> 'pinvarray':
        """Interchange two axes in a copy

        Parameters
        ----------
        axis1/axis2 : int
            First/Second axis.

        Returns
        -------
        a_swapped : ndarray
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
        """Effective shape of pinvarray in matmul etc.
        """
        # Matrix operations are allowed with x.inv when allowed for x.t
        return self._to_invert.t.shape

    @property
    def ndim(self) -> int:
        """Number of dimensions
        """
        return self._to_invert.ndim

    @property
    def size(self) -> int:
        """Number of elements
        """
        return self._to_invert.size

    @property
    def dtype(self) -> np.dtype:
        """Data type
        """
        return self._to_invert.dtype

    @property
    def pinv(self) -> lnarray:
        """Uninverted matrix
        """
        return self._to_invert

    @property
    def t(self) -> 'pinvarray':
        """A copy of object, but view of data"""
        return type(self)(self._to_invert.t)

    @property
    def h(self) -> 'pinvarray':
        """A copy of object, but view of data"""
        return type(self)(self._to_invert.h)

    def copy(self, order='C', **kwds) -> 'pinvarray':
        """Copy data"""
        _to_invert = kwds.pop('_to_invert', None)
        if _to_invert is None:
            _to_invert = self._to_invert.copy(order=order)
        return type(self)(_to_invert, **kwds)

    def _invert(self):
        """Actually perform (pseudo)inverse
        """
        if self.ndim < 2:
            # scalar or vector
            self._inverted = self._to_invert / gf.norm(self._to_invert)**2
        elif self.ndim >= 2:
            # pinv broadcasts
            self._inverted = np.linalg.pinv(self._to_invert).view(lnarray)
        else:
            raise ValueError('Nothing to invert? ' + str(self._to_invert))


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
    modifications of a `pinvarray` object will affect the original `lnarray`
    object.

    Methods
    -------
    self() -> lnarray
        Returns the actual, concrete inverse, calculating it if it has
        not already been done.
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

    It uses `gufuncs.solve`, `gufuncs.rsolve` for @ and `np.linalg.inv` for ().

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
    LinAlgError
        If original matrix is not full rank.
    ValueError
        if original matrix is not square.

    See also
    --------
    `lnarray` : the array class used.
    `pinvarray` : class that provides an interface for matrix pseudo-division.
    """
    # _gufunc_map[arg1][arg2] -> gufunc_out, where:
    # ar1/arg2 = is the first/second argument an array to be lazily inverted?
    # ufunc_out = gufunc to use instead of original in matrix operation
    _gufunc_map = gf.solve_family

    def __init__(self, to_invert: lnarray):
        super().__init__(to_invert)

        if to_invert.ndim < 2 or (to_invert.shape[-1] != to_invert.shape[-2]):
            msg = ("Array to be inverted must have ndim >= 2 and be square, "
                   + "but to_invert.shape = {}. ".format(to_invert.shape)
                   + "Must be a matrix, or stack of matrices.")
            raise ValueError(msg)

    def __str__(self) -> str:
        return str(self._to_invert) + '**(-1)'

    @property
    def pinv(self) -> lnarray:
        """Uninverted matrix
        """
        raise TypeError('This is an invarray, not a pinvarray!')

    @property
    def inv(self) -> lnarray:
        """Uninverted matrix
        """
        return self._to_invert

    def _invert(self):
        """Actually perform inverse
        """
        if self.ndim >= 2 and self.shape[-2] == self.shape[-1]:
            # square
            self._inverted = np.linalg.inv(self._to_invert)
        else:
            raise ValueError('Nothing to invert? ' + str(self._to_invert))