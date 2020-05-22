# -*- coding: utf-8 -*-
"""Customised unittest for numpy

Classes
-------
TestCaseNumpy
    A `unittest.TestCase` subclass with assert methods for testing array values
    and shapes. Create test cases by subclassing this.

Functions
---------
trnsp
    Modify a shape (tuple of ints) by swapping last two axes.
drop
    Modify a shape by removing an axis.
chop
    Modify a shape by setting the smaller of the last two axes to the larger.
grow
    Modify a shape by setting the larger of the last two axes to the smaller.
return_shape
    Calculated broadcasted shape of output from input shapes and signature.
array_return_shape
    Calculated broadcasted shape of output from input array and signature.

Constants
---------
broadcast_err
    Tuple of arguments for `unittest.TestCase.assertRaisesRegex` to match a
    error due to mismatched broadcasting dimensions.
core_dim_err
    Tuple of arguments to match a error due to mismatched core dimensions.
num_dim_err
    Tuple of arguments to match a error caused by too few dimensions.
invalid_err
    Tuple of arguments to match a numerical error.
"""
import contextlib as _cx
import unittest as _ut
from typing import Optional, Tuple, TypeVar, Union

import numpy as np

Tvar = TypeVar('Tvar')
OneOrTuple = Union[Tvar, Tuple[Tvar, ...]]
# pylint: disable=invalid-name
__all__ = [
    'TestCaseNumpy',
    'broadcast_err',
    'core_dim_err',
    'num_dim_err',
    'invalid_err',
    'trnsp',
    'drop',
    'chop',
    'grow',
    'return_shape',
    'array_return_shape',
]
# =============================================================================
# Error specs for assertRaisesRegex
# =============================================================================
broadcast_err = (ValueError, 'operands could not be broadcast')
core_dim_err = (ValueError, 'mismatch in its core dimension')
num_dim_err = (ValueError, 'does not have enough dimensions')
invalid_err = (FloatingPointError, 'invalid value encountered')

__unittest = True
# =============================================================================
# TestCaseNumpy base class
# =============================================================================


class TestCaseNumpy(_ut.TestCase):
    """Test case with methods for comparing numpy arrays.

    Subclass this class to make your own unit test suite.
    It has several assertArray... methods that call numpy functions and
    process the results like a unittest.TestCase method.
    If you write a setUp method, be sure to call super().setUp().

    Methods
    -------
    assertArrayAllClose
        Calls numpy.allclose (so it broadcasts, unlike
        numpy.testing.assert_allclose).
    assertArrayNotAllClose
        Calls numpy.allclose and negates result.
    assertArrayEqual
        Calls numpy.all(numpy.equal(...)).
    assertArrayNotEqual
        Calls numpy.all(numpy.not_equal(...)).
    assertArrayLess
        Calls numpy.all(numpy.less(...)).
    assertArrayNotLess
        Calls numpy.all(numpy.greater_equal(...)).
    assertArrayGreater
        Calls numpy.all(numpy.greater(...)).
    assertArrayNotGreater
        Calls numpy.all(numpy.less_equal(...)).
    """

    def setUp(self):
        # testing ndarray values (relative to np.float64's eps)
        self.all_close_opts = {'atol': 1e-10,
                               'rtol': 1e-10,
                               'equal_nan': False}
        self.addTypeEqualityFunc(np.ndarray, self.assertArrayAllClose)

    @_cx.contextmanager
    def _adjusted_tols(self, array: np.ndarray, cond: float = 1.):
        """Adjusting all_close tolerances for dtype"""
        try:
            cond = cond if np.isfinite(cond) else 1.
            old_opts = self.all_close_opts.copy()
            if np.issubdtype(array.dtype, np.inexact):
                epsratio = np.finfo(array.dtype).eps / np.finfo(np.float64).eps
                # single/double epsratio ~ 5.6e8
                self.all_close_opts['rtol'] *= epsratio * cond
                self.all_close_opts['atol'] *= epsratio * cond
            elif np.issubdtype(array.dtype, np.integer):
                self.all_close_opts['rtol'] = 0
                self.all_close_opts['atol'] = 0.5
            yield
        finally:
            self.all_close_opts = old_opts

    @np.errstate(all="ignore")
    def assertArrayAllClose(self, actual: np.ndarray, desired: np.ndarray,
                            cond: float = 1., msg: Optional[str] = None):
        """Calls numpy.allclose and processes the results like a
        unittest.TestCase method.

        It broadcasts, unlike numpy.testing.assert_allclose.
        It adjusts the tolerances according to the `desired.dtype`'s eps.
        The original tolerances are for `np.float64`.
        """
        # __unittest = True
        with self._adjusted_tols(np.array(desired), cond=cond):
            if not np.allclose(actual, desired, **self.all_close_opts):
                msg = '' if msg is None else f'{msg}\n'
                msg += miss_str(actual, desired, **self.all_close_opts)
                msg += f' (cond={cond})'
                self.fail(msg)

    @np.errstate(all="ignore")
    def assertArrayNotAllClose(self, actual: np.ndarray, desired: np.ndarray,
                               msg: Optional[str] = None):
        """Calls numpy.allclose (so it broadcasts, unlike
        numpy.testing.assert_allclose), negates and processes the results like
        a unittest.TestCase method.
        """
        # __unittest = True
        with self._adjusted_tols(np.array(desired)):
            if np.allclose(actual, desired, **self.all_close_opts):
                msg = '' if msg is None else f'{msg}\n'
                msg += miss_str(actual, desired, **self.all_close_opts)
                msg.replace("not all close", "are all close")
                self.fail(msg)

    def assertArrayEqual(self, actual: np.ndarray, desired: np.ndarray,
                         msg=None):
        """Calls numpy.all(numpy.equal(...)) and processes the results like a
        unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual != desired):
            self.fail(msg)

    def assertArrayLess(self, actual: np.ndarray, desired: np.ndarray,
                        msg: Optional[str] = None):
        """Calls numpy.all(numpy.less(...)) and processes the results like a
        unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual >= desired):
            self.fail(msg)

    def assertArrayGreater(self, actual: np.ndarray, desired: np.ndarray,
                           msg: Optional[str] = None):
        """Calls numpy.all(numpy.greater(...)) and processes the results like a
        unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual <= desired):
            self.fail(msg)

    def assertArrayNotEqual(self, actual: np.ndarray, desired: np.ndarray,
                            msg: Optional[str] = None):
        """Calls numpy.all(numpy.not_equal(...)) and processes the results like
        a unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual == desired):
            self.fail(msg)

    def assertArrayNotLess(self, actual: np.ndarray, desired: np.ndarray,
                           msg: Optional[str] = None):
        """Calls numpy.all(numpy.greater_equal(...)) and processes the results
        like a unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual < desired):
            self.fail(msg)

    def assertArrayNotGreater(self, actual: np.ndarray, desired: np.ndarray,
                              msg: Optional[str] = None):
        """Calls numpy.all(numpy.less_equal(...)) and processes the results
        like a unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual > desired):
            self.fail(msg)

    def assertArrayShape(self, array: np.ndarray, shape: Tuple[int, ...],
                         msg: Optional[str] = None):
        """Calls self.assertEqual(array.shape, shape).
        """
        # __unittest = True
        self.assertEqual(array.shape, shape, msg)

    def assertArrayShapesAre(self, arrays: Tuple[np.ndarray, ...],
                             shapes: Tuple[Tuple[int, ...], ...],
                             msg: Optional[str] = None):
        """Calls self.assertEqual(array.shape, shape).
        """
        # __unittest = True
        for array, shape in zip(arrays, shapes):
            self.assertEqual(array.shape, shape, msg)

    def assertArrayDtype(self,
                         array_dtype: Union[np.ndarray, np.dtype, type, str],
                         dtype: Union[np.dtype, type, str],
                         msg: Optional[str] = None):
        if hasattr(array_dtype, 'dtype'):
            array_dtype = array_dtype.dtype
        if not np.issubsctype(array_dtype, dtype):
            msg = '' if msg is None else msg + "\n"
            msg += f"expected: {dtype}, actual: {array_dtype}"
            self.fail(msg)


# =============================================================================
# Helpers for TestCaseNumpy methods
# =============================================================================


@np.errstate(all="ignore")
def miss_str(left: np.ndarray, right: np.ndarray, atol: float = 1e-8,
             rtol: float = 1e-5, equal_nan: bool = True) -> str:
    """Returns a string describing the maximum deviation of left and right

    Parameters
    ----------
    left,right: ndarray[float]
        actual/desired value, must broadcast.
    atol, rtol, equal_nan
        arguments of ``np.allclose``.

    Returns
    -------
    msg: str
        A string that looks lik the following with <> placeholders
        'Arrays not all close.
        Largest mismatch: <maximum devation> with dtype=<scalar type>,
        or: <relative-max dev> = <tolerance> * <max dev relative to tolerance>'
    """
    shape = np.broadcast(left, right).shape
    thresh = atol + rtol * np.abs(np.broadcast_to(right, shape))
    mismatch = np.abs(left - right)
    mask = mismatch > thresh
    mis_frac = np.full_like(mismatch, np.NINF)
    mis_frac[mask] = (np.log(mismatch[mask]) - np.log(thresh[mask]))/np.log(10)

    if equal_nan:
        argmax = np.nanargmax
    else:
        argmax = np.argmax
    if np.all(np.isnan(mismatch)) or np.all(np.isnan(mis_frac)):
        def argmax(val):
            return 0
    a_ind = np.unravel_index(argmax(mismatch), mismatch.shape)
    r_ind = np.unravel_index(argmax(mis_frac), mis_frac.shape)

    a_worst, r_worst = mismatch[a_ind], mismatch[r_ind]
    thresh, mis_f = thresh[r_ind], mis_frac[r_ind]
    dtype = left.dtype

    return f"""Arrays not all close.
    Largest mismatch: {a_worst:.2g} at {a_ind} with dtype={dtype},
    or: {r_worst:.2g} = {thresh:.2g} * 1e{mis_f:.1f} at {r_ind}."""


# =============================================================================
# Helpers for shape checks
# =============================================================================


def drop(shape: Tuple[int, ...], axis: int = -2) -> Tuple[int, ...]:
    """Shape -> shape with one axis dropped"""
    return shape[:axis] + shape[axis+1:]


def trnsp(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Shape -> shape of transposed array"""
    return shape[:-2] + shape[:-3:-1]


def chop(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Shape -> shape with last axes reduced to square"""
    return shape[:-2] + (min(shape[-2:]),) * 2


def grow(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Shape -> shape with last axes expanded to square"""
    return shape[:-2] + (max(shape[-2:]),) * 2


def make_bad_broadcast(left, right, cores=(2, 2)):
    """Stack arrays so they no longer broadcast"""
    axis = (left.ndim - cores[0]) - (right.ndim - cores[1])
    new_left = np.stack((left,) * 3)[np.s_[:, ] + (None,) * (-axis)]
    new_right = np.stack((right,) * 2)[np.s_[:, ] + (None,) * axis]
    return new_left, new_right


def make_off_by_one(matrices, vectors):
    """Arrange so that matrices.ndim = vectors.ndim + 1"""
    off_by_one = matrices.ndim - vectors.ndim - 1
    return (None,)*(-off_by_one), (None,)*off_by_one


def _split_signature(signature: str) -> Tuple[Tuple[str, ...], ...]:
    """Convert text signature into tuples of axes size names

    Parameters
    ----------
    signature : str
        Text signature of matrix operation, without optional axes or spaces,

    Returns
    -------
    axes_sizes : Tuple[Tuple[str, ...], ...]
        Tuples of core axes sizes as string variablr names,
        e.g. `(('a','b'),('b','c'),('a','c'))`
    """
    if '->' in signature:
        inputs, outputs = signature.split('->')
        return _split_signature(inputs), _split_signature(outputs)
    signature = signature.lstrip('(').rstrip(')').replace('->', ',')
    arrays = []
    for array in signature.split('),('):
        if array:
            arrays.append(tuple(array.split(',')))
        else:
            arrays.append(())
    return tuple(arrays)


def return_shape(signature: str,
                 *shapes: Tuple[int, ...]) -> OneOrTuple[Tuple[int, ...]]:
    """Shape of result of broadcasted matrix operation

    Parameters
    ----------
    signature : Tuple[str]
        Signature of the operation, without optional axes or spaces,
        e.g. `'(a,b),(b,c)->(a,c)'`
    shapes : Tuple[int, ...]
        Shapes of arguments of matrix operation.

    Returns
    -------
    output_shape : Tuple[int]
        Shape of result of broadcasted matrix operation.

    Raises
    ------
    ValueError
        If `arrays.shape`s do not match signatures.
    """
    msg = (f'dimensions: Shape: {shapes}. Signature: {signature}.')
    sigs_in, sigs_out = _split_signature(signature)
    dims = [len(sig) for sig in sigs_in]
    broads, cores, sizes = [], [], {}
    if any(len(shape) < dim for shape, dim in zip(shapes, dims)):
        raise ValueError('Core array does not have enough ' + msg)
    for shape, dim in zip(shapes, dims):
        if dim:
            broads.append(shape[:-dim])
            cores.append(shape[-dim:])
        else:
            broads.append(shape)
            cores.append(())
    for sig, core in zip(sigs_in, cores):
        for name, siz in zip(sig, core):
            sizes.setdefault(name, siz)
            if sizes[name] != siz:
                raise ValueError('Array mismatch in its core ' + msg)
    broad_out = np.broadcast(*(np.empty(broad) for broad in broads)).shape
    shapes_out = []
    for sig in sigs_out:
        core = [sizes[name] for name in sig]
        shapes_out.append(broad_out + tuple(core))
    return shapes_out[0] if len(shapes_out) == 1 else tuple(shapes_out)


def array_return_shape(signature: str,
                       *arrays: np.ndarray) -> OneOrTuple[Tuple[int, ...]]:
    """Shape of result of broadcasted matrix operation

    Parameters
    ----------
    signature : str
        Signature of the operation, without optional axes or spaces,
        e.g. `'(a,b),(b,c)->(a,c)'`
    arrays : np.ndarray
        Arguments of matrix operation.

    Returns
    -------
    output_shape : Tuple[int] or Tuple[Tuple[int, ...], ...]
        Shape of result of broadcasted matrix operation.

    Raises
    ------
    ValueError
        If `arrays.shape`s do not match signatures.
    """
    return return_shape(signature, *(array.shape for array in arrays))
