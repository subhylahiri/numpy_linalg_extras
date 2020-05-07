# -*- coding: utf-8 -*-
"""Customised unittest for numpy
"""
import unittest as _ut
import functools as _ft
import contextlib as _cx
from typing import Tuple
from fnmatch import fnmatchcase
import numpy as np
# pylint: disable=invalid-name
__all__ = [
    'TestCaseNumpy',
    'miss_str',
    'loop_test',
    'asa',
    'randn_asa',
    'zeros_asa',
    'ones_asa',
    'broadcast_err',
    'core_dim_err',
    'num_dim_err',
    'invalid_err',
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
        # Variables used in testing:
        self.varnames = []
        # Scalar types:
        # can be extended in Subclass: assign before calling super().setUp()
        extra_sctypes = getattr(self, 'sctype', [])
        self.sctype = ['f', 'd', 'F', 'D'] + extra_sctypes
        # testing ndarray values (relative to np.float64's eps)
        self.all_close_opts = {'atol': 1e-10, 'rtol': 1e-10, 'equal_nan': False}
        self.addTypeEqualityFunc(np.ndarray, self.assertArrayAllClose)

    def pick_var_type(self, sctype):
        """Set scalar types of member variables.

        If `self.varnames` is `['a', 'b', ...]`, it sets `self.a, self.b, ...`
        to `self._a[sctype], self._b[sctype], ...`.
        """
        for var in self.varnames:
            setattr(self, var, getattr(self, '_' + var)[sctype])

    def assertArrayAllClose(self, actual, desired, msg=None):
        """Calls numpy.allclose and processes the results like a
        unittest.TestCase method.

        It broadcasts, unlike numpy.testing.assert_allclose.
        It adjusts the tolerances according to the `desired.dtype`'s eps.
        The original tolerances are for `np.float64`.
        """
        # __unittest = True
        opts = self.all_close_opts.copy()
        if np.issubdtype(actual.dtype, np.inexact):
            epsratio = np.finfo(actual.dtype).eps / np.finfo(np.float64).eps
            opts['rtol'] *= epsratio
            opts['atol'] *= epsratio
        if not np.allclose(actual, desired, **opts):
            msg = '' if msg is None else f'{msg}\n'
            msg += miss_str(actual, desired, **opts)
            self.fail(msg)

    def assertArrayEqual(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.equal(...)) and processes the results like a
        unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual != desired):
            self.fail(msg)

    def assertArrayLess(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.less(...)) and processes the results like a
        unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual >= desired):
            self.fail(msg)

    def assertArrayGreater(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.greater(...)) and processes the results like a
        unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual <= desired):
            self.fail(msg)

    def assertArrayNotAllClose(self, actual, desired, msg=None):
        """Calls numpy.allclose (so it broadcasts, unlike
        numpy.testing.assert_allclose), negates and processes the results like
        a unittest.TestCase method.
        """
        # __unittest = True
        if np.allclose(actual, desired, **self.all_close_opts):
            if msg is None:
                msg = ''
            msg += miss_str(actual, desired, **self.all_close_opts)
            msg.replace("Should be", "Shouldn't be")
            self.fail(msg)

    def assertArrayNotEqual(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.not_equal(...)) and processes the results like
        a unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual == desired):
            self.fail(msg)

    def assertArrayNotLess(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.greater_equal(...)) and processes the results
        like a unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual < desired):
            self.fail(msg)

    def assertArrayNotGreater(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.less_equal(...)) and processes the results
        like a unittest.TestCase method.
        """
        # __unittest = True
        if np.any(actual > desired):
            self.fail(msg)

    def assertArrayShape(self, array, shape, msg=None):
        """Calls self.assertEqual(array.shape, shape).
        """
        # __unittest = True
        self.assertEqual(array.shape, shape, msg)

    def assertArrayShapesAre(self, arrays, shapes, msg=None):
        """Calls self.assertEqual(array.shape, shape).
        """
        # __unittest = True
        for array, shape in zip(arrays, shapes):
            self.assertEqual(array.shape, shape, msg)


# =============================================================================
# Helpers for TestCaseNumpy methods
# =============================================================================


def miss_str(left, right, atol=1e-8, rtol=1e-5, equal_nan=True):
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
        'Should be zero: <maximum devation>
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
    a_ind = np.unravel_index(argmax(mismatch), mismatch.shape)
    r_ind = np.unravel_index(argmax(mis_frac), mis_frac.shape)

    a_worst, r_worst = mismatch[a_ind], mismatch[r_ind]
    thresh, mis_frac = thresh[r_ind], mis_frac[r_ind]

    return f"""Should be zero: {a_worst:.2g} at {a_ind},
    or: {r_worst:.2g} = {thresh:.2g} * 1e{mis_frac:.1f} at {r_ind}."""


# =============================================================================
# Helpers for shape checks
# =============================================================================


def trnsp(shape):
    """Shape -> shape of transposed array"""
    return shape[:-2] + shape[:-3:-1]


def drop(shape, axis=-2):
    """Shape -> shape with one axis dropped"""
    return shape[:axis] + shape[axis+1:]


def chop(shape, axis=-1):
    """Shape -> shape with last axes reduced to square"""
    return shape[:-2] + (min(shape[-2:]),) * 2


def grow(shape, axis=-1):
    """Shape -> shape with last axes expanded to square"""
    return shape[:-2] + (max(shape[-2:]),) * 2
