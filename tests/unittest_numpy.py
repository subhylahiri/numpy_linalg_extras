# -*- coding: utf-8 -*-
"""Customised unittest for numpy
"""
import unittest
import contextlib
import functools
import numpy as np
import sl_py_tools.numpy_tricks.linalg._lnarray as la

__all__ = [
        'TestCaseNumpy',
        'loop_test',
        'miss_str',
        'asa',
        'randn_asa',
        'zeros_asa',
        'ones_asa',
        'errstate',
        'broadcast_err',
        'core_dim_err',
        'invalid_err',
        ]
# =============================================================================
# %% Error specs for assertRaisesRegex
# =============================================================================
broadcast_err = (ValueError, 'operands could not be broadcast')
core_dim_err = (ValueError, 'mismatch in its core dimension')
invalid_err = (FloatingPointError, 'invalid value encountered')

# =============================================================================
# %% Trying to customise traceback display
# =============================================================================
__unittest = True


class TestResultNumpy(unittest.TextTestResult):
    def _is_relevant_tb_level(self, tb):
        f_vars = tb.tb_frame.f_globals.copy()
        f_vars.update(tb.tb_frame.f_locals)
        return '__unittest' in f_vars and f_vars['__unittest']

# =============================================================================
# %% TestCaseNumpy base class
# =============================================================================


class TestCaseNumpy(unittest.TestCase):
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
        # Scalar types:
        self.sctype = ['f', 'd', 'F', 'D']
        self.all_close_opts = {'atol': 1e-5, 'rtol': 1e-5, 'equal_nan': False}
        self.addTypeEqualityFunc(np.ndarray, self.assertArrayAllClose)
        self.addTypeEqualityFunc(la.lnarray, self.assertArrayAllClose)

#    def defaultTestResult(self):
#        return TestResultNumpy

    def assertArrayAllClose(self, actual, desired, msg=None):
        """Calls numpy.allclose (so it broadcasts, unlike
        numpy.testing.assert_allclose) and processes the results like a
        unittest.TestCase method.
        """
        if not np.allclose(actual, desired, **self.all_close_opts):
            if msg is None:
                msg = miss_str(actual, desired, **self.all_close_opts)
            self.fail(msg)

    def assertArrayEqual(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.equal(...)) and processes the results like a
        unittest.TestCase method.
        """
        if np.any(actual != desired):
            self.fail(msg)

    def assertArrayLess(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.less(...)) and processes the results like a
        unittest.TestCase method.
        """
        if np.any(actual >= desired):
            self.fail(msg)

    def assertArrayGreater(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.greater(...)) and processes the results like a
        unittest.TestCase method.
        """
        if np.any(actual <= desired):
            self.fail(msg)

    def assertArrayNotAllClose(self, actual, desired, msg=None):
        """Calls numpy.allclose (so it broadcasts, unlike
        numpy.testing.assert_allclose), negates and processes the results like
        a unittest.TestCase method.
        """
        if np.allclose(actual, desired, **self.all_close_opts):
            if msg is None:
                msg = miss_str(actual, desired, **self.all_close_opts)
            self.fail(msg)

    def assertArrayNotEqual(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.not_equal(...)) and processes the results like
        a unittest.TestCase method.
        """
        if np.any(actual == desired):
            self.fail(msg)

    def assertArrayNotLess(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.greater_equal(...)) and processes the results
        like a unittest.TestCase method.
        """
        if np.any(actual < desired):
            self.fail(msg)

    def assertArrayNotGreater(self, actual, desired, msg=None):
        """Calls numpy.all(numpy.less_equal(...)) and processes the results
        like a unittest.TestCase method.
        """
        if np.any(actual > desired):
            self.fail(msg)


# =============================================================================
# %% Helpers for TestCaseNumpy methods
# =============================================================================


def loop_test(msg=None, attr_name='sctype', attr_inds=slice(None)):
    """Return decorator to loop a test over a sequence attribute of a TestCase.

    Decorated function must take ``attr_name`` as a keyword argument.

    Note: This is not a decorator - it is a function that returns a decorator.
    Even when there are no arguments, you must call it as ``@loop_test()``.

    Parameters
    ----------
    msg: str, optional
        message to pass to ``TestCase.subTest``.
    attr_name: str, default:'sctype'
        name of iterable, indexable attribute of ``TestCase`` to loop over.
    attr_inds: int, slice, default:slice(None)
        which elements of ``TestCase.attr_name`` to loop over.
    """
    def loop_dec(func):
        @functools.wraps(func)
        def loop_func(self, *args, **kwds):
            the_attr = getattr(self, attr_name)
#                __unittest = False
            for val in the_attr[attr_inds]:
                opts = {attr_name: val}
                with self.subTest(msg=msg, **opts):
                    func(self, *args, **opts, **kwds)
        return loop_func
    return loop_dec


def miss_str(x, y, atol=1e-8, rtol=1e-5, equal_nan=True):
    """Returns a string describing the maximum deviation of x and y

    Parameters
    ----------
    x,y: ndarray[float]
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
    shape = np.broadcast(x, y).shape
    thresh = atol + rtol * np.abs(np.broadcast_to(y, shape))
    mismatch = np.abs(x - y)
    mis_frac = np.log(mismatch) - np.log(thresh)
    ind = np.unravel_index(np.argmax(mis_frac), mis_frac.shape)
    formatter = 'Should be zero: {:.2g}\nor: {:.2g} = {:.2g} * 1e{:.1f} at {}'
    if equal_nan:
        worst = np.nanmax(mismatch)
    else:
        worst = np.amax(mismatch)

    return formatter.format(worst, mismatch[ind], thresh[ind],
                            mis_frac[ind] / np.log(10), ind)


cmplx = {'b': 0, 'h': 0, 'i': 0, 'l': 0, 'p': 0, 'q': 0,
         'f': 0, 'd': 0, 'g': 0, 'F': 1j, 'D': 1j, 'G': 1j}


def asa(x, y, sctype):
    """Convert x + iy to sctype

    Parameters
    ----------
    x,y: ndarray[float]
        real & imaginary parts, must broadcast
    sctype
        a numpy scalar type code, e.g. 'f,d,g,F,D,G'
    """
    imag = cmplx.get(sctype, 0)
    return (x + imag * y).astype(sctype)


def randn_asa(shape, sctype):
    """standard normal array with scalar type

    Parameters
    ----------
    shape: tuple[int]
        shape of arrray
    sctype
        a numpy scalar type code, e.g. 'f,d,g,F,D,G'
    """
    return asa(np.random.standard_normal(shape),
               np.random.standard_normal(shape), sctype)


def zeros_asa(shape, sctype):
    """standard normal array with scalar type

    Parameters
    ----------
    shape: tuple[int]
        shape of arrray
    sctype
        a numpy scalar type code, e.g. 'f,d,g,F,D,G'
    """
    return asa(np.zeros(shape), np.zeros(shape), sctype)


def ones_asa(shape, sctype):
    """standard normal array with scalar type

    Parameters
    ----------
    shape: tuple[int]
        shape of arrray
    sctype
        a numpy scalar type code, e.g. 'f,d,g,F,D,G'
    """
    return asa(np.ones(shape), np.zeros(shape), sctype)


@contextlib.contextmanager
def errstate(*args, **kwds):
    """Context manager like np.errstate that can also be used as a decorator
    """
    call = kwds.pop('call', None)
    old_errstate = np.geterr()
    try:
        old_errstate = np.seterr(*args, **kwds)
        if call is not None:
            old_call = np.seterrcall(call)
        yield np.geterr()
    finally:
        np.seterr(**old_errstate)
        if call is not None:
            np.seterrcall(old_call)
