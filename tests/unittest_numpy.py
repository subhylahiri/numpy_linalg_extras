# -*- coding: utf-8 -*-
"""Customised unittest for numpy
"""
import unittest
import contextlib
import functools
from fnmatch import fnmatchcase
import numpy as np

__all__ = [
        'TestCaseNumpy',
        'NosortTestLoader',
        'TestResultStopTB',
        'TestRunnerStopTB',
        'main',
        'loop_test',
        'miss_str',
        'asa',
        'randn_asa',
        'zeros_asa',
        'ones_asa',
        'errstate',
        'unique_unsorted',
        'broadcast_err',
        'core_dim_err',
        'num_dim_err',
        'invalid_err',
        ]
# =============================================================================
# %% Error specs for assertRaisesRegex
# =============================================================================
broadcast_err = (ValueError, 'operands could not be broadcast')
core_dim_err = (ValueError, 'mismatch in its core dimension')
num_dim_err = (ValueError, 'does not have enough dimensions')
invalid_err = (FloatingPointError, 'invalid value encountered')

# =============================================================================
# %% Customise test & traceback display
# =============================================================================
__unittest = True


class NosortTestLoader(unittest.TestLoader):
    """Test loader that does not sort test methods by default

    Use in place of `unittest.TestLoader` or `unittest.defaultTestLoader`.
    """
    sortTestMethodsUsing = None

    def getTestCaseNames(self, testCaseClass):
        """Return an unsorted sequence of method names found in testCaseClass
        """
        def shouldIncludeMethod(attrname):
            if not attrname.startswith(self.testMethodPrefix):
                return False
            testFunc = getattr(testCaseClass, attrname)
            if not callable(testFunc):
                return False
            if self.testNamePatterns is None:
                return True
            fullName = f'{testCaseClass.__module__}.{testFunc.__qualname__}'
            return any(fnmatchcase(fullName, pattern)
                       for pattern in self.testNamePatterns)
        try:
            testFnNames = testCaseClass.get_names()
        except AttributeError:
            testFnNames = dir(testCaseClass)
        testFnNames = list(filter(shouldIncludeMethod, testFnNames))
        if self.sortTestMethodsUsing:
            key_fn = functools.cmp_to_key(self.sortTestMethodsUsing)
            testFnNames.sort(key=key_fn)
        return testFnNames

    def loadTestsFromModule(self, module, *args, pattern=None, **kws):
        """Return a suite of all test cases contained in the given module

        If module has an `__all__` attribute but no `load_tests` function,
        `TesCase`s will be loaded in the order they appear there.
        """
        tests = super().loadTestsFromModule(self, module, *args, pattern=None,
                                            **kws)
        all_names = getattr(module, '__all__', None)
        if all_names is not None and not hasattr(module, 'load_tests'):
            tests = []
            for name in all_names:
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, unittest.TestCase):
                    tests.append(self.loadTestsFromTestCase(obj))
            tests = self.suiteClass(tests)
        return tests

    def copy_from(self, other):
        """Copy instance attributes from other loader"""
        self._loading_packages = other._loading_packages


nosortTestLoader = NosortTestLoader()


class TestResultStopTB(unittest.TextTestResult):
    """TestResult that does not print beyond certain frames in tracebacks

    Use in place of `unittest.TextTestResult`.

    You can stop traceback display at any particular point by writing
    ``__unittest = True``. This can be done at the function level or at the
    module level. If ``__unittest = True`` appears at the module level, it can
    be overridden in specific functions by writing ``__unittest = False``.

    Checks if there is a variable name ending with `__unittest` in the frame
    and if that variable evaluates as True. Only the last variable satisfying
    the first criterion is tested for the second, with locals added after
    globals and otherwise appearing in the order they were added to the dicts.
    """

    def addSubTest(self, test, subtest, err):
        """Called at the end of a subtest.
        'err' is None if the subtest ended successfully, otherwise it's a
        tuple of values as returned by sys.exc_info().
        """
        super().addSubTest(test, subtest, err)
        if err is not None:
            if issubclass(err[0], test.failureException):
                super().addFailure(test, err)
                del self.failures[-1]
            else:
                super().addError(test, err)
                del self.errors[-1]

    def _is_relevant_tb_level(self, tb):
        f_vars = tb.tb_frame.f_globals.copy()
        f_vars.update(tb.tb_frame.f_locals)  # locals after/overwrite globals
        flags = [v for k, v in f_vars.items() if k.endswith('__unittest')]
        # locals have precedence over globals, so we look at last element.
        # any matches & is the last match's frame variable True?
        return flags and flags[-1]
        # This would fail because key in f_locals is '_TestCase????__unittest':
        # return '__unittest' in f_vars and f_vars['__unittest']


class TestRunnerStopTB(unittest.TextTestRunner):
    """TestRunner that does not print certain frames in tracebacks

    Use in place of `unittest.TextTestRunner`. It uses `TestResultStopTB` by
    default.

    You can stop traceback display at any particular point by writing
    ``__unittest = True``. This can be done at the function level or at the
    module level. If ``__unittest = True`` appears at the module level, it can
    be overridden in specific functions by writing ``__unittest = False``.

    Checks if there is a variable name ending with `__unittest` in the frame
    and if that variable evaluates as True. Only the last variable satisfying
    the first criterion is tested for the second, with locals appearing after
    globals and otherwise appearing in the order they were added to the dicts.
    """

    def __init__(self, resultclass=None, **kwargs):
        if resultclass is None:
            resultclass = TestResultStopTB
        super().__init__(resultclass=resultclass, **kwargs)


def main(testLoader=nosortTestLoader, testRunner=None, **kwds):
    """Run tests in order without printing certain frames in tracebacks.

    Use in place of `unittest.main`. It uses `nosortTestLoader` and
    `TestRunnerStopTB` by default.

    You can stop traceback display at any particular point by writing
    ``__unittest = True``. This can be done at the function level or at the
    module level. If ``__unittest = True`` appears at the module level, it can
    be overridden in specific functions by writing ``__unittest = False``.

    Checks if there is a variable name ending with `__unittest` in the frame
    and if that variable evaluates as True. Only the last variable satisfying
    the first criterion is tested for the second, with locals appearing after
    globals and otherwise appearing in the order they were added to the dicts.
    """
    if testRunner is None:
        testRunner = TestRunnerStopTB
    unittest.main(testLoader=testLoader, testRunner=testRunner, **kwds)


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
        # Variables used in testing:
        self.varnames = []
        # Scalar types:
        # can be extended in Subclass: assign before calling super().setUp()
        extra_sctypes = getattr(self, 'sctype', [])
        self.sctype = ['f', 'd', 'F', 'D'] + extra_sctypes
        # testing ndarray values
        self.all_close_opts = {'atol': 1e-5, 'rtol': 1e-5, 'equal_nan': False}
        self.addTypeEqualityFunc(np.ndarray, self.assertArrayAllClose)

    def pick_var_type(self, sctype):
        """Set scalar types of member variables.

        If `self.varnames` is `['a', 'b', ...]`, it sets `self.a, self.b, ...`
        to `self._a[sctype], self._b[sctype], ...`.
        """
        for var in self.varnames:
            setattr(self, var, getattr(self, '_' + var)[sctype])

    def assertArrayAllClose(self, actual, desired, msg=None):
        """Calls numpy.allclose (so it broadcasts, unlike
        numpy.testing.assert_allclose) and processes the results like a
        unittest.TestCase method.
        """
        # __unittest = True
        if not np.allclose(actual, desired, **self.all_close_opts):
            if msg is None:
                msg = miss_str(actual, desired, **self.all_close_opts)
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
                msg = miss_str(actual, desired, **self.all_close_opts)
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

    @classmethod
    def get_names(cls):
        my_attr = []
        for base in cls.__bases__:
            try:
                my_attr += base.get_names()
            except AttributeError:
                my_attr += dir(base)
        my_attr.extend(cls.__dict__)
        return unique_unsorted(my_attr)


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
            if isinstance(attr_name, str):
                the_attr = getattr(self, attr_name)
    #                __unittest = False
                for val in the_attr[attr_inds]:
                    opts = {attr_name: val}
                    with self.subTest(msg=msg, **opts):
                        func(self, *args, **opts, **kwds)
            else:
                the_attr = [getattr(self, nam)[attr_inds] for nam in attr_name]
                for vals in zip(*the_attr):
                    opts = {name: val for name, val in zip(attr_name, vals)}
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
    mis_frac = (np.log(mismatch) - np.log(thresh)) / np.log(10)

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
    """array of zeros with scalar type

    Parameters
    ----------
    shape: tuple[int]
        shape of arrray
    sctype
        a numpy scalar type code, e.g. 'f,d,g,F,D,G'
    """
    return asa(np.zeros(shape), np.zeros(shape), sctype)


def ones_asa(shape, sctype):
    """array of ones with scalar type

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


def unique_unsorted(sequence):
    """remove repetitions without changing the order"""
    sequence = np.asarray(sequence)
    _, inds = np.unique(sequence, return_index=True)
    inds.sort()
    return sequence[inds].tolist()
