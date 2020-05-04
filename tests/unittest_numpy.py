# -*- coding: utf-8 -*-
"""Customised unittest for numpy
"""
import unittest as _ut
import functools as _ft
import contextlib as _cx
from typing import Tuple
from fnmatch import fnmatchcase
import numpy as np
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyn
# pylint: disable=invalid-name
__all__ = [
        'TestCaseNumpy',
        'NosortTestLoader',
        'TestResultStopTB',
        'TestProgramNoSort',
        'nosortTestLoader',
        'main',
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
        'complex_numbers',
        'numeric_dtypes',
        'signature_shapes',
        'broadcastable',
        'constant',
        'non_singular',
        'all_non_singular',
        'core_only',
        ]
# =============================================================================
# Error specs for assertRaisesRegex
# =============================================================================
broadcast_err = (ValueError, 'operands could not be broadcast')
core_dim_err = (ValueError, 'mismatch in its core dimension')
num_dim_err = (ValueError, 'does not have enough dimensions')
invalid_err = (FloatingPointError, 'invalid value encountered')

# =============================================================================
# Customise test & traceback display
# =============================================================================
__unittest = True


def _dir_dict(cls):
    my_dir = {}
    for base in cls.__bases__:
        my_dir.update(_dir_dict(base))
    # could use cls.__dir__() or dir(cls)
    my_dir.update((k, True) for k in cls.__dict__)
    return my_dir


def dir_nosort(cls: type):
    """Get the unsorted directory of a class (like dir())
    """
    return list(_dir_dict(cls))


class NosortTestLoader(_ut.TestLoader):
    """Test loader that does not sort test methods by default

    Use in place of `unittest.TestLoader` or `unittest.defaultTestLoader`.
    """
    sortTestMethodsUsing = None

    def getTestCaseNames(self, testCaseClass):
        """Return an unsorted sequence of method names found in testCaseClass
        """
        def should_include_method(attrname):
            if not attrname.startswith(self.testMethodPrefix):
                return False
            test_func = getattr(testCaseClass, attrname)
            if not callable(test_func):
                return False
            if self.testNamePatterns is None:
                return True
            full_name = f'{testCaseClass.__module__}.{test_func.__qualname__}'
            return any(fnmatchcase(full_name, pattern)
                       for pattern in self.testNamePatterns)
        test_fn_names = dir_nosort(testCaseClass)
        test_fn_names = list(filter(should_include_method, test_fn_names))
        if self.sortTestMethodsUsing:
            key_fn = _ft.cmp_to_key(self.sortTestMethodsUsing)
            test_fn_names.sort(key=key_fn)
        return test_fn_names

    def loadTestsFromModule(self, module, *args, pattern=None, **kws):
        """Return a suite of all test cases contained in the given module

        If module has an `__all__` attribute but no `load_tests` function,
        `TesCase`s will be loaded in the order they appear there.

        Extends `unittest.TestLoader.loadTestsFromModule`.
        """
        tests = super().loadTestsFromModule(self, module, *args,
                                            pattern=pattern, **kws)
        all_names = getattr(module, '__all__', None)
        if all_names is not None and not hasattr(module, 'load_tests'):
            tests = []
            for name in all_names:
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, _ut.TestCase):
                    tests.append(self.loadTestsFromTestCase(obj))
            tests = self.suiteClass(tests)
        return tests

    def copy_from(self, other: _ut.TestLoader):
        """Copy instance attributes from other loader"""
        # Ugh
        self._loading_packages = other._loading_packages


nosortTestLoader = NosortTestLoader()


class TestResultStopTB(_ut.TextTestResult):
    """TestResult that does not print beyond certain frames in tracebacks

    Use in place of `unittest.TextTestResult`.

    You can stop traceback display at any particular point by writing
    ``__unittest = True``. This can be done at the function level or at the
    module level. If ``__unittest = True`` appears at the module level, it can
    be overridden in specific functions by writing ``__unittest = False``.
    This relies on undocumented private internals of `unittest`, so it could
    stop working with a future update of `unittest`.

    It checks if there is a variable name ending with `__unittest` in the frame
    and if that variable evaluates as True. Only the last variable satisfying
    the first criterion is tested for the second, with locals added after
    globals and otherwise appearing in the order they were added to the dicts.
    """
    # names of variables that tell us if this traceback level should be dropped
    stoppers: list = ["__unittest", "the_error_hypothesis_found"]

    def addSubTest(self, test, subtest, err):
        """Called at the end of a subtest.
        'err' is None if the subtest ended successfully, otherwise it's a
        tuple of values as returned by sys.exc_info().
        """
        super().addSubTest(test, subtest, err)
        if err is not None:
            if issubclass(err[0], test.failureException):
                msg = "FAIL"
            else:
                msg = "ERROR"
            if self.showAll:
                self.stream.writeln(msg)
            elif self.dots:
                self.stream.write(msg[0])
                self.stream.flush()

    def _is_relevant_tb_level(self, tb):
        """Should this level of traceback be dropped from message?"""
        f_vars = tb.tb_frame.f_globals.copy()
        f_vars.update(tb.tb_frame.f_locals)  # locals after/overwrite globals
        flags = [bool(v) for k, v in f_vars.items()
                 if any(k.endswith(stop) for stop in self.stoppers)]
        # locals have precedence over globals, so we look at last element.
        # any matches & is the last match's frame variable True?
        return flags and flags[-1]
        # # This fails because key in f_locals is '_TestCase????__unittest':
        # return '__unittest' in f_vars and f_vars['__unittest']


class TestProgramNoSort(_ut.TestProgram):
    """Run tests in order without printing certain frames in tracebacks.

    Use in place of `unittest.main`. It uses `nosortTestLoader` and
    `TextTestRunner` by default. It allows you to choose the `TestResult class
    by passing it as an argument, rather than subclassing `TextTestRunner`.
    The default is `TestResultStopTB`.

    You can stop traceback display at any particular point by writing
    ``__unittest = True``. This can be done at the function level or at the
    module level. If ``__unittest = True`` appears at the module level, it can
    be overridden in specific functions by writing ``__unittest = False``.

    Checks if there is a variable name ending with `__unittest` in the frame
    and if that variable evaluates as True. Only the last variable satisfying
    the first criterion is tested for the second, with locals appearing after
    globals and otherwise appearing in the order they were added to the dicts.
    """
    def __init__(self, *args, resultclass=TestResultStopTB, **kwds):
        kwds.setdefault('testLoader', nosortTestLoader)
        kwds.setdefault('testRunner', _ut.TextTestRunner)
        super().__init__(*args, **kwds)
        if resultclass is not None:
            self.testRunner.resultclass = resultclass


main = TestProgramNoSort

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
            if msg is None:
                msg = miss_str(actual, desired, **opts)
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
# Strategies for Hypothesis generated test examples
# =============================================================================


def _extract_kwds(kwds: dict, **defaults) -> dict:
    """Take keys in defaults and pop from kwds, return as a dict"""
    extracted = {}
    for key, value in defaults.items():
        extracted[key] = kwds.pop(key, value)
    return extracted


def complex_numbers(**kwds) -> st.SearchStrategy[complex]:
    """Strategy to generate complex numbers of specified width

    Takes any keyword arguments for `hypothesis.strategies.floats`

    Returns
    -------
    complex_strategy : st.SearchStrategy[complex]
        Strategy for complex numbers that applies float options to real and
        imaginary parts.
    """
    if 'width' in kwds:
        kwds['width'] //= 2
    return st.builds(complex, st.floats(**kwds), st.floats(**kwds))


_DTYPES = {
    'f': (np.float32, st.floats),
    'd': (np.float64, st.floats),
    'F': (np.complex64, complex_numbers),
    'D': (np.complex128, complex_numbers),
}


@st.composite
def numeric_dtypes(draw, code_st=None, **kwds):
    """Strategy to generate dtypes codes

    Parameters
    ----------
    code_st : None|str|Sequence[str]|SearchStrategy[str], optional
        Strategy for dtype-code of numbers: a choice, or a list to choose from,
        or `None` to choose from {'f','d','F','D'} or a custom strategy.
        By default: `None`.
    Also takes any keyword arguments for `hypothesis.strategies.floats`.

    Returns
    -------
    dtype_strategy : st.SearchStrategy[np.dtype]
        Strategy for dtypes that are recognised by BLAS/LAPACK.
    elements_strategy : st.SearchStrategy[Number]
        Strategy for numbers of that dtype.
    """
    opts = {'min_value': -1e10, 'max_value': 1e10, 'allow_infinity': False,
            'allow_nan': False, 'exclude_min': False, 'exclude_max': False}
    opts.update(kwds)
    if code_st is None:
        code_st = st.sampled_from(['f', 'd', 'F', 'D'])
    elif isinstance(code_st, str):
        code_st = st.just(code_st)
    elif isinstance(code_st, (list, tuple)):
        code_st = st.sampled_from(code_st)
    code = draw(code_st)
    dtype, element_st = _DTYPES[code]
    opts['width'] = dtype().itemsize * 8
    return dtype, element_st(**opts)


@st.composite
def signature_shapes(draw, signature: str,
                     **kwds) -> st.SearchStrategy[Tuple[Tuple[int, ...], ...]]:
    """Create a hypothesis strategy for a tuple of shapes with the signature

    Parameters
    ----------
    signature : str
        Signature of array core dimension, without the return
    Also takes any keyword arguments (excluding `num_shapes`) for
    `hypothesis.extra.numpy.mutually_broadcastable_shapes`.

    Returns
    -------
    shape_strategy : st.SearchStrategy[Tuple[Tuple[int, ...], ...]]
        strategy to produce a tuple of tuples of ints that broadcast with the
        given core dimension signature.
    """
    opts = {'signature': signature + '->()', 'base_shape': (),
            'min_dims': 0, 'max_dims': None, 'min_side': 1, max_side: None}
    opts.update(kwds)
    return draw(hyn.mutually_broadcastable_shapes(**opts)).input_shapes


@st.composite
def _arrays_args(draw, signature, code_st, kwds):
    """Generate inputs for hyn.arrays strategy
    """
    num_opts = _extract_kwds(kwds, min_value=-1e10, max_value=1e10,
                             allow_nan=False, allow_infinity=False,
                             exclude_min=False, exclude_max=False)
    dtype, elements = draw(numeric_dtypes(code_st, **num_opts))
    shape_opts = _extract_kwds(kwds, base_shape=(), min_dims=0, max_dims=None,
                               min_side=1, max_side=None)
    shapes = draw(signature_shapes(signature, **shape_opts))
    return dtype, shapes, elements


@st.composite
def broadcastable(draw, signature: str, code_st=None,
                  **kwds) -> st.SearchStrategy[Tuple[np.ndarray, ...]]:
    """Create a hypothesis strategy for a tuple of arrays with the signature

    Parameters
    ----------
    signature : str
        Signature of array core dimension, without the return
    code_st : None|str|Sequence[str]|SearchStrategy[str], optional
        Strategy for dtype code of numbers: a choice, or a list to choose from,
        or `None` to choose from {'f','d','F','D'} or a custom strategy.
        By default: `None`.
    Also takes any keyword arguments for `hypothesis.strategies.floats` or
    `hypothesis.extra.numpy.mutually_broadcastable_shapes`, except `num_shapes`.

    Returns
    -------
    strategy : st.SearchStrategy[Tuple[np.ndarray, ...]]
        Strategy to produce a tuple of arrays that broadcast with the given
        core dimension signature.
    """
    dtype, shapes, elements = draw(_arrays_args(signature, code_st, kwds))
    kwds.update(dtype=dtype, elements=elements, fill=st.nothing())
    result = tuple(draw(hyn.arrays(shape=shape, **kwds)) for shape in shapes)
    return result[0] if len(result) == 1 else result


@st.composite
def constant(draw, signature: str, code_st=None,
             **kwds) -> st.SearchStrategy[np.ndarray]:
    """Create a hypothesis strategy for a constant array with the signature

    Parameters
    ----------
    signature : str
        Signature of array core dimension, without the return
    code_st : None|str|Sequence[str]|SearchStrategy[str], optional
        Strategy for dtype code of numbers: a choice, or a list to choose from,
        or `None` to choose from {'f','d','F','D'} or a custom strategy.
        By default: `None`.
    Also takes any keyword arguments for `hypothesis.strategies.floats` or
    `hypothesis.extra.numpy.mutually_broadcastable_shapes`, except `num_shapes`.

    Returns
    -------
    strategy : st.SearchStrategy[np.ndarray]
        Strategy to produce an array that broadcasts with the given core
        dimension signature, with a constant value of thet dtype.
    """
    dtype, shapes, elements = draw(_arrays_args(signature, code_st, kwds))
    fill = draw(elements)
    return np.full(shapes[0], fill, dtype)


# =============================================================================
# Helpers for TestCaseNumpy with Hypothesis
# =============================================================================


def non_singular(matrix: np.ndarray) -> np.ndarray:
    """Check that matrix/matrices are non-singular

    Parameters
    ----------
    matrix : np.ndarray
        square matrix/array of square matrices whose determinant to check.

    Returns
    -------
    is_not_singular : np.ndarray
        bool/array of bools that are True if the matrix is non-singular.
    """
    return np.abs(np.linalg.slogdet(matrix)[1]) < 500


def all_non_singular(*matrices: np.ndarray) -> bool:
    """Check that matrices are non-singular

    Parameters
    ----------
    matrices : np.ndarray
        square matrices whose determinant to check.

    Returns
    -------
    are_not_singular : bool
        True if all of the the matrices are non-singular.
    """
    return all(np.all(non_singular(mat)) for mat in matrices)


def core_only(*arrays: np.ndarray, dims: int = 2) -> np.ndarray:
    """Strip all non-core dimensions from arrays

    Parameters
    ----------
    arrays : np.ndarray
        Arrays to remove dimensions from.
    dims : int, optional
        Number of core dimensions to leave, by default 2.

    Returns
    -------
    stripped : np.ndarray
        Arrays with only core dimensions left.
    """
    result = tuple(arr[(0,) * (arr.ndim - dims)] for arr in arrays)
    return result[0] if len(result) == 1 else result


# =============================================================================
# Helpers for TestCaseNumpy methods
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
        @_ft.wraps(func)
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
                    # opts = {name: val for name, val in zip(attr_name, vals)}
                    opts = dict(zip(attr_name, vals))
                    with self.subTest(msg=msg, **opts):
                        func(self, *args, **opts, **kwds)
        return loop_func
    return loop_dec


CMPLX = {'b': 0, 'h': 0, 'i': 0, 'l': 0, 'p': 0, 'q': 0,
         'f': 0, 'd': 0, 'g': 0, 'F': 1j, 'D': 1j, 'G': 1j}


def asa(left, right, sctype):
    """Convert left + iy to sctype

    Parameters
    ----------
    left,right: ndarray[float]
        real & imaginary parts, must broadcast
    sctype
        a numpy scalar type code, e.g. 'f,d,g,F,D,G'
    """
    imag = CMPLX.get(sctype, 0)
    return (left + imag * right).astype(sctype)


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
