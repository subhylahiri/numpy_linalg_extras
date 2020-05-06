"""Some tweaks to the unittest module from the standard library

Classes
-------
NosortTestLoader
    `TestLoader` that does not sort `TestCase`s or their methods by name.
TestResultStopTB
    `TestResult` that gives some control of how the traceback is displayed.
TestProgramNoSort
    `TestProgram` that uses the previous two classes by default.

Functions etc.
--------------
nosortTestLoader : NosortTestLoader
    The instance that is the default for `TestProgramNoSort`.
main
    An alias for `TestProgramNoSort`.
dir_nosort
    Similar to the built in function `dir`, except entries appear in the order
    they were added to the class/module dictionary.
"""
import unittest as _ut
import functools as _ft
import contextlib as _cx
from typing import Tuple
from fnmatch import fnmatchcase

__all__ = [
    'NosortTestLoader',
    'TestResultStopTB',
    'TestProgramNoSort',
    'nosortTestLoader',
    'main',
    'dir_nosort,'
]
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

    def getTestCaseNames(self, testCaseClass: type):
        """Return an unsorted sequence of method names found in testCaseClass
        """
        def should_include_method(attrname: str):
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
    stoppers: list = ["__unittest"]

    def addSubTest(self, test, subtest, err):
        """Called at the end of a subtest.
        'err' is None if the subtest ended successfully, otherwise it's a
        tuple of values as returned by sys.exc_info().
        """
        super().addSubTest(test, subtest, err)
        if err is not None:
            msg = ("fail" if issubclass(err[0], test.failureException)
                   else "error")
            # writes after every subTest :(
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
    The default is `TestResultStopTB`, unless you pass a `TestRunner` as a
    positional argument.

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
        if len(args) < 4:
            kwds.setdefault('testRunner', _ut.TextTestRunner)
            if resultclass is not None:
                kwds['testRunner'].resultclass = resultclass
        if len(args) < 5:
            kwds.setdefault('testLoader', nosortTestLoader)
        super().__init__(*args, **kwds)


main = TestProgramNoSort
