"""Some tweaks to the unittest module from the standard library

This module changes the order in which tests are run, defaulting to code order.
It also allows you to halt the tracebacks from failed tests right before any of
your functions. This is useful if you write your own `assert...` methods.

If you import `main` from here, and use in place of `unittest.main`, everything
else follows. If you want to customise test discovery, you need tu use the `load_tests` protocol. You can do this by including the following function:

```
def load_tests(loader, standard_tests, pattern):
    \"\"\"Change the loader so that tests are in the original order
    \"\"\"
    this_dir = os.path.dirname(__file__)
    load_tests_helper(this_dir, loader, standard_tests, patter)
```

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
main
    An alias for `TestProgramNoSort`.
load_tests_helper
    Implements the `load_tests` protocol.
nosortTestLoader : NosortTestLoader
    The instance that is the default for `TestProgramNoSort`.
dir_nosort
    Similar to the built in function `dir`, except entries appear in the order
    they were added to the class/module dictionary.
"""
import unittest as _ut
import functools as _ft
from typing import Tuple, List, Dict, Optional, Type
from types import TracebackType, ModuleType
from fnmatch import fnmatchcase
Error = Tuple[Type[Exception], Exception, TracebackType]
__all__ = [
    'NosortTestLoader',
    'TestResultStopTB',
    'TestProgramNoSort',
    'nosortTestLoader',
    'main',
    'dir_nosort',
    'load_tests_helper',
]
# =============================================================================
# Customise test & traceback display
# =============================================================================
__unittest = True


def _dir_dict(cls: type) -> Dict[str, bool]:
    my_dir = {}
    for base in cls.__bases__:
        my_dir.update(_dir_dict(base))
    # could use cls.__dir__() or dir(cls)
    my_dir.update((k, True) for k in cls.__dict__)
    return my_dir


def dir_nosort(cls: type) -> List[str]:
    """Get the unsorted directory of a class (like dir())
    """
    return list(_dir_dict(cls))


class NosortTestLoader(_ut.TestLoader):
    """Test loader that does not sort test methods by default

    Use in place of `unittest.TestLoader` or `unittest.defaultTestLoader`.

    This loader leaves the test methods in the order they appear in the
    `TestCase` class dictionary (unless you change the `sortTestMethodsUsing`
    attribute of the `TestLoader` class). Normally, this would be the order
    they appear in the code.

    If a module has an `__all__` attribute, only the listed `TestCase`s will be
    used, in the order they appear there.

    See Also
    --------
    `unittest.TestLoader`
    """
    sortTestMethodsUsing = None

    def getTestCaseNames(self, testCaseClass: Type[_ut.TestCase]) -> List[str]:
        """Return an unsorted sequence of method names found in testCaseClass
        """
        def should_include_method(attrname: str) -> bool:
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

    def loadTestsFromModule(self,
                            module: ModuleType,
                            *args,
                            pattern: Optional[List[str]] = None,
                            **kws) -> _ut.TestSuite:
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
                obj = getattr(module, name, None)
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

    def addSubTest(self, test: _ut.TestCase, subtest: _ut.TestCase,
                   err: Optional[Error]):
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

    def _is_relevant_tb_level(self, tb: TracebackType) -> bool:
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
    `TextTestRunner` by default.

     The loader leaves the test methods in the order they appear in the code
    (unless you change the `sortTestMethodsUsing`attribute of `TestLoader`)

    It allows you to choose the `TestResult class by passing it as an argument,
    rather than subclassing `TextTestRunner`. Bu default: `TestResultStopTB`,
    unless you pass a `TestRunner` as a positional argument.

    You can stop traceback display at any particular point by writing
    ``__unittest = True``. This can be done at the function level or at the
    module level. If ``__unittest = True`` appears at the module level, it can
    be overridden in specific functions by writing ``__unittest = False``.
    """
    def __init__(self, *args,
                 resultclass: Type[_ut.TestResult] = TestResultStopTB,
                 **kwds):
        if len(args) < 4:
            kwds.setdefault('testRunner', _ut.TextTestRunner)
            if resultclass is not None:
                kwds['testRunner'].resultclass = resultclass
        if len(args) < 5:
            kwds.setdefault('testLoader', nosortTestLoader)
        super().__init__(*args, **kwds)


main = TestProgramNoSort


def load_tests_helper(this_dir: str,
                      loader: _ut.TestLoader,
                      standard_tests: _ut.TestSuite,
                      pattern: Optional[List[str]]) -> _ut.TestSuite:
    """Change the loader so that tests are in the original order.

    This implements the `load_tests` protocol. It uses `NosortTestLoader`,
    so the test methods are run in code order. You can use it like this:
    ```
    def load_tests(loader, standard_tests, pattern):
        \"\"\"Change the loader so that tests are in the original order
        \"\"\"
        this_dir = os.path.dirname(__file__)
        return load_tests_helper(this_dir, loader, standard_tests, pattern)
    ```
    """
    nosortTestLoader.copy_from(loader)
    return nosortTestLoader.discover(start_dir=this_dir, pattern=pattern)
