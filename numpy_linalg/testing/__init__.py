"""Tools for writing unit tests.

main
    Runs unit tests when called. By default it does not sort the tests.
load_tests_helper
    Implements the `load_tests` protocol to load tests in order of definition.
TestCaseNumpy
    A subclass of `unittest.TestCase` with methods for testing arrays.
    It can be used as a base class for your test cases.
unittest_tweaks
    Module with classes related to `main` and `load_tests_helper`.
unittest_numpy
    Module with tools for testing arrays.
hypothesis_numpy
    Module with tools for `hypothesis` to generate examples for unit tests.
"""
from . import unittest_tweaks, unittest_numpy, hypothesis_numpy
from .unittest_tweaks import main, load_tests_helper
from .unittest_numpy import TestCaseNumpy

assert any((True, unittest_tweaks, unittest_numpy, hypothesis_numpy))
assert any((True, main, load_tests_helper, TestCaseNumpy))
