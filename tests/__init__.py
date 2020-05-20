# -*- coding: utf-8 -*-
"""Customise test loading
"""
import os
from numpy_linalg.testing import load_tests_helper
from . import (test_gufunc, test_gulusolve, test_guqrlqpinv, test_guqrlstsq,
               test_linalg, test_lnarray, test_creation)

assert any((True, test_gufunc, test_gulusolve, test_guqrlqpinv, test_guqrlstsq,
            test_linalg, test_lnarray, test_creation))


def load_tests(loader, standard_tests, pattern):
    """Change the loader so that tests are in the original order
    """
    this_dir = os.path.dirname(__file__)
    return load_tests_helper(this_dir, loader, standard_tests, pattern)
