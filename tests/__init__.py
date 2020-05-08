# -*- coding: utf-8 -*-
"""Customise test loading
"""
import os
from numpy_linalg.testing.unittest_tweaks import load_tests_helper
from . import (test_gufunc, test_gusolve, test_gulstsq, test_gulqr,
               test_linalg, test_lnarray)

assert any((True, test_gufunc, test_gusolve, test_gulstsq, test_gulqr,
            test_linalg, test_lnarray))


def load_tests(loader, standard_tests, pattern):
    """Change the loader so that tests are in the original order
    """
    this_dir = os.path.dirname(__file__)
    return load_tests_helper(this_dir, loader, standard_tests, pattern)
