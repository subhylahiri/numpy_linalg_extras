# -*- coding: utf-8 -*-
"""Customise test loading
"""
import os
from .unittest_numpy import nosortTestLoader as nosort_loader
from . import (
        test_gufunc, test_gusolve, test_gulstsq, test_linalg, test_lnarray)

assert any((
        test_gufunc, test_gusolve, test_gulstsq, test_linalg, test_lnarray))


def load_tests(loader, standard_tests, pattern):
    """Change the loader so that tests are in the original order
    """
    # top level directory cached on loader instance
    this_dir = os.path.dirname(__file__)
    nosort_loader.copy_from(loader)
    package_tests = nosort_loader.discover(start_dir=this_dir, pattern=pattern)
    return package_tests
