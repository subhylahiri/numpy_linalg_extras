# -*- coding: utf-8 -*-
"""Customise test loading
"""
import os
from .unittest_numpy import nosortTestLoader as nosort_loader


def load_tests(loader, standard_tests, pattern):
    """Change the loader so that tests are in the original order
    """
    # top level directory cached on loader instance
    this_dir = os.path.dirname(__file__)
    nosort_loader.copy_from(loader)
    package_tests = nosort_loader.discover(start_dir=this_dir, pattern=pattern)
    return package_tests
