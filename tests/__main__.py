"""Main entry point"""
import sys
import os.path
from numpy_linalg.testing import main

this_dir = os.path.dirname(__file__)
executable = os.path.basename(sys.executable)
argv = [executable + ' -m tests', 'discover', '-s', this_dir] + sys.argv[1:]

main(module=None, argv=argv)
