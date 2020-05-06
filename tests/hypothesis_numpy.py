"""Hypothesis strategies for numpy arrays, and some tools for the generated data
"""
import collections.abc
from numbers import Number
from typing import Tuple, Union, Sequence
import hypothesis.strategies as st
import hypothesis.extra.numpy as hyn
import numpy as np

__all__ = [
    'complex_numbers',
    'numeric_dtypes',
    'signature_shapes',
    'broadcastable',
    'constant',
    'core_only',
    'non_singular',
    'all_non_singular',
    'wide',
    'tall',
    'nonsquare',
]
Shape = Tuple[int, ...]
CodeStrategy = Union[None, str, Sequence[str], st.SearchStrategy[str]]
# =============================================================================
# Strategies for Hypothesis generated test examples
# =============================================================================


def _extract_kwds(kwds: dict, **defaults) -> dict:
    """Take keys in defaults and pop from kwds, return as a dict"""
    extracted = {}
    for key, value in defaults.items():
        extracted[key] = kwds.pop(key, value)
    return extracted


def _default_opts(kind: str) -> dict:
    """Get default options for dtype/shape strategies"""
    if kind == "dtype":
        return {'min_value': -1e10, 'max_value': 1e10, 'allow_infinity': False,
                'allow_nan': False, 'exclude_min': False, 'exclude_max': False}
    if kind == "shape":
        return {'min_dims': 0, 'max_dims': 6, 'min_side': 1, 'max_side': 10,
                'base_shape': ()}
    raise ValueError(f"Unknown option kind: {kind}")


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
def numeric_dtypes(draw, code_st: CodeStrategy = None,
                   **kwds) -> Tuple[np.dtype, Number]:
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
    dtype_strategy : np.dtype
        Strategy for dtypes that are recognised by BLAS/LAPACK.
    elements_strategy : Number
        Strategy for numbers of that dtype.
    """
    opts = _default_opts("dtype")
    opts.update(kwds)
    if code_st is None:
        code_st = st.sampled_from(['f', 'd', 'F', 'D'])
    elif isinstance(code_st, str):
        code_st = st.just(code_st)
    elif isinstance(code_st, collections.abc.Sequence):
        code_st = st.sampled_from(code_st)
    code = draw(code_st)
    dtype, element_st = _DTYPES[code]
    opts['width'] = dtype().itemsize * 8
    return dtype, element_st(**opts)


@st.composite
def signature_shapes(draw, signature: str, **kwds) -> Tuple[Shape, ...]:
    """Create a hypothesis strategy for a tuple of shapes with the signature

    Parameters
    ----------
    signature : str
        Signature of array core dimension, without the return
    Also takes any keyword arguments (excluding `num_shapes`) for
    `hypothesis.extra.numpy.mutually_broadcastable_shapes`.

    Returns
    -------
    shape_strategy : Tuple[Tuple[int, ...], ...]
        strategy to produce a tuple of tuples of ints that broadcast with the
        given core dimension signature.
    """
    opts = _default_opts("shape")
    opts.update(kwds)
    opts['signature'] = signature + '->()'
    return draw(hyn.mutually_broadcastable_shapes(**opts)).input_shapes


@st.composite
def _arrays_args(draw, signature: str, code_st: CodeStrategy,
                 kwds: dict) -> Tuple[np.dtype, Tuple[Shape, ...], Number]:
    """Generate inputs for hyn.arrays strategy
    """
    num_opts = _extract_kwds(kwds, **_default_opts("dtype"))
    dtype, elements = draw(numeric_dtypes(code_st, **num_opts))
    shape_opts = _extract_kwds(kwds, **_default_opts("shape"))
    shapes = draw(signature_shapes(signature, **shape_opts))
    if kwds:
        raise ValueError(f"Unknown keywords: {list(kwds)}")
    return dtype, shapes, elements


@st.composite
def broadcastable(draw, signature: str, code_st: CodeStrategy = None,
                  **kwds) -> Tuple[np.ndarray, ...]:
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
    strategy : Tuple[np.ndarray, ...]
        Strategy to produce a tuple of arrays that broadcast with the given
        core dimension signature.
    """
    dtype, shapes, elements = draw(_arrays_args(signature, code_st, kwds))
    kwds.update(dtype=dtype, elements=elements, fill=st.nothing())
    result = tuple(draw(hyn.arrays(shape=shape, **kwds)) for shape in shapes)
    return result[0] if len(result) == 1 else result


@st.composite
def constant(draw, signature: str, code_st: CodeStrategy = None,
             **kwds) -> np.ndarray:
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
    strategy : np.ndarray
        Strategy to produce an array that broadcasts with the given core
        dimension signature, with a constant value of thet dtype.
    """
    dtype, shapes, elements = draw(_arrays_args(signature, code_st, kwds))
    fill = draw(elements)
    return np.full(shapes[0], fill, dtype)


# =============================================================================
# Helpers for TestCaseNumpy with Hypothesis
# =============================================================================


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


# -----------------------------------------------------------------------------
# Testing values
# -----------------------------------------------------------------------------


def non_singular(matrix: np.ndarray) -> Union[np.ndarray, bool]:
    """Check that matrix/matrices are non-singular (invertible)

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


# -----------------------------------------------------------------------------
# Testing shapes
# -----------------------------------------------------------------------------


def wide(array: np.ndarray) -> bool:
    """Check if it is an array of wide matrices.

    By `numpy.linalg`'s broadcasting rules, these are the last two axes.

    Parameters
    ----------
    array : np.ndarray
        matrix (or array of them) whose shape we are checking.

    Returns
    -------
    wide : bool
        Is it a wide (array of) matrix?
    """
    return array.shape[-2] < array.shape[-1]

def tall(array: np.ndarray) -> bool:
    """Check if it is an array of tall matrices.

    By `numpy.linalg`'s broadcasting rules, these are the last two axes.

    Parameters
    ----------
    array : np.ndarray
        matrix (or array of them) whose shape we are checking.

    Returns
    -------
    tall : bool
        Is it a tall (array of) matrix?
    """
    return array.shape[-2] > array.shape[-1]

def nonsquare(array: np.ndarray) -> bool:
    """Check if it is an array of non-square matrices.

    By `numpy.linalg`'s broadcasting rules, these are the last two axes.

    Parameters
    ----------
    array : np.ndarray
        matrix (or array of them) whose shape we are checking.

    Returns
    -------
    nonsquare : bool
        Is it a non-square (array of) matrix?
    """
    return array.shape[-2] != array.shape[-1]
