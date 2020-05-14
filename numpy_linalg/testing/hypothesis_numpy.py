"""Hypothesis strategies for numpy arrays, and tools for the generated data

Strategies
----------
complex_numbers
    Similar to `hypothesis.strategies.complex_numbers`, but it takes the same
    options as `hypothesis.strategies.floats`.
real_numbers
    Wrapper of `hypothesis.strategies.floats` with different default options.
integers
    Wrapper of `hypothesis.strategies.integers` to ignore irrelevant options.
numeric_dtypes
    Generate `dtype` and `elements` values for `hypothesis.extra.numpy.arrays`
signature_shapes
    Generate `shape` values for `hypothesis.extra.numpy.arrays` that broadcast
    with a given signature.
broadcastable
    Generate a tuple of arrays of the same generated `dtype` with generated
    `shape`s that broadcast with a given signature.
constant
    Generate a single array with all elements equal, with a generated `dtype`
    and a `shape` that broadcasts with a given signature.
matrices_b
    Generate broadcasting matrices (ndim >= 2) with float entries.
matrices_c
    Generate core-only matrices (ndim == 2) with float entries.
vectors
    Generate vectors (ndim == 1) with float entries.

Functions
---------
core_only
    Romove non-core dimensions from arrays for functions that do not broadcast.
For `hypothesis.assume`:

non_singular
Check if e
ach matrix in an array has finite nonzero determinant.
all_non_singular
    Check if every matrix in some arrays has finite nonzero determinant.
full_rank
    Check if each matrix in an array has the maximum rank given its shape.
all_full_rank
    Check if every matrix in some arrays has the maximum rank given its shape.
wide
    Check if a matrix has more columns than rows.
tall
    Check if a matrix has more rows than columns.
nonsquare
    Check if a matrix has different numbers of rows and columns.
"""
import collections.abc
from numbers import Number
from typing import Sequence, Tuple, Union

import hypothesis.extra.numpy as hyn
import hypothesis.strategies as st
import numpy as np

__all__ = [
    'complex_numbers',
    'real_numbers',
    'integers',
    'numeric_dtypes',
    'signature_shapes',
    'broadcastable',
    'constant',
    'matrices_b',
    'matrices_c',
    'vectors',
    'core_only',
    'non_singular',
    'all_non_singular',
    'full_rank',
    'all_full_rank',
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
    # Lapack functions raise runtime warnings if passed inf or nan, so exclude
    if kind == "dtype":
        return {'min_value': -1e10, 'max_value': 1e10, 'allow_infinity': False,
                'allow_nan': False, 'exclude_min': False, 'exclude_max': False}
    if kind == "shape":
        return {'min_dims': 0, 'max_dims': 3, 'min_side': 1, 'max_side': 5,
                'base_shape': ()}
    raise ValueError(f"Unknown option kind: {kind}")


def integers(**kwds) -> st.SearchStrategy[float]:
    """Strategy to generate real numbers of specified width

    This is a wrapper for `hypothesis.strategies.integers` that ignores
    irrelevant keywords, instead of raising an exception.

    See Also
    --------
    `hypothesis.strategies.integers`
    """
    min_value = kwds.get('min_value', None)
    max_value = kwds.get('max_value', None)
    return st.integers(min_value=min_value, max_value=max_value)


def real_numbers(**kwds) -> st.SearchStrategy[float]:
    """Strategy to generate real numbers of specified width

    This is a wrapper for `hypothesis.strategies.floats` with changed defaults

    See Also
    --------
    `hypothesis.strategies.floats`
   """
    opts = _default_opts("dtype")
    opts.update(kwds)
    return st.floats(**opts)


def complex_numbers(**kwds) -> st.SearchStrategy[complex]:
    """Strategy to generate complex numbers of specified width

    Takes any keyword arguments for `hypothesis.strategies.floats`

    Returns
    -------
    complex_strategy : st.SearchStrategy[complex]
        Strategy for complex numbers that applies float options to real and
        imaginary parts.

    See Also
    --------
    `hypothesis.strategies.complex_numbers`
    """
    if 'width' in kwds:
        kwds['width'] //= 2
    return st.builds(complex, st.floats(**kwds), st.floats(**kwds))


_DTYPES = {
    'f': (np.float32, real_numbers),
    'd': (np.float64, real_numbers),
    'F': (np.complex64, complex_numbers),
    'D': (np.complex128, complex_numbers),
    'i': (np.int32, integers)
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

    See Also
    --------
    `hypothesis.extra.numpy.arrays`
    """
    if code_st is None:
        code_st = st.sampled_from(['f', 'd', 'F', 'D'])
    elif isinstance(code_st, str):
        code_st = st.just(code_st)
    elif isinstance(code_st, collections.abc.Sequence):
        code_st = st.sampled_from(code_st)
    code = draw(code_st)
    dtype, element_st = _DTYPES[code]
    kwds['width'] = dtype().itemsize * 8
    return dtype, element_st(**kwds)


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

    See Also
    --------
    `hypothesis.extra.numpy.arrays`
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
    shape_opts = _extract_kwds(kwds, **_default_opts("shape"))
    if kwds:
        raise ValueError(f"Unknown keywords: {list(kwds)}")
    dtype, elements = draw(numeric_dtypes(code_st, **num_opts))
    shapes = draw(signature_shapes(signature, **shape_opts))
    return dtype, shapes, elements


@st.composite
def broadcastable(draw: st.DataObject,
                  signature: str,
                  code_st: CodeStrategy = None,
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
    `hypothesis.extra.numpy.mutually_broadcastable_shapes` except `num_shapes`.

    Returns
    -------
    strategy : Tuple[np.ndarray, ...]
        Strategy to produce a tuple of arrays that broadcast with the given
        core dimension signature.

    See Also
    --------
    `hypothesis.extra.numpy.arrays`
    """
    dtype, shapes, elements = draw(_arrays_args(signature, code_st, kwds))
    kwds.update(dtype=dtype, elements=elements, fill=st.nothing())
    result = tuple(draw(hyn.arrays(shape=shape, **kwds)) for shape in shapes)
    return result[0] if len(result) == 1 else result


@st.composite
def constant(draw: st.DataObject,
             signature: str,
             code_st: CodeStrategy = None,
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
    `hypothesis.extra.numpy.mutually_broadcastable_shapes` except `num_shapes`.

    Returns
    -------
    strategy : np.ndarray
        Strategy to produce an array that broadcasts with the given core
        dimension signature, with a constant value of thet dtype.

    See Also
    --------
    `hypothesis.extra.numpy.arrays`
    """
    dtype, shapes, elements = draw(_arrays_args(signature, code_st, kwds))
    fill = draw(elements)
    return np.full(shapes[0], fill, dtype)


matrices_b = hyn.arrays(dtype=np.float64,
                        shape=hyn.array_shapes(min_dims=2),
                        elements=real_numbers())
matrices_c = hyn.arrays(dtype=np.float64,
                        shape=hyn.array_shapes(min_dims=2, max_dims=2),
                        elements=real_numbers())
vectors = hyn.arrays(dtype=np.float64,
                     shape=hyn.array_shapes(min_dims=1, max_dims=1),
                     elements=real_numbers())

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
        Keyword only.

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
    is_not_singular : np.ndarray|bool
        bool/array of bools that are True if the matrix is non-singular.
    """
    return np.abs(np.linalg.slogdet(matrix)[1]) < 500


def all_non_singular(*matrices: np.ndarray) -> bool:
    """Check that all matrices are non-singular

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


def full_rank(matrix: np.ndarray) -> Union[np.ndarray, bool]:
    """Check that matrix/matrices have full rank

    Parameters
    ----------
    matrix : np.ndarray
        square matrix/array of square matrices whose rank to check.

    Returns
    -------
    is_full_rank : np.ndarray|bool
        bool/array of bools that are True if the matrix has full rank.
    """
    if tall(matrix):
        return non_singular(matrix.swapaxes(-2, -1) @ matrix)
    if wide(matrix):
        return non_singular(matrix @ matrix.swapaxes(-2, -1))
    return non_singular(matrix)


def all_full_rank(*matrices: np.ndarray) -> bool:
    """Check that all matrices have full rank

    Parameters
    ----------
    matrices : np.ndarray
        square matrices whose rank to check.

    Returns
    -------
    are_full_rank : bool
        True if all of the the matrices have full rank.
    """
    return all(np.all(full_rank(mat)) for mat in matrices)


def well_conditioned(matrix: np.ndarray) -> Union[np.ndarray, bool]:
    """Check condition number of matrix/matrices

    Parameters
    ----------
    matrix : np.ndarray
        matrix/array of matrices whose condition number to check

    Returns
    -------
    is_well_conditioned : np.ndarray|bool
        bool/array of bools that are True if the matrix is well conditioned.
    """
    return np.linalg.cond(matrix) < 1e3


def all_well_conditioned(*matrices: np.ndarray) -> bool:
    """Check condition number of all matrices

    Parameters
    ----------
    matrices : np.ndarray
        matrices whose condition number to check

    Returns
    -------
    are_well_conditioned : bool
        True if all of the the matrices are well conditioned.
    """
    return all(np.all(well_conditioned(mat)) for mat in matrices)


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
