"""Test array creation routines
"""
import os
import sys
import tempfile
from numbers import Real
from typing import Sequence

import hypothesis as hy
import hypothesis.extra.numpy as hn
import hypothesis.strategies as st
import numpy as np

import numpy_linalg as nl
import numpy_linalg.testing.hypothesis_numpy as hyn
from numpy_linalg.testing import TestCaseNumpy, main

# =============================================================================
# pylint: disable=missing-function-docstring
errstate = np.errstate(invalid='ignore')
hy.settings.register_profile("slow",
                             suppress_health_check=(hy.HealthCheck.too_slow,))
# hy.settings.load_profile('slow')
np.set_printoptions(precision=2, threshold=10, edgeitems=2)
print_full = np.printoptions(precision=None, threshold=sys.maxsize)
# =============================================================================
__all__ = ['TestCreation']
# =============================================================================
# Strategies
# =============================================================================
some_shape = hn.array_shapes()
some_dtype = st.one_of(
        # hn.boolean_dtypes(),
        hn.integer_dtypes(),
        hn.unsigned_integer_dtypes(),
        hn.floating_dtypes(),
        hn.complex_number_dtypes(),
        # hn.datetime64_dtypes(),
        # hn.timedelta64_dtypes(),
    )
some_array = hn.arrays(dtype=some_dtype, shape=some_shape)


@st.composite
def sliceish(draw: st.DataObject,
             stgy: st.SearchStrategy[float] = hyn.real_numbers(min_value=1e-5),
             non_empty: bool = False) -> slice:
    """Strategy for a [start:stop:step] slice, not necessarily ints

    Parameters
    ----------
    num_st : SearchStrategy[Number], optional
        Strategy to generate numbers, by default params['r']
    non_empty : bool, optional
        Ensure that slice is non empty, i.e. sgns of stop - start and step
        match? By default False

    Returns
    -------
    sliceish : slice
        slice with parameters start, stop, step that are of whatever type
        given by `num_st`.
    """
    start, stop, step = draw(stgy), draw(stgy), draw(stgy)
    hy.assume(step != 0)
    if non_empty:
        if (stop - start) * step < 0:
            start, stop = stop, start
    return slice(start, stop, step)


def ind_func(*inds):
    """to test fromfunction"""
    return sum(inds)


# -----------------------------------------------------------------------------
# creation routines
# -----------------------------------------------------------------------------
ones_and_zeros = st.sampled_from([
    'empty', 'eye', 'identity', 'ones', 'zeros', 'full'])
from_existing_data = st.sampled_from([
    'array', 'asarray', 'asanyarray', 'ascontiguousarray', 'asfortranarray',
    'asarray_chkfinite', 'copy', 'fromfunction', 'fromiter', 'frombuffer',
    'fromstring',
])
numerical_ranges = st.sampled_from([
    'arange', 'linspace', 'logspace', 'geomspace'])
indexing = [
    'ravel_multi_index', 'unravel_index', 'diag_indices', 'mask_indices',
    'tril_indices', 'triu_indices', 'indices',
]
# -----------------------------------------------------------------------------
# random numbers
# -----------------------------------------------------------------------------
params = {
    'i': st.integers(),
    'iop': st.integers(min_value=0),
    'ip': st.integers(min_value=1),
    'r': hyn.real_numbers(),
    'rop': hyn.real_numbers(min_value=0),
    'rp': hyn.real_numbers(min_value=1e-5),
    'p': hyn.real_numbers(min_value=0, max_value=1),
    'mcov': hyn.broadcastable('(n)(n,n)', 'd', max_dims=0),
}
params['vrp'] = hn.arrays(float, params['ip'], elements=params['rp'])
params['vp'] = hn.arrays(float, params['ip'], elements=params['p'])
params['viop'] = hn.arrays(int, params['ip'], elements=params['iop'])
#  grouped by paraneter signature
rnd_r_rop = st.sampled_from([
    'gumbel', 'laplace', 'logistic', 'lognormal', 'normal', 'vonmises'])
rnd_rp_rp = st.sampled_from(['beta', 'f', 'wald'])
rnd_rop = st.sampled_from([
    'exponential', 'poisson', 'power', 'rayleigh', 'standard_gamma', 'weibull'
])
rnd_rp = st.sampled_from(['chisquare', 'pareto', 'standard_t', 'zipf'])
rnd_p = st.sampled_from(['geometric', 'logseries'])
rnd_none = st.sampled_from([
    'random', 'standard_cauchy', 'standard_exponential', 'standard_normal'])
rnd_params = {
    'integers': ('i', 'i'),
    'binomial': ('iop', 'p'),
    'gamma': ('rop', 'rop'),
    'hypergeometric': ('iop', 'iop', 'iop'),
    'negative_binomial': ('rp', 'p'),
    'noncentral_chisquare': ('rp', 'rop'),
    'noncentral_f': ('rp', 'rp', 'rop'),
    'triangular': ('r', 'r', 'r'),
    'uniform': ('r', 'r'),
    'dirichlet': ('vrp',),
    'multinomial': ('iop', 'vp'),
    'multivariate_hypergeometric': ('viop', 'iop'),
    'multivariate_normal': ('mcov',),
}
rnd_others = st.sampled_from(list(rnd_params))


# =============================================================================
# Test BLAS ufuncs
# =============================================================================


class TestWrappers(TestCaseNumpy):
    """Testing norm, matmul and rmatmul"""

    def setUp(self):
        super().setUp()
        self.all_close_opts['equal_nan'] = True

    def assertArraysMatch(self, nl_array: nl.lnarray, np_array: np.ndarray,
                          val: bool = True, msg: str = None):
        """Assert that array type, shape, value are all correct

        Parameters
        ----------
        nl_array : lnarray
            Array we are testing
        np_array : ndarray
            Template array

        Raises
        ------
        AssertionError
            If `nl_array` is not an lnarray, if it doesn't have the same shape
            as `np_array` or the same values.
        """
        self.assertIsInstance(nl_array, nl.lnarray, msg=msg)
        self.assertArrayShape(nl_array, np_array.shape, msg=msg)
        self.assertArrayDtype(nl_array, np_array.dtype, msg=msg)
        if val:
            self.assertArrayAllClose(nl_array, np_array, msg=msg)

    def assertArraysAllMatch(self, nl_arrays: Sequence[nl.lnarray],
                             np_arrays: Sequence[np.ndarray],
                             val: bool = True, msg: str = None):
        """Assert that array type, shape, value are all correct

        Parameters
        ----------
        nl_arrays : Sequence[lnarray]
            Arrays we are testing
        np_arrays : Sequence[ndarray]
            Template arrays

        Raises
        ------
        AssertionError
            If `nl_array` is not an lnarray, if it doesn't have the same shape
            as `np_array` or the same values, for ever pair of arrays.
        """
        for nla, npa in zip(nl_arrays, np_arrays):
            self.assertArraysMatch(nla, npa, val=val, msg=msg)


class TestCreation(TestWrappers):
    """Testing array creatin routines"""

    @errstate
    @hy.given(ones_and_zeros, some_shape, st.integers(), some_dtype)
    def test_ones_and_zeros(self, func, shape, val, dtype):
        np_func, nl_func = getattr(np, func), getattr(nl, func)
        args = [shape]
        if func == "full":
            args.append(val)
        elif func == "eye":
            args = list(shape[:3])
        elif func == "identity":
            args = [shape[0]]
        kwds = {'dtype': dtype}
        self.assertArraysMatch(nl_func(*args, **kwds), np_func(*args, **kwds),
                               func != "empty", msg=func)

    @errstate
    @hy.given(from_existing_data, hyn.vectors)
    def test_existing(self, func: str, array: np.ndarray):
        np_func, nl_func = getattr(np, func), getattr(nl, func)
        args = []
        kwds = {}
        if func != "copy":
            kwds['dtype'] = array.dtype
        if func == "fromstring":
            with print_full:
                args.append(str(array.ravel())[1:-1])
            kwds['sep'] = ' '
        elif func == "frombuffer":
            args.append(array.tobytes())
        elif func == "fromfunction":
            args.extend([ind_func, array.shape])
        elif func == "fromiter":
            args.append(x for x in array.ravel())
        else:
            args.append(array.tolist())
        try:
            np_array = np_func(*args, **kwds)
        except ValueError:
            with self.assertRaises(ValueError):
                nl_func(*args, **kwds)
        else:
            if func == "fromiter":
                args = []
                args.append(x for x in array.ravel())
            self.assertArraysMatch(nl_func(*args, **kwds), np_array, msg=func)

    @errstate
    @hy.given(some_array, st.sampled_from("FCAWOE"))
    def test_require(self, array, requirement):
        self.assertArraysMatch(nl.require(array, None, requirement),
                               np.require(array, None, requirement))

    @print_full
    @hy.given(hyn.matrices_c)
    def test_fromfile(self, array: np.ndarray):
        hy.assume(np.all(np.isfinite(array)))
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.bin')
            array.tofile(fname, "")
            nl_array = nl.fromfile(fname, dtype=array.dtype, sep="")
            self.assertArraysMatch(nl_array, array.ravel())

            fname = os.path.join(tmpdir, 'test.str')
            array.tofile(fname, " ")
            nl_array = nl.fromfile(fname, dtype=array.dtype, sep=" ")
            self.assertArraysMatch(nl_array, array.ravel())

            fname = os.path.join(tmpdir, 'test.txt')
            np.savetxt(fname, array, delimiter=' ')
            nl_array = nl.loadtxt(fname, dtype=array.dtype, delimiter=" ")
            self.assertArraysMatch(nl_array, array.squeeze())

    @hy.given(numerical_ranges, sliceish(), params['iop'], params['rp'])
    def test_numerical_ranges(self, func: str, slicey: slice, num, base):
        np_func, nl_func = getattr(np, func), getattr(nl, func)
        args = [slicey.start, slicey.stop]
        if func == "arange":
            args.append(slicey.step)
        elif func == "logspace":
            args.extend([num, True, base])
        else:
            args.append(num)
        self.assertArraysMatch(nl_func(*args), np_func(*args))

    @hy.given(st.lists(sliceish(non_empty=True)))
    def test_nd_grids(self, slice_list):
        args = tuple(slice_list)
        meshed = nl.mgrid[args]
        opened = nl.ogrid[args]
        self.assertArraysMatch(meshed, np.mgrid[args])
        self.assertArraysAllMatch(opened, np.ogrid[args])
        meshed = tuple(meshed)
        self.assertArraysAllMatch(nl.meshgrid(*opened[::-1]), meshed[::-1])
        self.assertArraysMatch(nl.c_[meshed], np.c_[meshed])
        self.assertArraysMatch(nl.r_[meshed], np.r_[meshed])
        self.assertArraysMatch(nl.r_[args], np.r_[args])

    @hy.given(params['ip'], params['rp'])
    def test_fft(self, length, spacing):
        with self.assertRaises(ValueError):
            raise ValueError
        self.assertArraysMatch(nl.fft.fftfreq(length, spacing),
                               np.fft.fftfreq(length, spacing))
        self.assertArraysMatch(nl.fft.rfftfreq(length, spacing),
                               np.fft.rfftfreq(length, spacing))

    @hy.given(st.data(), some_shape)
    def test_indexing(self, data: st.DataObject, shape: Sequence[int]):
        mult_ind = data.draw(hn.integer_array_indices(shape))
        ravl_ind = np.ravel_multi_index(mult_ind, shape)
        self.assertArraysMatch(nl.ravel_multi_index(mult_ind, shape), ravl_ind)
        self.assertArraysMatch(nl.unravel_index(ravl_ind, shape), mult_ind)
        self.assertArraysMatch(nl.indices(shape), np.indices(shape))
        self.assertArraysMatch(nl.diag_indices(shape[0], len(shape)),
                               np.diag_indices(shape[0], len(shape)))
        if len(shape) > 0:
            n = shape[0]
            m = shape[-1]
            kmax = min(m, n)
            k = data.draw(st.integers(min_value=-kmax, max_value=kmax))
            self.assertArraysMatch(nl.tril_indices(n, k, m),
                                   np.tril_indices(n, k, m))
            self.assertArraysMatch(nl.triu_indices(n, k, m),
                                   np.triu_indices(n, k, m))
            self.assertArraysMatch(nl.mask_indices(n, np.triu, k),
                                   np.mask_indices(n, np.triu, k))


class TestRandom(TestWrappers):
    """Testing randon number generTION"""
    np_rng: np.random.Generator
    nl_rng: nl.random.LnGenerator

    def setUp(self):
        super().setUp()
        self.np_rng = nl.random.default_rng()
        self.nl_rng = nl.random.default_rng()

    def assertArraysMatch(self, nl_array, np_array, val: bool = False,
                          msg: str = None):
        """Assert that array type, shape, value are all correct

        Parameters
        ----------
        nl_array : lnarray
            Array we are testing
        np_array : ndarray
            Template array

        Raises
        ------
        AssertionError
            If `nl_array` is not an lnarray, if it doesn't have the same shape
            as `np_array` or the same values.
        """
        super().assertArraysMatch(nl_array, nl.lnarray, val=val, msg=msg)

    def assertRandomMatch(self, func: str, args: Sequence[float]):
        """Assert that outputs of random.Generator have correct type shape

        Parameters
        ----------
        func : str
            Name of `Generator` method to test
        args : Sequence[float, ...]
            Arguments for generator method
        """
        np_method = getattr(self.np_rng, func)
        nl_method = getattr(self.nl_rng, func)
        np_array, nl_array = np_method(*args), nl_method(*args)
        msg = f"{func=}, {args=}"
        self.assertArraysMatch(nl_array, np_array, val=False, msg=msg)
        self.assertIsInstance(nl_method(*args[:-1]), Real, msg=msg)

    def test_random_basic(self, vec: np.ndarray, num: int):
        nl_bytes = self.nl_rng.bytes(num)
        self.assertIsInstance(nl_bytes, bytes)
        self.assertEqual(len(nl_bytes), num)

        nl_perm = self.nl_rng.permutation(num)
        self.assertIsInstance(nl_perm, nl.lnarray)
        self.assertArrayShape(nl_perm, (num,))
        self.assertArrayDtype(nl_perm, int)

        nl_perm = self.nl_rng.permutation(vec)
        self.assertIsInstance(nl_perm, nl.lnarray)
        self.assertArrayShape(nl_perm, vec.shape)
        self.assertArrayDtype(nl_perm, vec.dtype)

        nl_choice = self.nl_rng.choice(vec, size=num)
        self.assertIsInstance(nl_choice, nl.lnarray)
        self.assertArrayShape(nl_choice, (num,))
        self.assertArrayDtype(nl_choice, vec.dtype)

        shape, dtype = vec.shape, vec.dtype
        nl_shuffle = self.nl_rng.shuffle(vec)
        self.assertIsNone(nl_shuffle)
        self.assertIsInstance(vec, np.ndarray)
        self.assertArrayShape(vec, shape)
        self.assertArrayDtype(vec, dtype)

    @hy.given(rnd_r_rop, params['r'], params['rop'], some_shape)
    def test_random_r_rop(self, func, alpha, beta, shape):
        args = (alpha, beta, shape)
        self.assertRandomMatch(func, args)

    @hy.given(rnd_rp_rp, params['rp'], params['rp'], some_shape)
    def test_random_rp_rp(self, func, alpha, beta, shape):
        args = (alpha, beta, shape)
        self.assertRandomMatch(func, args)

    @hy.given(rnd_rop, params['rop'], some_shape)
    def test_random_rop(self, func, alpha, shape):
        if func == "power":
            alpha += 1
        args = (alpha, shape)
        self.assertRandomMatch(func, args)

    @hy.given(rnd_rp, params['rp'], some_shape)
    def test_random_rp(self, func, alpha, shape):
        if func == "zipf":
            alpha += 1
        args = (alpha, shape)
        self.assertRandomMatch(func, args)

    @hy.given(rnd_p, params['p'], some_shape)
    def test_random_p(self, func, prob, shape):
        args = (prob, shape)
        self.assertRandomMatch(func, args)

    @hy.given(rnd_none, some_shape)
    def test_random_none(self, func, shape):
        args = (shape,)
        self.assertRandomMatch(func, args)

    @hy.given(st.data(), rnd_others)
    def test_random_others(self, data: st.DataObject, func: str):
        args = [data.draw(params[s]) for s in rnd_params[func]]
        self.assertRandomMatch(func, args)


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
