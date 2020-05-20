"""Test array creation routines
"""
import tempfile
import os.path as osp
import hypothesis as hy
import hypothesis.strategies as st
import hypothesis.extra.numpy as hn
import numpy as np
import numpy_linalg as nl
from numpy_linalg.testing import main, TestCaseNumpy
import numpy_linalg.testing.hypothesis_numpy as hyn
# =============================================================================
# pylint: disable=missing-function-docstring
errstate = np.errstate(invalid='ignore')
hy.settings.register_profile("slow",
                             suppress_health_check=(hy.HealthCheck.too_slow,))
hy.settings.load_profile('slow')
np.set_printoptions(precision=2, threshold=10, edgeitems=2)
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
ropts = {'allow_nan': False, 'allow_infinity': False}
params = {
    'i': st.integers(),
    'iop': st.integers(min_value=0),
    'ip': st.integers(min_value=1),
    'r': hyn.real_numbers(**ropts),
    'rop': hyn.real_numbers(min_value=0),
    'rp': hyn.real_numbers(min_value=1e-5, exclude_min=True),
    'p': hyn.real_numbers(min_value=0, max_value=1),
}


@st.composite
def sliceish(draw: st.DataObject, num_st: st.SearchStrategy = params['rp'],
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
    start, stop, step = draw(num_st), draw(num_st), draw(num_st)
    hy.assume(step != 0)
    if non_empty:
        if (stop - start) * step < 0:
            start, stop = stop, start
    return slice(start, stop, step)


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
other_numerical_ranges = 'meshgrid'
instances = ['mgrid', 'ogrid', 'r_', 'c_']
indexing = [
    'ravel_multi_index', 'unravel_index', 'diag_indices', 'mask_indices',
    'tril_indices', 'triu_indices', 'indices',
]
# -----------------------------------------------------------------------------
# random numbers
# -----------------------------------------------------------------------------
rng = nl.random.default_rng()
random = ['bytes', 'choice']
rnd_seq = ['permutation', 'shuffle']
rnd = {
    'integers': ('i', 'i'),
    'binomial': ('iop', 'p'),
    'gamma': ('rop', 'rop'),
    'hypergeometric': ('iop', 'iop', 'iop'),
    'negative_binomial': ('rp', 'p'),
    'noncentral_chisquare': ('rp', 'rop'),
    'noncentral_f': ('rp', 'rp', 'rop'),
    'triangular': ('r', 'r', 'r'),
    'uniform': ('r', 'r'),
}
rnd_r_rop = [
    'gumbel',
    'laplace',
    'logistic',
    'lognormal',
    'normal',
    'vonmises',
]
rnd_rp_rp = ['beta', 'f', 'wald']
rnd_rp = ['chisquare', 'pareto', 'standard_t', 'zipf']
rnd_rop = [
    'exponential',
    'poisson',
    'power',
    'rayleigh',
    'standard_gamma',
    'weibull',
]
rnd_ = [
    'random',
    'standard_cauchy',
    'standard_exponential',
    'standard_normal',
]
rnd_m = [
    'dirichlet',
    'multinomial',
    'multivariate_hypergeometric',
    'multivariate_normal',
]
rnd_p = [
    'geometric',
    'logseries',
]


def ind_func(*inds):
    """to test fromfunction"""
    return sum(inds)


# =============================================================================
# Test BLAS ufuncs
# =============================================================================


class TestCreation(TestCaseNumpy):
    """Testing norm, matmul and rmatmul"""

    def setUp(self):
        super().setUp()
        self.all_close_opts['equal_nan'] = True

    def assertArraysMatch(self, nl_array, np_array, val: bool = True,
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
        self.assertIsInstance(nl_array, nl.lnarray, msg=msg)
        self.assertArrayShape(nl_array, np_array.shape, msg=msg)
        self.assertArrayDtype(nl_array, np_array.dtype, msg=msg)
        if val:
            self.assertArrayAllClose(nl_array, np_array, msg=msg)

    def assertArraysAllMatch(self, nl_arrays, np_arrays, val: bool = True,
                             msg: str = None):
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

    @hy.given(hyn.matrices_c)
    def test_fromfile(self, array: np.ndarray):
        hy.assume(np.all(np.isfinite(array)))
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = osp.join(tmpdir, 'test.bin')
            array.tofile(fname, "")
            nl_array = nl.fromfile(fname, dtype=array.dtype, sep="")
            self.assertArraysMatch(nl_array, array.ravel())

            fname = osp.join(tmpdir, 'test.str')
            array.tofile(fname, " ")
            nl_array = nl.fromfile(fname, dtype=array.dtype, sep=" ")
            self.assertArraysMatch(nl_array, array.ravel())

            fname = osp.join(tmpdir, 'test.txt')
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
        hy.assume(spacing != 0.0)
        self.assertArraysMatch(nl.fft.fftfreq(length, spacing),
                               np.fft.fftfreq(length, spacing))
        self.assertArraysMatch(nl.fft.rfftfreq(length, spacing),
                               np.fft.rfftfreq(length, spacing))


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
