"""Test array creation routines
"""
import os
import re
import sys
import tempfile
from numbers import Real
from typing import List, Mapping, Sequence, Tuple, Union

import hypothesis as hy
import hypothesis.extra.numpy as hn
import hypothesis.strategies as st
import numpy as np

import numpy_linalg as nl
import numpy_linalg.testing.hypothesis_numpy as hyn
from numpy_linalg.testing import TestCaseNumpy, main

# =============================================================================
# pylint: disable=missing-function-docstring
errstate = np.errstate(invalid='ignore', over='ignore', under='ignore')
hy.settings.register_profile("slow", deadline=300,
                             suppress_health_check=(hy.HealthCheck.too_slow,))
hy.settings.load_profile('slow')
np.set_printoptions(precision=2, threshold=10, edgeitems=2)
# =============================================================================
__all__ = ['TestCreation', 'TestRandom']
# =============================================================================
# Strategies
# =============================================================================


def swap_ab(seq: Sequence) -> Sequence:
    """Swap first two elements of sequence
    """
    return seq[1::-1] + seq[2:]


def str_sample(*str_list: str) -> st.SearchStrategy[str]:
    """Strategy for substrings from string"""
    return st.sampled_from(" ".join(str_list).split(" "))


def dtype_str(dtype: np.dtype) -> (List[str], re.Pattern):
    """Format string and regex for structured dtype"""
    format, regexp = [], []
    for name in dtype.names:
        sctype = dtype.fields[name][0]
        fmt = f'{name}: '
        rex = re.escape(f'{name}: ')
        if np.issubsctype(sctype, np.integer):
            fmt += '%d'
            rex += '([-0-9]+)'
        elif np.issubsctype(sctype, np.floating):
            fmt += '%.11e'
            rex += '([-0-9.e+]+|-?inf|nan)'
        elif np.issubsctype(sctype, np.complexfloating):
            fmt += '%r'
            rex += '\\(?([-0-9.e+ ]*[+-]?[-0-9.e+ ]+j)\\)?'
        else:
            raise ValueError(f"unknown dtype: {sctype}")
        fmt += ','
        format.append(fmt)
        rex += ','
        regexp.append(rex)
    return format, re.compile(" ".join(regexp))


def make_cov(cholesky: np.ndarray) -> np.ndarray:
    """Make a covariance matrix from its Cholesky decomposition"""
    lower = np.tril(cholesky)
    # in a future version, np.diagonal will return a writable view
    diag = np.diagonal(lower).copy()
    diag[np.abs(diag) < 1e-3] = 1
    lower[np.diag_indices_from(lower)] = diag
    return lower @ lower.T


# Pre-defined strategies
some_shape = hn.array_shapes(max_dims=4, max_side=20)
simple_shape = hn.array_shapes(max_dims=2, max_side=10)
some_dtype = st.one_of(
    # hn.boolean_dtypes(),
    hn.integer_dtypes(),
    hn.unsigned_integer_dtypes(),
    hn.floating_dtypes(),
    hn.complex_number_dtypes(),
    # hn.datetime64_dtypes(),
    # hn.timedelta64_dtypes(),
)
real_dtype = st.one_of(
    # hn.boolean_dtypes(),
    hn.integer_dtypes(),
    hn.unsigned_integer_dtypes(),
    hn.floating_dtypes(),
    # hn.complex_number_dtypes(),
    # hn.datetime64_dtypes(),
    # hn.timedelta64_dtypes(),
)
some_array = hn.arrays(dtype=some_dtype, shape=some_shape)
# -----------------------------------------------------------------------------
# Creation routines
# -----------------------------------------------------------------------------
ones_and_zeros = str_sample('empty eye identity ones zeros full')
from_existing_data = str_sample(
    'array asarray asanyarray ascontiguousarray asfortranarray',
    'asarray_chkfinite copy frombuffer fromstring')
numerical_ranges = str_sample('arange linspace logspace geomspace')
# -----------------------------------------------------------------------------
# Random number generators
# -----------------------------------------------------------------------------
# Parameter types:
# (v)ector of (i)ntegers or (r)eals that are zer(o) and/or (p)ositive
# Or (m)ean and Cholesky deccomposition of (cov)ariance
params = {
    'i': st.integers(min_value=-100, max_value=100),
    'iop': st.integers(min_value=0, max_value=100),
    'ip': st.integers(min_value=1, max_value=100),
    'r': hyn.real_numbers(),
    'rop': hyn.real_numbers(min_value=0),
    'rp': hyn.real_numbers(min_value=1e-5),
    'p': hyn.real_numbers(min_value=1e-5, max_value=1),
    'mcov': hyn.broadcastable('(n),(n,n)', 'd', max_dims=0, max_side=20),
    'z': hyn.real_numbers(min_value=1.1, max_value=100),
    'none':  st.none(),
}
params['vrp'] = hn.arrays(float, params['ip'], elements=params['rp'])
params['vp'] = hn.arrays(float, params['ip'], elements=params['p'])
params['viop'] = hn.arrays(int, params['ip'], elements=params['iop'])
# -----------------------------------------------------------------------------
# Method names, grouped by paraneter signature
rnd_methods = {
    # ('_' -> 'standard_'):
    'r_rop': str_sample('gumbel laplace logistic lognormal normal vonmises'),
    'rp_rp': str_sample('beta f wald'),
    'rop': str_sample('exponential poisson rayleigh _gamma weibull'),
    'rp': str_sample('chisquare pareto _t'),
    'p': str_sample('geometric logseries'),
    'z': str_sample('power zipf'),
    'none': str_sample('random _cauchy _exponential _normal'),
    # sui generis ('_' -> 'noncentral_'):
    'others': str_sample('integers binomial gamma hypergeometric',
                         'negative_binomial _chisquare _f triangular uniform'),
    # each sample is a vector ('_' -> 'multivariate_'):
    'multi': str_sample('dirichlet multinomial _hypergeometric _normal'),
}
# signatures of the last two sets
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
# -----------------------------------------------------------------------------
# Composite strategies
# -----------------------------------------------------------------------------


@st.composite
def sliceish(draw: st.DataObject,
             stgy: st.SearchStrategy[float] = params['rp'],
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
    hy.assume(abs(stop - start) / step < 100)
    if non_empty:
        if (stop - start) * step < 0:
            start, stop = stop, start
    return slice(start, stop, step)


@st.composite
def struct_array(draw: st.DataObject,
                 subdtype: st.SearchStrategy[np.dtype],
                 shape: st.SearchStrategy[Tuple[int, ...]],
                 **kwargs) -> st.SearchStrategy[np.ndarray]:
    """Strategy for structured arrays (1-level only)"""
    dtype = draw(hn.array_dtypes(subdtype, **kwargs))
    if dtype.names is not None:
        hy.assume(all(x.isascii() and '%' not in x for x in dtype.names))
    return draw(hn.arrays(dtype, shape))


@st.composite
def array_inds(draw: st.DataObject,
               int_type: type = np.intp
               ) -> (Tuple[int, ...], np.ndarray, np.ndarray, Tuple[int, ...]):
    """Strategy for array shape and valid index arrays"""
    shape = draw(some_shape)
    multi_ind = draw(hn.integer_array_indices(shape, dtype=int_type))
    ravel_ind = np.ravel_multi_index(multi_ind, shape, mode='wrap')
    if shape:
        n, m = shape[0], shape[-1]
        k = draw(st.integers(min_value=-n, max_value=m))
    else:
        n, k, m = (0, 0, 0)
    return shape, multi_ind, ravel_ind, (n, k, m)


@st.composite
def func_params(draw: st.DataObject,
                func_group: str,
                func_prefix: str = 'standard',
                func_sts: Mapping[str, st.SearchStrategy[str]] = rnd_methods,
                func_sigs: Mapping[str, Sequence[str]] = rnd_params,
                param_sts: Mapping[str, st.SearchStrategy[Real]] = params,
                ) -> (str, List[Union[Real, Tuple[int, ...]]]):
    """Strategy for Generator method name and parameters"""
    func = draw(func_sts[func_group])
    if func.startswith('_'):
        func = func_prefix + func
    sig = func_sigs.get(func, func_group.split('_'))
    args = [draw(param_sts[s]) for s in sig]
    if func == 'zipf':
        shape = draw(simple_shape)
    else:
        shape = draw(some_shape)
    args.append(shape)
    return func, args


# =============================================================================
# Test Array creation
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
        val : bool, optional
            Should we check array vallues>, by default True
        msg : str, optional
            Text for error message, by default None

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
            if np_array.dtype.names is None:
                self.assertArrayAllClose(nl_array, np_array, msg=msg)
            else:
                self.assertStructArrayClose(nl_array, np_array, msg=msg)

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
        if not isinstance(nl_arrays, (tuple, list)):
            return self.assertArraysMatch(nl_arrays, np_arrays,
                                          val=val, msg=msg)
        self.assertEqual(len(nl_arrays), len(np_arrays))
        for nla, npa in zip(nl_arrays, np_arrays):
            self.assertArraysMatch(nla, npa, val=val, msg=msg)


class TestCreation(TestWrappers):
    """Testing array creatin routines"""

    @errstate
    @hy.given(ones_and_zeros, some_shape, params['i'], some_dtype)
    def test_ones_zeros(self, func, shape, val, dtype):
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
                               func != "empty", f"{func}(*{args},**{kwds})")

    @errstate
    @hy.given(from_existing_data, hyn.vectors)
    def test_existing(self, func: str, array: np.ndarray):
        np_func, nl_func = getattr(np, func), getattr(nl, func)
        args = []
        kwds = {}
        if func != "copy":
            kwds['dtype'] = array.dtype
        if func == "fromstring":
            with np.printoptions(precision=None, threshold=sys.maxsize):
                args.append(str(array.ravel())[1:-1])
            kwds['sep'] = ' '
        elif func == "frombuffer":
            args.append(array.tobytes())
        else:
            args.append(array.tolist())
        try:
            np_array = np_func(*args, **kwds)
        except ValueError:
            with self.assertRaises(ValueError):
                nl_func(*args, **kwds)
        else:
            self.assertArraysMatch(nl_func(*args, **kwds), np_array,
                                   msg=f"{func}(*{args},**{kwds})")

    @errstate
    @hy.given(some_shape, some_dtype)
    def test_fromfunc(self, shape, dtype):
        np_array = np.fromfunction(lambda *args: sum(args), shape, dtype=dtype)
        nl_array = nl.fromfunction(lambda *args: sum(args), shape, dtype=dtype)
        self.assertArraysMatch(nl_array, np_array)

    @errstate
    @hy.given(hyn.vectors)
    def test_fromiter(self, array):
        np_array = np.fromiter((x for x in array), array.dtype)
        nl_array = nl.fromiter((x for x in array), array.dtype)
        self.assertArraysMatch(nl_array, np_array)

    @errstate
    @hy.given(some_array, st.sampled_from("FCAWOE"))
    def test_require(self, array, requirement):
        self.assertArraysMatch(nl.require(array, None, requirement),
                               np.require(array, None, requirement))

    @hy.given(hyn.matrices_c)
    def test_fromfile_b(self, array: np.ndarray):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.bin')
            array.tofile(fname, "")
            nl_array = nl.fromfile(fname, dtype=array.dtype, sep="")
            self.assertArraysMatch(nl_array, array.ravel())

    @hy.given(hyn.matrices_c)
    def test_fromfile_t(self, array: np.ndarray):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.txt')
            array.tofile(fname, " ")
            nl_array = nl.fromfile(fname, dtype=array.dtype, sep=" ")
            self.assertArraysMatch(nl_array, array.ravel())

    @hy.given(hyn.matrices_c)
    def test_fromtxt(self, array: np.ndarray):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.txt')
            np.savetxt(fname, array, delimiter=' ')
            nl_array = nl.loadtxt(fname, dtype=array.dtype, delimiter=" ")
            self.assertArraysMatch(nl_array, array.squeeze())

            nl_array = nl.genfromtxt(fname, dtype=array.dtype, delimiter=" ")
            self.assertArraysMatch(nl_array, array.squeeze())

    @hy.given(struct_array(real_dtype, simple_shape, min_size=2))
    def test_fromregex(self, array: np.ndarray):
        # numpy.complex_ can't convert strings, needs extra step via np.complex
        format, pattern = dtype_str(array.dtype)
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.txt')
            np.savetxt(fname, array.ravel(), delimiter=' ', fmt=format)
            nl_arrray = nl.fromregex(fname, pattern, array.dtype)
            self.assertArraysMatch(nl_arrray, array.ravel())

    @hy.given(hyn.matrices_c)
    def test_fromload(self, array: np.ndarray):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, 'test.npy')
            np.save(fname, array)
            nl_array = nl.load(fname)
            self.assertArraysMatch(nl_array, array)

            fname = os.path.join(tmpdir, 'test.npz')
            np.savez(fname, x=array, y=2 * array)
            with nl.load(fname) as nl_file:
                self.assertArraysMatch(nl_file['x'], array)
                self.assertArraysMatch(nl_file['y'], 2 * array)

    @hy.given(array_inds())
    def test_indexing(self, args):
        shape, mult_ind, ravl_ind, nkm = args
        self.assertArraysMatch(nl.ravel_multi_index(mult_ind, shape,
                                                    mode='wrap'), ravl_ind)
        self.assertArraysAllMatch(nl.unravel_index(ravl_ind, shape),
                                  np.unravel_index(ravl_ind, shape))
        self.assertArraysAllMatch(nl.indices(shape), np.indices(shape))
        self.assertArraysAllMatch(nl.diag_indices(shape[0], len(shape)),
                                  np.diag_indices(shape[0], len(shape)))
        if shape:
            n, k, m = nkm
            self.assertArraysAllMatch(nl.tril_indices(n, k, m),
                                      np.tril_indices(n, k, m))
            self.assertArraysAllMatch(nl.triu_indices(n, k, m),
                                      np.triu_indices(n, k, m))
            self.assertArraysAllMatch(nl.mask_indices(n, np.triu, k),
                                      np.mask_indices(n, np.triu, k))

    @errstate
    @hy.given(numerical_ranges, sliceish(), params['iop'], params['rp'])
    def test_num_ranges(self, func: str, slicey: slice, num: int, base: float):
        np_func, nl_func = getattr(np, func), getattr(nl, func)
        args = [slicey.start, slicey.stop]
        if func == "arange":
            args.append(slicey.step)
        elif func == "logspace":
            args.extend([num, True, base])
        else:
            args.append(num)
        self.assertArraysMatch(nl_func(*args), np_func(*args))

    @hy.given(st.lists(sliceish(non_empty=True), min_size=1, max_size=4))
    def test_nd_grids(self, slice_list):
        args = tuple(slice_list)
        meshed = nl.mgrid[args]
        opened = nl.ogrid[args]
        self.assertArraysMatch(meshed, np.mgrid[args])
        self.assertArraysAllMatch(opened, np.ogrid[args])
        opened = swap_ab(opened)
        meshed = tuple(meshed)
        self.assertArraysAllMatch(nl.meshgrid(*opened), swap_ab(meshed))
        self.assertArraysMatch(nl.c_[meshed], np.c_[meshed])
        self.assertArraysMatch(nl.r_[meshed], np.r_[meshed])
        self.assertArraysMatch(nl.r_[args], np.r_[args])

    @hy.given(params['ip'], params['rp'])
    def test_fft_freq(self, length, spacing):
        with self.assertRaises(ValueError):
            raise ValueError
        self.assertArraysMatch(nl.fft.fftfreq(length, spacing),
                               np.fft.fftfreq(length, spacing))
        self.assertArraysMatch(nl.fft.rfftfreq(length, spacing),
                               np.fft.rfftfreq(length, spacing))


# @unittest.skip("Freezes")
class TestRandom(TestWrappers):
    """Testing randon number generation"""
    np_rng: np.random.Generator
    nl_rng: nl.random.LnGenerator

    def setUp(self):
        super().setUp()
        self.np_rng = nl.random.default_rng()
        self.nl_rng = nl.random.default_rng()

    def assertRandomMatch(self, func: str, args: Sequence[float],
                          scalar: bool = True):
        """Assert outputs of random.Generator have correct type, shape, etc.

        Parameters
        ----------
        func : str
            Name of `Generator` method to test
        args : Sequence[float, ...]
            Arguments for generator method
        scalar : bool
            Without a shape parameter, should the output be a scalar?
            By default: `True`
        """
        np_method = getattr(self.np_rng, func)
        nl_method = getattr(self.nl_rng, func)
        np_array, nl_array = np_method(*args), nl_method(*args)
        msg = f"func={func}, args={args[:-1]}, shape={args[-1]}"
        self.assertArraysMatch(nl_array, np_array, val=False, msg=msg)
        if scalar:
            self.assertIsInstance(nl_method(*args[:-1]), Real, msg=msg)

    @hy.given(hyn.vectors, params['iop'])
    def test_random_bytes(self, vec: np.ndarray, num: int):
        nl_bytes = self.nl_rng.bytes(num)
        self.assertIsInstance(nl_bytes, bytes)
        self.assertEqual(len(nl_bytes), num)

    @hy.given(hyn.vectors, params['iop'])
    def test_random_permute(self, vec: np.ndarray, num: int):
        nl_perm = self.nl_rng.permutation(num)
        self.assertIsInstance(nl_perm, nl.lnarray)
        self.assertArrayShape(nl_perm, (num,))
        self.assertArrayDtype(nl_perm, int)

        nl_perm = self.nl_rng.permutation(vec)
        self.assertIsInstance(nl_perm, nl.lnarray)
        self.assertArrayShape(nl_perm, vec.shape)
        self.assertArrayDtype(nl_perm, vec.dtype)

    @hy.given(hyn.vectors, params['iop'])
    def test_random_choice(self, vec: np.ndarray, num: int):
        nl_choice = self.nl_rng.choice(vec, size=num)
        self.assertIsInstance(nl_choice, nl.lnarray)
        self.assertArrayShape(nl_choice, (num,))
        self.assertArrayDtype(nl_choice, vec.dtype)

    @hy.given(hyn.vectors, params['iop'])
    def test_random_shuffle(self, vec: np.ndarray, num: int):
        shape, dtype = vec.shape, vec.dtype
        nl_shuffle = self.nl_rng.shuffle(vec)
        self.assertIsNone(nl_shuffle)
        self.assertIsInstance(vec, np.ndarray)
        self.assertArrayShape(vec, shape)
        self.assertArrayDtype(vec, dtype)

    @hy.given(func_params('r_rop'))
    def test_random_r_rop(self, func_args):
        func, args = func_args
        self.assertRandomMatch(func, args)

    @hy.given(func_params('rp_rp'))
    def test_random_rp_rp(self, func_args):
        func, args = func_args
        self.assertRandomMatch(func, args)

    @hy.given(func_params('rop'))
    def test_random_rop(self, func_args):
        func, args = func_args
        self.assertRandomMatch(func, args)

    @hy.given(func_params('rp'))
    def test_random_rp(self, func_args):
        func, args = func_args
        self.assertRandomMatch(func, args)

    @hy.given(func_params('z'))
    def test_random_z(self, func_args):
        func, args = func_args
        self.assertRandomMatch(func, args)

    @hy.given(func_params('p'))
    def test_random_p(self, func_args):
        func, args = func_args
        self.assertRandomMatch(func, args)

    @hy.given(func_params('none'))
    def test_random_none(self, func_args):
        func, args = func_args
        args = args[-1:]
        self.assertRandomMatch(func, args)

    @hy.given(func_params('others', 'noncentral'))
    def test_random_others(self, func_args):
        func, args = func_args
        if func == "integers":
            args = sorted(args[:-1]) + args[-1:]
            hy.assume(args[0] < args[1])
        elif func == "hypergeometric":
            if args[0] + args[1] <= args[2]:
                args = sorted(args[:-1], reverse=True) + args[-1:]
            hy.assume(args[0] + args[1] > args[2])
        elif func == "triangular":
            args = sorted(args[:-1]) + args[-1:]
            hy.assume(args[0] < args[2])
        self.assertRandomMatch(func, args)

    @hy.given(func_params('multi', 'multivariate'))
    @hy.settings(deadline=None)
    def test_random_multi(self, func_args):
        func, args = func_args
        if func == "multinomial":
            args[1] /= args[1].sum()
        elif func == "multivariate_hypergeometric":
            hy.assume(args[0].sum() >= args[1])
        elif func == "multivariate_normal":
            args = list(args[0]) + args[1:]
            args[1] = make_cov(args[1])
        self.assertRandomMatch(func, args, scalar=False)


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
