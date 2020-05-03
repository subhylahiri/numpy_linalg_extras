# -*- coding: utf-8 -*-
"""Test lnarray class
"""
import hypothesis as hy
import hypothesis.extra.numpy as hyn
import numpy as np
import numpy.linalg as npl
import numpy_linalg as la
import numpy_linalg.gufuncs as gf
from numpy_linalg.gufuncs import array_return_shape as return_shape
if __name__.find('tests.') < 0:
    # pylint: disable=import-error
    import unittest_numpy as utn
    from test_linalg import trnsp, insert
else:
    from . import unittest_numpy as utn
    from .test_linalg import trnsp, insert
# pylint: disable=missing-function-docstring
# =============================================================================
__all__ = ['TestArray', 'TestPinvarray']
# =============================================================================
# new class helper
# =============================================================================


def view_as(*arrays: np.ndarray, kind: type = la.lnarray) -> la.lnarray:
    """Convert array types

    Parameters
    ----------
    arrays : np.ndarray
        Arrays to convert.
    kind : type, optional
        Number of core dimensions to leave, by default `la.lnarray`.

    Returns
    -------
    views : la.lnarray
        Converted arrays.
    """
    result = tuple(arr.view(kind) for arr in arrays)
    return result[0] if len(result) == 1 else result


# =============================================================================
# Test python classes
# =============================================================================


class TestArray(utn.TestCaseNumpy):
    """Testing lnarray"""

    def setUp(self):
        self.sctype = ['i']
        super().setUp()

    @hy.given(utn.broadcastable('(a,b),(b,a),(a,a),(b,b)', 'd'))
    def test_return_array_types(self, arrays):
        wide_n, tall_n = arrays[:2]
        wide, tall, smol, big = view_as(*arrays)
        tall_m, small_m, big_m = utn.core_only(tall, smol, big)
        hy.assume(np.all(utn.non_singular(smol)))
        hy.assume(np.all(utn.non_singular(big)))
        hy.assume(wide.ndim != smol.ndim - 1)  # np..solve's broadcasting issue

        self.assertIsInstance(wide @ tall, la.lnarray)
        self.assertIsInstance(wide_n @ tall, la.lnarray)
        tw_o = np.empty(return_shape('(a,b),(b,c)->(a,c)', tall, wide), 'd')
        tw_r = la.matmul(tall, wide_n, tw_o)
        self.assertIsInstance(tw_r, np.ndarray)
        self.assertIsInstance(tw_o, np.ndarray)
        self.assertIsInstance(np.matmul(tall, wide_n), np.ndarray)
        self.assertIsInstance(la.solve(smol, wide_n), la.lnarray)
        self.assertIsInstance(npl.solve(smol, wide_n), np.ndarray)
        self.assertIsInstance(la.lstsq(tall, big), la.lnarray)
        self.assertIsInstance(npl.lstsq(tall_m, big_m, rcond=None)[0],
                              np.ndarray)
        self.assertIsInstance(la.lu(smol)[0], la.lnarray)
        self.assertIsInstance(la.lu(tall_n)[0], np.ndarray)
        self.assertIsInstance(la.qr(smol)[0], la.lnarray)
        self.assertIsInstance(la.qr(tall_n)[0], np.ndarray)
        self.assertIsInstance(la.lq(smol)[0], la.lnarray)
        self.assertIsInstance(la.lq(tall_n)[0], np.ndarray)
        self.assertIsInstance(la.lqr(smol)[0], la.lnarray)
        self.assertIsInstance(la.lqr(tall_n)[0], np.ndarray)
        self.assertIsInstance(npl.qr(small_m)[0], np.ndarray)

    @hy.given(utn.broadcastable('(a,b),(b,b)', 'D'))
    def test_lnarray_shape_methods(self, arrays):
        tall, square = view_as(*arrays)
        tall_sh, square_sh = tall.shape, square.shape
        hy.assume(tall_sh[-2] > tall_sh[-1])
        hy.assume(np.max(np.abs(tall.imag)) > .01)
        hy.assume(np.max(np.abs(tall.real)) / np.max(np.abs(tall.imag)) < 1e3)

        expect = trnsp(tall_sh)
        self.assertArrayShape(tall.t, expect)
        self.assertArrayShape(tall.h, expect)
        self.assertArrayNotAllClose(tall.t, tall.h)
        self.assertArrayShape(square.c, square_sh + (1,))
        self.assertArrayShape(tall.c.uc, tall_sh)
        expect = insert(square_sh)
        self.assertArrayShape(square.r, expect)
        self.assertArrayShape(tall.r.ur, tall_sh)
        self.assertArrayShape(square.s, square_sh + (1, 1))
        self.assertArrayShape(tall.s.us, tall_sh)

        # expect = square_sh[:1] + (1,) + square_sh[1:2] + (1,) + square_sh[2:]
        expect = insert(insert(square_sh, 2), 1)
        self.assertArrayShape(square.expand_dims(1, 3), expect)
        expect = tall_sh[:1] + (np.prod(tall_sh[1:4]),) + tall_sh[4:]
        self.assertArrayShape(tall.flattish(1, 4), expect)
        with self.assertRaisesRegex(ValueError, "repeated axes"):
            tall.expand_dims(tall.ndim - 1, -3)
        half = (tall.ndim + 2) // 2 + 1
        with self.assertRaises(ValueError):
            (tall.s).flattish(half, -half)

    @hy.given(utn.broadcastable('(a,b),(b,b),(b)', None))
    def test_lnarray_operations_return_expected_values(self, arrays):
        tall, smol, vec = view_as(*arrays)
        tall_m = utn.core_only(tall)
        vec = utn.core_only(vec, dims=1)
        hy.assume(tall.shape[-2] > tall.shape[-1])
        hy.assume(smol.ndim != 3)  # causes np..solve's broadcasting issue
        hy.assume(np.all(utn.non_singular(smol)))

        ts_o = np.empty(return_shape('(a,b),(b,c)->(a,c)', tall, smol),
                        smol.dtype)
        ts_r = la.matmul(tall, smol, ts_o)
        self.assertArrayAllClose(ts_r, ts_o)
        self.assertArrayAllClose(tall @ smol, np.matmul(tall, smol))
        self.assertArrayAllClose(tall @ smol, np.matmul(tall, smol))
        self.assertArrayAllClose(tall @ vec, np.matmul(tall, vec))
        self.assertArrayAllClose(gf.solve(smol, vec), npl.solve(smol, vec.c).uc)
        self.assertArrayAllClose(gf.lstsq(tall_m.t, vec),
                                 npl.lstsq(tall_m.t, vec, rcond=None)[0])
        self.assertArrayAllClose(gf.rmatmul(smol, tall), np.matmul(tall, smol))
        # tall @= smol
        # self.assertArrayAllClose(ts_r, tall)


class TestPinvarray(utn.TestCaseNumpy):
    """test pinvarray & invarray classes
    """

    @hy.given(utn.broadcastable('(a,a),(b,a)', ['d', 'D']))
    def test_pinvarray_attribute_types(self, arrays):
        smol, tall = view_as(*arrays)
        hy.assume(tall.shape[-2] > tall.shape[-1])
        self.assertIsInstance(smol.pinv, la.pinvarray)
        self.assertIsInstance(smol.inv, la.invarray)
        self.assertIs(smol.pinv.dtype, tall.dtype)
        self.assertIsInstance(smol.pinv.pinv, la.lnarray)
        self.assertIsInstance(smol.inv.inv, la.lnarray)
        self.assertIsInstance(smol.pinv(), la.lnarray)
        self.assertIsInstance(smol.inv(), la.lnarray)

        tall_p = la.pinvarray(tall)
        self.assertIsInstance(tall_p, la.pinvarray)
        self.assertIsInstance(tall_p.pinv, la.lnarray)
        self.assertIsInstance(2 * tall_p, la.pinvarray)
        self.assertIsInstance((2 * tall_p).pinv, la.lnarray)
        pout = la.pinvarray(la.empty_like(tall))
        np.multiply(2, tall_p, pout)
        self.assertIsInstance(pout, la.pinvarray)
        self.assertIsInstance(pout.pinv, la.lnarray)
        with self.assertRaises(AttributeError):
            tall_p.inv  # pylint: disable=no-member,pointless-statement
        with self.assertRaises(TypeError):
            smol.inv.pinv  # pylint: disable=pointless-statement

    @hy.given(hyn.arrays('d', hyn.array_shapes(min_dims=2)))
    def test_pinvarray_shape_methods(self, array):
        tall = array.view(la.lnarray)
        hy.assume(tall.shape[-2] > tall.shape[-1])
        tall_p = tall.pinv
        expect = trnsp(tall.shape)
        self.assertEqual(tall_p.ndim, len(expect))
        self.assertEqual(tall_p.shape, expect)
        self.assertEqual(tall_p.size, np.prod(expect))
        self.assertArrayShape(tall_p(), expect)
        with self.assertRaises(ValueError):
            tall.inv  # pylint: disable=pointless-statement
        tall_p = tall.c.pinv
        expect = insert(tall.shape)
        now_expect = expect[1::-1] + expect[2:]
        self.assertArrayShape(tall_p.swapaxes(0, 1), now_expect)
        now_expect = expect[2::-1] + expect[3:]
        self.assertArrayShape(tall_p.swapaxes(0, 2), now_expect)
        now_expect = trnsp(expect)
        self.assertArrayShape(tall_p.swapaxes(-1, -2), now_expect)

    @hy.given(utn.broadcastable('(a,b),(b,a),(b,a)', None))
    def test_pinvarray_in_functions(self, arrays):
        wide, high, tall = view_as(*arrays)
        hy.assume(tall.shape[-2] > tall.shape[-1])

        self.assertArrayAllClose(gf.matmul(tall.pinv, high),
                                 gf.lstsq(tall, high))
        self.assertArrayAllClose(gf.matmul(wide, tall.pinv.t),
                                 gf.rlstsq(wide, tall.t))
        xpout = la.pinvarray(la.empty_like(tall))
        tall_p = np.multiply(tall.pinv, 2, out=xpout)
        self.assertArrayAllClose(tall_p.pinv, xpout.pinv)
        self.assertArrayAllClose(tall_p.pinv, tall / 2)
        with self.assertRaises(TypeError):
            gf.matmul(tall.pinv, wide.pinv)
        self.assertArrayAllClose(gf.lstsq(wide.pinv, high),
                                 gf.matmul(wide, high))
        with self.assertRaises(TypeError):
            gf.lstsq(high, wide.pinv)
        self.assertArrayAllClose(gf.rlstsq(high.t, wide.t.pinv),
                                 gf.matmul(high.t, wide.t))
        with self.assertRaises(TypeError):
            gf.rlstsq(wide.t.pinv, high.t)
        with self.assertRaises(TypeError):
            gf.rmatmul(wide.pinv, tall.pinv)
        with self.assertRaises(TypeError):
            gf.solve(wide.pinv, high)
        with self.assertRaises(TypeError):
            gf.rsolve(wide, tall.pinv)

    @hy.given(utn.broadcastable('(a,a),(b,a),(a,b)', None))
    def test_invarray_in_functions(self, arrays):
        smol, tall, wide = view_as(*arrays)
        mini = tall[..., :smol.shape[-1], :]
        hy.assume(tall.shape[-2] > tall.shape[-1])
        hy.assume(np.all(utn.non_singular(smol)))

        self.assertArrayAllClose(gf.matmul(smol.inv, wide),
                                 gf.solve(smol, wide))
        self.assertArrayAllClose(gf.matmul(tall, smol.inv),
                                 gf.rsolve(tall, smol))
        self.assertArrayAllClose(gf.matmul(smol.inv, mini.inv).inv, mini @ smol)
        self.assertArrayAllClose(gf.solve(smol.inv, wide),
                                 gf.matmul(smol, wide))
        self.assertArrayAllClose(gf.solve(mini, smol.inv).inv,
                                 gf.matmul(smol, mini))
        self.assertArrayAllClose(gf.solve(mini.inv, smol.inv),
                                 gf.rsolve(mini, smol))
        self.assertArrayAllClose(gf.rsolve(smol, mini.inv),
                                 gf.matmul(smol, mini))
        self.assertArrayAllClose(gf.rsolve(mini.inv, smol).inv,
                                 gf.matmul(smol, mini))
        self.assertArrayAllClose(gf.rsolve(mini.inv, smol.inv),
                                 gf.solve(mini, smol))
        self.assertArrayAllClose(gf.rmatmul(smol, mini.inv),
                                 gf.solve(mini, smol))
        self.assertArrayAllClose(gf.rmatmul(mini.inv, smol),
                                 gf.rsolve(smol, mini))
        self.assertArrayAllClose(gf.rmatmul(mini.inv, smol.inv).inv, mini @ smol)

    @hy.given(utn.broadcastable('(a,a),(b,a),(a,b)', None))
    def test_bad_p_invarray_combos_in_functions(self, arrays):
        smol, tall, wide = view_as(*arrays)
        mini = tall[..., :smol.shape[-1], :]
        hy.assume(tall.shape[-2] > tall.shape[-1])

        with self.assertRaises(TypeError):
            la.solve(wide.pinv, mini)
        with self.assertRaises(TypeError):
            la.rsolve(mini, tall.pinv)
        with self.assertRaises(TypeError):
            la.solve(mini, tall.pinv)
        with self.assertRaises(TypeError):
            la.rsolve(wide.pinv, mini)
        with self.assertRaises(TypeError):
            la.solve(wide.pinv, tall.pinv)
        with self.assertRaises(TypeError):
            la.rsolve(wide.pinv, tall.pinv)
        with self.assertRaises(TypeError):
            la.matmul(smol.inv, tall.pinv)
        with self.assertRaises(TypeError):
            la.matmul(wide.pinv, mini.inv)
        with self.assertRaises(TypeError):
            la.solve(tall.pinv, mini.inv)
        with self.assertRaises(TypeError):
            la.rsolve(mini.inv, wide.pinv)

    @hy.given(utn.broadcastable('(a,a),(b,a),(a,b)', None))
    def test_good_p_invarray_combos_in_lstsq(self, arrays):
        smol, tall, wide = view_as(*arrays)
        mini = tall[..., :smol.shape[-1], :]
        hy.assume(tall.shape[-2] > tall.shape[-1])
        hy.assume(np.all(utn.non_singular(smol)))
        hy.assume(np.all(utn.non_singular(mini)))

        self.assertArrayAllClose(la.lstsq(mini.inv, wide),
                                 la.matmul(mini, wide))
        self.assertArrayAllClose(la.rlstsq(tall, mini.inv),
                                 la.matmul(tall, mini))
        self.assertArrayAllClose(la.lstsq(mini.inv, tall.pinv),
                                 la.rlstsq(mini, tall))
        self.assertArrayAllClose(la.rlstsq(mini.inv, wide.pinv),
                                 la.solve(mini, wide))
        self.assertArrayAllClose(la.lstsq(mini.inv, smol.inv),
                                 la.rsolve(mini, smol))
        self.assertArrayAllClose(la.rlstsq(mini.inv, smol.inv),
                                 la.solve(mini, smol))
        self.assertArrayAllClose(la.lstsq(tall.pinv, smol.inv),
                                 la.rsolve(tall, smol))
        self.assertArrayAllClose(la.rlstsq(wide.pinv, smol.inv),
                                 la.lstsq(wide, smol))

    @hy.given(utn.broadcastable('(a,a),(b,a),(a,b)', None))
    def test_good_p_invarray_combos_in_solve(self, arrays):
        smol, tall, wide = view_as(*arrays)
        mini = tall[..., :smol.shape[-1], :]
        hy.assume(tall.shape[-2] > tall.shape[-1])
        hy.assume(np.all(utn.non_singular(smol)))
        hy.assume(np.all(utn.non_singular(mini)))

        self.assertArrayAllClose(la.solve(smol.inv, tall.pinv),
                                 la.rlstsq(smol, tall))
        self.assertArrayAllClose(la.rsolve(wide.pinv, mini.inv),
                                 la.lstsq(wide, mini))

    @hy.given(utn.broadcastable('(a,b),(b,a),(b,a),()', None))
    def test_pinvarray_operators(self, arrays):
        wide, high, tall, scal = view_as(*arrays)
        scal = scal.s
        hy.assume(tall.shape[-2] > tall.shape[-1])

        self.assertArrayAllClose(tall.pinv @ high, gf.lstsq(tall, high))
        self.assertArrayAllClose(tall.pinv() @ high, gf.lstsq(tall, high))
        self.assertArrayAllClose(wide @ tall.pinv.t, gf.rlstsq(wide, tall.t))
        with self.assertRaises(TypeError):
            tall.pinv @ wide.pinv  # pylint: disable=pointless-statement
        self.assertArrayAllClose((tall.pinv * 3.5).pinv, tall / 3.5)
        self.assertArrayAllClose((2.4 * tall.pinv).pinv, tall / 2.4)
        self.assertArrayAllClose((tall.pinv / 3.564).pinv, tall * 3.564)
        with self.assertRaises(TypeError):
            65 / tall.pinv  # pylint: disable=pointless-statement
        self.assertArrayAllClose((tall.pinv * scal).pinv, tall / scal)
        self.assertArrayAllClose((scal * tall.pinv).pinv, tall / scal)
        self.assertArrayAllClose((tall.pinv / scal).pinv, tall * scal)
        with self.assertRaises(TypeError):
            scal / tall.pinv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            scal.pinv * tall.pinv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            tall.pinv + wide  # pylint: disable=pointless-statement
        told = 1. * tall
        tall_p = tall.pinv
        tall_p *= 2
        self.assertArrayAllClose(tall, told / 2)

    @hy.given(utn.broadcastable('(a,a),(b,a),(a,b),()', None))
    def test_invarray_operators(self, arrays):
        smol, tall, wide, scal = view_as(*arrays)
        scal = scal.s
        mini = tall[..., :smol.shape[-1], :]
        hy.assume(tall.shape[-2] > tall.shape[-1])
        hy.assume(np.all(utn.non_singular(smol)))
        hy.assume(np.all(utn.non_singular(mini)))

        self.assertArrayAllClose(smol.inv @ wide, gf.solve(smol, wide))
        self.assertArrayAllClose(smol.inv() @ wide, gf.solve(smol, wide))
        self.assertArrayAllClose(tall @ smol.inv, gf.rsolve(tall, smol))
        self.assertArrayAllClose((smol.inv @ mini.inv).inv, mini @ smol)
        self.assertArrayAllClose((smol.inv * 3.5).inv, smol / 3.5)
        self.assertArrayAllClose((2.4 * smol.inv).inv, smol / 2.4)
        self.assertArrayAllClose((smol.inv / 3.564).inv, smol * 3.564)
        with self.assertRaises(TypeError):
            45.564 / smol.inv  # pylint: disable=pointless-statement
        self.assertArrayAllClose((mini.inv * scal).inv, mini / scal)
        self.assertArrayAllClose((scal * mini.inv).inv, mini / scal)
        self.assertArrayAllClose((mini.inv / scal).inv, mini * scal)
        with self.assertRaises(TypeError):
            scal / mini.inv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            scal.inv * mini.inv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            smol + mini.inv  # pylint: disable=pointless-statement
        mini_old = 1. * mini
        mini_i = mini.inv
        mini_i @= smol.inv
        self.assertArrayAllClose(mini, smol @ mini_old)


if __name__ == '__main__':
    utn.main(verbosity=2)
