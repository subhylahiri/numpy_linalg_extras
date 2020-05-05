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
    from test_gufunc import utn, hn, main
    from test_linalg import trnsp, insert
else:
    from .test_gufunc import utn, hn, main
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

    @hy.given(hn.broadcastable('(a,b),(b,a),(a,a),(b,b)', 'd'))
    def test_return_array_types(self, arrays):
        m_sb_n, m_bs_n = arrays[:2]
        m_sb, m_bs, m_ss, m_bb = view_as(*arrays)
        m_bs_m, m_ss_m, m_bb_m = hn.core_only(m_bs, m_ss, m_bb)
        hy.assume(hn.all_non_singular(m_ss))
        hy.assume(hn.all_non_singular(m_bb))
        hy.assume(m_sb.ndim != m_ss.ndim - 1)  # np..solve's broadcasting issue

        self.assertIsInstance(m_sb @ m_bs, la.lnarray)
        self.assertIsInstance(m_sb_n @ m_bs, la.lnarray)
        tw_o = np.empty(return_shape('(a,b),(b,c)->(a,c)', m_bs, m_sb), 'd')
        tw_r = la.matmul(m_bs, m_sb_n, tw_o)
        self.assertIsInstance(tw_r, np.ndarray)
        self.assertIsInstance(tw_o, np.ndarray)
        self.assertIsInstance(np.matmul(m_bs, m_sb_n), np.ndarray)
        self.assertIsInstance(la.solve(m_ss, m_sb_n), la.lnarray)
        self.assertIsInstance(npl.solve(m_ss, m_sb_n), np.ndarray)
        self.assertIsInstance(la.lstsq(m_bs, m_bb), la.lnarray)
        self.assertIsInstance(npl.lstsq(m_bs_m, m_bb_m, rcond=None)[0],
                              np.ndarray)
        self.assertIsInstance(la.lu(m_ss)[0], la.lnarray)
        self.assertIsInstance(la.lu(m_bs_n)[0], np.ndarray)
        self.assertIsInstance(la.qr(m_ss)[0], la.lnarray)
        self.assertIsInstance(la.qr(m_bs_n)[0], np.ndarray)
        self.assertIsInstance(la.lq(m_ss)[0], la.lnarray)
        self.assertIsInstance(la.lq(m_bs_n)[0], np.ndarray)
        self.assertIsInstance(la.lqr(m_ss)[0], la.lnarray)
        self.assertIsInstance(la.lqr(m_bs_n)[0], np.ndarray)
        self.assertIsInstance(npl.qr(m_ss_m)[0], np.ndarray)

    @hy.given(hn.broadcastable('(a,b),(b,b)', 'D'))
    def test_lnarray_shape_methods(self, arrays):
        m_bs, m_ss = view_as(*arrays)
        tall, smol = m_bs.shape, m_ss.shape
        hy.assume(hn.tall(m_bs))
        hy.assume(np.max(np.abs(m_bs.imag)) > .01)
        hy.assume(np.max(np.abs(m_bs.real)) / np.max(np.abs(m_bs.imag)) < 1e3)

        expect = trnsp(tall)
        self.assertArrayShape(m_bs.t, expect)
        self.assertArrayShape(m_bs.h, expect)
        self.assertArrayNotAllClose(m_bs.t, m_bs.h)
        self.assertArrayShape(m_ss.c, smol + (1,))
        self.assertArrayShape(m_bs.c.uc, tall)
        expect = insert(smol)
        self.assertArrayShape(m_ss.r, expect)
        self.assertArrayShape(m_bs.r.ur, tall)
        self.assertArrayShape(m_ss.s, smol + (1, 1))
        self.assertArrayShape(m_bs.s.us, tall)

        # expect = smol[:1] + (1,) + smol[1:2] + (1,) + smol[2:]
        expect = insert(insert(smol, 2), 1)
        self.assertArrayShape(m_ss.expand_dims(1, 3), expect)
        expect = tall[:1] + (np.prod(tall[1:4]),) + tall[4:]
        self.assertArrayShape(m_bs.flattish(1, 4), expect)
        with self.assertRaisesRegex(ValueError, "repeated axes"):
            m_bs.expand_dims(m_bs.ndim - 1, -3)
        half = (m_bs.ndim + 2) // 2 + 1
        with self.assertRaises(ValueError):
            (m_bs.s).flattish(half, -half)

    @hy.given(hn.broadcastable('(a,b),(b,b),(b)', None))
    def test_lnarray_operations_return_expected_values(self, arrays):
        m_bs, m_ss, vec = view_as(*arrays)
        m_bs_m = hn.core_only(m_bs)
        vec = hn.core_only(vec, dims=1)
        hy.assume(hn.tall(m_bs))
        hy.assume(m_ss.ndim != 3)  # causes np..solve's broadcasting issue
        hy.assume(hn.all_non_singular(m_ss))

        ts_o = np.empty(return_shape('(a,b),(b,c)->(a,c)', m_bs, m_ss),
                        m_ss.dtype)
        ts_r = la.matmul(m_bs, m_ss, ts_o)
        self.assertArrayAllClose(ts_r, ts_o)
        self.assertArrayAllClose(m_bs @ m_ss, np.matmul(m_bs, m_ss))
        self.assertArrayAllClose(m_bs @ m_ss, np.matmul(m_bs, m_ss))
        self.assertArrayAllClose(m_bs @ vec, np.matmul(m_bs, vec))
        self.assertArrayAllClose(gf.solve(m_ss, vec), npl.solve(m_ss, vec.c).uc)
        self.assertArrayAllClose(gf.lstsq(m_bs_m.t, vec),
                                 npl.lstsq(m_bs_m.t, vec, rcond=None)[0])
        self.assertArrayAllClose(gf.rmatmul(m_ss, m_bs), np.matmul(m_bs, m_ss))
        # m_bs @= m_ss
        # self.assertArrayAllClose(ts_r, m_bs)


class TestPinvarray(utn.TestCaseNumpy):
    """test pinvarray & invarray classes
    """

    @hy.given(hn.broadcastable('(a,a),(b,a)', ['d', 'D']))
    def test_pinvarray_attribute_types(self, arrays):
        m_ss, m_bs = view_as(*arrays)
        hy.assume(hn.tall(m_bs))
        self.assertIsInstance(m_ss.pinv, la.pinvarray)
        self.assertIsInstance(m_ss.inv, la.invarray)
        self.assertIs(m_ss.pinv.dtype, m_bs.dtype)
        self.assertIsInstance(m_ss.pinv.pinv, la.lnarray)
        self.assertIsInstance(m_ss.inv.inv, la.lnarray)
        self.assertIsInstance(m_ss.pinv(), la.lnarray)
        self.assertIsInstance(m_ss.inv(), la.lnarray)

        m_bs_p = la.pinvarray(m_bs)
        self.assertIsInstance(m_bs_p, la.pinvarray)
        self.assertIsInstance(m_bs_p.pinv, la.lnarray)
        self.assertIsInstance(2 * m_bs_p, la.pinvarray)
        self.assertIsInstance((2 * m_bs_p).pinv, la.lnarray)
        pout = la.pinvarray(la.empty_like(m_bs))
        np.multiply(2, m_bs_p, pout)
        self.assertIsInstance(pout, la.pinvarray)
        self.assertIsInstance(pout.pinv, la.lnarray)
        with self.assertRaises(AttributeError):
            m_bs_p.inv  # pylint: disable=no-member,pointless-statement
        with self.assertRaises(TypeError):
            m_ss.inv.pinv  # pylint: disable=pointless-statement

    @hy.given(hyn.arrays('d', hyn.array_shapes(min_dims=2)))
    def test_pinvarray_shape_methods(self, array):
        m_bs = array.view(la.lnarray)
        hy.assume(hn.tall(m_bs))
        m_bs_p = m_bs.pinv
        expect = trnsp(m_bs.shape)
        self.assertEqual(m_bs_p.ndim, len(expect))
        self.assertEqual(m_bs_p.shape, expect)
        self.assertEqual(m_bs_p.size, np.prod(expect))
        self.assertArrayShape(m_bs_p(), expect)
        with self.assertRaises(ValueError):
            m_bs.inv  # pylint: disable=pointless-statement
        m_bs_p = m_bs.c.pinv
        expect = insert(m_bs.shape)
        now_expect = expect[1::-1] + expect[2:]
        self.assertArrayShape(m_bs_p.swapaxes(0, 1), now_expect)
        now_expect = expect[2::-1] + expect[3:]
        self.assertArrayShape(m_bs_p.swapaxes(0, 2), now_expect)
        now_expect = trnsp(expect)
        self.assertArrayShape(m_bs_p.swapaxes(-1, -2), now_expect)

    @hy.given(hn.broadcastable('(a,b),(b,a),(b,a)', None))
    def test_pinvarray_in_functions(self, arrays):
        m_sb, high, m_bs = view_as(*arrays)
        hy.assume(hn.tall(m_bs))

        self.assertArrayAllClose(gf.matmul(m_bs.pinv, high),
                                 gf.lstsq(m_bs, high))
        self.assertArrayAllClose(gf.matmul(m_sb, m_bs.pinv.t),
                                 gf.rlstsq(m_sb, m_bs.t))
        xpout = la.pinvarray(la.empty_like(m_bs))
        m_bs_p = np.multiply(m_bs.pinv, 2, out=xpout)
        self.assertArrayAllClose(m_bs_p.pinv, xpout.pinv)
        self.assertArrayAllClose(m_bs_p.pinv, m_bs / 2)
        with self.assertRaises(TypeError):
            gf.matmul(m_bs.pinv, m_sb.pinv)
        self.assertArrayAllClose(gf.lstsq(m_sb.pinv, high),
                                 gf.matmul(m_sb, high))
        with self.assertRaises(TypeError):
            gf.lstsq(high, m_sb.pinv)
        self.assertArrayAllClose(gf.rlstsq(high.t, m_sb.t.pinv),
                                 gf.matmul(high.t, m_sb.t))
        with self.assertRaises(TypeError):
            gf.rlstsq(m_sb.t.pinv, high.t)
        with self.assertRaises(TypeError):
            gf.rmatmul(m_sb.pinv, m_bs.pinv)
        with self.assertRaises(TypeError):
            gf.solve(m_sb.pinv, high)
        with self.assertRaises(TypeError):
            gf.rsolve(m_sb, m_bs.pinv)

    @hy.given(hn.broadcastable('(a,a),(b,a),(a,b)', None))
    def test_invarray_in_functions(self, arrays):
        m_ss, m_bs, m_sb = view_as(*arrays)
        mini = m_bs[..., :m_ss.shape[-1], :]
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_non_singular(m_ss))

        self.assertArrayAllClose(gf.matmul(m_ss.inv, m_sb),
                                 gf.solve(m_ss, m_sb))
        self.assertArrayAllClose(gf.matmul(m_bs, m_ss.inv),
                                 gf.rsolve(m_bs, m_ss))
        self.assertArrayAllClose(gf.matmul(m_ss.inv, mini.inv).inv, mini @ m_ss)
        self.assertArrayAllClose(gf.solve(m_ss.inv, m_sb),
                                 gf.matmul(m_ss, m_sb))
        self.assertArrayAllClose(gf.solve(mini, m_ss.inv).inv,
                                 gf.matmul(m_ss, mini))
        self.assertArrayAllClose(gf.solve(mini.inv, m_ss.inv),
                                 gf.rsolve(mini, m_ss))
        self.assertArrayAllClose(gf.rsolve(m_ss, mini.inv),
                                 gf.matmul(m_ss, mini))
        self.assertArrayAllClose(gf.rsolve(mini.inv, m_ss).inv,
                                 gf.matmul(m_ss, mini))
        self.assertArrayAllClose(gf.rsolve(mini.inv, m_ss.inv),
                                 gf.solve(mini, m_ss))
        self.assertArrayAllClose(gf.rmatmul(m_ss, mini.inv),
                                 gf.solve(mini, m_ss))
        self.assertArrayAllClose(gf.rmatmul(mini.inv, m_ss),
                                 gf.rsolve(m_ss, mini))
        self.assertArrayAllClose(gf.rmatmul(mini.inv, m_ss.inv).inv, mini @ m_ss)

    @hy.given(hn.broadcastable('(a,a),(b,a),(a,b)', None))
    def test_bad_p_invarray_combos_in_functions(self, arrays):
        m_ss, m_bs, m_sb = view_as(*arrays)
        mini = m_bs[..., :m_ss.shape[-1], :]
        hy.assume(hn.tall(m_bs))

        with self.assertRaises(TypeError):
            la.solve(m_sb.pinv, mini)
        with self.assertRaises(TypeError):
            la.rsolve(mini, m_bs.pinv)
        with self.assertRaises(TypeError):
            la.solve(mini, m_bs.pinv)
        with self.assertRaises(TypeError):
            la.rsolve(m_sb.pinv, mini)
        with self.assertRaises(TypeError):
            la.solve(m_sb.pinv, m_bs.pinv)
        with self.assertRaises(TypeError):
            la.rsolve(m_sb.pinv, m_bs.pinv)
        with self.assertRaises(TypeError):
            la.matmul(m_ss.inv, m_bs.pinv)
        with self.assertRaises(TypeError):
            la.matmul(m_sb.pinv, mini.inv)
        with self.assertRaises(TypeError):
            la.solve(m_bs.pinv, mini.inv)
        with self.assertRaises(TypeError):
            la.rsolve(mini.inv, m_sb.pinv)

    @hy.given(hn.broadcastable('(a,a),(b,a),(a,b)', None))
    def test_good_p_invarray_combos_in_lstsq(self, arrays):
        m_ss, m_bs, m_sb = view_as(*arrays)
        mini = m_bs[..., :m_ss.shape[-1], :]
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_non_singular(m_ss))
        hy.assume(hn.all_non_singular(mini))

        self.assertArrayAllClose(la.lstsq(mini.inv, m_sb),
                                 la.matmul(mini, m_sb))
        self.assertArrayAllClose(la.rlstsq(m_bs, mini.inv),
                                 la.matmul(m_bs, mini))
        self.assertArrayAllClose(la.lstsq(mini.inv, m_bs.pinv),
                                 la.rlstsq(mini, m_bs))
        self.assertArrayAllClose(la.rlstsq(mini.inv, m_sb.pinv),
                                 la.solve(mini, m_sb))
        self.assertArrayAllClose(la.lstsq(mini.inv, m_ss.inv),
                                 la.rsolve(mini, m_ss))
        self.assertArrayAllClose(la.rlstsq(mini.inv, m_ss.inv),
                                 la.solve(mini, m_ss))
        self.assertArrayAllClose(la.lstsq(m_bs.pinv, m_ss.inv),
                                 la.rsolve(m_bs, m_ss))
        self.assertArrayAllClose(la.rlstsq(m_sb.pinv, m_ss.inv),
                                 la.lstsq(m_sb, m_ss))

    @hy.given(hn.broadcastable('(a,a),(b,a),(a,b)', None))
    def test_good_p_invarray_combos_in_solve(self, arrays):
        m_ss, m_bs, m_sb = view_as(*arrays)
        mini = m_bs[..., :m_ss.shape[-1], :]
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_non_singular(m_ss))
        hy.assume(hn.all_non_singular(mini))

        self.assertArrayAllClose(la.solve(m_ss.inv, m_bs.pinv),
                                 la.rlstsq(m_ss, m_bs))
        self.assertArrayAllClose(la.rsolve(m_sb.pinv, mini.inv),
                                 la.lstsq(m_sb, mini))

    @hy.given(hn.broadcastable('(a,b),(b,a),(b,a),()', None))
    def test_pinvarray_operators(self, arrays):
        m_sb, high, m_bs, scal = view_as(*arrays)
        scal = scal.s
        hy.assume(hn.tall(m_bs))

        self.assertArrayAllClose(m_bs.pinv @ high, gf.lstsq(m_bs, high))
        self.assertArrayAllClose(m_bs.pinv() @ high, gf.lstsq(m_bs, high))
        self.assertArrayAllClose(m_sb @ m_bs.pinv.t, gf.rlstsq(m_sb, m_bs.t))
        with self.assertRaises(TypeError):
            m_bs.pinv @ m_sb.pinv  # pylint: disable=pointless-statement
        self.assertArrayAllClose((m_bs.pinv * 3.5).pinv, m_bs / 3.5)
        self.assertArrayAllClose((2.4 * m_bs.pinv).pinv, m_bs / 2.4)
        self.assertArrayAllClose((m_bs.pinv / 3.564).pinv, m_bs * 3.564)
        with self.assertRaises(TypeError):
            65 / m_bs.pinv  # pylint: disable=pointless-statement
        self.assertArrayAllClose((m_bs.pinv * scal).pinv, m_bs / scal)
        self.assertArrayAllClose((scal * m_bs.pinv).pinv, m_bs / scal)
        self.assertArrayAllClose((m_bs.pinv / scal).pinv, m_bs * scal)
        with self.assertRaises(TypeError):
            scal / m_bs.pinv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            scal.pinv * m_bs.pinv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            m_bs.pinv + m_sb  # pylint: disable=pointless-statement
        told = 1. * m_bs
        m_bs_p = m_bs.pinv
        m_bs_p *= 2
        self.assertArrayAllClose(m_bs, told / 2)

    @hy.given(hn.broadcastable('(a,a),(b,a),(a,b),()', None))
    def test_invarray_operators(self, arrays):
        m_ss, m_bs, m_sb, scal = view_as(*arrays)
        scal = scal.s
        mini = m_bs[..., :m_ss.shape[-1], :]
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_non_singular(m_ss))
        hy.assume(hn.all_non_singular(mini))

        self.assertArrayAllClose(m_ss.inv @ m_sb, gf.solve(m_ss, m_sb))
        self.assertArrayAllClose(m_ss.inv() @ m_sb, gf.solve(m_ss, m_sb))
        self.assertArrayAllClose(m_bs @ m_ss.inv, gf.rsolve(m_bs, m_ss))
        self.assertArrayAllClose((m_ss.inv @ mini.inv).inv, mini @ m_ss)
        self.assertArrayAllClose((m_ss.inv * 3.5).inv, m_ss / 3.5)
        self.assertArrayAllClose((2.4 * m_ss.inv).inv, m_ss / 2.4)
        self.assertArrayAllClose((m_ss.inv / 3.564).inv, m_ss * 3.564)
        with self.assertRaises(TypeError):
            45.564 / m_ss.inv  # pylint: disable=pointless-statement
        self.assertArrayAllClose((mini.inv * scal).inv, mini / scal)
        self.assertArrayAllClose((scal * mini.inv).inv, mini / scal)
        self.assertArrayAllClose((mini.inv / scal).inv, mini * scal)
        with self.assertRaises(TypeError):
            scal / mini.inv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            scal.inv * mini.inv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            m_ss + mini.inv  # pylint: disable=pointless-statement
        mini_old = 1. * mini
        mini_i = mini.inv
        mini_i @= m_ss.inv
        self.assertArrayAllClose(mini, m_ss @ mini_old)


if __name__ == '__main__':
    main(verbosity=2)
