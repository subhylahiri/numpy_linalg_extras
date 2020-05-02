# -*- coding: utf-8 -*-
"""Test lnarray class
"""
import hypothesis as hy
import numpy as np
import numpy.linalg as npl
import numpy_linalg as la
import numpy_linalg.gufuncs as gf
from numpy_linalg.gufuncs import return_shape
if __name__.find('tests.') < 0:
    # pylint: disable=import-error
    import unittest_numpy as utn
    from test_gufunc import TestMatsVecs
else:
    from . import unittest_numpy as utn
    from .test_gufunc import TestMatsVecs
# pylint: disable=missing-function-docstring
# =============================================================================
__all__ = ['TestArray', 'TestPinvarray']
# =============================================================================
# Test python classes
# =============================================================================


class TestNewClasses(TestMatsVecs):
    """Testing lnarray, pinvarray, etc"""
    lones_ss: la.lnarray
    lones_sb: la.lnarray
    lones_bs: la.lnarray
    lid_s: la.lnarray
    lid_b: la.lnarray
    lv_s: la.lnarray
    lv_b: la.lnarray
    m_bs: la.lnarray
    m_sb: la.lnarray
    m_ss: la.lnarray
    m_bb: la.lnarray
    a_bs: la.lnarray
    a_sb: la.lnarray
    a_ss: la.lnarray
    a_bb: la.lnarray

    def setUp(self):
        super().setUp()
        for vname in self.varnames:
            np_arrs = getattr(self, '_' + vname)
            ln_arrs = {}
            for sctype in self.sctype:
                ln_arrs[sctype] = np_arrs[sctype].view(la.lnarray)
            setattr(self, '_l' + vname, ln_arrs)
        self.varnames += ['l' + vname for vname in self.varnames]


class TestArray(utn.TestCaseNumpy):
    """Testing lnarray"""

    def setUp(self):
        self.sctype = ['i']
        super().setUp()

    @hy.given(utn.broadcastable('(a,b),(b,a),(a,a),(b,b)', 'd'))
    def test_return_array_types(self, arrays):
        wide_n, tall_n = arrays[:2]
        wide, tall, smol, big = utn.view_as(*arrays)
        tall_m, small_m, big_m = utn.core_only(tall, smol, big)
        hy.assume(np.all(utn.non_singular(smol)))
        hy.assume(np.all(utn.non_singular(big)))
        hy.assume(wide.ndim != smol.ndim - 1)  # np..solve's broadcasting issue

        self.assertIsInstance(wide @ tall, la.lnarray)
        self.assertIsInstance(wide_n @ tall, la.lnarray)
        xyout = np.empty(return_shape('(a,b),(b,c)->(a,c)', tall, wide), 'd')
        xyr = la.matmul(tall, wide_n, xyout)
        self.assertIsInstance(xyr, np.ndarray)
        self.assertIsInstance(xyout, np.ndarray)
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
        tall, square = utn.view_as(*arrays)
        tall_sh, square_sh = tall.shape, square.shape
        hy.assume(tall_sh[-2] > tall_sh[-1])
        hy.assume(np.max(np.abs(tall.imag)) > .01)
        hy.assume(np.max(np.abs(tall.real)) / np.max(np.abs(tall.imag)) < 1e3)

        expect = tall_sh[:-2] + tall_sh[:-3:-1]
        self.assertArrayShape(tall.t, expect)
        self.assertArrayShape(tall.h, expect)
        self.assertArrayNotAllClose(tall.t, tall.h)
        self.assertArrayShape(square.c, square_sh + (1,))
        self.assertArrayShape(tall.c.uc, tall_sh)
        expect = square_sh[:-1] + (1,) + square_sh[-1:]
        self.assertArrayShape(square.r, expect)
        self.assertArrayShape(tall.r.ur, tall_sh)
        self.assertArrayShape(square.s, square_sh + (1, 1))
        self.assertArrayShape(tall.s.us, tall_sh)

        expect = square_sh[:1] + (1,) + square_sh[1:2] + (1,) + square_sh[2:]
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
        tall, smol, vec = utn.view_as(*arrays)
        tall_m = utn.core_only(tall)
        vec = utn.core_only(vec, dims=1)
        hy.assume(tall.shape[-2] > tall.shape[-1])
        hy.assume(smol.ndim != 3)  # causes np..solve's broadcasting issue
        hy.assume(np.all(utn.non_singular(smol)))

        xwout = np.empty(return_shape('(a,b),(b,c)->(a,c)', tall, smol),
                         smol.dtype)
        xwr = la.matmul(tall, smol, xwout)
        self.assertArrayAllClose(xwr, xwout)
        self.assertArrayAllClose(tall @ smol, np.matmul(tall, smol))
        self.assertArrayAllClose(tall @ smol, np.matmul(tall, smol))
        self.assertArrayAllClose(tall @ vec, np.matmul(tall, vec))
        self.assertArrayAllClose(gf.solve(smol, vec), npl.solve(smol, vec.c).uc)
        self.assertArrayAllClose(gf.lstsq(tall_m.t, vec),
                                 npl.lstsq(tall_m.t, vec, rcond=None)[0])
        self.assertArrayAllClose(gf.rmatmul(smol, tall), np.matmul(tall, smol))
        # tall @= smol
        # self.assertArrayAllClose(xwr, tall)


class TestPinvarray(TestNewClasses):
    """test pinvarray & invarray classes
    """

    def test_pinvarray_attribute_types(self):
        self.pick_var_type('D')
        m_ss, a_bs = self.lm_ss, self.la_bs
        self.assertIsInstance(m_ss.pinv, la.pinvarray)
        self.assertIsInstance(m_ss.inv, la.invarray)
        self.assertIs(m_ss.pinv.dtype, np.dtype('D'))
        self.assertIsInstance(m_ss.pinv.pinv, la.lnarray)
        self.assertIsInstance(m_ss.inv.inv, la.lnarray)
        self.assertIsInstance(m_ss.pinv(), la.lnarray)
        self.assertIsInstance(m_ss.inv(), la.lnarray)
        self.pick_var_type('d')
        pn_bs = la.pinvarray(a_bs)
        self.assertIsInstance(pn_bs, la.pinvarray)
        self.assertIsInstance(pn_bs.pinv, la.lnarray)
        self.assertIsInstance(2 * pn_bs, la.pinvarray)
        self.assertIsInstance((2 * pn_bs).pinv, la.lnarray)
        pout = la.pinvarray(la.empty_like(a_bs))
        np.multiply(2, pn_bs, pout)
        self.assertIsInstance(pout, la.pinvarray)
        self.assertIsInstance(pout.pinv, la.lnarray)
        with self.assertRaises(AttributeError):
            pn_bs.inv  # pylint: disable=no-member,pointless-statement
        with self.assertRaises(TypeError):
            m_ss.inv.pinv  # pylint: disable=pointless-statement

    def test_pinvarray_shape_methods(self):
        self.pick_var_type('d')
        a_bs = self.la_bs
        a_bs_p = a_bs.pinv
        expect = a_bs.shape[:-2] + a_bs.shape[:-3:-1]
        self.assertEqual(a_bs_p.ndim, len(expect))
        self.assertEqual(a_bs_p.shape, expect)
        self.assertEqual(a_bs_p.size, np.prod(expect))
        self.assertArrayShape(a_bs_p(), expect)
        with self.assertRaises(ValueError):
            a_bs.inv  # pylint: disable=pointless-statement
        a_bs_p = a_bs.c.pinv
        expect = a_bs.shape[:-1] + (1,) + a_bs.shape[-1:]
        now_expect = expect[1::-1] +expect[2:]
        self.assertArrayShape(a_bs_p.swapaxes(0, 1), now_expect)
        now_expect = expect[2::-1] +expect[3:]
        self.assertArrayShape(a_bs_p.swapaxes(0, 2), now_expect)
        now_expect = expect[:-2] +expect[:-3:-1]
        self.assertArrayShape(a_bs_p.swapaxes(-1, -2),now_expect)

    @utn.loop_test()
    def test_pinvarray_in_functions(self, sctype):
        self.pick_var_type(sctype)
        m_sb, m_bs, a_bs = self.lm_sb, self.lm_bs, self.la_bs
        self.assertArrayAllClose(gf.matmul(a_bs.pinv, m_bs),
                                 gf.lstsq(a_bs, m_bs))
        self.assertArrayAllClose(gf.matmul(m_sb, a_bs.pinv.t),
                                 gf.rlstsq(m_sb, a_bs.t))
        xpout = la.pinvarray(la.empty_like(a_bs))
        a_bs_p = np.multiply(a_bs.pinv, 2, out=xpout)
        self.assertArrayAllClose(a_bs_p.pinv, xpout.pinv)
        self.assertArrayAllClose(a_bs_p.pinv, a_bs / 2)
        with self.assertRaises(TypeError):
            gf.matmul(a_bs.pinv, m_sb.pinv)
        self.assertArrayAllClose(gf.lstsq(m_sb.pinv, m_bs),
                                 gf.matmul(m_sb, m_bs))
        with self.assertRaises(TypeError):
            gf.lstsq(m_bs, m_sb.pinv)
        self.assertArrayAllClose(gf.rlstsq(m_bs.T, m_sb.t.pinv),
                                 gf.matmul(m_bs.T, m_sb.t))
        with self.assertRaises(TypeError):
            gf.rlstsq(m_sb.t.pinv, m_bs.t)
        with self.assertRaises(TypeError):
            gf.rmatmul(m_sb.pinv, a_bs.pinv)
        with self.assertRaises(TypeError):
            gf.solve(m_sb.pinv, m_bs)
        with self.assertRaises(TypeError):
            gf.rsolve(m_sb, a_bs.pinv)

    @utn.loop_test()
    def test_invarray_in_functions(self, sctype):
        self.pick_var_type(sctype)
        a_ss, a_bs, m_sb = self.la_ss, self.la_bs, self.lm_sb
        xwr = a_bs[..., :3, :]
        self.assertArrayAllClose(gf.matmul(a_ss.inv, m_sb),
                                 gf.solve(a_ss, m_sb))
        self.assertArrayAllClose(gf.matmul(a_bs, a_ss.inv),
                                 gf.rsolve(a_bs, a_ss))
        self.assertArrayAllClose(gf.matmul(a_ss.inv, xwr.inv).inv, xwr @ a_ss)
        self.assertArrayAllClose(gf.solve(a_ss.inv, m_sb),
                                 gf.matmul(a_ss, m_sb))
        self.assertArrayAllClose(gf.solve(xwr, a_ss.inv).inv,
                                 gf.matmul(a_ss, xwr))
        self.assertArrayAllClose(gf.solve(xwr.inv, a_ss.inv),
                                 gf.rsolve(xwr, a_ss))
        self.assertArrayAllClose(gf.rsolve(a_ss, xwr.inv),
                                 gf.matmul(a_ss, xwr))
        self.assertArrayAllClose(gf.rsolve(xwr.inv, a_ss).inv,
                                 gf.matmul(a_ss, xwr))
        self.assertArrayAllClose(gf.rsolve(xwr.inv, a_ss.inv),
                                 gf.solve(xwr, a_ss))
        self.assertArrayAllClose(gf.rmatmul(a_ss, xwr.inv),
                                 gf.solve(xwr, a_ss))
        self.assertArrayAllClose(gf.rmatmul(xwr.inv, a_ss),
                                 gf.rsolve(a_ss, xwr))
        self.assertArrayAllClose(gf.rmatmul(xwr.inv, a_ss.inv).inv, xwr @ a_ss)

    def test_bad_p_invarray_combos_in_functions(self):
        self.pick_var_type('d')
        a_ss, a_bs, m_sb = self.la_ss, self.la_bs, self.lm_sb
        xwr = a_bs[..., :3, :]
        with self.assertRaises(TypeError):
            la.solve(m_sb.pinv, xwr)
        with self.assertRaises(TypeError):
            la.rsolve(xwr, a_bs.pinv)
        with self.assertRaises(TypeError):
            la.solve(xwr, a_bs.pinv)
        with self.assertRaises(TypeError):
            la.rsolve(m_sb.pinv, xwr)
        with self.assertRaises(TypeError):
            la.solve(m_sb.pinv, a_bs.pinv)
        with self.assertRaises(TypeError):
            la.rsolve(m_sb.pinv, a_bs.pinv)
        with self.assertRaises(TypeError):
            la.matmul(a_ss.inv, a_bs.pinv)
        with self.assertRaises(TypeError):
            la.matmul(m_sb.pinv, xwr.inv)
        with self.assertRaises(TypeError):
            la.solve(a_bs.pinv, xwr.inv)
        with self.assertRaises(TypeError):
            la.rsolve(xwr.inv, m_sb.pinv)

    def test_good_p_invarray_combos_in_lstsq(self):
        self.pick_var_type('d')
        a_ss, a_bs, m_sb = self.la_ss, self.la_bs, self.lm_sb
        xwr = a_bs[..., :3, :]
        self.assertArrayAllClose(la.lstsq(xwr.inv, m_sb),
                                 la.matmul(xwr, m_sb))
        self.assertArrayAllClose(la.rlstsq(a_bs, xwr.inv),
                                 la.matmul(a_bs, xwr))
        self.assertArrayAllClose(la.lstsq(xwr.inv, a_bs.pinv),
                                 la.rlstsq(xwr, a_bs))
        self.assertArrayAllClose(la.rlstsq(xwr.inv, m_sb.pinv),
                                 la.solve(xwr, m_sb))
        self.assertArrayAllClose(la.lstsq(xwr.inv, a_ss.inv),
                                 la.rsolve(xwr, a_ss))
        self.assertArrayAllClose(la.rlstsq(xwr.inv, a_ss.inv),
                                 la.solve(xwr, a_ss))
        self.assertArrayAllClose(la.lstsq(a_bs.pinv, a_ss.inv),
                                 la.rsolve(a_bs, a_ss))
        self.assertArrayAllClose(la.rlstsq(m_sb.pinv, a_ss.inv),
                                 la.lstsq(m_sb, a_ss))

    def test_good_p_invarray_combos_in_solve(self):
        self.pick_var_type('d')
        a_ss, a_bs, m_sb = self.la_ss, self.la_bs, self.lm_sb
        xwr = a_bs[..., :3, :]
        self.assertArrayAllClose(la.solve(a_ss.inv, a_bs.pinv),
                                 la.rlstsq(a_ss, a_bs))
        self.assertArrayAllClose(la.rsolve(m_sb.pinv, xwr.inv),
                                 la.lstsq(m_sb, xwr))

    @utn.loop_test()
    def test_pinvarray_operators(self, sctype):
        self.pick_var_type(sctype)
        m_sb, m_bs, a_bs = self.lm_sb, self.m_bs, self.la_bs
        scal = self.lm_bs[..., :2].s
        self.assertArrayAllClose(a_bs.pinv @ m_bs, gf.lstsq(a_bs, m_bs))
        self.assertArrayAllClose(a_bs.pinv() @ m_bs, gf.lstsq(a_bs, m_bs))
        self.assertArrayAllClose(m_sb @ a_bs.pinv.t, gf.rlstsq(m_sb, a_bs.t))
        with self.assertRaises(TypeError):
            a_bs.pinv @ m_sb.pinv  # pylint: disable=pointless-statement
        self.assertArrayAllClose((a_bs.pinv * 3.5).pinv, a_bs / 3.5)
        self.assertArrayAllClose((2.4 * a_bs.pinv).pinv, a_bs / 2.4)
        self.assertArrayAllClose((a_bs.pinv / 3.564).pinv, a_bs * 3.564)
        with self.assertRaises(TypeError):
            65 / a_bs.pinv  # pylint: disable=pointless-statement
        self.assertArrayAllClose((a_bs.pinv * scal).pinv, a_bs / scal)
        self.assertArrayAllClose((scal * a_bs.pinv).pinv, a_bs / scal)
        self.assertArrayAllClose((a_bs.pinv / scal).pinv, a_bs * scal)
        with self.assertRaises(TypeError):
            scal / a_bs.pinv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            scal.pinv * a_bs.pinv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            a_bs.pinv + m_sb  # pylint: disable=pointless-statement
        xold = 1. * a_bs
        a_bs_p = a_bs.pinv
        a_bs_p *= 2
        self.assertArrayAllClose(a_bs, xold / 2)

    @utn.loop_test()
    def test_invarray_operators(self, sctype):
        self.pick_var_type(sctype)
        scal, m_ss = self.lm_bs[..., :2].s, self.lm_ss
        a_bs, m_sb = self.la_bs, self.lm_sb
        xwr = a_bs[..., :3, :]
        self.assertArrayAllClose(m_ss.inv @ m_sb, gf.solve(m_ss, m_sb))
        self.assertArrayAllClose(m_ss.inv() @ m_sb, gf.solve(m_ss, m_sb))
        self.assertArrayAllClose(a_bs @ m_ss.inv, gf.rsolve(a_bs, m_ss))
        self.assertArrayAllClose((m_ss.inv @ xwr.inv).inv, xwr @ m_ss)
        self.assertArrayAllClose((m_ss.inv * 3.5).inv, m_ss / 3.5)
        self.assertArrayAllClose((2.4 * m_ss.inv).inv, m_ss / 2.4)
        self.assertArrayAllClose((m_ss.inv / 3.564).inv, m_ss * 3.564)
        with self.assertRaises(TypeError):
            45.564 / m_ss.inv  # pylint: disable=pointless-statement
        self.assertArrayAllClose((xwr.inv * scal).inv, xwr / scal)
        self.assertArrayAllClose((scal * xwr.inv).inv, xwr / scal)
        self.assertArrayAllClose((xwr.inv / scal).inv, xwr * scal)
        with self.assertRaises(TypeError):
            scal / xwr.inv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            scal.inv * xwr.inv  # pylint: disable=pointless-statement
        with self.assertRaises(TypeError):
            m_ss + xwr.inv  # pylint: disable=pointless-statement
        xwold = 1. * xwr
        xwi = xwr.inv
        xwi @= m_ss.inv
        self.assertArrayAllClose(xwr, m_ss @ xwold)


if __name__ == '__main__':
    utn.main(verbosity=2)
