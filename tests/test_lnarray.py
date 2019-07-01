# -*- coding: utf-8 -*-
"""Test lnarray class
"""
import unittest
import numpy as np
import numpy.linalg as npl
import numpy_linalg as la
import numpy_linalg.gufuncs as gf
if __name__.find('tests.') < 0:
    import unittest_numpy as utn
    from test_gufunc import TestMatsVecs
else:
    from . import unittest_numpy as utn
    from .test_gufunc import TestMatsVecs

# =============================================================================
__all__ = ['TestArray', 'TestPinvarray']
# =============================================================================
# %% Test python classes
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
    lm_bs: la.lnarray
    lm_sb: la.lnarray
    lm_ss: la.lnarray
    lm_bb: la.lnarray
    la_bs: la.lnarray
    la_sb: la.lnarray
    la_ss: la.lnarray
    la_bb: la.lnarray

    def setUp(self):
        super().setUp()
        for v in self.varnames:
            np_arrs = getattr(self, '_' + v)
            ln_arrs = {}
            for sctype in self.sctype:
                ln_arrs[sctype] = np_arrs[sctype].view(la.lnarray)
            setattr(self, '_l' + v, ln_arrs)
        self.varnames += ['l' + v for v in self.varnames]


class TestArray(TestNewClasses):
    """Testing lnarray"""

    def setUp(self):
        self.sctype = ['i']
        super().setUp()

    def test_return_array_types(self):
        self.pick_var_type('d')
        m_sb, m_bs, m_ss, m_bb = self.lm_sb, self.lm_bs, self.lm_ss, self.lm_bb
        a_bs = self.la_bs
        self.assertIsInstance(m_sb @ m_bs, la.lnarray)
        self.assertIsInstance(self.m_sb @ m_bs, la.lnarray)
        xyout = np.empty((2, 7, 7), 'd')
        xy = la.matmul(a_bs, self.m_sb, xyout)
        self.assertIsInstance(xy, np.ndarray)
        self.assertIsInstance(xyout, np.ndarray)
        self.assertIsInstance(np.matmul(a_bs, self.m_sb), np.ndarray)
        self.assertIsInstance(la.solve(m_ss, self.m_sb), la.lnarray)
        self.assertIsInstance(npl.solve(m_ss, self.m_sb), np.ndarray)
        self.assertIsInstance(la.lstsq(a_bs, m_bb), la.lnarray)
        self.assertIsInstance(npl.lstsq(a_bs[0], m_bb, rcond=None)[0],
                              np.ndarray)
        self.assertIsInstance(la.lu(m_ss)[0], la.lnarray)
        self.assertIsInstance(la.lu(self.m_bs)[0], np.ndarray)
        self.assertIsInstance(la.qr(m_ss)[0], la.lnarray)
        self.assertIsInstance(la.qr(self.m_bs)[0], np.ndarray)
        self.assertIsInstance(la.lq(m_ss)[0], la.lnarray)
        self.assertIsInstance(la.lq(self.m_bs)[0], np.ndarray)
        self.assertIsInstance(la.lqr(m_ss)[0], la.lnarray)
        self.assertIsInstance(la.lqr(self.m_bs)[0], np.ndarray)
        self.assertIsInstance(npl.qr(m_ss)[0], np.ndarray)

    def test_lnarray_shape_methods(self):
        self.pick_var_type('D')
        a_bs, m_ss = self.la_bs, self.lm_ss
        self.assertArrayShape(a_bs.t, (2, 3, 7))
        self.assertArrayShape(a_bs.h, (2, 3, 7))
        self.assertArrayNotAllClose(a_bs.t, a_bs.h)
        self.assertArrayShape(m_ss.c, (3, 3, 1))
        self.assertArrayShape(a_bs.c.uc, (2, 7, 3))
        self.assertArrayShape(m_ss.r, (3, 1, 3))
        self.assertArrayShape(a_bs.r.ur, (2, 7, 3))
        self.assertArrayShape(m_ss.s, (3, 3, 1, 1))
        self.assertArrayShape(a_bs.s.us, (2, 7, 3))
        self.assertArrayShape(m_ss.expand_dims(1, 3), (3, 1, 3, 1))
        self.assertArrayShape((a_bs.s * m_ss).flattish(1, 4), (2, 63, 3))
        with self.assertRaisesRegex(ValueError, "repeated axes"):
            a_bs.expand_dims(2, -3)
        with self.assertRaises(ValueError):
            (a_bs.s * m_ss).flattish(3, -3)

    @utn.loop_test(attr_inds=slice(4))
    def test_lnarray_operations_return_expected_values(self, sctype):
        self.pick_var_type(sctype)
        m_ss, a_bs, lv_s = self.lm_ss, self.la_bs, self.lv_s
        xwout = np.empty((2, 7, 3), sctype)
        xw = la.matmul(a_bs, m_ss, xwout)
        self.assertArrayAllClose(xw, xwout)
        self.assertArrayAllClose(a_bs @ m_ss, np.matmul(a_bs, m_ss))
        self.assertArrayAllClose(a_bs @ m_ss, np.matmul(a_bs, m_ss))
        self.assertArrayAllClose(a_bs @ lv_s, np.matmul(a_bs, lv_s))
        self.assertArrayAllClose(gf.solve(m_ss, lv_s), npl.solve(m_ss, lv_s))
        self.assertArrayAllClose(gf.lstsq(a_bs.t[0], lv_s),
                                 npl.lstsq(a_bs[0].t, lv_s, rcond=None)[0])
        self.assertArrayAllClose(gf.rmatmul(m_ss, a_bs), np.matmul(a_bs, m_ss))
        a_bs @= m_ss
        self.assertArrayAllClose(xw, a_bs)


class TestPinvarray(TestNewClasses):
    """test pinvarray & invarray classes
    """

    def test_pinvarray_attribute_types(self):
        self.pick_var_type('D')
        m_ss = self.lm_ss
        self.assertIsInstance(m_ss.pinv, la.pinvarray)
        self.assertIsInstance(m_ss.inv, la.invarray)
        self.assertIs(m_ss.pinv.dtype, np.dtype('D'))
        self.assertIsInstance(m_ss.pinv.pinv, la.lnarray)
        self.assertIsInstance(m_ss.inv.inv, la.lnarray)
        self.assertIsInstance(m_ss.pinv(), la.lnarray)
        self.assertIsInstance(m_ss.inv(), la.lnarray)
        self.pick_var_type('d')
        pn_bs = la.pinvarray(self.la_bs)
        self.assertIsInstance(pn_bs, la.pinvarray)
        self.assertIsInstance(pn_bs.pinv, la.lnarray)
        self.assertIsInstance(2 * pn_bs, la.pinvarray)
        self.assertIsInstance((2 * pn_bs).pinv, la.lnarray)
        pout = la.pinvarray(la.empty_like(self.la_bs))
        np.multiply(2, pn_bs, pout)
        self.assertIsInstance(pout, la.pinvarray)
        self.assertIsInstance(pout.pinv, la.lnarray)
        with self.assertRaises(AttributeError):
            pn_bs.inv
        with self.assertRaises(TypeError):
            m_ss.inv.pinv

    def test_pinvarray_shape_methods(self):
        self.pick_var_type('d')
        a_bs_p = self.la_bs.pinv
        self.assertEqual(a_bs_p.ndim, 3)
        self.assertEqual(a_bs_p.shape, (2, 3, 7))
        self.assertEqual(a_bs_p.size, 42)
        self.assertArrayShape(a_bs_p(), (2, 3, 7))
        with self.assertRaises(ValueError):
            self.la_bs.inv
        a_bs_p = self.la_bs.c.pinv
        self.assertArrayShape(a_bs_p.swapaxes(0, 1), (7, 2, 1, 3))
        self.assertArrayShape(a_bs_p.swapaxes(0, 2), (1, 7, 2, 3))
        self.assertArrayShape(a_bs_p.swapaxes(-1, -2), (2, 7, 3, 1))

    @utn.loop_test()
    def test_pinvarray_in_functions(self, sctype):
        self.pick_var_type(sctype)
        m_sb, m_bs, a_bs = self.lm_sb, self.lm_bs, self.la_bs
        self.assertArrayAllClose(gf.matmul(a_bs.pinv, m_bs),
                                 gf.lstsq(a_bs, m_bs))
        self.assertArrayAllClose(gf.matmul(m_sb, a_bs.pinv.t),
                                 gf.rlstsq(m_sb, a_bs.t))
        xpout = la.pinvarray(la.empty_like(a_bs))
        xp = np.multiply(a_bs.pinv, 2, out=xpout)
        self.assertArrayAllClose(xp.pinv, xpout.pinv)
        self.assertArrayAllClose(xp.pinv, a_bs / 2)
        with self.assertRaises(TypeError):
            gf.matmul(a_bs.pinv, m_sb.pinv)
        self.assertArrayAllClose(gf.lstsq(m_sb.pinv, m_bs),
                                 gf.matmul(m_sb, m_bs))
        with self.assertRaises(TypeError):
            gf.lstsq(m_bs, m_sb.pinv)
        self.assertArrayAllClose(gf.rlstsq(m_bs.T, m_sb.t.pinv),
                                 gf.matmul(m_bs.T, m_sb.t))
        with self.assertRaises(TypeError):
            gf.rlstsq(m_sb.t.pinv, m_bs.T)
        with self.assertRaises(TypeError):
            gf.rmatmul(m_sb.pinv, a_bs.pinv)
        with self.assertRaises(TypeError):
            gf.solve(m_sb.pinv, m_bs)
        with self.assertRaises(TypeError):
            gf.rsolve(m_sb, a_bs.pinv)

    @utn.loop_test()
    def test_invarray_in_functions(self, sctype):
        self.pick_var_type(sctype)
        la_ss, la_bs, lm_sb = self.la_ss, self.la_bs, self.lm_sb
        xw = la_bs[:, :3]
        self.assertArrayAllClose(gf.matmul(la_ss.inv, lm_sb),
                                 gf.solve(la_ss, lm_sb))
        self.assertArrayAllClose(gf.matmul(la_bs, la_ss.inv),
                                 gf.rsolve(la_bs, la_ss))
        self.assertArrayAllClose(gf.matmul(la_ss.inv, xw.inv).inv, xw @ la_ss)
        self.assertArrayAllClose(gf.solve(la_ss.inv, lm_sb),
                                 gf.matmul(la_ss, lm_sb))
        self.assertArrayAllClose(gf.solve(xw, la_ss.inv).inv,
                                 gf.matmul(la_ss, xw))
        self.assertArrayAllClose(gf.solve(xw.inv, la_ss.inv),
                                 gf.rsolve(xw, la_ss))
        self.assertArrayAllClose(gf.rsolve(la_ss, xw.inv),
                                 gf.matmul(la_ss, xw))
        self.assertArrayAllClose(gf.rsolve(xw.inv, la_ss).inv,
                                 gf.matmul(la_ss, xw))
        self.assertArrayAllClose(gf.rsolve(xw.inv, la_ss.inv),
                                 gf.solve(xw, la_ss))
        self.assertArrayAllClose(gf.rmatmul(la_ss, xw.inv),
                                 gf.solve(xw, la_ss))
        self.assertArrayAllClose(gf.rmatmul(xw.inv, la_ss),
                                 gf.rsolve(la_ss, xw))
        self.assertArrayAllClose(gf.rmatmul(xw.inv, la_ss.inv).inv, xw @ la_ss)
        with self.assertRaises(TypeError):
            gf.lstsq(la_ss.inv, lm_sb)
        with self.assertRaises(TypeError):
            gf.rlstsq(la_bs, la_ss.inv)

    def test_bad_p_invarray_combos_in_functions(self):
        self.pick_var_type('d')
        la_ss, la_bs, lm_sb = self.la_ss, self.la_bs, self.lm_sb
        xw = la_bs[:, :3]
        with self.assertRaises(TypeError):
            la.solve(lm_sb.pinv, xw)
        with self.assertRaises(TypeError):
            la.solve(xw, la_bs.pinv)
        with self.assertRaises(TypeError):
            la.solve(lm_sb.pinv, la_bs.pinv)
        with self.assertRaisesRegex(ValueError, 'not square'):
            la.matmul(la_ss.inv, la_bs.pinv)
        with self.assertRaisesRegex(ValueError, 'not square'):
            la.matmul(lm_sb.pinv, xw.inv)

    @unittest.expectedFailure
    def test_good_p_invarray_combos_in_functions(self):
        self.pick_var_type('d')
        la_ss, la_bs, lm_sb = self.la_ss, self.la_bs, self.lm_sb
        xw = la_bs[:, :3]
        self.assertIsInstance(la.lstsq(xw.inv, lm_sb), la.lnarray)
        self.assertIsInstance(la.lstsq(xw.inv, la_ss.pinv), la.lnarray)
        self.assertIsInstance(la.lstsq(xw.inv, la_ss.inv), la.lnarray)
        self.assertIsInstance(la.lstsq(la_bs.pinv, la_ss.inv), la.lnarray)
        self.assertIsInstance(la.solve(xw.inv, la_bs.pinv), la.lnarray)
        with self.assertRaisesRegex(*utn.core_dim_err):
            la.solve(la_bs.pinv, xw.inv)

    @utn.loop_test()
    def test_pinvarray_operators(self, sctype):
        self.pick_var_type(sctype)
        m_sb, m_bs, a_bs = self.lm_sb, self.m_bs, self.la_bs
        vs = self.lm_bs[..., :2].s
        self.assertArrayAllClose(a_bs.pinv @ m_bs, gf.lstsq(a_bs, m_bs))
        self.assertArrayAllClose(a_bs.pinv() @ m_bs, gf.lstsq(a_bs, m_bs))
        self.assertArrayAllClose(m_sb @ a_bs.pinv.t, gf.rlstsq(m_sb, a_bs.t))
        with self.assertRaises(TypeError):
            a_bs.pinv @ m_sb.pinv
        self.assertArrayAllClose((a_bs.pinv * 3.5).pinv, a_bs / 3.5)
        self.assertArrayAllClose((2.4 * a_bs.pinv).pinv, a_bs / 2.4)
        self.assertArrayAllClose((a_bs.pinv / 3.564).pinv, a_bs * 3.564)
        with self.assertRaises(TypeError):
            65 / a_bs.pinv
        self.assertArrayAllClose((a_bs.pinv * vs).pinv, a_bs / vs)
        self.assertArrayAllClose((vs * a_bs.pinv).pinv, a_bs / vs)
        self.assertArrayAllClose((a_bs.pinv / vs).pinv, a_bs * vs)
        with self.assertRaises(TypeError):
            vs / a_bs.pinv
        with self.assertRaises(TypeError):
            vs.pinv * a_bs.pinv
        with self.assertRaises(TypeError):
            a_bs.pinv + self.lm_sb
        xold = 1. * a_bs
        xp = a_bs.pinv
        xp *= 2
        self.assertArrayAllClose(a_bs, xold / 2)

    @utn.loop_test()
    def test_invarray_operators(self, sctype):
        self.pick_var_type(sctype)
        vs, m_ss = self.lm_bs[..., :2].s, self.lm_ss
        a_bs, m_sb = self.la_bs, self.lm_sb
        xw = a_bs[:, :3]
        self.assertArrayAllClose(m_ss.inv @ m_sb, gf.solve(m_ss, m_sb))
        self.assertArrayAllClose(m_ss.inv() @ m_sb, gf.solve(m_ss, m_sb))
        self.assertArrayAllClose(a_bs @ m_ss.inv, gf.rsolve(a_bs, m_ss))
        self.assertArrayAllClose((m_ss.inv @ xw.inv).inv, xw @ m_ss)
        self.assertArrayAllClose((m_ss.inv * 3.5).inv, m_ss / 3.5)
        self.assertArrayAllClose((2.4 * m_ss.inv).inv, m_ss / 2.4)
        self.assertArrayAllClose((m_ss.inv / 3.564).inv, m_ss * 3.564)
        with self.assertRaises(TypeError):
            45.564 / m_ss.inv
        self.assertArrayAllClose((xw.inv * vs).inv, xw / vs)
        self.assertArrayAllClose((vs * xw.inv).inv, xw / vs)
        self.assertArrayAllClose((xw.inv / vs).inv, xw * vs)
        with self.assertRaises(TypeError):
            vs / xw.inv
        with self.assertRaises(TypeError):
            vs.inv * xw.inv
        with self.assertRaises(TypeError):
            m_ss + xw.inv
        xwold = 1. * xw
        xwi = xw.inv
        xwi @= m_ss.inv
        self.assertArrayAllClose(xw, m_ss @ xwold)


if __name__ == '__main__':
    utn.main(verbosity=2)
