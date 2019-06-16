# -*- coding: utf-8 -*-
"""Test python wrappers of gufuncs.
It has been established, in test_gu*.py, that the gufuncs return the correct
values. We just check that the python wrappers call the correct ones.
"""
import numpy as np
import numpy_linalg as la
import numpy_linalg._linalg as lr
import numpy_linalg.gufuncs as gf
import unittest_numpy as utn
from test_gufunc import TestMatsVecs

errstate = utn.errstate(invalid='raise')
# =============================================================================
__all__ = ['TestShape', 'TestValue']
# =============================================================================
# %% Test python funcs
# =============================================================================


class TestLinalg(TestMatsVecs):
    """Testing row, col, scal and transpose"""

    def setUp(self):
        super().setUp()
        self.varnames += ['u', 'v', 'w', 'x', 'y', 'z', 'ones', 'zeros']
        self._u = {}
        self._v = {}
        self._w = {}
        self._x = {}
        self._y = {}
        self._z = {}
        self._ones = {}
        self._zeros = {}
        for sctype in self.sctype:
            self._u[sctype] = utn.randn_asa((7, 5), sctype)
            self._v[sctype] = utn.randn_asa((5, 2), sctype)
            self._w[sctype] = utn.randn_asa((2, 3, 3), sctype)
            self._x[sctype] = utn.randn_asa((2, 5, 3), sctype)
            self._y[sctype] = utn.randn_asa((3, 5), sctype)
            self._z[sctype] = utn.randn_asa((3,), sctype)
            self._ones[sctype] = utn.ones_asa((3, 3), sctype)
            self._zeros[sctype] = utn.zeros_asa((3, 3), sctype)


class TestShape(TestLinalg):
    """Testing shapes returned by linalg functions"""

    def test_shape_fn(self):
        """Check transpose, row, col, scal returns arrays of expected shape
        """
        self.pick_var_type('d')
        # shape
        self.assertArrayShape(la.transpose(self.a_bs), (2, 3, 7))
        self.assertArrayShape(la.row(self.a_bs), (2, 7, 1, 3))
        self.assertArrayShape(la.col(self.a_bs), (2, 7, 3, 1))
        self.assertArrayShape(la.scalar(self.a_bs), (2, 7, 3, 1, 1))

    def test_la_fn(self):
        """Check (r)matmul, (r)solve, (r)lstsq, return arrays of correct shape
        """
        self.pick_var_type('d')
        # matmul
        self.assertArrayShape(la.matmul(self.a_bs, self.m_sb), (2, 7, 7))
        self.assertArrayShape(la.matmul(self.a_bs, self.v_s), (2, 7))
        self.assertArrayShape(la.matmul(self.v_b, self.a_bs), (2, 3))
        self.assertArrayShape(la.matmul(self.v_s, self.v_s), ())
        # rmatmul
        self.assertArrayShape(lr.rmatmul(self.a_bs, self.m_sb), (2, 3, 3))
        self.assertArrayShape(lr.rmatmul(self.m_sb, self.v_s), (7,))
        self.assertArrayShape(lr.rmatmul(self.v_b, self.a_sb), (4, 1, 3))
        self.assertArrayShape(lr.rmatmul(self.v_b, self.v_b), ())
        # solve
        self.assertArrayShape(la.solve(self.a_ss, self.m_sb), (5, 1, 3, 7))
        self.assertArrayShape(la.solve(self.a_ss, self.v_s), (5, 1, 3))
        # rsolve
        self.assertArrayShape(la.rsolve(self.a_bs, self.a_ss), (5, 2, 7, 3))
        self.assertArrayShape(la.rsolve(self.v_s, self.a_ss), (5, 1, 3))
        # lstsq
        self.assertArrayShape(la.lstsq(self.a_bs, self.m_bb), (2, 3, 7))
        self.assertArrayShape(la.lstsq(self.m_sb, self.a_ss), (5, 1, 7, 3))
        self.assertArrayShape(la.lstsq(self.m_sb, self.v_s), (7,))
        self.assertArrayShape(la.lstsq(self.v_s, self.a_ss), (5, 1, 3))
        # rlstsq
        self.assertArrayShape(la.rlstsq(self.a_ss, self.a_bs), (5, 2, 3, 7))
        self.assertArrayShape(la.rlstsq(self.a_bb, self.a_sb), (4, 3, 7, 3))
        self.assertArrayShape(la.rlstsq(self.v_b, self.m_sb), (3,))
        self.assertArrayShape(la.rlstsq(self.a_ss, self.v_s), (5, 1, 3))
        self.assertArrayShape(la.rlstsq(self.v_s, self.a_bs), (2, 7))

    def test_div_fn(self):
        """Check matldiv, matrdiv return correct shape
        """
        self.pick_var_type('d')
        # solve
        self.assertArrayShape(la.matldiv(self.a_ss, self.m_sb), (5, 1, 3, 7))
        self.assertArrayShape(la.matldiv(self.a_ss, self.v_s), (5, 1, 3))
        # rsolve
        self.assertArrayShape(la.matrdiv(self.a_bs, self.a_ss), (5, 2, 7, 3))
        self.assertArrayShape(la.matrdiv(self.v_s, self.a_ss), (5, 1, 3))
        # lstsq
        self.assertArrayShape(la.matldiv(self.a_bs, self.m_bb), (2, 3, 7))
        self.assertArrayShape(la.matldiv(self.m_sb, self.a_ss), (5, 1, 7, 3))
        self.assertArrayShape(la.matldiv(self.m_sb, self.v_s), (7,))
        self.assertArrayShape(la.matldiv(self.v_s, self.a_ss), (5, 1, 3))
        # rlstsq
        self.assertArrayShape(la.matrdiv(self.a_ss, self.a_bs), (5, 2, 3, 7))
        self.assertArrayShape(la.matrdiv(self.v_b, self.m_sb), (3,))
        self.assertArrayShape(la.matrdiv(self.a_ss, self.v_s), (5, 1, 3))
        self.assertArrayShape(la.matrdiv(self.v_s, self.a_bs), (2, 7))

    def test_qr(self):
        """Check that qr returns correct shape in each mode
        """
        self.pick_var_type('d')
        self.assertArrayShapesAre(la.qr(self.a_bs, 'reduced'),
                                  ((2, 7, 3), (2, 3, 3)))
        self.assertArrayShapesAre(la.qr(self.m_sb, 'reduced'),
                                  ((3, 3, ), (3, 7)))
        self.assertArrayShapesAre(la.qr(self.a_bs, 'complete'),
                                  ((2, 7, 7), (2, 7, 3)))
        self.assertArrayShapesAre(la.qr(self.m_sb, 'complete'),
                                  ((3, 3), (3, 7)))
        self.assertArrayShape(la.qr(self.a_bs, 'r'), (2, 3, 3))
        self.assertArrayShape(la.qr(self.m_sb, 'r'), (3, 7))
        self.assertArrayShapesAre(la.qr(self.a_bs, 'raw'), ((2, 3, 7), (2, 3)))
        self.assertArrayShapesAre(la.qr(self.m_sb, 'raw'), ((7, 3), (3,)))

    def test_lq(self):
        """Check that lq returns correct shape in each mode
        """
        self.pick_var_type('d')
        self.assertArrayShapesAre(la.lq(self.a_bs, 'reduced'),
                                  ((2, 7, 3), (2, 3, 3)))
        self.assertArrayShapesAre(la.lq(self.m_sb, 'reduced'),
                                  ((3, 3), (3, 7)))
        self.assertArrayShapesAre(la.lq(self.a_bs, 'complete'),
                                  ((2, 7, 3), (2, 3, 3)))
        self.assertArrayShapesAre(la.lq(self.m_sb, 'complete'),
                                  ((3, 7), (7, 7)))
        self.assertArrayShape(la.lq(self.a_bs, 'l'), (2, 7, 3))
        self.assertArrayShape(la.lq(self.m_sb, 'l'), (3, 3))
        self.assertArrayShapesAre(la.lq(self.a_bs, 'raw'), ((2, 3, 7), (2, 3)))
        self.assertArrayShapesAre(la.lq(self.m_sb, 'raw'), ((7, 3), (3,)))

    def test_lqr(self):
        """Check that lqr returns correct shape in each mode
        """
        self.pick_var_type('d')
        self.assertArrayShapesAre(la.lqr(self.a_bs, 'reduced'),
                                  ((2, 7, 3), (2, 3, 3)))
        self.assertArrayShapesAre(la.lqr(self.m_sb, 'reduced'),
                                  ((3, 3), (3, 7)))
        self.assertArrayShapesAre(la.lqr(self.a_bs, 'complete'),
                                  ((2, 7, 7), (2, 7, 3)))
        self.assertArrayShapesAre(la.lqr(self.m_sb, 'complete'),
                                  ((3, 7), (7, 7)))
        self.assertArrayShape(la.lqr(self.a_bs, 'l'), (2, 3, 3))
        self.assertArrayShape(la.lqr(self.m_sb, 'l'), (3, 3))
        self.assertArrayShape(la.lqr(self.a_bs, 'r'), (2, 3, 3))
        self.assertArrayShape(la.lqr(self.m_sb, 'r'), (3, 3))
        self.assertArrayShapesAre(la.lqr(self.a_bs, 'raw'),
                                  ((2, 3, 7), (2, 3)))
        self.assertArrayShapesAre(la.lqr(self.m_sb, 'raw'), ((7, 3), (3,)))

    def test_lu(self):
        """Check that lu returns correct shape in each mode
        """
        self.pick_var_type('d')
        self.assertArrayShapesAre(la.lu(self.a_ss, 'separate'),
                                  ((5, 1, 3, 3), (5, 1, 3, 3), (5, 1, 3)))
        self.assertArrayShapesAre(la.lu(self.a_bs, 'separate'),
                                  ((2, 7, 3), (2, 3, 3), (2, 3)))
        self.assertArrayShapesAre(la.lu(self.m_sb, 'separate'),
                                  ((3, 3), (3, 7), (3,)))
        self.assertArrayShapesAre(la.lu(self.a_ss, 'raw'),
                                  ((5, 1, 3, 3), (5, 1, 3)))
        self.assertArrayShapesAre(la.lu(self.a_bs, 'raw'), ((2, 3, 7), (2, 3)))
        self.assertArrayShapesAre(la.lu(self.m_sb, 'raw'), ((7, 3), (3,)))


class TestValue(TestLinalg):
    """Testing values returned by linalg functions"""

    @utn.loop_test()
    def test_la_fn(self, sctype):
        """Check (r)matmul, (r)solve, (r)lstsq, return arrays of correct value
        """
        self.pick_var_type(sctype)
        # matmul
        self.assertArrayAllClose(la.matmul(self.a_bs, self.m_sb),
                                 gf.matmul(self.a_bs, self.m_sb))
        # rmatmul
        self.assertArrayAllClose(lr.rmatmul(self.a_bs, self.m_sb),
                                 gf.rmatmul(self.a_bs, self.m_sb))
        # solve
        self.assertArrayAllClose(la.solve(self.a_ss, self.m_sb),
                                 gf.solve(self.a_ss, self.m_sb))
        wyout = np.empty((5, 1, 3, 7), sctype)
        wy = la.solve(self.a_ss, self.m_sb, out=wyout)
        self.assertArrayAllClose(wyout, wy)
        # rsolve
        self.assertArrayAllClose(la.rsolve(self.a_bs, self.a_ss),
                                 gf.rsolve(self.a_bs, self.a_ss))
        # lstsq
        self.assertArrayAllClose(la.lstsq(self.a_bs, self.m_bb),
                                 gf.lstsq(self.a_bs, self.m_bb))
        self.assertArrayAllClose(la.lstsq(self.m_sb, self.a_ss),
                                 gf.lstsq(self.m_sb, self.a_ss))
        xvout = np.empty((2, 3, 7), sctype)
        xv = la.lstsq(self.a_bs, self.m_bb, out=xvout)
        self.assertArrayAllClose(xvout, xv)
        # rlstsq
        self.assertArrayAllClose(la.rlstsq(self.a_ss, self.a_bs),
                                 gf.rlstsq(self.a_ss, self.a_bs))
        self.assertArrayAllClose(la.rlstsq(self.v_b, self.m_sb),
                                 gf.rlstsq(la.row(self.v_b), self.m_sb))

    @utn.loop_test()
    def test_div_fn(self, sctype):
        """Check matldiv, matrdiv return correct value
        """
        self.pick_var_type(sctype)
        # solve
        self.assertArrayAllClose(la.matldiv(self.a_ss, self.m_sb),
                                 gf.solve(self.a_ss, self.m_sb))
        wyout = np.empty((5, 1, 3, 7), sctype)
        wy = la.matldiv(self.a_ss, self.m_sb, out=wyout)
        self.assertArrayAllClose(wyout, wy)
        # rsolve
        self.assertArrayAllClose(la.matrdiv(self.a_bs, self.a_ss),
                                 gf.rsolve(self.a_bs, self.a_ss))
        # lstsq
        self.assertArrayAllClose(la.matldiv(self.a_bs, self.m_bb),
                                 gf.lstsq(self.a_bs, self.m_bb))
        self.assertArrayAllClose(la.matldiv(self.m_sb, self.a_ss),
                                 gf.lstsq(self.m_sb, self.a_ss))
        # rlstsq
        self.assertArrayAllClose(la.matrdiv(self.a_ss, self.a_bs),
                                 gf.rlstsq(self.a_ss, self.a_bs))
        self.assertArrayAllClose(la.matrdiv(self.v_b, self.m_sb),
                                 gf.rlstsq(la.row(self.v_b), self.m_sb))

    @utn.loop_test()
    def test_qr(self, sctype):
        """Check that qr returns correct value in each mode
        """
        self.pick_var_type(sctype)
        q, r = la.qr(self.a_bs, 'reduced')
        self.assertArrayAllClose(q @ r, self.a_bs)
        q, r = la.qr(self.m_sb, 'reduced')
        self.assertArrayAllClose(q @ r, self.m_sb)
        q, r = la.qr(self.a_bs, 'complete')
        self.assertArrayAllClose(q @ r, self.a_bs)
        q, r = la.qr(self.m_sb, 'complete')
        self.assertArrayAllClose(q @ r, self.m_sb)
        r = la.qr(self.a_bs, 'r')
        h, tau = la.qr(self.a_bs, 'raw')
        self.assertArrayAllClose(r, np.triu(la.transpose(h))[:, :3])
        r = la.qr(self.m_sb, 'r')
        h, tau = la.qr(self.m_sb, 'raw')
        self.assertArrayAllClose(r, np.triu(la.transpose(h)))

    @utn.loop_test()
    def test_lq(self, sctype):
        """Check that lq returns correct value in each mode
        """
        self.pick_var_type(sctype)
        lo, q = la.lq(self.a_bs, 'reduced')
        self.assertArrayAllClose(lo @ q, self.a_bs)
        lo, q = la.lq(self.m_sb, 'reduced')
        self.assertArrayAllClose(lo @ q, self.m_sb)
        lo, q = la.lq(self.a_bs, 'complete')
        self.assertArrayAllClose(lo @ q, self.a_bs)
        lo, q = la.lq(self.m_sb, 'complete')
        self.assertArrayAllClose(lo @ q, self.m_sb)
        lo = la.lq(self.a_bs, 'l')
        h, tau = la.lq(self.a_bs, 'raw')
        self.assertArrayAllClose(lo, np.tril(la.transpose(h)))
        lo = la.lq(self.m_sb, 'l')
        h, tau = la.lq(self.m_sb, 'raw')
        self.assertArrayAllClose(lo, np.tril(la.transpose(h))[..., :3])

    @utn.loop_test()
    def test_lqr(self, sctype):
        """Check that lqr returns correct value in each mode
        """
        self.pick_var_type(sctype)
        q, r = la.lqr(self.a_bs, 'reduced')
        self.assertArrayAllClose(q @ r, self.a_bs)
        lo, q = la.lqr(self.m_sb, 'reduced')
        self.assertArrayAllClose(lo @ q, self.m_sb)
        q, r = la.lqr(self.a_bs, 'complete')
        self.assertArrayAllClose(q @ r, self.a_bs)
        lo, q = la.lqr(self.m_sb, 'complete')
        self.assertArrayAllClose(lo @ q, self.m_sb)
        r = la.lqr(self.a_bs, 'r')
        h, tau = la.lqr(self.a_bs, 'raw')
        self.assertArrayAllClose(r, np.triu(la.transpose(h))[:, :3])
        lo = la.lqr(self.m_sb, 'r')
        h, tau = la.lqr(self.m_sb, 'raw')
        self.assertArrayAllClose(lo, np.tril(la.transpose(h))[..., :3])

    @utn.loop_test()
    def test_lu(self, sctype):
        """Check that lu returns correct value in each mode
        """
        self.pick_var_type(sctype)
        low, up, piv = la.lu(self.a_ss, 'separate')
        luf, piv = la.lu(self.a_ss, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(low @ up, gf.pivot(self.a_ss, piv))
        self.assertArrayAllClose(np.tril(low, -1), np.tril(luf, -1))
        self.assertArrayAllClose(up, np.triu(luf))
        low, up, piv = la.lu(self.a_bs, 'separate')
        luf, piv = la.lu(self.a_bs, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(np.tril(low, -1), np.tril(luf, -1))
        self.assertArrayAllClose(up, np.triu(luf)[:, :3])
        low, up, piv = la.lu(self.m_sb, 'separate')
        luf, piv = la.lu(self.m_sb, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(np.tril(low, -1), np.tril(luf, -1)[:, :3])
        self.assertArrayAllClose(up, np.triu(luf))

    @utn.loop_test()
    def test_low_rank(self, sctype):
        """Check low rank matrices are handled appropriately
        """
        self.pick_var_type(sctype)
        with self.assertRaises(np.linalg.LinAlgError):
            la.solve(self.ones, self.m_sb)
        q, r = la.qr(self.ones)
        self.assertArrayAllClose(q @ r, self.ones)
        low, up, piv = la.lu(self.ones)
        self.assertArrayAllClose(low @ up, self.ones)


# =============================================================================
if __name__ == '__main__':
    utn.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
