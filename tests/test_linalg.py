# -*- coding: utf-8 -*-
"""Test python wrappers of gufuncs.
It has been established, in test_gu*.py, that the gufuncs return the correct
values. We just check that the python wrappers call the correct ones.
"""
import unittest
import numpy as np
import unittest_numpy as utn
import numpy_linalg as la
import numpy_linalg._linalg as lr
import numpy_linalg.gufuncs as gf

errstate = utn.errstate(invalid='raise')
# =============================================================================

# =============================================================================
# %% Test python funcs
# =============================================================================


class TestLinalg(utn.TestCaseNumpy):
    """Testing row, col, scal and transpose"""

    def setUp(self):
        super().setUp()
        self.varnames = ['u', 'v', 'w', 'x', 'y', 'z', 'ones', 'zeros']
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
        self.assertEqual(la.transpose(self.x).shape, (2, 3, 5))
        self.assertEqual(la.row(self.x).shape, (2, 5, 1, 3))
        self.assertEqual(la.col(self.x).shape, (2, 5, 3, 1))
        self.assertEqual(la.scalar(self.x).shape, (2, 5, 3, 1, 1))

    def test_la_fn(self):
        """Check (r)matmul, (r)solve, (r)lstsq, return arrays of correct shape
        """
        self.pick_var_type('d')
        # matmul
        self.assertEqual(la.matmul(self.x, self.y).shape, (2, 5, 5))
        self.assertEqual(la.matmul(self.x, self.z).shape, (2, 5))
        # rmatmul
        self.assertEqual(lr.rmatmul(self.x, self.y).shape, (2, 3, 3))
        self.assertEqual(lr.rmatmul(self.y, self.z).shape, (5,))
        # solve
        self.assertEqual(la.solve(self.w, self.y).shape, (2, 3, 5))
        self.assertEqual(la.solve(self.w, self.z).shape, (2, 3))
        # rsolve
        self.assertEqual(la.rsolve(self.x, self.w).shape, (2, 5, 3))
        self.assertEqual(la.rsolve(self.z, self.w).shape, (2, 3))
        # lstsq
        self.assertEqual(la.lstsq(self.x, self.v).shape, (2, 3, 2))
        self.assertEqual(la.lstsq(self.y, self.w).shape, (2, 5, 3))
        self.assertEqual(la.lstsq(self.y, self.z).shape, (5,))
        self.assertEqual(la.lstsq(self.z, self.w).shape, (2, 3))
        # rlstsq
        self.assertEqual(la.rlstsq(self.w, self.x).shape, (2, 3, 5))
        self.assertEqual(la.rlstsq(self.u, self.y).shape, (7, 3))
        self.assertEqual(la.rlstsq(self.w, self.z).shape, (2, 3))
        self.assertEqual(la.rlstsq(self.z, self.x).shape, (2, 5))

    def test_div_fn(self):
        """Check matldiv, matrdiv return correct shape
        """
        self.pick_var_type('d')
        # solve
        self.assertEqual(la.matldiv(self.w, self.y).shape, (2, 3, 5))
        self.assertEqual(la.matldiv(self.w, self.z).shape, (2, 3))
        # rsolve
        self.assertEqual(la.matrdiv(self.x, self.w).shape, (2, 5, 3))
        self.assertEqual(la.matrdiv(self.z, self.w).shape, (2, 3))
        # lstsq
        self.assertEqual(la.matldiv(self.x, self.v).shape, (2, 3, 2))
        self.assertEqual(la.matldiv(self.y, self.w).shape, (2, 5, 3))
        self.assertEqual(la.matldiv(self.y, self.z).shape, (5,))
        self.assertEqual(la.matldiv(self.z, self.w).shape, (2, 3))
        # rlstsq
        self.assertEqual(la.matrdiv(self.w, self.x).shape, (2, 3, 5))
        self.assertEqual(la.matrdiv(self.u, self.y).shape, (7, 3))
        self.assertEqual(la.matrdiv(self.w, self.z).shape, (2, 3))
        self.assertEqual(la.matrdiv(self.z, self.x).shape, (2, 5))

    def test_qr(self):
        """Check that qr returns correct shape in each mode
        """
        self.pick_var_type('d')
        q, r = la.qr(self.x, 'reduced')
        self.assertEqual((q.ndim, r.ndim), (3, 3))
        self.assertEqual(q.shape + r.shape, (2, 5, 3, 2, 3, 3))
        q, r = la.qr(self.y, 'reduced')
        self.assertEqual((q.ndim, r.ndim), (2, 2))
        self.assertEqual(q.shape + r.shape, (3, 3, 3, 5))
        q, r = la.qr(self.x, 'complete')
        self.assertEqual((q.ndim, r.ndim), (3, 3))
        self.assertEqual(q.shape + r.shape, (2, 5, 5, 2, 5, 3))
        q, r = la.qr(self.y, 'complete')
        self.assertEqual((q.ndim, r.ndim), (2, 2))
        self.assertEqual(q.shape + r.shape, (3, 3, 3, 5))
        r = la.qr(self.x, 'r')
        self.assertEqual(r.shape, (2, 3, 3))
        r = la.qr(self.y, 'r')
        self.assertEqual(r.shape, (3, 5))
        h, tau = la.qr(self.x, 'raw')
        self.assertEqual((h.ndim, tau.ndim), (3, 2))
        self.assertEqual(h.shape + tau.shape, (2, 3, 5, 2, 3))
        h, tau = la.qr(self.y, 'raw')
        self.assertEqual((h.ndim, tau.ndim), (2, 1))
        self.assertEqual(h.shape + tau.shape, (5, 3, 3))

    def test_lq(self):
        """Check that lq returns correct shape in each mode
        """
        self.pick_var_type('d')
        lo, q = la.lq(self.x, 'reduced')
        self.assertEqual((lo.ndim, q.ndim), (3, 3))
        self.assertEqual(lo.shape + q.shape, (2, 5, 3, 2, 3, 3))
        lo, q = la.lq(self.y, 'reduced')
        self.assertEqual((lo.ndim, q.ndim), (2, 2))
        self.assertEqual(lo.shape + q.shape, (3, 3, 3, 5))
        lo, q = la.lq(self.x, 'complete')
        self.assertEqual((lo.ndim, q.ndim), (3, 3))
        self.assertEqual(lo.shape + q.shape, (2, 5, 3, 2, 3, 3))
        lo, q = la.lq(self.y, 'complete')
        self.assertEqual((lo.ndim, q.ndim), (2, 2))
        self.assertEqual(lo.shape + q.shape, (3, 5, 5, 5))
        lo = la.lq(self.x, 'l')
        self.assertEqual(lo.shape, (2, 5, 3))
        lo = la.lq(self.y, 'l')
        self.assertEqual(lo.shape, (3, 3))
        h, tau = la.lq(self.x, 'raw')
        self.assertEqual((h.ndim, tau.ndim), (3, 2))
        self.assertEqual(h.shape + tau.shape, (2, 3, 5, 2, 3))
        h, tau = la.lq(self.y, 'raw')
        self.assertEqual((h.ndim, tau.ndim), (2, 1))
        self.assertEqual(h.shape + tau.shape, (5, 3, 3))

    def test_lqr(self):
        """Check that lqr returns correct shape in each mode
        """
        self.pick_var_type('d')
        q, r = la.lqr(self.x, 'reduced')
        self.assertEqual((q.ndim, r.ndim), (3, 3))
        self.assertEqual(q.shape + r.shape, (2, 5, 3, 2, 3, 3))
        lo, q = la.lqr(self.y, 'reduced')
        self.assertEqual((lo.ndim, q.ndim), (2, 2))
        self.assertEqual(lo.shape + q.shape, (3, 3, 3, 5))
        q, r = la.lqr(self.x, 'complete')
        self.assertEqual((q.ndim, r.ndim), (3, 3))
        self.assertEqual(q.shape + r.shape, (2, 5, 5, 2, 5, 3))
        lo, q = la.lqr(self.y, 'complete')
        self.assertEqual((lo.ndim, q.ndim), (2, 2))
        self.assertEqual(lo.shape + q.shape, (3, 5, 5, 5))
        r = la.lqr(self.x, 'l')
        self.assertEqual(r.shape, (2, 3, 3))
        lo = la.lqr(self.y, 'l')
        self.assertEqual(lo.shape, (3, 3))
        r = la.lqr(self.x, 'r')
        self.assertEqual(r.shape, (2, 3, 3))
        lo = la.lqr(self.y, 'r')
        self.assertEqual(lo.shape, (3, 3))
        h, tau = la.lqr(self.x, 'raw')
        self.assertEqual((h.ndim, tau.ndim), (3, 2))
        self.assertEqual(h.shape + tau.shape, (2, 3, 5, 2, 3))
        h, tau = la.lqr(self.y, 'raw')
        self.assertEqual((h.ndim, tau.ndim), (2, 1))
        self.assertEqual(h.shape + tau.shape, (5, 3, 3))

    def test_lu(self):
        """Check that lu returns correct shape in each mode
        """
        self.pick_var_type('d')
        low, up, piv = la.lu(self.w, 'separate')
        self.assertEqual((low.ndim, up.ndim, piv.ndim), (3, 3, 2))
        self.assertEqual(low.shape + up.shape + piv.shape,
                         (2, 3, 3, 2, 3, 3, 2, 3))
        low, up, piv = la.lu(self.x, 'separate')
        self.assertEqual((low.ndim, up.ndim, piv.ndim), (3, 3, 2))
        self.assertEqual(low.shape + up.shape + piv.shape,
                         (2, 5, 3, 2, 3, 3, 2, 3))
        low, up, piv = la.lu(self.y, 'separate')
        self.assertEqual((low.ndim, up.ndim, piv.ndim), (2, 2, 1))
        self.assertEqual(low.shape + up.shape + piv.shape, (3, 3, 3, 5, 3))
        luf, piv = la.lu(self.w, 'raw')
        self.assertEqual((luf.ndim, piv.ndim), (3, 2))
        self.assertEqual(luf.shape + piv.shape, (2, 3, 3, 2, 3))
        luf, piv = la.lu(self.x, 'raw')
        self.assertEqual((luf.ndim, piv.ndim), (3, 2))
        self.assertEqual(luf.shape + piv.shape, (2, 3, 5, 2, 3))
        luf, piv = la.lu(self.y, 'raw')
        self.assertEqual((luf.ndim, piv.ndim), (2, 1))
        self.assertEqual(luf.shape + piv.shape, (5, 3, 3))


class TestValue(TestLinalg):
    """Testing values returned by linalg functions"""

    @utn.loop_test()
    def test_la_fn(self, sctype):
        """Check (r)matmul, (r)solve, (r)lstsq, return arrays of correct value
        """
        self.pick_var_type(sctype)
        # matmul
        self.assertArrayAllClose(la.matmul(self.x, self.y),
                                 gf.matmul(self.x, self.y))
        # rmatmul
        self.assertArrayAllClose(lr.rmatmul(self.x, self.y),
                                 gf.rmatmul(self.x, self.y))
        # solve
        self.assertArrayAllClose(la.solve(self.w, self.y),
                                 gf.solve(self.w, self.y))
        wyout = np.empty((2, 3, 5), sctype)
        wy = la.solve(self.w, self.y, out=wyout)
        self.assertArrayAllClose(wyout, wy)
        # rsolve
        self.assertArrayAllClose(la.rsolve(self.x, self.w),
                                 gf.rsolve(self.x, self.w))
        # lstsq
        self.assertArrayAllClose(la.lstsq(self.x, self.v),
                                 gf.lstsq(self.x, self.v))
        self.assertArrayAllClose(la.lstsq(self.y, self.w),
                                 gf.lstsq(self.y, self.w))
        xvout = np.empty((2, 3, 2), sctype)
        xv = la.lstsq(self.x, self.v, out=xvout)
        self.assertArrayAllClose(xvout, xv)
        # rlstsq
        self.assertArrayAllClose(la.rlstsq(self.w, self.x),
                                 gf.rlstsq(self.w, self.x))
        self.assertArrayAllClose(la.rlstsq(self.u, self.y),
                                 gf.rlstsq(self.u, self.y))

    @utn.loop_test()
    def test_div_fn(self, sctype):
        """Check matldiv, matrdiv return correct value
        """
        self.pick_var_type(sctype)
        # solve
        self.assertArrayAllClose(la.matldiv(self.w, self.y),
                                 gf.solve(self.w, self.y))
        wyout = np.empty((2, 3, 5), sctype)
        wy = la.matldiv(self.w, self.y, out=wyout)
        self.assertArrayAllClose(wyout, wy)
        # rsolve
        self.assertArrayAllClose(la.matrdiv(self.x, self.w),
                                 gf.rsolve(self.x, self.w))
        # lstsq
        self.assertArrayAllClose(la.matldiv(self.x, self.v),
                                 gf.lstsq(self.x, self.v))
        self.assertArrayAllClose(la.matldiv(self.y, self.w),
                                 gf.lstsq(self.y, self.w))
        # rlstsq
        self.assertArrayAllClose(la.matrdiv(self.w, self.x),
                                 gf.rlstsq(self.w, self.x))
        self.assertArrayAllClose(la.matrdiv(self.u, self.y),
                                 gf.rlstsq(self.u, self.y))

    @utn.loop_test()
    def test_qr(self, sctype):
        """Check that qr returns correct value in each mode
        """
        self.pick_var_type(sctype)
        q, r = la.qr(self.x, 'reduced')
        self.assertArrayAllClose(q @ r, self.x)
        q, r = la.qr(self.y, 'reduced')
        self.assertArrayAllClose(q @ r, self.y)
        q, r = la.qr(self.x, 'complete')
        self.assertArrayAllClose(q @ r, self.x)
        q, r = la.qr(self.y, 'complete')
        self.assertArrayAllClose(q @ r, self.y)
        r = la.qr(self.x, 'r')
        h, tau = la.qr(self.x, 'raw')
        self.assertArrayAllClose(r, np.triu(la.transpose(h))[:, :3])
        r = la.qr(self.y, 'r')
        h, tau = la.qr(self.y, 'raw')
        self.assertArrayAllClose(r, np.triu(la.transpose(h)))

    @utn.loop_test()
    def test_lq(self, sctype):
        """Check that lq returns correct value in each mode
        """
        self.pick_var_type(sctype)
        lo, q = la.lq(self.x, 'reduced')
        self.assertArrayAllClose(lo @ q, self.x)
        lo, q = la.lq(self.y, 'reduced')
        self.assertArrayAllClose(lo @ q, self.y)
        lo, q = la.lq(self.x, 'complete')
        self.assertArrayAllClose(lo @ q, self.x)
        lo, q = la.lq(self.y, 'complete')
        self.assertArrayAllClose(lo @ q, self.y)
        lo = la.lq(self.x, 'l')
        h, tau = la.lq(self.x, 'raw')
        self.assertArrayAllClose(lo, np.tril(la.transpose(h)))
        lo = la.lq(self.y, 'l')
        h, tau = la.lq(self.y, 'raw')
        self.assertArrayAllClose(lo, np.tril(la.transpose(h))[..., :3])

    @utn.loop_test()
    def test_lqr(self, sctype):
        """Check that lqr returns correct value in each mode
        """
        self.pick_var_type(sctype)
        q, r = la.lqr(self.x, 'reduced')
        self.assertArrayAllClose(q @ r, self.x)
        lo, q = la.lqr(self.y, 'reduced')
        self.assertArrayAllClose(lo @ q, self.y)
        q, r = la.lqr(self.x, 'complete')
        self.assertArrayAllClose(q @ r, self.x)
        lo, q = la.lqr(self.y, 'complete')
        self.assertArrayAllClose(lo @ q, self.y)
        r = la.lqr(self.x, 'r')
        h, tau = la.lqr(self.x, 'raw')
        self.assertArrayAllClose(r, np.triu(la.transpose(h))[:, :3])
        lo = la.lqr(self.y, 'r')
        h, tau = la.lqr(self.y, 'raw')
        self.assertArrayAllClose(lo, np.tril(la.transpose(h))[..., :3])

    @utn.loop_test()
    def test_lu(self, sctype):
        """Check that lu returns correct value in each mode
        """
        self.pick_var_type(sctype)
        low, up, piv = la.lu(self.w, 'separate')
        luf, piv = la.lu(self.w, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(low @ up, gf.pivot(self.w, piv))
        self.assertArrayAllClose(np.tril(low, -1), np.tril(luf, -1))
        self.assertArrayAllClose(up, np.triu(luf))
        low, up, piv = la.lu(self.x, 'separate')
        luf, piv = la.lu(self.x, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(np.tril(low, -1), np.tril(luf, -1))
        self.assertArrayAllClose(up, np.triu(luf)[:, :3])
        low, up, piv = la.lu(self.y, 'separate')
        luf, piv = la.lu(self.y, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(np.tril(low, -1), np.tril(luf, -1)[:, :3])
        self.assertArrayAllClose(up, np.triu(luf))

    @utn.loop_test()
    def test_low_rank(self, sctype):
        """Check low rank matrices are handled appropriately
        """
        self.pick_var_type(sctype)
        with self.assertRaises(np.linalg.LinAlgError):
            la.solve(self.ones, self.y)
        q, r = la.qr(self.ones)
        self.assertArrayAllClose(q @ r, self.ones)
        low, up, piv = la.lu(self.ones)
        self.assertArrayAllClose(low @ up, self.ones)


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
