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
        self.u = {}
        self.v = {}
        self.w = {}
        self.x = {}
        self.y = {}
        self.z = {}
        self.ones = {}
        self.zeros = {}
        for sctype in self.sctype:
            self.u[sctype] = utn.randn_asa((7, 5), sctype)
            self.v[sctype] = utn.randn_asa((5, 2), sctype)
            self.w[sctype] = utn.randn_asa((2, 3, 3), sctype)
            self.x[sctype] = utn.randn_asa((2, 5, 3), sctype)
            self.y[sctype] = utn.randn_asa((3, 5), sctype)
            self.z[sctype] = utn.randn_asa((3,), sctype)
            self.ones[sctype] = utn.ones_asa((3, 3), sctype)
            self.zeros[sctype] = utn.zeros_asa((3, 3), sctype)


class TestShape(TestLinalg):
    """Testing shapes returned by linalg functions"""

    def test_shape_fn(self):
        """Check transpose, row, col, scal returns arrays of expected shape
        """
        # shape
        self.assertEqual(la.transpose(self.x['d']).shape, (2, 3, 5))
        self.assertEqual(la.row(self.x['d']).shape, (2, 5, 1, 3))
        self.assertEqual(la.col(self.x['d']).shape, (2, 5, 3, 1))
        self.assertEqual(la.scalar(self.x['d']).shape, (2, 5, 3, 1, 1))

    def test_la_fn(self):
        """Check (r)matmul, (r)solve, (r)lstsq, return arrays of correct shape
        """
        # matmul
        self.assertEqual(la.matmul(self.x['d'], self.y['d']).shape, (2, 5, 5))
        self.assertEqual(la.matmul(self.x['d'], self.z['d']).shape, (2, 5))
        # rmatmul
        self.assertEqual(lr.rmatmul(self.x['d'], self.y['d']).shape, (2, 3, 3))
        self.assertEqual(lr.rmatmul(self.y['d'], self.z['d']).shape, (5,))
        # solve
        self.assertEqual(la.solve(self.w['d'], self.y['d']).shape, (2, 3, 5))
        self.assertEqual(la.solve(self.w['d'], self.z['d']).shape, (2, 3))
        # rsolve
        self.assertEqual(la.rsolve(self.x['d'], self.w['d']).shape, (2, 5, 3))
        self.assertEqual(la.rsolve(self.z['d'], self.w['d']).shape, (2, 3))
        # lstsq
        self.assertEqual(la.lstsq(self.x['d'], self.v['d']).shape, (2, 3, 2))
        self.assertEqual(la.lstsq(self.y['d'], self.w['d']).shape, (2, 5, 3))
        self.assertEqual(la.lstsq(self.y['d'], self.z['d']).shape, (5,))
        self.assertEqual(la.lstsq(self.z['d'], self.w['d']).shape, (2, 3))
        # rlstsq
        self.assertEqual(la.rlstsq(self.w['d'], self.x['d']).shape, (2, 3, 5))
        self.assertEqual(la.rlstsq(self.u['d'], self.y['d']).shape, (7, 3))
        self.assertEqual(la.rlstsq(self.w['d'], self.z['d']).shape, (2, 3))
        self.assertEqual(la.rlstsq(self.z['d'], self.x['d']).shape, (2, 5))

    def test_div_fn(self):
        """Check matldiv, matrdiv return correct shape
        """
        # solve
        self.assertEqual(la.matldiv(self.w['d'], self.y['d']).shape, (2, 3, 5))
        self.assertEqual(la.matldiv(self.w['d'], self.z['d']).shape, (2, 3))
        # rsolve
        self.assertEqual(la.matrdiv(self.x['d'], self.w['d']).shape, (2, 5, 3))
        self.assertEqual(la.matrdiv(self.z['d'], self.w['d']).shape, (2, 3))
        # lstsq
        self.assertEqual(la.matldiv(self.x['d'], self.v['d']).shape, (2, 3, 2))
        self.assertEqual(la.matldiv(self.y['d'], self.w['d']).shape, (2, 5, 3))
        self.assertEqual(la.matldiv(self.y['d'], self.z['d']).shape, (5,))
        self.assertEqual(la.matldiv(self.z['d'], self.w['d']).shape, (2, 3))
        # rlstsq
        self.assertEqual(la.matrdiv(self.w['d'], self.x['d']).shape, (2, 3, 5))
        self.assertEqual(la.matrdiv(self.u['d'], self.y['d']).shape, (7, 3))
        self.assertEqual(la.matrdiv(self.w['d'], self.z['d']).shape, (2, 3))
        self.assertEqual(la.matrdiv(self.z['d'], self.x['d']).shape, (2, 5))

    def test_qr(self):
        """Check that qr returns correct shape in each mode
        """
        q, r = la.qr(self.x['d'], 'reduced')
        self.assertEqual((q.ndim, r.ndim), (3, 3))
        self.assertEqual(q.shape + r.shape, (2, 5, 3, 2, 3, 3))
        q, r = la.qr(self.y['d'], 'reduced')
        self.assertEqual((q.ndim, r.ndim), (2, 2))
        self.assertEqual(q.shape + r.shape, (3, 3, 3, 5))
        q, r = la.qr(self.x['d'], 'complete')
        self.assertEqual((q.ndim, r.ndim), (3, 3))
        self.assertEqual(q.shape + r.shape, (2, 5, 5, 2, 5, 3))
        q, r = la.qr(self.y['d'], 'complete')
        self.assertEqual((q.ndim, r.ndim), (2, 2))
        self.assertEqual(q.shape + r.shape, (3, 3, 3, 5))
        r = la.qr(self.x['d'], 'r')
        self.assertEqual(r.shape, (2, 3, 3))
        r = la.qr(self.y['d'], 'r')
        self.assertEqual(r.shape, (3, 5))
        h, tau = la.qr(self.x['d'], 'raw')
        self.assertEqual((h.ndim, tau.ndim), (3, 2))
        self.assertEqual(h.shape + tau.shape, (2, 3, 5, 2, 3))
        h, tau = la.qr(self.y['d'], 'raw')
        self.assertEqual((h.ndim, tau.ndim), (2, 1))
        self.assertEqual(h.shape + tau.shape, (5, 3, 3))

    def test_lu(self):
        """Check that lu returns correct shape in each mode
        """
        low, up, piv = la.lu(self.w['d'], 'separate')
        self.assertEqual((low.ndim, up.ndim, piv.ndim), (3, 3, 2))
        self.assertEqual(low.shape + up.shape + piv.shape,
                         (2, 3, 3, 2, 3, 3, 2, 3))
        low, up, piv = la.lu(self.x['d'], 'separate')
        self.assertEqual((low.ndim, up.ndim, piv.ndim), (3, 3, 2))
        self.assertEqual(low.shape + up.shape + piv.shape,
                         (2, 5, 3, 2, 3, 3, 2, 3))
        low, up, piv = la.lu(self.y['d'], 'separate')
        self.assertEqual((low.ndim, up.ndim, piv.ndim), (2, 2, 1))
        self.assertEqual(low.shape + up.shape + piv.shape, (3, 3, 3, 5, 3))
        luf, piv = la.lu(self.w['d'], 'raw')
        self.assertEqual((luf.ndim, piv.ndim), (3, 2))
        self.assertEqual(luf.shape + piv.shape, (2, 3, 3, 2, 3))
        luf, piv = la.lu(self.x['d'], 'raw')
        self.assertEqual((luf.ndim, piv.ndim), (3, 2))
        self.assertEqual(luf.shape + piv.shape, (2, 3, 5, 2, 3))
        luf, piv = la.lu(self.y['d'], 'raw')
        self.assertEqual((luf.ndim, piv.ndim), (2, 1))
        self.assertEqual(luf.shape + piv.shape, (5, 3, 3))


class TestValue(TestLinalg):
    """Testing values returned by linalg functions"""

    @utn.loop_test()
    def test_la_fn(self, sctype):
        """Check (r)matmul, (r)solve, (r)lstsq, return arrays of correct value
        """
        # matmul
        self.assertArrayAllClose(la.matmul(self.x[sctype], self.y[sctype]),
                                 gf.matmul(self.x[sctype], self.y[sctype]))
        # rmatmul
        self.assertArrayAllClose(lr.rmatmul(self.x[sctype], self.y[sctype]),
                                 gf.rmatmul(self.x[sctype], self.y[sctype]))
        # solve
        self.assertArrayAllClose(la.solve(self.w[sctype], self.y[sctype]),
                                 gf.solve(self.w[sctype], self.y[sctype]))
        wyout = np.empty((2, 3, 5), sctype)
        wy = la.solve(self.w[sctype], self.y[sctype], out=wyout)
        self.assertArrayAllClose(wyout, wy)
        # rsolve
        self.assertArrayAllClose(la.rsolve(self.x[sctype], self.w[sctype]),
                                 gf.rsolve(self.x[sctype], self.w[sctype]))
        # lstsq
        self.assertArrayAllClose(la.lstsq(self.x[sctype], self.v[sctype]),
                                 gf.lstsq(self.x[sctype], self.v[sctype]))
        self.assertArrayAllClose(la.lstsq(self.y[sctype], self.w[sctype]),
                                 gf.lstsq(self.y[sctype], self.w[sctype]))
        xvout = np.empty((2, 3, 2), sctype)
        xv = la.lstsq(self.x[sctype], self.v[sctype], out=xvout)
        self.assertArrayAllClose(xvout, xv)
        # rlstsq
        self.assertArrayAllClose(la.rlstsq(self.w[sctype], self.x[sctype]),
                                 gf.rlstsq(self.w[sctype], self.x[sctype]))
        self.assertArrayAllClose(la.rlstsq(self.u[sctype], self.y[sctype]),
                                 gf.rlstsq(self.u[sctype], self.y[sctype]))

    @utn.loop_test()
    def test_div_fn(self, sctype):
        """Check matldiv, matrdiv return correct value
        """
        # solve
        self.assertArrayAllClose(la.matldiv(self.w[sctype], self.y[sctype]),
                                 gf.solve(self.w[sctype], self.y[sctype]))
        wyout = np.empty((2, 3, 5), sctype)
        wy = la.matldiv(self.w[sctype], self.y[sctype], out=wyout)
        self.assertArrayAllClose(wyout, wy)
        # rsolve
        self.assertArrayAllClose(la.matrdiv(self.x[sctype], self.w[sctype]),
                                 gf.rsolve(self.x[sctype], self.w[sctype]))
        # lstsq
        self.assertArrayAllClose(la.matldiv(self.x[sctype], self.v[sctype]),
                                 gf.lstsq(self.x[sctype], self.v[sctype]))
        self.assertArrayAllClose(la.matldiv(self.y[sctype], self.w[sctype]),
                                 gf.lstsq(self.y[sctype], self.w[sctype]))
        # rlstsq
        self.assertArrayAllClose(la.matrdiv(self.w[sctype], self.x[sctype]),
                                 gf.rlstsq(self.w[sctype], self.x[sctype]))
        self.assertArrayAllClose(la.matrdiv(self.u[sctype], self.y[sctype]),
                                 gf.rlstsq(self.u[sctype], self.y[sctype]))

    @utn.loop_test()
    def test_qr(self, sctype):
        """Check that qr returns correct value in each mode
        """
        q, r = la.qr(self.x[sctype], 'reduced')
        self.assertArrayAllClose(q @ r, self.x[sctype])
        q, r = la.qr(self.y[sctype], 'reduced')
        self.assertArrayAllClose(q @ r, self.y[sctype])
        q, r = la.qr(self.x[sctype], 'complete')
        self.assertArrayAllClose(q @ r, self.x[sctype])
        q, r = la.qr(self.y[sctype], 'complete')
        self.assertArrayAllClose(q @ r, self.y[sctype])
        r = la.qr(self.x[sctype], 'r')
        h, tau = la.qr(self.x[sctype], 'raw')
        self.assertArrayAllClose(r, np.triu(la.transpose(h))[:, :3])
        r = la.qr(self.y[sctype], 'r')
        h, tau = la.qr(self.y[sctype], 'raw')
        self.assertArrayAllClose(r, np.triu(la.transpose(h)))

    @utn.loop_test()
    def test_lu(self, sctype):
        """Check that lu returns correct value in each mode
        """
        low, up, piv = la.lu(self.w[sctype], 'separate')
        luf, piv = la.lu(self.w[sctype], 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(low @ up, gf.pivot(self.w[sctype], piv))
        self.assertArrayAllClose(np.tril(low, -1), np.tril(luf, -1))
        self.assertArrayAllClose(up, np.triu(luf))
        low, up, piv = la.lu(self.x[sctype], 'separate')
        luf, piv = la.lu(self.x[sctype], 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(np.tril(low, -1), np.tril(luf, -1))
        self.assertArrayAllClose(up, np.triu(luf)[:, :3])
        low, up, piv = la.lu(self.y[sctype], 'separate')
        luf, piv = la.lu(self.y[sctype], 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(np.tril(low, -1), np.tril(luf, -1)[:, :3])
        self.assertArrayAllClose(up, np.triu(luf))

    @utn.loop_test()
    def test_low_rank(self, sctype):
        """Check low rank matrices are handled appropriately
        """
        with self.assertRaises(np.linalg.LinAlgError):
            la.solve(self.ones[sctype], self.y[sctype])
        q, r = la.qr(self.ones[sctype])
        self.assertArrayAllClose(q @ r, self.ones[sctype])
        low, up, piv = la.lu(self.ones[sctype])
        self.assertArrayAllClose(low @ up, self.ones[sctype])


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
