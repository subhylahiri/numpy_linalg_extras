# -*- coding: utf-8 -*-
"""Test qr & lstsq families of gufuncs
"""
import unittest
import numpy as np
import unittest_numpy as utn
import numpy_linalg.gufuncs._gufuncs_qr_lstsq as gfl
import numpy_linalg.gufuncs._gufuncs_blas as gfb
from numpy_linalg import transpose

errstate = utn.errstate(invalid='raise')
# =============================================================================
# %% Test qr
# =============================================================================


class TestQR(utn.TestCaseNumpy):
    """Testing gufuncs_lapack.qr_*
    """

    def setUp(self):
        super().setUp()
        self.wide = {}
        self.tall = {}
        self.wide = {}
        self.id_small = {}
        self.id_big = {}
        for sctype in self.sctype:
            x = np.random.randn(120, 10, 5, 16)
            y = np.random.randn(120, 10, 16, 5)
            self.wide[sctype] = utn.asa(x, y.swapaxes(-2, -1), sctype)
            self.tall[sctype] = utn.asa(x.swapaxes(-2, -1), y, sctype)
            self.id_small[sctype] = np.eye(5, dtype=sctype)
            self.id_big[sctype] = np.eye(16, dtype=sctype)

    @errstate
    def test_qr_shape(self):
        """Check that qr_* all return arrays with the expected shape
        """
        with self.subTest(msg='wide'):
            q, r = gfl.qr_m(self.wide['d'])
            self.assertEqual(q.shape, (120, 10, 5, 5))
            self.assertEqual(r.shape, (120, 10, 5, 16))
        with self.subTest(msg='tall'):
            q, r = gfl.qr_n(self.tall['d'])
            self.assertEqual(q.shape, (120, 10, 16, 5))
            self.assertEqual(r.shape, (120, 10, 5, 5))
            with self.assertRaisesRegex(*utn.invalid_err):
                gfl.qr_n(self.wide['d'])
        with self.subTest(msg='complete'):
            q, r = gfl.qr_m(self.tall['d'])
            self.assertEqual(q.shape, (120, 10, 16, 16))
            self.assertEqual(r.shape, (120, 10, 16, 5))
        with self.subTest(msg='raw'):
            h, tau = gfl.qr_rawn(self.tall['d'])
            self.assertEqual(h.shape, (120, 10, 5, 16))
            self.assertEqual(tau.shape, (120, 10, 5))

    @utn.loop_test(msg='wide')
    def test_qr_wide(self, sctype):
        """Check that qr_m returns the expected values on wide matrices
        """
        q, r = gfl.qr_m(self.wide[sctype])
        wide = q @ r
        eye = transpose(q.conj()) @ q
        eyet = q @ transpose(q.conj())
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(wide, self.wide[sctype])
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_small[sctype], eye)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_small[sctype], eyet)

    @utn.loop_test(msg='tall')
    def test_qr_tall(self, sctype):
        """Check that qr_n returns the expected values
        """
        q, r = gfl.qr_n(self.tall[sctype])
        tall = q @ r
        eye = transpose(q.conj()) @ q
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(tall, self.tall[sctype])
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_small[sctype], eye)

    @utn.loop_test(msg='complete')
    def test_qr_complete(self, sctype):
        """Check that qr_m returns the expected values on tall matrices
        """
        q, r = gfl.qr_m(self.tall[sctype])
        tall = q @ r
        eye = transpose(q.conj()) @ q
        eyet = q @ transpose(q.conj())
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(tall, self.tall[sctype])
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_big[sctype], eye)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_big[sctype], eyet)

    @utn.loop_test()
    def test_qr_r(self, sctype):
        """Check that qr_rm, qr_rn return the expected values
        """
        with self.subTest(msg='r_m'):
            r = gfl.qr_rm(self.wide[sctype])
            rr = gfl.qr_m(self.wide[sctype])[1]
            self.assertArrayAllClose(r, rr)
        with self.subTest(msg='r_n'):
            r = gfl.qr_rn(self.tall[sctype])
            rr = gfl.qr_n(self.tall[sctype])[1]
            self.assertArrayAllClose(r, rr)

    @utn.loop_test(attr_inds=1)
    def test_qr_rawm(self, sctype):
        """Check that qr_rawm returns the expected values
        """
        rr = gfl.qr_m(self.wide[sctype])[1]
        n = rr.shape[-2]
        ht, tau = gfl.qr_rawm(self.wide[sctype])
        h = transpose(ht)
        v = np.tril(h[..., :n], -1)
        v[(...,) + np.diag_indices(n)] = 1
        vn = gfb.norm(v, axis=-2)**2 * tau
        r = np.triu(h)
        with self.subTest(msg='raw_m'):
            self.assertArrayAllClose(r, rr)
            self.assertArrayAllClose(vn[..., :-1], 2)
            self.assertArrayAllClose(vn[..., -1], 0)
        for k in range(2, n+1):
            vr = v[..., None, :, -k] @ r
            r -= tau[..., None, None, -k] * v[..., -k, None] * vr
        with self.subTest(msg='h_m'):
            self.assertArrayAllClose(r, self.wide[sctype])

    @utn.loop_test(attr_inds=1)
    def test_qr_rawn(self, sctype):
        """Check that qr_rawn returns the expected values
        """
        rr = gfl.qr_n(self.tall[sctype])[1]
        n = rr.shape[-1]
        ht, tau = gfl.qr_rawn(self.tall[sctype])
        h = transpose(ht)
        v = np.tril(h, -1)
        v[(...,) + np.diag_indices(n)] = 1
        vn = gfb.norm(v, axis=-2)**2 * tau
        r = np.triu(h)
        with self.subTest(msg='raw_n'):
            self.assertArrayAllClose(r[..., :5, :], rr)
            self.assertArrayAllClose(vn, 2)
        for k in range(1, n+1):
            vr = v[..., None, :, -k] @ r
            r -= tau[..., None, None, -k] * v[..., -k, None] * vr
        with self.subTest(msg='h_n'):
            self.assertArrayAllClose(r, self.tall[sctype])


# =============================================================================
# %% Test lstsq
# =============================================================================
sh_ufuncs = [gfl.lstsq_qrm, gfl.lstsq_qrn]
rsh_ufuncs = [gfl.rlstsq_qrm, gfl.rlstsq_qrn]


class TestLstsq(utn.TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    def setUp(self):
        super().setUp()
        self.u = {}
        self.v = {}
        self.w = {}
        self.x = {}
        self.y = {}
        self.z = {}
        self.wt = {}
        self.xt = {}
        self.yt = {}
        self.zt = {}
        for sctype in self.sctype:
            self.u[sctype] = utn.randn_asa((5, 4), sctype)
            self.v[sctype] = utn.randn_asa((4, 5), sctype)
            self.w[sctype] = utn.randn_asa((3, 1, 1, 5), sctype)
            self.x[sctype] = utn.randn_asa((2, 8, 5), sctype)
            self.y[sctype] = utn.randn_asa((8, 2), sctype)
            self.z[sctype] = utn.randn_asa((3, 1, 8, 4), sctype)
            self.wt[sctype] = transpose(self.w[sctype]).conj()
            self.xt[sctype] = transpose(self.x[sctype]).conj()
            self.yt[sctype] = transpose(self.y[sctype]).conj()
            self.zt[sctype] = transpose(self.z[sctype]).conj()


class TestLstsqShape(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @errstate
    def test_lstsq_shape(self):
        """Check if (r)lstsq return arrays with the expected shape
        """
        # overconstrained
        a = gfl.lstsq(self.x['d'], self.y['d'])
        self.assertEqual(a.shape, (2, 5, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(self.x['d'], self.yt['d'])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(self.yt['d'], self.xt['d'])
        # underconstrained
        b = gfl.rlstsq(self.yt['d'], self.xt['d'])
        self.assertEqual(b.shape, (2, 2, 5))

    def test_lstsqqr_shape(self):
        """Check if lstsq_qrn, (r)qr_lstsq return arrays of the expected shape
        """
        # overconstrained
        a, xf, tau = gfl.lstsq_qrm(self.x['d'], self.y['d'])
        self.assertArrayEqual(a.shape, (2, 5, 2))
        self.assertArrayEqual(xf.shape, (2, 8, 5))
        self.assertArrayEqual(tau.shape, (2, 8))
        a, xf, tau = gfl.lstsq_qrn(self.x['d'], self.y['d'])
        self.assertArrayEqual(a.shape, (2, 5, 2))
        self.assertArrayEqual(xf.shape, (2, 8, 5))
        self.assertArrayEqual(tau.shape, (2, 5))
        # overconstrained
        b = gfl.qr_lstsq(xf, tau, self.z['d'])
        self.assertArrayEqual(b.shape, (3, 2, 5, 4))
        # underconstrained
        c = gfl.rqr_lstsq(self.w['d'], xf, tau)
        self.assertArrayEqual(c.shape, (3, 2, 1, 8))

    def test_rlstsqqr_shape(self):
        """Check if rlstsq_qrm, (r)qr_lstsq return arrays of the expected shape
        """
        # underconstrained
        a, xf, tau = gfl.rlstsq_qrm(self.w['d'], self.x['d'])
        self.assertArrayEqual(a.shape, (3, 2, 1, 8))
        self.assertArrayEqual(xf.shape, (3, 2, 8, 5))
        self.assertArrayEqual(tau.shape, (3, 2, 5))
        a, xf, tau = gfl.rlstsq_qrn(self.w['d'], self.x['d'])
        self.assertArrayEqual(a.shape, (3, 2, 1, 8))
        self.assertArrayEqual(xf.shape, (3, 2, 8, 5))
        self.assertArrayEqual(tau.shape, (3, 2, 8))
        # underconstrained
        b = gfl.rqr_lstsq(self.v['d'], xf, tau)
        self.assertArrayEqual(b.shape, (3, 2, 4, 8))


class TestLstsqVal(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @utn.loop_test(attr_inds=3)
    def test_lstsq_val(self, sctype):
        """Check if (r)lstsq return the expected values
        """
        # overconstrained
        a = gfl.lstsq(self.x[sctype], self.y[sctype])
        with self.subTest(msg='lstsq(over)'):
            self.assertArrayAllClose(self.xt[sctype] @ self.x[sctype] @ a,
                                     self.xt[sctype] @ self.y[sctype])
        # underconstrained
        b = gfl.rlstsq(self.v[sctype], self.x[sctype])
        with self.subTest(msg='rlstsq(under)'):
            self.assertArrayAllClose(b @ self.x[sctype], self.v[sctype])

    @utn.loop_test(attr_inds=3)
    def test_lstsqqr_val(self, sctype):
        """Check lstsq_qr{m,n}, (r)qr_lstsq return the expected values (tall)
        """
        # overconstrained
        a0 = gfl.lstsq(self.x[sctype], self.y[sctype])
        # overconstrained
        for ufunc, suffix in zip(sh_ufuncs, [',cross)', ')']):
            a, xf, tau = ufunc(self.x[sctype], self.y[sctype])
            with self.subTest('lstsq(qr,over' + suffix):
                self.assertArrayAllClose(a, a0)
            # overconstrained
            aa = gfl.qr_lstsq(xf, tau, self.y[sctype])
            with self.subTest('(qr)lstsq(over' + suffix):
                self.assertArrayAllClose(aa, a0)
            # underconstrained
            b = gfl.rqr_lstsq(self.v[sctype], xf, tau)
            with self.subTest('(rqr)lstsq(under' + suffix):
                self.assertArrayAllClose(b @ self.x[sctype], self.v[sctype])

    @utn.loop_test(attr_inds=3)
    def test_rlstsqqr_val(self, sctype):
        """Check rlstsq_qr{m,n}, (r)qr_lstsq return the expected values (wide)
        """
        # underconstrained
        a0 = gfl.rlstsq(self.w[sctype], self.x[sctype])
        # underconstrained
        for ufunc, suffix in zip(rsh_ufuncs, [')', ',cross)']):
            a, xf, tau = ufunc(self.w[sctype], self.x[sctype])
            with self.subTest('rlstsq(qr,under' + suffix):
                self.assertArrayAllClose(a, a0)
            # underconstrained
            aa = gfl.rqr_lstsq(self.w[sctype], xf, tau)
            with self.subTest('(rqr)rlstsq(under' + suffix):
                self.assertArrayAllClose(aa, a0)
            # overconstrained
            b = gfl.qr_lstsq(xf, tau, self.z[sctype])
            with self.subTest('(qr)rlstsq(over' + suffix):
                self.assertArrayAllClose(self.xt[sctype] @ self.x[sctype] @ b,
                                         self.xt[sctype] @ self.z[sctype])

    @utn.loop_test(attr_inds=3)
    def test_lstsqqrt_val(self, sctype):
        """Check lstsq_qr{m,n}, (r)qr_lstsq return the expected values (wide)
        """
        # underconstrained
        a0 = gfl.lstsq(self.xt[sctype], self.u[sctype])
        # underconstrained
        for ufunc, suffix in zip(sh_ufuncs, [')', ',cross)']):
            a, xf, tau = ufunc(self.xt[sctype], self.u[sctype])
            with self.subTest('lstsq(qr,under' + suffix):
                self.assertArrayAllClose(a, a0)
            # underconstrained
            aa = gfl.qr_lstsq(xf, tau, self.u[sctype])
            with self.subTest('(qr)lstsq(under' + suffix):
                self.assertArrayAllClose(aa, a0)
            # overconstrained
            b = gfl.rqr_lstsq(self.zt[sctype], xf, tau)
            with self.subTest('(rqr)lstsq(over' + suffix):
                self.assertArrayAllClose(b @ self.xt[sctype] @ self.x[sctype],
                                         self.zt[sctype] @ self.x[sctype])

    @utn.loop_test(attr_inds=3)
    def test_rlstsqqrt_val(self, sctype):
        """Check rlstsq_qr{m,n}, (r)qr_lstsq return the expected values (tall)
        """
        # overconstrained
        a0 = gfl.rlstsq(self.yt[sctype], self.xt[sctype])
        # overconstrained
        for ufunc, suffix in zip(rsh_ufuncs, [',cross)', ')']):
            a, xf, tau = ufunc(self.yt[sctype], self.xt[sctype])
            with self.subTest('rlstsq(qr,over' + suffix):
                self.assertArrayAllClose(a, a0)
            # overconstrained
            aa = gfl.rqr_lstsq(self.yt[sctype], xf, tau)
            with self.subTest('(rqr)rlstsq(over' + suffix):
                self.assertArrayAllClose(aa, a0)
            # underconstrained
            b = gfl.qr_lstsq(xf, tau, self.wt[sctype])
            with self.subTest('(qr)rlstsq(under' + suffix):
                self.assertArrayAllClose(self.xt[sctype] @ b, self.wt[sctype])

    @unittest.expectedFailure
    @errstate
    @utn.loop_test(msg='rank')
    def test_rank(self, sctype):
        """Check if lstsq_qr raises an exception when divisor is rank deficient
        """
        yy = self.y[sctype] @ self.yt[sctype]
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lstsq_qr(yy, self.z[sctype])


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
