# -*- coding: utf-8 -*-
"""Test qr & lstsq families of gufuncs
"""
import unittest
import numpy as np
import unittest_numpy as utn
import numpy_linalg.gufuncs._gufuncs_qr_lstsq as gfl
import numpy_linalg.gufuncs._gufuncs_blas as gfb
from numpy_linalg import transpose, dagger, row, col, scalar

errstate = utn.errstate(invalid='raise')
# =============================================================================
# %% Test qr
# =============================================================================


class TestQRPinv(utn.TestCaseNumpy):
    """Testing gufuncs_lapack.qr_*, lq_*, pinv, pinv_qr and qr_pinv
    """

    def setUp(self):
        super().setUp()
        self.varnames = ['tall', 'wide', 'id_small', 'id_big']
        self._wide = {}
        self._tall = {}
        self._wide = {}
        self._id_small = {}
        self._id_big = {}
        for sctype in self.sctype:
            x = np.random.randn(120, 10, 5, 16)
            y = np.random.randn(120, 10, 16, 5)
            self._wide[sctype] = utn.asa(x, y.swapaxes(-2, -1), sctype)
            self._tall[sctype] = utn.asa(x.swapaxes(-2, -1), y, sctype)
            self._id_small[sctype] = np.eye(5, dtype=sctype)
            self._id_big[sctype] = np.eye(16, dtype=sctype)


class TestQRPinvShape(TestQRPinv):
    """Testing gufuncs_lapack.qr_*, lq_*, pinv, pinv_qr and qr_pinv
    """

    @errstate
    def test_qr_shape(self):
        """Check that qr_* all return arrays with the expected shape
        """
        self.pick_var_type('d')
        with self.subTest(msg='wide'):
            q, r = gfl.qr_m(self.wide)
            self.assertArrayShaped(q, (120, 10, 5, 5))
            self.assertArrayShaped(r, (120, 10, 5, 16))
            r = gfl.qr_rm(self.wide)
            self.assertArrayShaped(r, (120, 10, 5, 16))
            with self.assertRaisesRegex(*utn.invalid_err):
                gfl.qr_n(self.wide)
        with self.subTest(msg='tall'):
            q, r = gfl.qr_n(self.tall)
            self.assertArrayShaped(q, (120, 10, 16, 5))
            self.assertArrayShaped(r, (120, 10, 5, 5))
            r = gfl.qr_rn(self.tall)
            self.assertArrayShaped(r, (120, 10, 5, 5))
        with self.subTest(msg='complete'):
            q, r = gfl.qr_m(self.tall)
            self.assertArrayShaped(q, (120, 10, 16, 16))
            self.assertArrayShaped(r, (120, 10, 16, 5))
        with self.subTest(msg='raw'):
            h, tau = gfl.qr_rawn(self.tall)
            self.assertArrayShaped(h, (120, 10, 5, 16))
            self.assertArrayShaped(tau, (120, 10, 5))

    @errstate
    def test_lq_shape(self):
        """Check that lq_* all return arrays with the expected shape
        """
        self.pick_var_type('d')
        with self.subTest(msg='wide'):
            lo, q = gfl.lq_m(self.wide)
            self.assertArrayShaped(lo, (120, 10, 5, 5))
            self.assertArrayShaped(q, (120, 10, 5, 16))
            lo = gfl.lq_lm(self.wide)
            self.assertArrayShaped(lo, (120, 10, 5, 5))
        with self.subTest(msg='tall'):
            lo, q = gfl.lq_n(self.tall)
            self.assertArrayShaped(lo, (120, 10, 16, 5))
            self.assertArrayShaped(q, (120, 10, 5, 5))
            lo = gfl.lq_ln(self.tall)
            self.assertArrayShaped(lo, (120, 10, 16, 5))
            with self.assertRaisesRegex(*utn.invalid_err):
                gfl.lq_m(self.tall)
        with self.subTest(msg='complete'):
            lo, q = gfl.lq_n(self.wide)
            self.assertArrayShaped(lo, (120, 10, 5, 16))
            self.assertArrayShaped(q, (120, 10, 16, 16))
        with self.subTest(msg='raw'):
            h, tau = gfl.lq_rawn(self.tall)
            self.assertArrayShaped(h, (120, 10, 5, 16))
            self.assertArrayShaped(tau, (120, 10, 5))

    def test_pinv_shape(self):
        """Check that pinv gufuncs all return arrays with the expected shape
        """
        self.pick_var_type('d')
        with self.subTest(msg='wide'):
            wide_p = gfl.pinv(self.wide)
            self.assertArrayShaped(wide_p, (120, 10, 16, 5))
        with self.subTest(msg='tall'):
            tall_p = gfl.pinv(self.tall)
            self.assertArrayShaped(tall_p, (120, 10, 5, 16))
        with self.subTest(msg='wide,+qr'):
            wide_p, wide_f, wide_tau = gfl.pinv_qrm(self.wide)
            self.assertArrayShaped(wide_p, (120, 10, 16, 5))
            self.assertArrayShaped(wide_f, (120, 10, 16, 5))
            self.assertArrayShaped(wide_tau, (120, 10, 5))
        with self.subTest(msg='tall,+qr'):
            tall_p, tall_f, tall_tau = gfl.pinv_qrn(self.tall)
            self.assertArrayShaped(tall_p, (120, 10, 5, 16))
            self.assertArrayShaped(tall_f, (120, 10, 5, 16))
            self.assertArrayShaped(tall_tau, (120, 10, 5))
        with self.subTest(msg='wide,-qr'):
            wide_p = gfl.qr_pinv(wide_f, wide_tau)
            self.assertArrayShaped(wide_p, (120, 10, 16, 5))
        with self.subTest(msg='tall,-qr'):
            tall_p = gfl.qr_pinv(tall_f, tall_tau)
            self.assertArrayShaped(tall_p, (120, 10, 5, 16))


class TestQR(TestQRPinv):
    """Testing gufuncs_lapack.qr_*
    """

    @utn.loop_test(msg='wide')
    def test_qr_wide(self, sctype):
        """Check that qr_m returns the expected values on wide matrices
        """
        self.pick_var_type(sctype)
        q, r = gfl.qr_m(self.wide)
        wide = q @ r
        eye = dagger(q) @ q
        eyet = q @ dagger(q)
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(wide, self.wide)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_small, eye)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_small, eyet)

    @utn.loop_test(msg='tall')
    def test_qr_tall(self, sctype):
        """Check that qr_n returns the expected values
        """
        self.pick_var_type(sctype)
        q, r = gfl.qr_n(self.tall)
        tall = q @ r
        eye = dagger(q) @ q
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(tall, self.tall)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_small, eye)

    @utn.loop_test(msg='complete')
    def test_qr_complete(self, sctype):
        """Check that qr_m returns the expected values on tall matrices
        """
        self.pick_var_type(sctype)
        q, r = gfl.qr_m(self.tall)
        tall = q @ r
        eye = dagger(q) @ q
        eyet = q @ dagger(q)
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(tall, self.tall)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_big, eye)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_big, eyet)

    @utn.loop_test(msg='r')
    def test_qr_r(self, sctype):
        """Check that qr_rm, qr_rn return the expected values
        """
        self.pick_var_type(sctype)
        with self.subTest(msg='r_m'):
            r = gfl.qr_rm(self.wide)
            rr = gfl.qr_m(self.wide)[1]
            self.assertArrayAllClose(r, rr)
        with self.subTest(msg='r_n'):
            r = gfl.qr_rn(self.tall)
            rr = gfl.qr_n(self.tall)[1]
            self.assertArrayAllClose(r, rr)

    @utn.loop_test(msg='rawm')
    def test_qr_rawm(self, sctype):
        """Check that qr_rawm returns the expected values
        """
        self.pick_var_type(sctype)
        rr = gfl.qr_m(self.wide)[1]
        n = rr.shape[-2]
        ht, tau = gfl.qr_rawm(self.wide)
        h = transpose(ht)
        v = np.tril(h, -1)
        v[(...,) + np.diag_indices(n)] = 1
        vn = gfb.norm(row(tau) * v[..., :n], axis=-2)**2
        r = np.triu(h)
        with self.subTest(msg='raw_m'):
            self.assertArrayAllClose(r, rr)
            self.assertArrayAllClose(vn, 2 * tau.real)
        for k in range(n):
            vv = v[..., n-k-1:n-k]
            r -= scalar(tau[..., -k-1]) * vv * (dagger(vv) @ r)
        with self.subTest(msg='h_m'):
            self.assertArrayAllClose(r, self.wide)

    @utn.loop_test(msg='rawn')
    def test_qr_rawn(self, sctype):
        """Check that qr_rawn returns the expected values
        """
        self.pick_var_type(sctype)
        rr = gfl.qr_n(self.tall)[1]
        n = rr.shape[-1]
        ht, tau = gfl.qr_rawn(self.tall)
        h = transpose(ht)
        v = np.tril(h, -1)
        v[(...,) + np.diag_indices(n)] = 1
        vn = gfb.norm(row(tau) * v, axis=-2)**2
        r = np.triu(h)
        with self.subTest(msg='raw_n'):
            self.assertArrayAllClose(r[..., :n, :], rr)
            self.assertArrayAllClose(vn, 2 * tau.real)
        for k in range(n):
            vv = v[..., n-k-1:n-k]
            r -= scalar(tau[..., -k-1]) * vv * (dagger(vv) @ r)
        with self.subTest(msg='h_n'):
            self.assertArrayAllClose(r, self.tall)


class TestLQ(TestQRPinv):
    """Testing gufuncs_lapack.lq_*
    """

    @utn.loop_test(msg='wide')
    def test_lq_wide(self, sctype):
        """Check that lq_m returns the expected values
        """
        self.pick_var_type(sctype)
        lo, q = gfl.lq_m(self.wide)
        wide = lo @ q
        eye = q @ dagger(q)
        with self.subTest(msg='lq'):
            self.assertArrayAllClose(wide, self.wide)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_small, eye)

    @utn.loop_test(msg='tall')
    def test_lq_tall(self, sctype):
        """Check that lq_n returns the expected values on tall matrices
        """
        self.pick_var_type(sctype)
        lo, q = gfl.lq_n(self.tall)
        tall = lo @ q
        eye = q @ dagger(q)
        eyet = dagger(q) @ q
        with self.subTest(msg='lq'):
            self.assertArrayAllClose(tall, self.tall)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_small, eye)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_small, eyet)

    # @unittest.skip('wrong size')
    @utn.loop_test(msg='complete')
    def test_lq_complete(self, sctype):
        """Check that lq_n returns the expected values on wide matrices
        """
        self.pick_var_type(sctype)
        lo, q = gfl.lq_n(self.wide)
        wide = lo @ q
        eye = q @ dagger(q)
        eyet = dagger(q) @ q
        with self.subTest(msg='lq'):
            self.assertArrayAllClose(wide, self.wide)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_big, eye)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_big, eyet)

    @utn.loop_test(msg='l')
    def test_lq_l(self, sctype):
        """Check that lq_lm, lq_ln return the expected values
        """
        self.pick_var_type(sctype)
        with self.subTest(msg='l_m'):
            lo = gfl.lq_lm(self.wide)
            llo = gfl.lq_m(self.wide)[0]
            self.assertArrayAllClose(lo, llo)
        with self.subTest(msg='l_n'):
            lo = gfl.lq_ln(self.tall)
            llo = gfl.lq_n(self.tall)[0]
            self.assertArrayAllClose(lo, llo)

    @utn.loop_test(msg='rawm')
    def test_lq_rawm(self, sctype):
        """Check that lq_rawm returns the expected values
        """
        self.pick_var_type(sctype)
        llo = gfl.lq_m(self.wide)[0]
        n = llo.shape[-2]
        ht, tau = gfl.lq_rawm(self.wide)
        h = transpose(ht)
        v = np.triu(h, 1)
        v[(...,) + np.diag_indices(n)] = 1
        vn = gfb.norm(col(tau) * v, axis=-1)**2
        lo = np.tril(h)
        with self.subTest(msg='raw_m'):
            self.assertArrayAllClose(lo[..., :n], llo)
        with self.subTest(msg='tau_m'):
            self.assertArrayAllClose(vn, 2 * tau.real)
        for k in range(n):
            vv = v[..., n-k-1:n-k, :]
            lo -= scalar(tau[..., -k-1].conj()) * (lo @ dagger(vv)) * vv
        with self.subTest(msg='h_m'):
            self.assertArrayAllClose(lo, self.wide)

    @utn.loop_test(msg='rawn')
    def test_lq_rawn(self, sctype):
        """Check that lq_rawn returns the expected values
        """
        self.pick_var_type(sctype)
        llo = gfl.lq_n(self.tall)[0]
        n = llo.shape[-1]
        ht, tau = gfl.lq_rawn(self.tall)
        h = transpose(ht)
        v = np.triu(h, 1)
        v[(...,) + np.diag_indices(n)] = 1
        vn = gfb.norm(col(tau) * v[..., :n, :], axis=-1)**2
        lo = np.tril(h)
        with self.subTest(msg='raw_n'):
            self.assertArrayAllClose(lo, llo)
        with self.subTest(msg='tau_n'):
            self.assertArrayAllClose(vn, 2 * tau.real)
        for k in range(n):
            vv = v[..., n-k-1:n-k, :]
            lo -= scalar(tau[..., -k-1].conj()) * (lo @ dagger(vv)) * vv
        with self.subTest(msg='h_n'):
            self.assertArrayAllClose(lo, self.tall)


class TestPinv(TestQRPinv):
    """Testing gufuncs_lapack.pinv, pinv_qr and qr_pinv
    """
    @utn.loop_test(msg='pinv')
    def test_pinv_val(self, sctype):
        """Check that pinv gufuncs all return arrays with the expected values
        """
        self.pick_var_type(sctype)
        with self.subTest(msg='wide'):
            wide_p = gfl.pinv(self.wide)
            self.assertArrayAllClose(self.wide @ wide_p,
                                     self.id_small)
        with self.subTest(msg='tall'):
            tall_p = gfl.pinv(self.tall)
            self.assertArrayAllClose(tall_p @ self.tall,
                                     self.id_small)
        with self.subTest(msg='wide,+qr'):
            wide_pq, wide_f, wide_tau = gfl.pinv_qrm(self.wide)
            # actually want lq here
            qrf, tau = gfl.lq_rawm(self.wide)
            # qrf = dagger(qrf)
            self.assertArrayAllClose(wide_pq, wide_p)
            # self.assertArrayAllClose(wide_f, qrf)
            # self.assertArrayAllClose(wide_tau, tau)
        with self.subTest(msg='tall,+qr'):
            tall_pq, tall_f, tall_tau = gfl.pinv_qrn(self.tall)
            qrf, tau = gfl.qr_rawn(self.tall)
            self.assertArrayAllClose(tall_pq, tall_p)
            self.assertArrayAllClose(tall_f, qrf)
            self.assertArrayAllClose(tall_tau, tau)
        with self.subTest(msg='wide,-qr'):
            wide_qp = gfl.qr_pinv(wide_f, wide_tau)
            self.assertArrayAllClose(wide_qp, wide_p)
        with self.subTest(msg='tall,-qr'):
            tall_qp = gfl.qr_pinv(tall_f, tall_tau)
            self.assertArrayAllClose(tall_qp, tall_p)


# =============================================================================
# %% Test lstsq
# =============================================================================
sh_ufuncs = [gfl.lstsq_qrm, gfl.lstsq_qrn]
rsh_ufuncs = [gfl.rlstsq_qrm, gfl.rlstsq_qrn]


class TestLstsq(utn.TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    def setUp(self):
        super().setUp()
        self.varnames = ['u', 'v', 'w', 'x', 'y', 'z', 'wt', 'xt', 'yt', 'zt',
                         'ones']
        self._u = {}
        self._v = {}
        self._w = {}
        self._x = {}
        self._y = {}
        self._z = {}
        self._wt = {}
        self._xt = {}
        self._yt = {}
        self._zt = {}
        self._ones = {}
        for sctype in self.sctype:
            self._u[sctype] = utn.randn_asa((5, 4), sctype)
            self._v[sctype] = utn.randn_asa((4, 5), sctype)
            self._w[sctype] = utn.randn_asa((3, 1, 1, 5), sctype)
            self._x[sctype] = utn.randn_asa((2, 8, 5), sctype)
            self._y[sctype] = utn.randn_asa((8, 2), sctype)
            self._z[sctype] = utn.randn_asa((3, 1, 8, 4), sctype)
            self._wt[sctype] = dagger(self._w[sctype])
            self._xt[sctype] = dagger(self._x[sctype])
            self._yt[sctype] = dagger(self._y[sctype])
            self._zt[sctype] = dagger(self._z[sctype])
            self._ones[sctype] = utn.ones_asa((8, 3), sctype)


class TestLstsqShape(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @errstate
    def test_lstsq_shape(self):
        """Check if (r)lstsq return arrays with the expected shape
        """
        self.pick_var_type('d')
        # overconstrained
        a = gfl.lstsq(self.x, self.y)
        self.assertArrayShaped(a, (2, 5, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(self.x, self.yt)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(self.yt, self.xt)
        # underconstrained
        b = gfl.rlstsq(self.yt, self.xt)
        self.assertArrayShaped(b, (2, 2, 5))

    def test_lstsqqr_shape(self):
        """Check if lstsq_qrn, (r)qr_lstsq return arrays of the expected shape
        """
        self.pick_var_type('d')
        # overconstrained
        a, xf, tau = gfl.lstsq_qrm(self.x, self.y)
        self.assertArrayShaped(a, (2, 5, 2))
        self.assertArrayShaped(xf, (2, 5, 8))
        self.assertArrayShaped(tau, (2, 8))
        a, xf, tau = gfl.lstsq_qrn(self.x, self.y)
        self.assertArrayShaped(a, (2, 5, 2))
        self.assertArrayShaped(xf, (2, 5, 8))
        self.assertArrayShaped(tau, (2, 5))
        # overconstrained
        b = gfl.qr_lstsq(xf, tau, self.z)
        self.assertArrayShaped(b, (3, 2, 5, 4))
        # underconstrained
        c = gfl.rqr_lstsq(self.w, xf, tau)
        self.assertArrayShaped(c, (3, 2, 1, 8))

    def test_rlstsqqr_shape(self):
        """Check if rlstsq_qrm, (r)qr_lstsq return arrays of the expected shape
        """
        self.pick_var_type('d')
        # underconstrained
        a, xf, tau = gfl.rlstsq_qrm(self.w, self.x)
        self.assertArrayShaped(a, (3, 2, 1, 8))
        self.assertArrayShaped(xf, (3, 2, 5, 8))
        self.assertArrayShaped(tau, (3, 2, 5))
        a, xf, tau = gfl.rlstsq_qrn(self.w, self.x)
        self.assertArrayShaped(a, (3, 2, 1, 8))
        self.assertArrayShaped(xf, (3, 2, 5, 8))
        self.assertArrayShaped(tau, (3, 2, 8))
        # underconstrained
        b = gfl.rqr_lstsq(self.v, xf, tau)
        self.assertArrayShaped(b, (3, 2, 4, 8))


class TestLstsqVal(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @utn.loop_test()
    def test_lstsq_val(self, sctype):
        """Check if (r)lstsq return the expected values
        """
        self.pick_var_type(sctype)
        # overconstrained
        a = gfl.lstsq(self.x, self.y)
        with self.subTest(msg='lstsq(over)'):
            self.assertArrayAllClose(self.xt @ self.x @ a,
                                     self.xt @ self.y)
        # underconstrained
        b = gfl.rlstsq(self.v, self.x)
        with self.subTest(msg='rlstsq(under)'):
            self.assertArrayAllClose(b @ self.x, self.v)

    @utn.loop_test()
    def test_lstsqqr_val(self, sctype):
        """Check lstsq_qr{m,n}, (r)qr_lstsq return the expected values (tall)
        """
        self.pick_var_type(sctype)
        # overconstrained
        a0 = gfl.lstsq(self.x, self.y)
        # overconstrained
        for ufunc, suffix in zip(sh_ufuncs, [',cross)', ')']):
            a, xf, tau = ufunc(self.x, self.y)
            with self.subTest('lstsq_qr(over' + suffix):
                self.assertArrayAllClose(a, a0)
            # overconstrained
            aa = gfl.qr_lstsq(xf, tau, self.y)
            with self.subTest('qr_lstsq(over' + suffix):
                self.assertArrayAllClose(aa, a0)
            # underconstrained
            b = gfl.rqr_lstsq(self.v, xf, tau)
            with self.subTest('rqr_lstsq(under' + suffix):
                self.assertArrayAllClose(b @ self.x, self.v)

    @utn.loop_test()
    def test_rlstsqqr_val(self, sctype):
        """Check rlstsq_qr{m,n}, (r)qr_lstsq return the expected values (wide)
        """
        self.pick_var_type(sctype)
        # underconstrained
        a0 = gfl.rlstsq(self.w, self.x)
        # underconstrained
        for ufunc, suffix in zip(rsh_ufuncs, [')', ',cross)']):
            a, xf, tau = ufunc(self.w, self.x)
            with self.subTest('rlstsq_qr(under' + suffix):
                self.assertArrayAllClose(a, a0)
            # underconstrained
            aa = gfl.rqr_lstsq(self.w, xf, tau)
            with self.subTest('rqr_rlstsq(under' + suffix):
                self.assertArrayAllClose(aa, a0)
            # overconstrained
            b = gfl.qr_lstsq(xf, tau, self.z)
            with self.subTest('qr_rlstsq(over' + suffix):
                self.assertArrayAllClose(self.xt @ self.x @ b,
                                         self.xt @ self.z)

    @utn.loop_test()
    def test_lstsqqrt_val(self, sctype):
        """Check lstsq_qr{m,n}, (r)qr_lstsq return the expected values (wide)
        """
        self.pick_var_type(sctype)
        # underconstrained
        a0 = gfl.lstsq(self.xt, self.u)
        # underconstrained
        for ufunc, suffix in zip(sh_ufuncs, [')', ',cross)']):
            a, xf, tau = ufunc(self.xt, self.u)
            with self.subTest('lstsq_qr(under' + suffix):
                self.assertArrayAllClose(a, a0)
            # underconstrained
            aa = gfl.qr_lstsq(xf, tau, self.u)
            with self.subTest('qr_lstsq(under' + suffix):
                self.assertArrayAllClose(aa, a0)
            # overconstrained
            b = gfl.rqr_lstsq(self.zt, xf, tau)
            with self.subTest('rqr_lstsq(over' + suffix):
                self.assertArrayAllClose(b @ self.xt @ self.x,
                                         self.zt @ self.x)

    @utn.loop_test()
    def test_rlstsqqrt_val(self, sctype):
        """Check rlstsq_qr{m,n}, (r)qr_lstsq return the expected values (tall)
        """
        self.pick_var_type(sctype)
        # overconstrained
        a0 = gfl.rlstsq(self.yt, self.xt)
        # overconstrained
        for ufunc, suffix in zip(rsh_ufuncs, [',cross)', ')']):
            a, xf, tau = ufunc(self.yt, self.xt)
            with self.subTest('rlstsq_qr(over' + suffix):
                self.assertArrayAllClose(a, a0)
            # overconstrained
            aa = gfl.rqr_lstsq(self.yt, xf, tau)
            with self.subTest('rqr_rlstsq(over' + suffix):
                self.assertArrayAllClose(aa, a0)
            # underconstrained
            b = gfl.qr_lstsq(xf, tau, self.wt)
            with self.subTest('qr_rlstsq(under' + suffix):
                self.assertArrayAllClose(self.xt @ b, self.wt)

    @unittest.expectedFailure
    @errstate
    @utn.loop_test(msg='rank')
    def test_rank(self, sctype):
        """Check if lstsq_qr raises an exception when divisor is rank deficient
        """
        self.pick_var_type(sctype)
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lstsq_qr(self.ones, self.z)


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
