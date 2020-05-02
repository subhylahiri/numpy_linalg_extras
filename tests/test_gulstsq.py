# -*- coding: utf-8 -*-
"""Test qr & lstsq families of gufuncs
"""
import unittest
import numpy as np
import numpy_linalg.gufuncs._gufuncs_qr_lstsq as gfl
import numpy_linalg.gufuncs._gufuncs_blas as gfb
from numpy_linalg import transpose, dagger, row, col, scalar
if __name__.find('tests.') < 0:
    # pylint: disable=import-error
    import unittest_numpy as utn
    from test_gufunc import TestMatsVecs
else:
    from . import unittest_numpy as utn
    from .test_gufunc import TestMatsVecs
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# pylint: disable=unsupported-assignment-operation
# pylint: disable=invalid-sequence-index
errstate = np.errstate(invalid='raise')
# =============================================================================
__all__ = ['TestQRPinvShape', 'TestQR', 'TestLQ', 'TestPinv',
           'TestLstsqShape', 'TestLstsqVectors', 'TestLstsqVal']
# =============================================================================
# Test qr
# =============================================================================


class TestQRPinvShape(TestMatsVecs):
    """Testing gufuncs_lapack.qr_*, lq_*, pinv, pinv_qr and qr_pinv
    """

    @errstate
    def test_qr_returns_expected_shapes(self):
        self.pick_var_type('d')
        with self.subTest(msg='wide'):
            self.assertArrayShapesAre(gfl.qr_m(self.a_sb),
                                      ((4, 1, 3, 3), (4, 1, 3, 7)))
            self.assertArrayShape(gfl.qr_rm(self.a_sb), (4, 1, 3, 7))
            with self.assertRaisesRegex(*utn.invalid_err):
                gfl.qr_n(self.a_sb)
        with self.subTest(msg='tall'):
            self.assertArrayShapesAre(gfl.qr_n(self.a_bs),
                                      ((2, 7, 3), (2, 3, 3)))
            self.assertArrayShape(gfl.qr_rn(self.a_bs), (2, 3, 3))
        with self.subTest(msg='complete'):
            self.assertArrayShapesAre(gfl.qr_m(self.a_bs),
                                      ((2, 7, 7), (2, 7, 3)))
        with self.subTest(msg='raw'):
            self.assertArrayShapesAre(gfl.qr_rawm(self.a_sb),
                                      ((4, 1, 7, 3), (4, 1, 3)))
            self.assertArrayShapesAre(gfl.qr_rawn(self.a_bs),
                                      ((2, 3, 7), (2, 3)))

    @errstate
    def test_lq_returns_expected_shapes(self):
        self.pick_var_type('d')
        with self.subTest(msg='wide'):
            self.assertArrayShapesAre(gfl.lq_m(self.a_sb),
                                      ((4, 1, 3, 3), (4, 1, 3, 7)))
            self.assertArrayShape(gfl.lq_lm(self.a_sb), (4, 1, 3, 3))
        with self.subTest(msg='tall'):
            self.assertArrayShapesAre(gfl.lq_n(self.a_bs),
                                      ((2, 7, 3), (2, 3, 3)))
            self.assertArrayShape(gfl.lq_ln(self.a_bs), (2, 7, 3))
            with self.assertRaisesRegex(*utn.invalid_err):
                gfl.lq_m(self.a_bs)
        with self.subTest(msg='complete'):
            self.assertArrayShapesAre(gfl.lq_n(self.a_sb),
                                      ((4, 1, 3, 7), (4, 1, 7, 7)))
        with self.subTest(msg='raw'):
            self.assertArrayShapesAre(gfl.lq_rawm(self.a_sb),
                                      ((4, 1, 7, 3), (4, 1, 3)))
            self.assertArrayShapesAre(gfl.lq_rawn(self.a_bs),
                                      ((2, 3, 7), (2, 3)))

    def test_pinv_returns_expected_shapes(self):
        self.pick_var_type('d')
        with self.subTest(msg='wide'):
            self.assertArrayShape(gfl.pinv(self.a_sb), (4, 1, 7, 3))
        with self.subTest(msg='tall'):
            self.assertArrayShape(gfl.pinv(self.a_bs), (2, 3, 7))
        with self.subTest(msg='wide,+qr'):
            self.assertArrayShapesAre(gfl.pinv_qrm(self.a_sb), (
                (4, 1, 7, 3), (4, 1, 7, 3), (4, 1, 3)))
        with self.subTest(msg='tall,+qr'):
            self.assertArrayShapesAre(gfl.pinv_qrn(self.a_bs), (
                (2, 3, 7), (2, 3, 7), (2, 3)))
        with self.subTest(msg='wide,-qr'):
            _, wide_f, wide_tau = gfl.pinv_qrm(self.a_sb)
            self.assertArrayShape(gfl.qr_pinv(wide_f, wide_tau), (4, 1, 7, 3))
        with self.subTest(msg='tall,-qr'):
            _, tall_f, tall_tau = gfl.pinv_qrn(self.a_bs)
            self.assertArrayShape(gfl.qr_pinv(tall_f, tall_tau), (2, 3, 7))


class TestQR(TestMatsVecs):
    """Testing gufuncs_lapack.qr_*
    """

    @utn.loop_test(msg='wide')
    def test_qr_returns_expected_values_with_wide(self, sctype):
        self.pick_var_type(sctype)
        q, r = gfl.qr_m(self.a_sb)
        wide = q @ r
        eye = dagger(q) @ q
        eyet = q @ dagger(q)
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(wide, self.a_sb)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_s, eye)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_s, eyet)

    @utn.loop_test(msg='tall')
    def test_qr_returns_expected_values_with_tall(self, sctype):
        self.pick_var_type(sctype)
        q, r = gfl.qr_n(self.a_bs)
        tall = q @ r
        eye = dagger(q) @ q
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(tall, self.a_bs)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_s, eye)

    @utn.loop_test(msg='complete')
    def test_qr_complete_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        q, r = gfl.qr_m(self.a_bs)
        tall = q @ r
        eye = dagger(q) @ q
        eyet = q @ dagger(q)
        with self.subTest(msg='qr'):
            self.assertArrayAllClose(tall, self.a_bs)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_b, eye)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_b, eyet)

    @utn.loop_test(msg='r')
    def test_qr_r_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        with self.subTest(msg='r_m'):
            r = gfl.qr_rm(self.a_sb)
            rr = gfl.qr_m(self.a_sb)[1]
            self.assertArrayAllClose(r, rr)
        with self.subTest(msg='r_n'):
            r = gfl.qr_rn(self.a_bs)
            rr = gfl.qr_n(self.a_bs)[1]
            self.assertArrayAllClose(r, rr)

    @utn.loop_test(msg='rawm')
    def test_qr_rawm_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        rr = gfl.qr_m(self.a_sb)[1]
        n = rr.shape[-2]
        ht, tau = gfl.qr_rawm(self.a_sb)
        h = transpose(ht)
        v: np.ndarray = np.tril(h, -1)
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
            self.assertArrayAllClose(r, self.a_sb)

    @utn.loop_test(msg='rawn')
    def test_qr_rawn_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        rr = gfl.qr_n(self.a_bs)[1]
        n = rr.shape[-1]
        ht, tau = gfl.qr_rawn(self.a_bs)
        h = transpose(ht)
        v: np.ndarray = np.tril(h, -1)
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
            self.assertArrayAllClose(r, self.a_bs)


class TestLQ(TestMatsVecs):
    """Testing gufuncs_lapack.lq_*
    """

    @utn.loop_test(msg='wide')
    def test_lq_returns_expected_values_with_wide(self, sctype):
        self.pick_var_type(sctype)
        lo, q = gfl.lq_m(self.a_sb)
        wide = lo @ q
        eye = q @ dagger(q)
        with self.subTest(msg='lq'):
            self.assertArrayAllClose(wide, self.a_sb)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_s, eye)

    @utn.loop_test(msg='tall')
    def test_lq_returns_expected_values_with_tall_matrices(self, sctype):
        self.pick_var_type(sctype)
        lo, q = gfl.lq_n(self.a_bs)
        tall = lo @ q
        eye = q @ dagger(q)
        eyet = dagger(q) @ q
        with self.subTest(msg='lq'):
            self.assertArrayAllClose(tall, self.a_bs)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_s, eye)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_s, eyet)

    @utn.loop_test(msg='complete')
    def test_lq_complete_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        lo, q = gfl.lq_n(self.a_sb)
        wide = lo @ q
        eye = q @ dagger(q)
        eyet = dagger(q) @ q
        with self.subTest(msg='lq'):
            self.assertArrayAllClose(wide, self.a_sb)
        with self.subTest(msg='q q^T'):
            self.assertArrayAllClose(self.id_b, eye)
        with self.subTest(msg='q^T q'):
            self.assertArrayAllClose(self.id_b, eyet)

    @utn.loop_test(msg='l')
    def test_lq_l_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        with self.subTest(msg='l_m'):
            lo = gfl.lq_lm(self.a_sb)
            llo = gfl.lq_m(self.a_sb)[0]
            self.assertArrayAllClose(lo, llo)
        with self.subTest(msg='l_n'):
            lo = gfl.lq_ln(self.a_bs)
            llo = gfl.lq_n(self.a_bs)[0]
            self.assertArrayAllClose(lo, llo)

    @utn.loop_test(msg='rawm')
    def test_lq_rawm_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        llo = gfl.lq_m(self.a_sb)[0]
        n = llo.shape[-2]
        ht, tau = gfl.lq_rawm(self.a_sb)
        h = transpose(ht)
        v: np.ndarray = np.triu(h, 1)
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
            self.assertArrayAllClose(lo, self.a_sb)

    @utn.loop_test(msg='rawn')
    def test_lq_rawn_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        llo = gfl.lq_n(self.a_bs)[0]
        n = llo.shape[-1]
        ht, tau = gfl.lq_rawn(self.a_bs)
        h = transpose(ht)
        v: np.ndarray = np.triu(h, 1)
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
            self.assertArrayAllClose(lo, self.a_bs)


class TestPinv(TestMatsVecs):
    """Testing gufuncs_lapack.pinv, pinv_qr and qr_pinv
    """
    @utn.loop_test(msg='pinv')
    def test_pinv_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        with self.subTest(msg='wide'):
            wide_p = gfl.pinv(self.a_sb)
            self.assertArrayAllClose(self.a_sb @ wide_p, self.id_s)
        with self.subTest(msg='tall'):
            tall_p = gfl.pinv(self.a_bs)
            self.assertArrayAllClose(tall_p @ self.a_bs, self.id_s)
        with self.subTest(msg='wide,+qr'):
            wide_pq, wide_f, wide_tau = gfl.pinv_qrm(self.a_sb)
            # actually want lq here
            qrf, tau = gfl.lq_rawm(self.a_sb)
            # qrf = dagger(qrf)
            self.assertArrayAllClose(wide_pq, wide_p)
            # self.assertArrayAllClose(wide_f, qrf)
            # self.assertArrayAllClose(wide_tau, tau)
        with self.subTest(msg='tall,+qr'):
            tall_pq, tall_f, tall_tau = gfl.pinv_qrn(self.a_bs)
            qrf, tau = gfl.qr_rawn(self.a_bs)
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
# Test lstsq
# =============================================================================


class TestLstsq(TestMatsVecs):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    def setUp(self):
        self.func = [gfl.lstsq_qrm, gfl.lstsq_qrn,
                     gfl.rlstsq_qrm, gfl.rlstsq_qrn]
        self.tau_len = [{'bs': (7,), 'sb': (3,), 'vs': (3,), 'vb': (7,)},
                        {'bs': (3,), 'sb': (7,), 'vs': (), 'vb': ()},
                        {'bs': (3,), 'sb': (7,), 'vs': (3,), 'vb': (7,)},
                        {'bs': (7,), 'sb': (3,), 'vs': (), 'vb': ()}]
        super().setUp()


class TestLstsqShape(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @errstate
    def test_lstsq_returns_expected_shape(self):
        self.pick_var_type('d')
        with self.subTest('overconstrained'):
            self.assertArrayShape(gfl.lstsq(self.m_bs, self.m_bb), (3, 7))
            self.assertArrayShape(gfl.lstsq(self.a_bs, self.m_bb), (2, 3, 7))
            self.assertArrayShape(gfl.lstsq(self.m_bs, self.a_bb), (3, 3, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lstsq(self.m_bs, self.m_sb)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.lstsq(self.a_bs, self.a_bb)
        with self.subTest('underconstrained'):
            self.assertArrayShape(gfl.lstsq(self.m_sb, self.m_ss), (7, 3))
            self.assertArrayShape(
                                gfl.lstsq(self.a_sb, self.m_ss), (4, 1, 7, 3))
            self.assertArrayShape(
                                gfl.lstsq(self.m_sb, self.a_ss), (5, 1, 7, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lstsq(self.m_sb, self.m_bs)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.lstsq(self.a_sb, self.a_ss)

    @errstate
    def test_rlstsq_returns_expected_shape(self):
        self.pick_var_type('d')
        with self.subTest('underconstrained'):
            self.assertArrayShape(gfl.rlstsq(self.m_ss, self.m_bs), (3, 7))
            self.assertArrayShape(
                                gfl.rlstsq(self.a_ss, self.m_bs), (5, 1, 3, 7))
            self.assertArrayShape(gfl.rlstsq(self.m_ss, self.a_bs), (2, 3, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlstsq(self.m_sb, self.m_bs)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.rlstsq(self.a_ss, transpose(self.a_sb))
        with self.subTest('overconstrained'):
            self.assertArrayShape(gfl.rlstsq(self.m_bb, self.m_sb), (7, 3))
            self.assertArrayShape(gfl.rlstsq(self.a_bb, self.m_sb), (3, 7, 3))
            self.assertArrayShape(
                                gfl.rlstsq(self.m_bb, self.a_sb), (4, 1, 7, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlstsq(self.m_bs, self.m_sb)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.rlstsq(self.a_bb, transpose(self.a_bs))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_shape_tall(self, func, tau_len):
        self.pick_var_type('d')
        tau = tau_len['bs']
        self.assertArrayShapesAre(func(self.m_bs, self.m_bb),
                                  ((3, 7), (3, 7), tau))
        self.assertArrayShapesAre(func(self.a_bs, self.m_bb),
                                  ((2, 3, 7), (2, 3, 7), (2,) + tau))
        self.assertArrayShapesAre(func(self.m_bs, self.a_bb),
                                  ((3, 3, 7), (3, 3, 7), (3,) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(self.m_bs, self.m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(self.a_bs, self.a_bb)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_returns_expected_shape_tall(self, func):
        self.pick_var_type('d')
        _, xf, tau = func(self.m_bs, self.m_bb)
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.m_bb), (3, 7))
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.a_bb), (3, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(self.m_ss, xf, tau), (3, 7))
        self.assertArrayShape(
                        gfl.rqr_lstsq(self.a_ss, xf, tau), (5, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(xf, tau, self.m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(self.m_sb, xf, tau)
        _, xf, tau = func(self.a_bs, self.m_bb)
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(self.m_ss, xf, tau), (2, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(xf, tau, self.a_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_shape_wide(self, func, tau_len):
        self.pick_var_type('d')
        tau = tau_len['sb']
        self.assertArrayShapesAre(func(self.m_sb, self.m_ss),
                                  ((7, 3), (7, 3), tau))
        self.assertArrayShapesAre(func(self.a_sb, self.m_ss), (
            (4, 1, 7, 3), (4, 1, 7, 3), (4, 1) + tau))
        self.assertArrayShapesAre(func(self.m_sb, self.a_ss), (
            (5, 1, 7, 3), (5, 1, 7, 3), (5, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(self.m_sb, self.m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(self.a_sb, self.a_ss)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_returns_expected_shape_wide(self, func):
        self.pick_var_type('d')
        _, xf, tau = func(self.m_bs, self.m_bb)
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.m_bb), (3, 7))
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.a_bb), (3, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(self.m_ss, xf, tau), (3, 7))
        self.assertArrayShape(
                        gfl.rqr_lstsq(self.a_ss, xf, tau), (5, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(xf, tau, self.m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(self.m_sb, xf, tau)
        _, xf, tau = func(self.a_bs, self.m_bb)
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(self.m_ss, xf, tau), (2, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(xf, tau, self.a_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_shape_tall(self, func, tau_len):
        self.pick_var_type('d')
        tau = tau_len['bs']
        self.assertArrayShapesAre(func(self.m_ss, self.m_bs),
                                  ((3, 7), (3, 7), tau))
        self.assertArrayShapesAre(func(self.m_ss, self.a_bs),
                                  ((2, 3, 7), (2, 3, 7), (2,) + tau))
        self.assertArrayShapesAre(func(self.a_ss, self.m_bs), (
            (5, 1, 3, 7), (5, 1, 3, 7), (5, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(self.m_sb, self.m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(self.a_ss, transpose(self.a_sb))

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_returns_expected_shape_tall(self, func):
        self.pick_var_type('d')
        _, xf, tau = func(self.m_ss, self.m_bs)
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.m_bb), (3, 7))
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.a_bb), (3, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(self.m_ss, xf, tau), (3, 7))
        self.assertArrayShape(
                        gfl.rqr_lstsq(self.a_ss, xf, tau), (5, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(xf, tau, self.m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(self.m_sb, xf, tau)
        _, xf, tau = func(self.m_ss, self.a_bs)
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(self.m_ss, xf, tau), (2, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(xf, tau, self.a_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_shape_wide(self, func, tau_len):
        self.pick_var_type('d')
        tau = tau_len['sb']
        self.assertArrayShapesAre(func(self.m_bb, self.m_sb),
                                  ((7, 3), (7, 3), tau))
        self.assertArrayShapesAre(func(self.a_bb, self.m_sb),
                                  ((3, 7, 3), (3, 7, 3), (3,) + tau))
        self.assertArrayShapesAre(func(self.m_bb, self.a_sb),
                                  ((4, 1, 7, 3), (4, 1, 7, 3), (4, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(self.m_bs, self.m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(self.a_bb, transpose(self.a_bs))

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_returns_expected_shape_wide(self, func):
        self.pick_var_type('d')
        _, xf, tau = func(self.m_bb, self.m_sb)
        self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.m_ss), (7, 3))
        self.assertArrayShape(
            gfl.qr_lstsq(xf, tau, self.a_ss), (5, 1, 7, 3))
        self.assertArrayShape(gfl.rqr_lstsq(self.m_bb, xf, tau), (7, 3))
        self.assertArrayShape(
            gfl.rqr_lstsq(self.a_bb, xf, tau), (3, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(xf, tau, self.m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(self.m_bs, xf, tau)
        _, xf, tau = func(self.m_bb, self.a_sb)
        self.assertArrayShape(
                        gfl.qr_lstsq(xf, tau, self.m_ss), (4, 1, 7, 3))
        self.assertArrayShape(
                        gfl.rqr_lstsq(self.m_bb, xf, tau), (4, 1, 7, 3))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(xf, tau, self.a_ss)


class TestLstsqVectors(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    def test_lstsq_flexible_signature_with_vectors(self):
        self.pick_var_type('d')
        with self.subTest('Matrix-Vector'):
            self.assertArrayShape(gfl.lstsq(self.m_bs, self.v_b), (3,))
            self.assertArrayShape(gfl.lstsq(self.m_sb, self.v_s), (7,))
            self.assertArrayShape(gfl.lstsq(self.a_bs, self.v_b), (2, 3))
            self.assertArrayShape(gfl.lstsq(self.a_sb, self.v_s), (4, 1, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lstsq(self.m_sb, self.v_b)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
                gfl.lstsq(self.a_sb, self.m_bs)
        with self.subTest('Vector-Matrix'):
            self.assertArrayShape(gfl.lstsq(self.v_s, self.m_sb), (7,))
            self.assertArrayShape(gfl.lstsq(self.v_s, self.a_sb), (4, 1, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lstsq(self.v_s, self.m_bs)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vM: (3)(7)\(3)(7,7)
                gfl.lstsq(self.m_sb, self.a_bb)
        with self.subTest('Vector-Vector'):
            self.assertArrayShape(gfl.lstsq(self.v_s, self.v_s), ())
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lstsq(self.v_s, self.v_b)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vv: ()(3)\(7)(3)
                gfl.lstsq(self.v_s, self.m_bs)

    def test_rlstsq_flexible_signature_with_vectors(self):
        self.pick_var_type('d')
        with self.subTest('Matrix-Vector'):
            self.assertArrayShape(gfl.rlstsq(self.m_sb, self.v_b), (3,))
            self.assertArrayShape(gfl.rlstsq(self.a_sb, self.v_b), (4, 1, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlstsq(self.m_bs, self.v_b)
            # If interpreted as Mv, this would be: (5,1)(3,3)/(7)(3)->(5,7)(3)
            self.assertArrayShape(gfl.rlstsq(self.a_ss, self.m_bs),
                                  (5, 1, 3, 7))
        with self.subTest('Vector-Matrix'):
            self.assertArrayShape(gfl.rlstsq(self.v_b, self.m_sb), (3,))
            self.assertArrayShape(gfl.rlstsq(self.v_s, self.m_bs), (7,))
            self.assertArrayShape(gfl.rlstsq(self.v_b, self.a_sb), (4, 1, 3))
            self.assertArrayShape(gfl.rlstsq(self.v_s, self.a_bs), (2, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlstsq(self.v_b, self.m_bs)
            # If interpreted as vM, this would be: (3)(7)/(4,1)(3,7)->(4,3)(3)
            self.assertArrayShape(gfl.rlstsq(self.m_sb, self.a_sb),
                                  (4, 1, 3, 3))
        with self.subTest('Vector-Vector'):
            self.assertArrayShape(gfl.rlstsq(self.v_s, self.v_s), ())
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlstsq(self.v_s, self.v_b)
            # If interpreted as vv, this would be: (7)(3)/(2,7)(3)->(2,7)
            self.assertArrayShape(gfl.rlstsq(self.m_bs, self.a_bs), (2, 7, 7))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_flexible_signature_with_vectors(self, func, tau_len):
        self.pick_var_type('d')
        with self.subTest('Matrix-Vector'):
            tau = tau_len['bs']
            self.assertArrayShapesAre(func(self.m_bs, self.v_b),
                                      ((3,), (3, 7), tau))
            self.assertArrayShapesAre(func(self.a_bs, self.v_b),
                                      ((2, 3), (2, 3, 7), (2,) + tau))
            tau = tau_len['sb']
            self.assertArrayShapesAre(func(self.m_sb, self.v_s),
                                      ((7,), (7, 3), tau))
            self.assertArrayShapesAre(func(self.a_sb, self.v_s),
                                      ((4, 1, 7), (4, 1, 7, 3), (4, 1) + tau))
            with self.assertRaisesRegex(*utn.core_dim_err):
                func(self.m_sb, self.v_b)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
                func(self.a_sb, self.m_bs)
        with self.subTest('Vector-Matrix'):
            tau = tau_len['vs']
            self.assertArrayShapesAre(func(self.v_s, self.m_sb),
                                      ((7,), (3,), tau))
            self.assertArrayShapesAre(func(self.v_s, self.a_sb),
                                      ((4, 1, 7), (4, 1, 3), (4, 1) + tau))
            with self.assertRaisesRegex(*utn.core_dim_err):
                func(self.v_s, self.m_bs)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vM: (3)(7)\(3)(7,7)
                func(self.m_sb, self.a_bb)
        with self.subTest('Vector-Vector'):
            self.assertArrayShapesAre(func(self.v_s, self.v_s),
                                      ((), (3,), tau))
            with self.assertRaisesRegex(*utn.core_dim_err):
                func(self.v_s, self.v_b)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vv: ()(3)\(7)(3)
                func(self.v_s, self.m_bs)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_flexible_signature_with_vectors(self, func, tau_len):
        self.pick_var_type('d')
        with self.subTest('Matrix-Vector'):
            tau = tau_len['vb']
            self.assertArrayShapesAre(func(self.m_sb, self.v_b),
                                      ((3,), (7,), tau))
            self.assertArrayShapesAre(func(self.a_sb, self.v_b),
                                      ((4, 1, 3), (4, 1, 7), (4, 1) + tau))
            with self.assertRaisesRegex(*utn.core_dim_err):
                func(self.m_bs, self.v_b)
            tau = tau_len['bs']
            # If interpreted as Mv, this would be: (5,1)(3,3)/(7)(3)->(5,7)(3)
            self.assertArrayShapesAre(func(self.a_ss, self.m_bs), (
                (5, 1, 3, 7), (5, 1, 3, 7), (5, 1) + tau))
        with self.subTest('Vector-Matrix'):
            tau = tau_len['sb']
            self.assertArrayShapesAre(func(self.v_b, self.m_sb),
                                      ((3,), (7, 3), tau))
            self.assertArrayShapesAre(func(self.v_b, self.a_sb),
                                      ((4, 1, 3), (4, 1, 7, 3), (4, 1) + tau))
            tau = tau_len['bs']
            self.assertArrayShapesAre(func(self.v_s, self.m_bs),
                                      ((7,), (3, 7), tau))
            self.assertArrayShapesAre(func(self.v_s, self.a_bs),
                                      ((2, 7), (2, 3, 7), (2,) + tau))
            with self.assertRaisesRegex(*utn.core_dim_err):
                func(self.v_b, self.m_bs)
            tau = tau_len['sb']
            # If interpreted as vM, this would be: (3)(7)/(4,1)(3,7)->(4,3)(3)
            self.assertArrayShapesAre(func(self.m_sb, self.a_sb), (
                (4, 1, 3, 3), (4, 1, 7, 3), (4, 1) + tau))
        with self.subTest('Vector-Vector'):
            tau = tau_len['vs']
            self.assertArrayShapesAre(func(self.v_s, self.v_s),
                                      ((), (3,), tau))
            with self.assertRaisesRegex(*utn.core_dim_err):
                func(self.v_s, self.v_b)
            tau = tau_len['bs']
            # If interpreted as vv, this would be: (7)(3)/(2,7)(3)->(2,7)
            self.assertArrayShapesAre(func(self.m_bs, self.a_bs),
                                      ((2, 7, 7), (2, 3, 7), (2,) + tau))

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_flexible_signature_with_vectors(self, func):
        self.pick_var_type('d')
        with self.subTest('Matrix-Vector'):
            _, xf, tau = func(self.a_bs, self.v_b)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_b), (2, 3))
            self.assertArrayShape(gfl.rqr_lstsq(self.v_s, xf, tau), (2, 7))

            _, xf, tau = func(self.m_bs, self.v_b)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_b), (3,))
            self.assertArrayShape(gfl.rqr_lstsq(self.v_s, xf, tau), (7,))

            _, xf, tau = func(self.a_sb, self.v_s)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_s), (4, 1, 7))
            self.assertArrayShape(gfl.rqr_lstsq(self.v_b, xf, tau), (4, 1, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
                gfl.qr_lstsq(xf, tau, self.m_bs)

            _, xf, tau = func(self.m_sb, self.v_s)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_s), (7,))
            self.assertArrayShape(gfl.rqr_lstsq(self.v_b, xf, tau), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                func(xf, tau, self.v_b)
        with self.subTest('Vector-Matrix'):
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vM: (3)(7)\(3)(7,7)
                gfl.qr_lstsq(xf, tau, self.a_bb)

            _, xf, tau = func(self.v_s, self.m_sb)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.m_sb), (7,))
            self.assertArrayShape(gfl.rqr_lstsq(self.m_bs, xf, tau), (7,))
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.a_sb), (4, 1, 7))
            self.assertArrayShape(gfl.rqr_lstsq(self.a_bs, xf, tau), (2, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.qr_lstsq(xf, tau, self.m_bs)
            self.assertArrayShape(gfl.rqr_lstsq(self.v_s, xf, tau), ())
        with self.subTest('Vector-Vector'):
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_s), ())
            self.assertArrayShape(gfl.rqr_lstsq(self.v_s, xf, tau), ())
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.qr_lstsq(xf, tau, self.v_b)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rqr_lstsq(self.v_b, xf, tau)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vv: ()(3)\(7)(3)
                gfl.qr_lstsq(xf, tau, self.m_bs)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vv: (7)(3)/()(3)
                gfl.rqr_lstsq(self.m_sb, xf, tau)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_flexible_signature_with_vectors(self, func):
        self.pick_var_type('d')
        with self.subTest('Matrix-Vector'):
            _, xf, tau = func(self.v_s, self.a_bs)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_b), (2, 3))
            self.assertArrayShape(gfl.rqr_lstsq(self.v_s, xf, tau), (2, 7))

            _, xf, tau = func(self.v_s, self.m_bs)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_b), (3,))
            self.assertArrayShape(gfl.rqr_lstsq(self.v_s, xf, tau), (7,))

            _, xf, tau = func(self.v_b, self.a_sb)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_s), (4, 1, 7))
            self.assertArrayShape(gfl.rqr_lstsq(self.v_b, xf, tau), (4, 1, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
                gfl.qr_lstsq(xf, tau, self.m_bs)

            _, xf, tau = func(self.v_b, self.m_sb)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_s), (7,))
            self.assertArrayShape(gfl.rqr_lstsq(self.v_b, xf, tau), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.qr_lstsq(xf, tau, self.v_b)
        with self.subTest('Vector-Matrix'):
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vM: (3)(7)\(3)(7,7)
                gfl.qr_lstsq(xf, tau, self.a_bb)

            _, xf, tau = func(self.m_bs, self.v_s)
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.m_sb), (7,))
            self.assertArrayShape(gfl.rqr_lstsq(self.m_bs, xf, tau), (7,))
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.a_sb), (4, 1, 7))
            self.assertArrayShape(gfl.rqr_lstsq(self.a_bs, xf, tau), (2, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.qr_lstsq(xf, tau, self.m_bs)
            self.assertArrayShape(gfl.rqr_lstsq(self.v_s, xf, tau), ())
        with self.subTest('Vector-Vector'):
            self.assertArrayShape(gfl.qr_lstsq(xf, tau, self.v_s), ())
            self.assertArrayShape(gfl.rqr_lstsq(self.v_s, xf, tau), ())
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.qr_lstsq(xf, tau, self.v_b)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rqr_lstsq(self.v_b, xf, tau)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vv: ()(3)\(7)(3)
                gfl.qr_lstsq(xf, tau, self.m_bs)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as vv: (7)(3)/()(3)
                gfl.rqr_lstsq(self.m_sb, xf, tau)


class TestLstsqVal(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @utn.loop_test()
    def test_lstsq_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        # overconstrained
        a = gfl.lstsq(self.a_bs, self.m_bb)
        a_bst = dagger(self.a_bs)
        with self.subTest(msg='lstsq(over)'):
            self.assertArrayAllClose(a_bst @ self.a_bs @ a, a_bst @ self.m_bb)
        a = gfl.rlstsq(self.a_bb, self.a_sb)
        a_sbt = dagger(self.a_sb)
        with self.subTest(msg='rlstsq(over)'):
            self.assertArrayAllClose(a @ self.a_sb @ a_sbt, self.a_bb @ a_sbt)
        # underconstrained
        a = gfl.lstsq(self.a_sb, self.m_ss)
        with self.subTest(msg='lstsq(under)'):
            self.assertArrayAllClose(self.a_sb @ a, self.m_ss)
        a = gfl.rlstsq(self.a_ss, self.a_bs)
        with self.subTest(msg='rlstsq(under)'):
            self.assertArrayAllClose(a @ self.a_bs, self.a_ss)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_values_with_tall(self, sctype, func):
        self.pick_var_type(sctype)
        suffix = func.__name__[-1] + ')'
        # overconstrained
        a0 = gfl.lstsq(self.a_bs, self.m_bb)
        # overconstrained
        a, xf, tau = func(self.a_bs, self.m_bb)
        with self.subTest('lstsq_qr(over,' + suffix):
            self.assertArrayAllClose(a, a0)
        # overconstrained
        aa = gfl.qr_lstsq(xf, tau, self.m_bb)
        with self.subTest('qr_lstsq(over,' + suffix):
            self.assertArrayAllClose(aa, a0)
        # underconstrained
        b = gfl.rqr_lstsq(self.a_ss, xf, tau)
        with self.subTest('rqr_lstsq(under,' + suffix):
            self.assertArrayAllClose(b @ self.a_bs, self.a_ss)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_values_with_wide(self, sctype, func):
        self.pick_var_type(sctype)
        suffix = func.__name__[-1] + ')'
        # overconstrained
        a0 = gfl.rlstsq(self.a_bb, self.a_sb)
        # overconstrained
        a, xf, tau = func(self.a_bb, self.a_sb)
        with self.subTest('rlstsq_qr(under,' + suffix):
            self.assertArrayAllClose(a, a0)
        # overconstrained
        aa = gfl.rqr_lstsq(self.a_bb, xf, tau)
        with self.subTest('rqr_rlstsq(under,' + suffix):
            self.assertArrayAllClose(aa, a0)
        # underconstrained
        b = gfl.qr_lstsq(xf, tau, self.m_ss)
        with self.subTest('qr_rlstsq(over,' + suffix):
            self.assertArrayAllClose(self.a_sb @ b, self.m_ss)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_values_with_wide(self, sctype, func):
        self.pick_var_type(sctype)
        suffix = func.__name__[-1] + ')'
        # underconstrained
        a0 = gfl.lstsq(self.a_sb, self.m_ss)
        # underconstrained
        a, xf, tau = func(self.a_sb, self.m_ss)
        with self.subTest('lstsq_qr(under,' + suffix):
            self.assertArrayAllClose(a, a0)
        # underconstrained
        aa = gfl.qr_lstsq(xf, tau, self.m_ss)
        with self.subTest('qr_lstsq(under,' + suffix):
            self.assertArrayAllClose(aa, a0)
        # overconstrained
        b = gfl.rqr_lstsq(self.a_bb, xf, tau)
        a_sbt = dagger(self.a_sb)
        with self.subTest('rqr_lstsq(over,' + suffix):
            self.assertArrayAllClose(b @ self.a_sb @ a_sbt, self.a_bb @ a_sbt)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_values_with_tall(self, sctype, func):
        self.pick_var_type(sctype)
        suffix = func.__name__[-1] + ')'
        # underconstrained
        a0 = gfl.rlstsq(self.a_ss, self.a_bs)
        # underconstrained
        a, xf, tau = func(self.a_ss, self.a_bs)
        with self.subTest('rlstsq_qr(over,' + suffix):
            self.assertArrayAllClose(a, a0)
        # underconstrained
        aa = gfl.rqr_lstsq(self.a_ss, xf, tau)
        with self.subTest('rqr_rlstsq(over,' + suffix):
            self.assertArrayAllClose(aa, a0)
        # overconstrained
        b = gfl.qr_lstsq(xf, tau, self.m_bb)
        a_bst = dagger(self.a_bs)
        with self.subTest('qr_rlstsq(under,' + suffix):
            self.assertArrayAllClose(a_bst @ self.a_bs @ b, a_bst @ self.m_bb)

    @unittest.expectedFailure
    @errstate
    @utn.loop_test(msg='rank')
    def test_lstsq_qr_raises_with_low_rank(self, sctype):
        self.pick_var_type(sctype)
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lstsq_qrn(self.ones_bs, self.a_bs)


# =============================================================================
if __name__ == '__main__':
    utn.main(verbosity=2)
