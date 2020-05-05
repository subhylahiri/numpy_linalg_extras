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
    from test_gufunc import utn, hn, main, TestMatsVecs
else:
    from .test_gufunc import utn, hn, main, TestMatsVecs
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
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='wide'):
        self.assertArrayShapesAre(gfl.qr_m(m_sb),
                                  ((4, 1, 3, 3), (4, 1, 3, 7)))
        self.assertArrayShape(gfl.qr_rm(m_sb), (4, 1, 3, 7))
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.qr_n(m_sb)
        # with self.subTest(msg='tall'):
        self.assertArrayShapesAre(gfl.qr_n(m_bs),
                                  ((2, 7, 3), (2, 3, 3)))
        self.assertArrayShape(gfl.qr_rn(m_bs), (2, 3, 3))
        # with self.subTest(msg='complete'):
        self.assertArrayShapesAre(gfl.qr_m(m_bs),
                                  ((2, 7, 7), (2, 7, 3)))
        # with self.subTest(msg='raw'):
        self.assertArrayShapesAre(gfl.qr_rawm(m_sb),
                                  ((4, 1, 7, 3), (4, 1, 3)))
        self.assertArrayShapesAre(gfl.qr_rawn(m_bs),
                                  ((2, 3, 7), (2, 3)))

    @errstate
    def test_lq_returns_expected_shapes(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='wide'):
        self.assertArrayShapesAre(gfl.lq_m(m_sb),
                                  ((4, 1, 3, 3), (4, 1, 3, 7)))
        self.assertArrayShape(gfl.lq_lm(m_sb), (4, 1, 3, 3))
        # with self.subTest(msg='tall'):
        self.assertArrayShapesAre(gfl.lq_n(m_bs),
                                  ((2, 7, 3), (2, 3, 3)))
        self.assertArrayShape(gfl.lq_ln(m_bs), (2, 7, 3))
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lq_m(m_bs)
        # with self.subTest(msg='complete'):
        self.assertArrayShapesAre(gfl.lq_n(m_sb),
                                  ((4, 1, 3, 7), (4, 1, 7, 7)))
        # with self.subTest(msg='raw'):
        self.assertArrayShapesAre(gfl.lq_rawm(m_sb),
                                  ((4, 1, 7, 3), (4, 1, 3)))
        self.assertArrayShapesAre(gfl.lq_rawn(m_bs),
                                  ((2, 3, 7), (2, 3)))

    def test_pinv_returns_expected_shapes(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='wide'):
        self.assertArrayShape(gfl.pinv(m_sb), (4, 1, 7, 3))
        # with self.subTest(msg='tall'):
        self.assertArrayShape(gfl.pinv(m_bs), (2, 3, 7))
        # with self.subTest(msg='wide,+qr'):
        self.assertArrayShapesAre(gfl.pinv_qrm(m_sb), (
            (4, 1, 7, 3), (4, 1, 7, 3), (4, 1, 3)))
        # with self.subTest(msg='tall,+qr'):
        self.assertArrayShapesAre(gfl.pinv_qrn(m_bs), (
            (2, 3, 7), (2, 3, 7), (2, 3)))
        # with self.subTest(msg='wide,-qr'):
        _, wide_f, wide_tau = gfl.pinv_qrm(m_sb)
        self.assertArrayShape(gfl.qr_pinv(wide_f, wide_tau), (4, 1, 7, 3))
        # with self.subTest(msg='tall,-qr'):
        _, tall_f, tall_tau = gfl.pinv_qrn(m_bs)
        self.assertArrayShape(gfl.qr_pinv(tall_f, tall_tau), (2, 3, 7))


class TestQR(TestMatsVecs):
    """Testing gufuncs_lapack.qr_*
    """

    @utn.loop_test(msg='wide')
    def test_qr_returns_expected_values_with_wide(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        unitary, right = gfl.qr_m(m_sb)
        wide = unitary @ right
        eye = dagger(unitary) @ unitary
        eyet = unitary @ dagger(unitary)
        # with self.subTest(msg='qr'):
        self.assertArrayAllClose(wide, m_sb)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(self.id_s, eye)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(self.id_s, eyet)

    @utn.loop_test(msg='tall')
    def test_qr_returns_expected_values_with_tall(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        unitary, right = gfl.qr_n(m_bs)
        tall = unitary @ right
        eye = dagger(unitary) @ unitary
        # with self.subTest(msg='qr'):
        self.assertArrayAllClose(tall, m_bs)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(self.id_s, eye)

    @utn.loop_test(msg='complete')
    def test_qr_complete_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        unitary, right = gfl.qr_m(m_bs)
        tall = unitary @ right
        eye = dagger(unitary) @ unitary
        eyet = unitary @ dagger(unitary)
        # with self.subTest(msg='qr'):
        self.assertArrayAllClose(tall, m_bs)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(self.id_b, eye)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(self.id_b, eyet)

    @utn.loop_test(msg='right')
    def test_qr_r_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='r_m'):
        right = gfl.qr_rm(m_sb)
        rrr = gfl.qr_m(m_sb)[1]
        self.assertArrayAllClose(right, rrr)
        # with self.subTest(msg='r_n'):
        right = gfl.qr_rn(m_bs)
        rrr = gfl.qr_n(m_bs)[1]
        self.assertArrayAllClose(right, rrr)

    @utn.loop_test(msg='rawm')
    def test_qr_rawm_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        rrr = gfl.qr_m(m_sb)[1]
        num = rrr.shape[-2]
        ht_sb, tau = gfl.qr_rawm(m_sb)
        h_sb = transpose(ht_sb)
        vecs = np.tril(h_sb, -1)
        vecs[(...,) + np.diag_indices(num)] = 1
        vnorm = gfb.norm(row(tau) * vecs[..., :num], axis=-2)**2
        right = np.triu(h_sb)
        # with self.subTest(msg='raw_m'):
        self.assertArrayAllClose(right, rrr)
        self.assertArrayAllClose(vnorm, 2 * tau.real)
        for k in range(num):
            vvv = vecs[..., num-k-1:num-k]
            right -= scalar(tau[..., -k-1]) * vvv * (dagger(vvv) @ right)
        # with self.subTest(msg='h_m'):
        self.assertArrayAllClose(right, m_sb)

    @utn.loop_test(msg='rawn')
    def test_qr_rawn_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        rrr = gfl.qr_n(m_bs)[1]
        num = rrr.shape[-1]
        ht_bs, tau = gfl.qr_rawn(m_bs)
        h_bs = transpose(ht_bs)
        vecs: np.ndarray = np.tril(h_bs, -1)
        vecs[(...,) + np.diag_indices(num)] = 1
        vnorm = gfb.norm(row(tau) * vecs, axis=-2)**2
        right = np.triu(h_bs)
        # with self.subTest(msg='raw_n'):
        self.assertArrayAllClose(right[..., :num, :], rrr)
        self.assertArrayAllClose(vnorm, 2 * tau.real)
        for k in range(num):
            vvv = vecs[..., num-k-1:num-k]
            right -= scalar(tau[..., -k-1]) * vvv * (dagger(vvv) @ right)
        # with self.subTest(msg='h_n'):
        self.assertArrayAllClose(right, m_bs)


class TestLQ(TestMatsVecs):
    """Testing gufuncs_lapack.lq_*
    """

    @utn.loop_test(msg='wide')
    def test_lq_returns_expected_values_with_wide(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        left, unitary = gfl.lq_m(m_sb)
        wide = left @ unitary
        eye = unitary @ dagger(unitary)
        # with self.subTest(msg='lq'):
        self.assertArrayAllClose(wide, m_sb)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(self.id_s, eye)

    @utn.loop_test(msg='tall')
    def test_lq_returns_expected_values_with_tall_matrices(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        left, unitary = gfl.lq_n(m_bs)
        tall = left @ unitary
        eye = unitary @ dagger(unitary)
        eyet = dagger(unitary) @ unitary
        # with self.subTest(msg='lq'):
        self.assertArrayAllClose(tall, m_bs)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(self.id_s, eye)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(self.id_s, eyet)

    @utn.loop_test(msg='complete')
    def test_lq_complete_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        left, unitary = gfl.lq_n(m_sb)
        wide = left @ unitary
        eye = unitary @ dagger(unitary)
        eyet = dagger(unitary) @ unitary
        # with self.subTest(msg='lq'):
        self.assertArrayAllClose(wide, m_sb)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(self.id_b, eye)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(self.id_b, eyet)

    @utn.loop_test(msg='l')
    def test_lq_l_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='l_m'):
        left = gfl.lq_lm(m_sb)
        llo = gfl.lq_m(m_sb)[0]
        self.assertArrayAllClose(left, llo)
        # with self.subTest(msg='l_n'):
        left = gfl.lq_ln(m_bs)
        llo = gfl.lq_n(m_bs)[0]
        self.assertArrayAllClose(left, llo)

    @utn.loop_test(msg='rawm')
    def test_lq_rawm_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        llo = gfl.lq_m(m_sb)[0]
        num = llo.shape[-2]
        ht_sb, tau = gfl.lq_rawm(m_sb)
        h_sb = transpose(ht_sb)
        vecs: np.ndarray = np.triu(h_sb, 1)
        vecs[(...,) + np.diag_indices(num)] = 1
        vnorm = gfb.norm(col(tau) * vecs, axis=-1)**2
        left = np.tril(h_sb)
        # with self.subTest(msg='raw_m'):
        self.assertArrayAllClose(left[..., :num], llo)
        # with self.subTest(msg='tau_m'):
        self.assertArrayAllClose(vnorm, 2 * tau.real)
        for k in range(num):
            vvv = vecs[..., num-k-1:num-k, :]
            left -= scalar(tau[..., -k-1].conj()) * (left @ dagger(vvv)) * vvv
        # with self.subTest(msg='h_m'):
        self.assertArrayAllClose(left, m_sb)

    @utn.loop_test(msg='rawn')
    def test_lq_rawn_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        llo = gfl.lq_n(m_bs)[0]
        num = llo.shape[-1]
        ht_bs, tau = gfl.lq_rawn(m_bs)
        h_bs = transpose(ht_bs)
        vecs = np.triu(h_bs, 1)
        vecs[(...,) + np.diag_indices(num)] = 1
        vnorm = gfb.norm(col(tau) * vecs[..., :num, :], axis=-1)**2
        left = np.tril(h_bs)
        # with self.subTest(msg='raw_n'):
        self.assertArrayAllClose(left, llo)
        # with self.subTest(msg='tau_n'):
        self.assertArrayAllClose(vnorm, 2 * tau.real)
        for k in range(num):
            vvv = vecs[..., num-k-1:num-k, :]
            left -= scalar(tau[..., -k-1].conj()) * (left @ dagger(vvv)) * vvv
        # with self.subTest(msg='h_n'):
        self.assertArrayAllClose(left, m_bs)


class TestPinv(TestMatsVecs):
    """Testing gufuncs_lapack.pinv, pinv_qr and qr_pinv
    """
    @utn.loop_test(msg='pinv')
    def test_pinv_returns_expected_values_wide(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='wide'):
        wide_p = gfl.pinv(m_sb)
        self.assertArrayAllClose(m_sb @ wide_p, self.id_s)
        # with self.subTest(msg='wide,+qr'):
        wide_pq, wide_f, wide_tau = gfl.pinv_qrm(m_sb)
        # actually want lq here
        qrf, tau = gfl.lq_rawm(m_sb)
        # qrf = dagger(qrf)
        self.assertArrayAllClose(wide_pq, wide_p)
        # self.assertArrayAllClose(wide_f, qrf)
        # self.assertArrayAllClose(wide_tau, tau)

    @utn.loop_test(msg='pinv')
    def test_pinv_returns_expected_values_tall(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='wide,-qr'):
        wide_qp = gfl.qr_pinv(wide_f, wide_tau)
        self.assertArrayAllClose(wide_qp, wide_p)
        # with self.subTest(msg='tall'):
        tall_p = gfl.pinv(m_bs)
        self.assertArrayAllClose(tall_p @ m_bs, self.id_s)
        # with self.subTest(msg='tall,+qr'):
        tall_pq, tall_f, tall_tau = gfl.pinv_qrn(m_bs)
        qrf, tau = gfl.qr_rawn(m_bs)
        self.assertArrayAllClose(tall_pq, tall_p)
        self.assertArrayAllClose(tall_f, qrf)
        self.assertArrayAllClose(tall_tau, tau)
        # with self.subTest(msg='tall,-qr'):
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
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest('overconstrained'):
        self.assertArrayShape(gfl.lstsq(m_bs, m_bb), (3, 7))
        self.assertArrayShape(gfl.lstsq(m_bs, m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.lstsq(m_bs, m_bb), (3, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.lstsq(m_bs, m_bb)
        # with self.subTest('underconstrained'):
        self.assertArrayShape(gfl.lstsq(m_sb, m_ss), (7, 3))
        self.assertArrayShape(
                            gfl.lstsq(m_sb, m_ss), (4, 1, 7, 3))
        self.assertArrayShape(
                            gfl.lstsq(m_sb, m_ss), (5, 1, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.lstsq(m_sb, m_ss)

    @errstate
    def test_rlstsq_returns_expected_shape(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest('underconstrained'):
        self.assertArrayShape(gfl.rlstsq(m_ss, m_bs), (3, 7))
        self.assertArrayShape(
                            gfl.rlstsq(m_ss, m_bs), (5, 1, 3, 7))
        self.assertArrayShape(gfl.rlstsq(m_ss, m_bs), (2, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rlstsq(m_ss, transpose(m_sb))
        # with self.subTest('overconstrained'):
        self.assertArrayShape(gfl.rlstsq(m_bb, m_sb), (7, 3))
        self.assertArrayShape(gfl.rlstsq(m_bb, m_sb), (3, 7, 3))
        self.assertArrayShape(
                            gfl.rlstsq(m_bb, m_sb), (4, 1, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rlstsq(m_bb, transpose(m_bs))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_shape_tall(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['bs']
        self.assertArrayShapesAre(func(m_bs, m_bb),
                                  ((3, 7), (3, 7), tau))
        self.assertArrayShapesAre(func(m_bs, m_bb),
                                  ((2, 3, 7), (2, 3, 7), (2,) + tau))
        self.assertArrayShapesAre(func(m_bs, m_bb),
                                  ((3, 3, 7), (3, 3, 7), (3,) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(m_bs, m_bb)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_returns_expected_shape_tall(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_bs, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 7))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (3, 7))
        self.assertArrayShape(
            gfl.rqr_lstsq(m_ss, x_f, tau), (5, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        _, x_f, tau = func(m_bs, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (2, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, tau, m_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_shape_wide(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['sb']
        self.assertArrayShapesAre(func(m_sb, m_ss),
                                  ((7, 3), (7, 3), tau))
        self.assertArrayShapesAre(func(m_sb, m_ss), (
            (4, 1, 7, 3), (4, 1, 7, 3), (4, 1) + tau))
        self.assertArrayShapesAre(func(m_sb, m_ss), (
            (5, 1, 7, 3), (5, 1, 7, 3), (5, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(m_sb, m_ss)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_returns_expected_shape_wide(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_bs, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 7))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (3, 7))
        self.assertArrayShape(
                        gfl.rqr_lstsq(m_ss, x_f, tau), (5, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        _, x_f, tau = func(m_bs, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (2, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, tau, m_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_shape_tall(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['bs']
        self.assertArrayShapesAre(func(m_ss, m_bs),
                                  ((3, 7), (3, 7), tau))
        self.assertArrayShapesAre(func(m_ss, m_bs),
                                  ((2, 3, 7), (2, 3, 7), (2,) + tau))
        self.assertArrayShapesAre(func(m_ss, m_bs), (
            (5, 1, 3, 7), (5, 1, 3, 7), (5, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(m_ss, transpose(m_sb))

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_returns_expected_shape_tall(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_ss, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 7))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (3, 7))
        self.assertArrayShape(
            gfl.rqr_lstsq(m_ss, x_f, tau), (5, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        _, x_f, tau = func(m_ss, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (2, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, tau, m_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_shape_wide(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['sb']
        self.assertArrayShapesAre(func(m_bb, m_sb),
                                  ((7, 3), (7, 3), tau))
        self.assertArrayShapesAre(func(m_bb, m_sb),
                                  ((3, 7, 3), (3, 7, 3), (3,) + tau))
        self.assertArrayShapesAre(func(m_bb, m_sb),
                                  ((4, 1, 7, 3), (4, 1, 7, 3), (4, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(m_bb, transpose(m_bs))

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_returns_expected_shape_wide(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_bb, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_ss), (7, 3))
        self.assertArrayShape(
            gfl.qr_lstsq(x_f, tau, m_ss), (5, 1, 7, 3))
        self.assertArrayShape(gfl.rqr_lstsq(m_bb, x_f, tau), (7, 3))
        self.assertArrayShape(
            gfl.rqr_lstsq(m_bb, x_f, tau), (3, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_bs, x_f, tau)
        _, x_f, tau = func(m_bb, m_sb)
        self.assertArrayShape(
                        gfl.qr_lstsq(x_f, tau, m_ss), (4, 1, 7, 3))
        self.assertArrayShape(
                        gfl.rqr_lstsq(m_bb, x_f, tau), (4, 1, 7, 3))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, tau, m_ss)


class TestLstsqVectors(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    def test_lstsq_flexible_signature_with_vectors_mv(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.lstsq(m_bs, v_b), (3,))
        self.assertArrayShape(gfl.lstsq(m_sb, v_s), (7,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_sb, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
            gfl.lstsq(m_sb, m_bs)

    def test_lstsq_flexible_signature_with_vectors_vm(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.lstsq(v_s, m_sb), (7,))
        self.assertArrayShape(gfl.lstsq(v_s, m_sb), (4, 1, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(v_s, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vM: (3)(7)\(3)(7,7)
            gfl.lstsq(m_sb, m_bb)

    def test_lstsq_flexible_signature_with_vectors_vv(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.lstsq(v_s, v_s), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(v_s, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: ()(3)\(7)(3)
            gfl.lstsq(v_s, m_bs)

    def test_rlstsq_flexible_signature_with_vectors_mv(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.rlstsq(m_sb, v_b), (3,))
        self.assertArrayShape(gfl.rlstsq(m_sb, v_b), (4, 1, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_bs, v_b)
        # If interpreted as Mv, this would be: (5,1)(3,3)/(7)(3)->(5,7)(3)
        self.assertArrayShape(gfl.rlstsq(m_ss, m_bs),
                                (5, 1, 3, 7))

    def test_rlstsq_flexible_signature_with_vectors_vm(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.rlstsq(v_b, m_sb), (3,))
        self.assertArrayShape(gfl.rlstsq(v_s, m_bs), (7,))
        self.assertArrayShape(gfl.rlstsq(v_b, m_sb), (4, 1, 3))
        self.assertArrayShape(gfl.rlstsq(v_s, m_bs), (2, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(v_b, m_bs)
        # If interpreted as vM, this would be: (3)(7)/(4,1)(3,7)->(4,3)(3)
        self.assertArrayShape(gfl.rlstsq(m_sb, m_sb),
                                (4, 1, 3, 3))

    def test_rlstsq_flexible_signature_with_vectors_vv(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.rlstsq(v_s, v_s), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(v_s, v_b)
        # If interpreted as vvv, this would be: (7)(3)/(2,7)(3)->(2,7)
        self.assertArrayShape(gfl.rlstsq(m_bs, m_bs), (2, 7, 7))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_flexible_signature_with_vectors_mv(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['bs']
        self.assertArrayShapesAre(func(m_bs, v_b),
                                    ((3,), (3, 7), tau))
        self.assertArrayShapesAre(func(m_bs, v_b),
                                    ((2, 3), (2, 3, 7), (2,) + tau))
        tau = tau_len['sb']
        self.assertArrayShapesAre(func(m_sb, v_s),
                                    ((7,), (7, 3), tau))
        self.assertArrayShapesAre(func(m_sb, v_s),
                                    ((4, 1, 7), (4, 1, 7, 3), (4, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_sb, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
            func(m_sb, m_bs)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_flexible_signature_with_vectors_vm(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['vs']
        self.assertArrayShapesAre(func(v_s, m_sb),
                                    ((7,), (3,), tau))
        self.assertArrayShapesAre(func(v_s, m_sb),
                                    ((4, 1, 7), (4, 1, 3), (4, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_s, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vM: (3)(7)\(3)(7,7)
            func(m_sb, m_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_flexible_signature_with_vectors_vv(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShapesAre(func(v_s, v_s),
                                    ((), (3,), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_s, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: ()(3)\(7)(3)
            func(v_s, m_bs)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_flexible_signature_with_vectors_mv(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['vb']
        self.assertArrayShapesAre(func(m_sb, v_b),
                                    ((3,), (7,), tau))
        self.assertArrayShapesAre(func(m_sb, v_b),
                                    ((4, 1, 3), (4, 1, 7), (4, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_bs, v_b)
        tau = tau_len['bs']
        # If interpreted as Mv, this would be: (5,1)(3,3)/(7)(3)->(5,7)(3)
        self.assertArrayShapesAre(func(m_ss, m_bs), (
            (5, 1, 3, 7), (5, 1, 3, 7), (5, 1) + tau))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_flexible_signature_with_vectors_vm(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['sb']
        self.assertArrayShapesAre(func(v_b, m_sb),
                                    ((3,), (7, 3), tau))
        self.assertArrayShapesAre(func(v_b, m_sb),
                                    ((4, 1, 3), (4, 1, 7, 3), (4, 1) + tau))
        tau = tau_len['bs']
        self.assertArrayShapesAre(func(v_s, m_bs),
                                    ((7,), (3, 7), tau))
        self.assertArrayShapesAre(func(v_s, m_bs),
                                    ((2, 7), (2, 3, 7), (2,) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_b, m_bs)
        tau = tau_len['sb']
        # If interpreted as vM, this would be: (3)(7)/(4,1)(3,7)->(4,3)(3)
        self.assertArrayShapesAre(func(m_sb, m_sb), (
            (4, 1, 3, 3), (4, 1, 7, 3), (4, 1) + tau))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_flexible_signature_with_vectors_vv(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['vs']
        self.assertArrayShapesAre(func(v_s, v_s),
                                  ((), (3,), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_s, v_b)
        tau = tau_len['bs']
        # If interpreted as vvv, this would be: (7)(3)/(2,7)(3)->(2,7)
        self.assertArrayShapesAre(func(m_bs, m_bs),
                                    ((2, 7, 7), (2, 3, 7), (2,) + tau))

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_flexible_signature_with_vectors_mv(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_bs, v_b)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), (2, 3))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), (2, 7))

        _, x_f, tau = func(m_bs, v_b)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), (3,))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), (7,))

        _, x_f, tau = func(m_sb, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), (4, 1, 7))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), (4, 1, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
            gfl.qr_lstsq(x_f, tau, m_bs)

        _, x_f, tau = func(m_sb, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), (7,))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), (3,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(x_f, tau, v_b)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_flexible_signature_with_vectors_vm(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vM: (3)(7)\(3)(7,7)
            gfl.qr_lstsq(x_f, tau, m_bb)

        _, x_f, tau = func(v_s, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), (7,))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), (7,))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), (4, 1, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), (2, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_flexible_signature_with_vectors_vv(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), ())
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_b, x_f, tau)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: ()(3)\(7)(3)
            gfl.qr_lstsq(x_f, tau, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: (7)(3)/()(3)
            gfl.rqr_lstsq(m_sb, x_f, tau)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_flexible_signature_with_vectors_mv(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(v_s, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), (2, 3))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), (2, 7))

        _, x_f, tau = func(v_s, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), (3,))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), (7,))

        _, x_f, tau = func(v_b, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), (4, 1, 7))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), (4, 1, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
            gfl.qr_lstsq(x_f, tau, m_bs)

        _, x_f, tau = func(v_b, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), (7,))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), (3,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_flexible_signature_with_vectors_vm(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vM: (3)(7)\(3)(7,7)
            gfl.qr_lstsq(x_f, tau, m_bb)

        _, x_f, tau = func(m_bs, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), (7,))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), (7,))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), (4, 1, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), (2, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_flexible_signature_with_vectors_vv(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), ())
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_b, x_f, tau)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: ()(3)\(7)(3)
            gfl.qr_lstsq(x_f, tau, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: (7)(3)/()(3)
            gfl.rqr_lstsq(m_sb, x_f, tau)


class TestLstsqVal(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @utn.loop_test()
    def test_lstsq_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # overconstrained
        x_sb = gfl.lstsq(m_bs, m_bb)
        a_bst = dagger(m_bs)
        # with self.subTest(msg='lstsq(over)'):
        self.assertArrayAllClose(a_bst @ m_bs @ x_sb, a_bst @ m_bb)
        x_bs = gfl.rlstsq(m_bb, m_sb)
        a_sbt = dagger(m_sb)
        # with self.subTest(msg='rlstsq(over)'):
        self.assertArrayAllClose(x_bs @ m_sb @ a_sbt, m_bb @ a_sbt)
        # underconstrained
        x_bs = gfl.lstsq(m_sb, m_ss)
        # with self.subTest(msg='lstsq(under)'):
        self.assertArrayAllClose(m_sb @ x_bs, m_ss)
        x_sb = gfl.rlstsq(m_ss, m_bs)
        # with self.subTest(msg='rlstsq(under)'):
        self.assertArrayAllClose(x_sb @ m_bs, m_ss)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_values_with_tall(self, sctype, func):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        suffix = func.__name__[-1] + ')'
        # overconstrained
        x0_sb = gfl.lstsq(m_bs, m_bb)
        # overconstrained
        x_sb, x_f, tau = func(m_bs, m_bb)
        # with self.subTest('lstsq_qr(over,' + suffix):
        self.assertArrayAllClose(x_sb, x0_sb)
        # overconstrained
        xx_sb = gfl.qr_lstsq(x_f, tau, m_bb)
        # with self.subTest('qr_lstsq(over,' + suffix):
        self.assertArrayAllClose(xx_sb, x0_sb)
        # underconstrained
        y_sb = gfl.rqr_lstsq(m_ss, x_f, tau)
        # with self.subTest('rqr_lstsq(under,' + suffix):
        self.assertArrayAllClose(y_sb @ m_bs, m_ss)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_values_with_wide(self, sctype, func):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        suffix = func.__name__[-1] + ')'
        # overconstrained
        x0_bs = gfl.rlstsq(m_bb, m_sb)
        # overconstrained
        x_bs, x_f, tau = func(m_bb, m_sb)
        # with self.subTest('rlstsq_qr(under,' + suffix):
        self.assertArrayAllClose(x_bs, x0_bs)
        # overconstrained
        xx_bs = gfl.rqr_lstsq(m_bb, x_f, tau)
        # with self.subTest('rqr_rlstsq(under,' + suffix):
        self.assertArrayAllClose(xx_bs, x0_bs)
        # underconstrained
        y_bs = gfl.qr_lstsq(x_f, tau, m_ss)
        # with self.subTest('qr_rlstsq(over,' + suffix):
        self.assertArrayAllClose(m_sb @ y_bs, m_ss)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_values_with_wide(self, sctype, func):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        suffix = func.__name__[-1] + ')'
        # underconstrained
        x0_bs = gfl.lstsq(m_sb, m_ss)
        # underconstrained
        x_bs, x_f, tau = func(m_sb, m_ss)
        # with self.subTest('lstsq_qr(under,' + suffix):
        self.assertArrayAllClose(x_bs, x0_bs)
        # underconstrained
        xx_bs = gfl.qr_lstsq(x_f, tau, m_ss)
        # with self.subTest('qr_lstsq(under,' + suffix):
        self.assertArrayAllClose(xx_bs, x0_bs)
        # overconstrained
        y_bs = gfl.rqr_lstsq(m_bb, x_f, tau)
        a_sbt = dagger(m_sb)
        # with self.subTest('rqr_lstsq(over,' + suffix):
        self.assertArrayAllClose(y_bs @ m_sb @ a_sbt, m_bb @ a_sbt)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_values_with_tall(self, sctype, func):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        suffix = func.__name__[-1] + ')'
        # underconstrained
        x0_sb = gfl.rlstsq(m_ss, m_bs)
        # underconstrained
        x_sb, x_f, tau = func(m_ss, m_bs)
        # with self.subTest('rlstsq_qr(over,' + suffix):
        self.assertArrayAllClose(x_sb, x0_sb)
        # underconstrained
        xx_sb = gfl.rqr_lstsq(m_ss, x_f, tau)
        # with self.subTest('rqr_rlstsq(over,' + suffix):
        self.assertArrayAllClose(xx_sb, x0_sb)
        # overconstrained
        y_sb = gfl.qr_lstsq(x_f, tau, m_bb)
        a_bst = dagger(m_bs)
        # with self.subTest('qr_rlstsq(under,' + suffix):
        self.assertArrayAllClose(a_bst @ m_bs @ y_sb, a_bst @ m_bb)

    @unittest.expectedFailure
    @errstate
    @utn.loop_test(msg='rank')
    def test_lstsq_qr_raises_with_low_rank(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lstsq_qrn(self.ones_bs, m_bs)


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
