# -*- coding: utf-8 -*-
"""Test qr & lstsq families of gufuncs
"""
import hypothesis as hy
import numpy as np
import numpy_linalg as la
import numpy_linalg.gufuncs._gufuncs_qr_lstsq as gfl
import numpy_linalg.gufuncs._gufuncs_blas as gfb
import numpy_linalg.testing.unittest_numpy as utn
import numpy_linalg.testing.hypothesis_numpy as hn
from numpy_linalg.testing import main, TestCaseNumpy
# =============================================================================
# pylint: disable=missing-function-docstring
# pylint: disable=unsupported-assignment-operation
# pylint: disable=invalid-sequence-index
errstate = np.errstate(invalid='raise')
hy.settings.register_profile("slow",
                             suppress_health_check=(hy.HealthCheck.too_slow,))
hy.settings.load_profile('slow')
np.set_printoptions(precision=2, threshold=10, edgeitems=2)
# =============================================================================
__all__ = ['TestQRPinvShape', 'TestQR', 'TestLQ', 'TestPinv']
# =============================================================================
# Test qr
# =============================================================================


class TestQRPinvShape(TestCaseNumpy):
    """Testing gufuncs_lapack.qr_*, lq_*, pinv, pinv_qr and qr_pinv
    """

    @errstate
    @hy.given(hn.broadcastable('(a,b),(b,a)', 'd'))
    def test_qr_returns_expected_shapes(self, arrays):
        m_sb, m_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='wide'):
        self.assertArrayShapesAre(gfl.qr_m(m_sb), (utn.chop(wide), wide))
        self.assertArrayShape(gfl.qr_rm(m_sb), wide)
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.qr_n(m_sb)
        # with self.subTest(msg='tall'):
        self.assertArrayShapesAre(gfl.qr_n(m_bs), (tall, utn.chop(tall)))
        self.assertArrayShape(gfl.qr_rn(m_bs), utn.chop(tall))
        # with self.subTest(msg='complete'):
        self.assertArrayShapesAre(gfl.qr_m(m_bs), (utn.grow(tall), tall))
        # with self.subTest(msg='raw'):
        self.assertArrayShapesAre(gfl.qr_rawm(m_sb),
                                  (utn.trnsp(wide), wide[:-1]))
        self.assertArrayShapesAre(gfl.qr_rawn(m_bs),
                                  (utn.trnsp(tall), utn.drop(tall)))

    @errstate
    @hy.given(hn.broadcastable('(a,b),(b,a)', 'd'))
    def test_lq_returns_expected_shapes(self, arrays):
        m_sb, m_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='wide'):
        self.assertArrayShapesAre(gfl.lq_m(m_sb), (utn.chop(wide), wide))
        self.assertArrayShape(gfl.lq_lm(m_sb), utn.chop(wide))
        # with self.subTest(msg='tall'):
        self.assertArrayShapesAre(gfl.lq_n(m_bs), (tall, utn.chop(tall)))
        self.assertArrayShape(gfl.lq_ln(m_bs), tall)
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lq_m(m_bs)
        # with self.subTest(msg='complete'):
        self.assertArrayShapesAre(gfl.lq_n(m_sb), (wide, utn.grow(wide)))
        # with self.subTest(msg='raw'):
        self.assertArrayShapesAre(gfl.lq_rawm(m_sb),
                                  (utn.trnsp(wide), wide[:-1]))
        self.assertArrayShapesAre(gfl.lq_rawn(m_bs),
                                  (utn.trnsp(tall), utn.drop(tall)))

    @hy.given(hn.broadcastable('(a,b),(b,a)', 'd'))
    def test_pinv_returns_expected_shapes(self, arrays):
        m_sb, m_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_bs, m_sb))

        # with self.subTest(msg='wide'):
        self.assertArrayShape(gfl.pinv(m_sb), utn.trnsp(wide))
        # with self.subTest(msg='tall'):
        self.assertArrayShape(gfl.pinv(m_bs), utn.trnsp(tall))
        # with self.subTest(msg='wide,+qr'):
        self.assertArrayShapesAre(
            gfl.pinv_qrm(m_sb), (utn.trnsp(wide), utn.trnsp(wide), wide[:-1]))
        # with self.subTest(msg='tall,+qr'):
        self.assertArrayShapesAre(
            gfl.pinv_qrn(m_bs),
            (utn.trnsp(tall), utn.trnsp(tall), utn.drop(tall)))
        # with self.subTest(msg='wide,-qr'):
        _, m_sb_f, m_sb_tau = gfl.pinv_qrm(m_sb)
        self.assertArrayShape(gfl.qr_pinv(m_sb_f, m_sb_tau), utn.trnsp(wide))
        # with self.subTest(msg='tall,-qr'):
        _, m_bs_f, m_bs_tau = gfl.pinv_qrn(m_bs)
        self.assertArrayShape(gfl.qr_pinv(m_bs_f, m_bs_tau), utn.trnsp(tall))


class TestQR(TestCaseNumpy):
    """Testing gufuncs_lapack.qr_*
    """

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_qr_returns_expected_values_with_wide(self, m_sb):
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        cond = np.linalg.cond(m_sb).max()

        unitary, right = gfl.qr_m(m_sb)
        wide = unitary @ right
        eye = la.dagger(unitary) @ unitary
        eyet = unitary @ la.dagger(unitary)
        id_s = np.identity(m_sb.shape[-2], m_sb.dtype)
        # with self.subTest(msg='qr'):
        self.assertArrayAllClose(wide, m_sb, cond=cond)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(id_s, eye, cond=cond)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(id_s, eyet, cond=cond)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_qr_returns_expected_values_with_tall(self, m_bs):
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))
        cond = np.linalg.cond(m_bs).max()

        unitary, right = gfl.qr_n(m_bs)
        tall = unitary @ right
        eye = la.dagger(unitary) @ unitary
        id_s = np.identity(m_bs.shape[-1], m_bs.dtype)
        # with self.subTest(msg='qr'):
        self.assertArrayAllClose(tall, m_bs, cond=cond)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(id_s, eye, cond=cond)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_qr_complete_returns_expected_values(self, m_bs):
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))
        cond = np.linalg.cond(m_bs).max()

        unitary, right = gfl.qr_m(m_bs)
        tall = unitary @ right
        eye = la.dagger(unitary) @ unitary
        eyet = unitary @ la.dagger(unitary)
        id_b = np.identity(m_bs.shape[-2], m_bs.dtype)
        # with self.subTest(msg='qr'):
        self.assertArrayAllClose(tall, m_bs, cond=cond)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(id_b, eye, cond=cond)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(id_b, eyet, cond=cond)

    @hy.given(hn.broadcastable('(a,b),(b,a)', None))
    def test_qr_r_returns_expected_values(self, arrays):
        m_sb, m_bs = arrays
        hy.assume(hn.all_well_behaved(m_sb, m_bs))
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='r_m'):
        cond = np.linalg.cond(m_sb).max()
        right = gfl.qr_rm(m_sb)
        rrr = gfl.qr_m(m_sb)[1]
        self.assertArrayAllClose(right, rrr, cond=cond)
        # with self.subTest(msg='r_n'):
        cond = np.linalg.cond(m_bs).max()
        right = gfl.qr_rn(m_bs)
        rrr = gfl.qr_n(m_bs)[1]
        self.assertArrayAllClose(right, rrr, cond=cond)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_qr_rawm_returns_expected_values(self, m_sb):
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        cond = np.linalg.cond(m_sb).max()

        rrr = gfl.qr_m(m_sb)[1]
        num = rrr.shape[-2]
        ht_sb, tau = gfl.qr_rawm(m_sb)
        h_sb = la.transpose(ht_sb)
        vecs = np.tril(h_sb, -1)
        vecs[(...,) + np.diag_indices(num)] = 1
        vnorm = gfb.norm(la.row(tau) * vecs[..., :num], axis=-2)**2
        right = np.triu(h_sb)
        # with self.subTest(msg='raw_m'):
        self.assertArrayAllClose(right, rrr, cond=cond)
        self.assertArrayAllClose(vnorm, 2 * tau.real, cond=cond)
        for k in range(num):
            vvv = vecs[..., num-k-1:num-k]
            ttt = la.scalar(tau[..., -k-1])
            right -= ttt * vvv * (la.dagger(vvv) @ right)
        # with self.subTest(msg='h_m'):
        self.assertArrayAllClose(right, m_sb, cond=cond)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_qr_rawn_returns_expected_values(self, m_bs):
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))
        cond = np.linalg.cond(m_bs).max()

        rrr = gfl.qr_n(m_bs)[1]
        num = rrr.shape[-1]
        ht_bs, tau = gfl.qr_rawn(m_bs)
        h_bs = la.transpose(ht_bs)
        vecs = np.tril(h_bs, -1)
        vecs[(...,) + np.diag_indices(num)] = 1
        vnorm = gfb.norm(la.row(tau) * vecs, axis=-2)**2
        right = np.triu(h_bs)
        # with self.subTest(msg='raw_n'):
        self.assertArrayAllClose(right[..., :num, :], rrr, cond=cond)
        self.assertArrayAllClose(vnorm, 2 * tau.real, cond=cond)
        for k in range(num):
            vvv = vecs[..., num-k-1:num-k]
            ttt = la.scalar(tau[..., -k-1])
            right -= ttt * vvv * (la.dagger(vvv) @ right)
        # with self.subTest(msg='h_n'):
        self.assertArrayAllClose(right, m_bs, cond=cond)


class TestLQ(TestCaseNumpy):
    """Testing gufuncs_lapack.lq_*
    """

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_lq_returns_expected_values_with_wide(self, m_sb):
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        cond = np.linalg.cond(m_sb).max()

        left, unitary = gfl.lq_m(m_sb)
        wide = left @ unitary
        eye = unitary @ la.dagger(unitary)
        id_s = np.identity(m_sb.shape[-2], m_sb.dtype)
        # with self.subTest(msg='lq'):
        self.assertArrayAllClose(wide, m_sb, cond=cond)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(id_s, eye, cond=cond)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_lq_returns_expected_values_with_tall(self, m_bs):
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))
        cond = np.linalg.cond(m_bs).max()

        left, unitary = gfl.lq_n(m_bs)
        tall = left @ unitary
        eye = unitary @ la.dagger(unitary)
        eyet = la.dagger(unitary) @ unitary
        id_s = np.identity(m_bs.shape[-1], m_bs.dtype)
        # with self.subTest(msg='lq'):
        self.assertArrayAllClose(tall, m_bs, cond=cond)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(id_s, eye, cond=cond)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(id_s, eyet, cond=cond)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_lq_complete_returns_expected_values(self, m_sb):
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        cond = np.linalg.cond(m_sb).max()

        left, unitary = gfl.lq_n(m_sb)
        wide = left @ unitary
        eye = unitary @ la.dagger(unitary)
        eyet = la.dagger(unitary) @ unitary
        id_b = np.identity(m_sb.shape[-1], m_sb.dtype)
        # with self.subTest(msg='lq'):
        self.assertArrayAllClose(wide, m_sb, cond=cond)
        # with self.subTest(msg='Q Q^T'):
        self.assertArrayAllClose(id_b, eye, cond=cond)
        # with self.subTest(msg='Q^T Q'):
        self.assertArrayAllClose(id_b, eyet, cond=cond)

    @hy.given(hn.broadcastable('(a,b),(b,a)', None))
    def test_lq_l_returns_expected_values(self, arrays):
        m_sb, m_bs = arrays
        hy.assume(hn.all_well_behaved(m_bs, m_sb))
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg='l_m'):
        cond = np.linalg.cond(m_sb).max()
        left = gfl.lq_lm(m_sb)
        llo = gfl.lq_m(m_sb)[0]
        self.assertArrayAllClose(left, llo, cond=cond)
        # with self.subTest(msg='l_n'):
        cond = np.linalg.cond(m_bs).max()
        left = gfl.lq_ln(m_bs)
        llo = gfl.lq_n(m_bs)[0]
        self.assertArrayAllClose(left, llo, cond=cond)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_lq_rawm_returns_expected_values(self, m_sb):
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        cond = np.linalg.cond(m_sb).max()

        llo = gfl.lq_m(m_sb)[0]
        num = llo.shape[-2]
        ht_sb, tau = gfl.lq_rawm(m_sb)
        h_sb = la.transpose(ht_sb)
        vecs = np.triu(h_sb, 1)
        vecs[(...,) + np.diag_indices(num)] = 1
        vnorm = gfb.norm(la.col(tau) * vecs, axis=-1)**2
        left = np.tril(h_sb)
        # with self.subTest(msg='raw_m'):
        self.assertArrayAllClose(left[..., :num], llo, cond=cond)
        # with self.subTest(msg='tau_m'):
        self.assertArrayAllClose(vnorm, 2 * tau.real, cond=cond)
        for k in range(num):
            vvv = vecs[..., num-k-1:num-k, :]
            ttt = la.scalar(tau[..., -k-1])
            left -= ttt.conj() * (left @ la.dagger(vvv)) * vvv
        # with self.subTest(msg='h_m'):
        self.assertArrayAllClose(left, m_sb, cond=cond)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_lq_rawn_returns_expected_values(self, m_bs):
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))
        cond = np.linalg.cond(m_bs).max()

        llo = gfl.lq_n(m_bs)[0]
        num = llo.shape[-1]
        ht_bs, tau = gfl.lq_rawn(m_bs)
        h_bs = la.transpose(ht_bs)
        vecs = np.triu(h_bs, 1)
        vecs[(...,) + np.diag_indices(num)] = 1
        vnorm = gfb.norm(la.col(tau) * vecs[..., :num, :], axis=-1)**2
        left = np.tril(h_bs)
        # with self.subTest(msg='raw_n'):
        self.assertArrayAllClose(left, llo, cond=cond)
        # with self.subTest(msg='tau_n'):
        self.assertArrayAllClose(vnorm, 2 * tau.real, cond=cond)
        for k in range(num):
            vvv = vecs[..., num-k-1:num-k, :]
            ttt = la.scalar(tau[..., -k-1])
            left -= ttt.conj() * (left @ la.dagger(vvv)) * vvv
        # with self.subTest(msg='h_n'):
        self.assertArrayAllClose(left, m_bs, cond=cond)


class TestPinv(TestCaseNumpy):
    """Testing gufuncs_lapack.pinv, pinv_qr and qr_pinv
    """
    @hy.given(hn.broadcastable('(a,b)', None))
    def test_pinv_returns_expected_values_wide(self, m_sb):
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        cond = np.linalg.cond(m_sb).max()

        id_s = np.identity(m_sb.shape[-2], m_sb.dtype)
        # with self.subTest(msg='wide'):
        wide_p = gfl.pinv(m_sb)
        self.assertArrayAllClose(m_sb @ wide_p, id_s, cond=cond)
        # with self.subTest(msg='wide,+qr'):
        wide_pq, wide_f, wide_tau = gfl.pinv_qrm(m_sb)
        # actually want lq here
        qrf, tau = gfl.lq_rawm(m_sb)
        # qrf = la.dagger(qrf)
        self.assertArrayAllClose(wide_pq, wide_p, cond=cond)
        self.assertArrayAllClose(wide_f, qrf, cond=cond)
        self.assertArrayAllClose(wide_tau, tau, cond=cond)
        # with self.subTest(msg='wide,-qr'):
        wide_qp = gfl.qr_pinv(wide_f, wide_tau)
        self.assertArrayAllClose(wide_qp, wide_p, cond=cond)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_pinv_returns_expected_values_tall(self, m_bs):
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))
        cond = np.linalg.cond(m_bs).max()

        # with self.subTest(msg='tall'):
        tall_p = gfl.pinv(m_bs)
        id_s = np.identity(m_bs.shape[-1], m_bs.dtype)
        self.assertArrayAllClose(tall_p @ m_bs, id_s, cond=cond)
        # with self.subTest(msg='tall,+qr'):
        tall_pq, tall_f, tall_tau = gfl.pinv_qrn(m_bs)
        qrf, tau = gfl.qr_rawn(m_bs)
        self.assertArrayAllClose(tall_pq, tall_p, cond=cond)
        self.assertArrayAllClose(tall_f, qrf, cond=cond)
        self.assertArrayAllClose(tall_tau, tau, cond=cond)
        # with self.subTest(msg='tall,-qr'):
        tall_qp = gfl.qr_pinv(tall_f, tall_tau)
        self.assertArrayAllClose(tall_qp, tall_p, cond=cond)


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
