# -*- coding: utf-8 -*-
"""Test qr & lstsq families of gufuncs
"""
from unittest import expectedFailure
import hypothesis as hy
import numpy as np
import numpy_linalg as la
import numpy_linalg.gufuncs._gufuncs_qr_lstsq as gfl
from numpy_linalg.gufuncs import unbroadcast_factors
import numpy_linalg.testing.unittest_numpy as utn
import numpy_linalg.testing.hypothesis_numpy as hn
from numpy_linalg.testing import main, TestCaseNumpy
# =============================================================================
# pylint: disable=missing-function-docstring
errstate = np.errstate(invalid='raise')
hy.settings.register_profile("slow",
                             suppress_health_check=(hy.HealthCheck.too_slow,))
hy.settings.load_profile('slow')
qr_funcs = hy.strategies.sampled_from([gfl.lstsq_qrm, gfl.lstsq_qrn])
rqr_funcs = hy.strategies.sampled_from([gfl.rlstsq_qrm, gfl.rlstsq_qrn])
np.set_printoptions(precision=2, threshold=10, edgeitems=2)
# =============================================================================
__all__ = ['TestLstsqShape', 'TestLstsqVectors', 'TestLstsqVal']
# =============================================================================


def tau_len(array: np.ndarray, fun: np.ufunc) -> int:
    cases = {'l': {'m': array.shape[-2], 'n': array.shape[-1]},
             'r': {'n': array.shape[-2], 'm': array.shape[-1]}}
    return (cases[fun.__name__[0]][fun.__name__[-1]],)


def tau_len_vec(vec: np.ndarray, fun: np.ufunc) -> int:
    cases = {'l': {'m': (len(vec),), 'n': ()},
             'r': {'n': (), 'm': (len(vec),)}}
    return cases[fun.__name__[0]][fun.__name__[-1]]


# =============================================================================
# Test lstsq
# =============================================================================


class TestLstsqShape(TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @errstate
    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_lstsq_returns_expected_shape(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))

        # with self.subTest('overconstrained'):
        expect = utn.array_return_shape('(m,n),(m,p)->(n,p)', m_bs, m_bb)
        self.assertArrayShape(gfl.lstsq(m_bs, m_bb), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.lstsq(*utn.make_bad_broadcast(m_bs, m_bb))
        # with self.subTest('underconstrained'):
        expect = utn.array_return_shape('(m,n),(m,p)->(n,p)', m_sb, m_ss)
        self.assertArrayShape(gfl.lstsq(m_sb, m_ss), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.lstsq(*utn.make_bad_broadcast(m_sb, m_ss))

    @errstate
    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_rlstsq_returns_expected_shape(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))

        # with self.subTest('underconstrained'):
        expect = utn.array_return_shape('(m,n),(p,n)->(m,p)', m_ss, m_bs)
        self.assertArrayShape(gfl.rlstsq(m_ss, m_bs), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rlstsq(*utn.make_bad_broadcast(m_ss, la.transpose(m_sb)))
        # with self.subTest('overconstrained'):
        expect = utn.array_return_shape('(m,n),(p,n)->(m,p)', m_bb, m_sb)
        self.assertArrayShape(gfl.rlstsq(m_bb, m_sb), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rlstsq(*utn.make_bad_broadcast(m_bb, la.transpose(m_bs)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), qr_funcs)
    def test_lstsq_qr_returns_expected_shape_tall(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))
        tall = m_bs.shape

        expect = utn.array_return_shape('(m,n),(m,p)->(n,p),(n,m)', m_bs, m_bb)
        tau = expect[1][:-2] + tau_len(m_bs, fun)
        result = fun(m_bs, m_bb)
        self.assertArrayShapesAre(result, expect + (tau,))
        self.assertArrayShapesAre(unbroadcast_factors(m_bs, *result[1:]),
                                  (utn.trnsp(tall), tall[:-2] + tau[-1:]))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            fun(*utn.make_bad_broadcast(m_bs, m_bb))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), qr_funcs)
    def test_qr_lstsq_returns_expected_shape_tall(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))

        _, x_f, tau = fun(m_bs, m_bb)
        expect = utn.array_return_shape('(n,m),(m,p)->(n,p)', x_f, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), expect)
        expect = utn.array_return_shape('(m,n),(n,p)->(m,p)', m_ss, x_f)
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, *utn.make_bad_broadcast(tau, m_bb, (1, 2)))
        x_f, tau = unbroadcast_factors(m_bs, x_f, tau)
        expect = utn.array_return_shape('(m,n),(p,n)->(m,p)', m_ss, m_bs)
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), expect)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), qr_funcs)
    def test_lstsq_qr_returns_expected_shape_wide(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        wide = m_sb.shape

        expect = utn.array_return_shape('(m,n),(m,p)->(n,p),(n,m)', m_sb, m_ss)
        tau = expect[1][:-2] + tau_len(m_sb, fun)
        result = fun(m_sb, m_ss)
        self.assertArrayShapesAre(result, expect + (tau,))
        self.assertArrayShapesAre(unbroadcast_factors(m_sb, *result[1:]),
                                  (utn.trnsp(wide), wide[:-2] + tau[-1:]))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            fun(*utn.make_bad_broadcast(m_sb, m_ss))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), qr_funcs)
    def test_qr_lstsq_returns_expected_shape_wide(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))

        _, x_f, tau = fun(m_sb, m_ss)
        expect = utn.array_return_shape('(n,m),(m,p)->(n,p)', x_f, m_ss)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_ss), expect)
        expect = utn.array_return_shape('(m,n),(n,p)->(m,p)', m_bb, x_f)
        self.assertArrayShape(gfl.rqr_lstsq(m_bb, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_ss, x_f, tau)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, *utn.make_bad_broadcast(tau, m_ss, (1, 2)))
        x_f, tau = unbroadcast_factors(m_sb, x_f, tau)
        expect = utn.array_return_shape('(m,n),(p,n)->(m,p)', m_bb, m_sb)
        self.assertArrayShape(gfl.rqr_lstsq(m_bb, x_f, tau), expect)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), rqr_funcs)
    def test_rlstsq_qr_returns_expected_shape_tall(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))
        tall = m_bs.shape

        expect = utn.array_return_shape('(m,n),(p,n)->(m,p),(n,p)', m_ss, m_bs)
        tau = expect[1][:-2] + tau_len(m_bs, fun)
        result = fun(m_ss, m_bs)
        self.assertArrayShapesAre(result, expect + (tau,))
        self.assertArrayShapesAre(unbroadcast_factors(m_bs, *result[1:]),
                                  (utn.trnsp(tall), tall[:-2] + tau[-1:]))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            fun(*utn.make_bad_broadcast(m_ss, la.transpose(m_sb)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), rqr_funcs)
    def test_rqr_lstsq_returns_expected_shape_tall(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_well_behaved(m_bs))

        _, x_f, tau = fun(m_ss, m_bs)
        expect = utn.array_return_shape('(n,m),(m,p)->(n,p)', x_f, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), expect)
        expect = utn.array_return_shape('(m,n),(n,p)->(m,p)', m_ss, x_f)
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, *utn.make_bad_broadcast(tau, m_bb, (1, 2)))
        x_f, tau = unbroadcast_factors(m_bs, x_f, tau)
        expect = utn.array_return_shape('(m,n),(m,p)->(n,p)', m_bs, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), expect)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), rqr_funcs)
    def test_rlstsq_qr_returns_expected_shape_wide(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        wide = m_sb.shape

        expect = utn.array_return_shape('(m,n),(p,n)->(m,p),(n,p)', m_bb, m_sb)
        tau = expect[1][:-2] + tau_len(m_sb, fun)
        result = fun(m_bb, m_sb)
        self.assertArrayShapesAre(result, expect + (tau,))
        self.assertArrayShapesAre(unbroadcast_factors(m_sb, *result[1:]),
                                  (utn.trnsp(wide), wide[:-2] + tau[-1:]))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            fun(*utn.make_bad_broadcast(m_bb, la.transpose(m_bs)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), rqr_funcs)
    def test_rqr_lstsq_returns_expected_shape_wide(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))

        _, x_f, tau = fun(m_bb, m_sb)
        expect = utn.array_return_shape('(n,m),(m,p)->(n,p)', x_f, m_ss)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_ss), expect)
        expect = utn.array_return_shape('(m,n),(n,p)->(m,p)', m_bb, x_f)
        self.assertArrayShape(gfl.rqr_lstsq(m_bb, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_bs, x_f, tau)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, *utn.make_bad_broadcast(tau, m_ss, (1, 2)))
        x_f, tau = unbroadcast_factors(m_sb, x_f, tau)
        expect = utn.array_return_shape('(m,n),(m,p)->(n,p)', m_sb, m_ss)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_ss), expect)


class TestLstsqVectors(TestCaseNumpy):
    """Testing (r)lstsq with vectors"""

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'))
    def test_lstsq_flexible_signature_with_vectors_mv(self, arrays):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        off_b, y_one = utn.make_off_by_one(m_sb, m_bs)

        self.assertArrayShape(gfl.lstsq(m_bs, v_b), utn.drop(tall))
        self.assertArrayShape(gfl.lstsq(m_sb, v_s), utn.drop(wide))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_sb, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as Mv:
            gfl.lstsq(m_sb[off_b], m_bs[y_one])

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'))
    def test_lstsq_flexible_signature_with_vectors_vm(self, arrays):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.nonsquare(m_sb))

        self.assertArrayShape(gfl.lstsq(v_s, m_sb), utn.drop(wide))
        self.assertArrayShape(gfl.lstsq(v_b, m_bs), utn.drop(tall))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(v_s, m_bs)

    @hy.given(hn.vectors, hn.vectors)
    def test_lstsq_flexible_signature_with_vectors_vv(self, v_s, v_b):
        hy.assume(len(v_s) != len(v_b))

        self.assertArrayShape(gfl.lstsq(v_s, v_s), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(v_s, v_b)

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'))
    def test_rlstsq_flexible_signature_with_vectors_mv(self, arrays):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.rlstsq(m_sb, v_b), wide[:-1])
        self.assertArrayShape(gfl.rlstsq(m_bs, v_s), tall[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_bs, v_b)

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'))
    def test_rlstsq_flexible_signature_with_vectors_vm(self, arrays):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        off_b, y_one = utn.make_off_by_one(m_sb, m_bs)

        self.assertArrayShape(gfl.rlstsq(v_b, m_sb), wide[:-1])
        self.assertArrayShape(gfl.rlstsq(v_s, m_bs), tall[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(v_b, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as vM:
            gfl.rlstsq(m_bs[y_one], m_sb[off_b])

    @hy.given(hn.vectors, hn.vectors)
    def test_rlstsq_flexible_signature_with_vectors_vv(self, v_s, v_b):
        hy.assume(len(v_s) != len(v_b))

        self.assertArrayShape(gfl.rlstsq(v_s, v_s), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(v_s, v_b)


class TestLstsqQRVectors(TestCaseNumpy):
    """Testing (r)lstsq_qr? with vectors"""

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), qr_funcs)
    def test_lstsq_qr_flexible_signature_with_vectors_mv(self, arrays, fun):
        m_sb, m_bs = arrays[:-2]
        wide, tall = [arr.shape for arr in arrays[:-2]]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_bs, m_sb))
        off_b, y_one = utn.make_off_by_one(m_sb, m_bs)

        tau = m_bs.shape[:-2] + tau_len(m_bs, fun)
        self.assertArrayShapesAre(fun(m_bs, v_b),
                                  (utn.drop(tall), utn.trnsp(tall), tau))
        tau = m_sb.shape[:-2] + tau_len(m_sb, fun)
        self.assertArrayShapesAre(fun(m_sb, v_s),
                                  (utn.drop(wide), utn.trnsp(wide), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(m_sb, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as Mv:
            fun(m_sb[off_b], m_bs[y_one])

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), qr_funcs)
    def test_lstsq_qr_flexible_signature_with_vectors_vm(self, arrays, fun):
        m_sb, m_bs = arrays[:-2]
        wide, tall = [arr.shape for arr in arrays[:-2]]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        hy.assume(hn.wide(m_sb))
        hy.assume(la.norm(v_s) > 0.)

        tau = m_sb.shape[:-2] + tau_len_vec(v_s, fun)
        self.assertArrayShapesAre(fun(v_s, m_sb),
                                  (utn.drop(wide), wide[:-1], tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(v_s, m_bs)

    @hy.given(hn.vectors, hn.vectors, qr_funcs)
    def test_lstsq_qr_flexible_signature_with_vectors_vv(self, v_s, v_b, fun):
        hy.assume(len(v_s) != len(v_b))
        hy.assume(la.norm(v_s) > 0.)

        tau = tau_len_vec(v_s, fun)
        self.assertArrayShapesAre(fun(v_s, v_s), ((), v_s.shape, tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(v_s, v_b)

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), rqr_funcs)
    def test_rlstsq_qr_flexible_signature_with_vectors_mv(self, arrays, fun):
        m_sb, m_bs = arrays[:-2]
        wide, tall = [arr.shape for arr in arrays[:-2]]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        hy.assume(hn.wide(m_sb))
        hy.assume(la.norm(v_b) > 0.)

        tau = m_sb.shape[:-2] + tau_len_vec(v_b, fun)
        self.assertArrayShapesAre(fun(m_sb, v_b),
                                  (wide[:-1], utn.drop(wide), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(m_bs, v_b)

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), rqr_funcs)
    def test_rlstsq_qr_flexible_signature_with_vectors_vm(self, arrays, fun):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb, m_bs))
        off_b, y_one = utn.make_off_by_one(m_sb, m_bs)

        tau = m_sb.shape[:-2] + tau_len(m_sb, fun)
        self.assertArrayShapesAre(fun(v_b, m_sb),
                                  (wide[:-1], utn.trnsp(wide), tau))
        tau = m_bs.shape[:-2] + tau_len(m_bs, fun)
        self.assertArrayShapesAre(fun(v_s, m_bs),
                                  (tall[:-1], utn.trnsp(tall), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(v_b, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as vM:
            fun(m_bs[y_one], m_sb[off_b])

    @hy.given(hn.vectors, hn.vectors, rqr_funcs)
    def test_rlstsq_qr_flexible_signature_with_vectors_vv(self, v_s, v_b, fun):
        hy.assume(len(v_s) != len(v_b))
        hy.assume(la.norm(v_s) > 0.)

        tau = tau_len_vec(v_s, fun)
        self.assertArrayShapesAre(fun(v_s, v_s), ((), v_s.shape, tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            fun(v_s, v_b)


class TestQRLstsqVectors(TestCaseNumpy):
    """Testing (r)qr_lstsq with vectors"""

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), qr_funcs)
    def test_qr_lstsq_flexible_signature_with_vectors_mv(self, arrays, fun):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_bs, m_sb))
        off_b, y_one = utn.make_off_by_one(m_sb, m_bs)

        _, x_f, tau = fun(m_bs, v_b)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), utn.drop(tall))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), tall[:-1])

        _, x_f, tau = fun(m_sb, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), utn.drop(wide))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), wide[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as Mv:
            gfl.qr_lstsq(x_f[off_b], tau[off_b], m_bs[y_one])

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), qr_funcs)
    def test_qr_lstsq_flexible_signature_with_vectors_vm(self, arrays, fun):
        m_sb, m_bs = hn.core_only(*arrays[:-2])
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape[-2:] for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        hy.assume(la.norm(v_s) > 0.)
        off_b, y_one = utn.make_off_by_one(m_sb, m_bs)

        _, x_f, tau = fun(v_s, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), utn.drop(wide))
        expect = utn.array_return_shape('(m,n),(n,p)->(m,p)', m_bs, m_sb)[:-1]
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), wide[:-2])

        m_sb, m_bs = arrays[:-2]
        wide, tall = [arr.shape for arr in arrays[:-2]]

        _, x_f, tau = fun(v_s, m_sb)
        x_f, tau = unbroadcast_factors(v_s, x_f, tau)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), utn.drop(wide))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), tall[:-1])

    @hy.given(hn.vectors, hn.vectors, qr_funcs)
    def test_qr_lstsq_flexible_signature_with_vectors_vv(self, v_s, v_b, fun):
        hy.assume(len(v_s) != len(v_b))
        hy.assume(la.norm(v_s) > 0.)

        _, x_f, tau = fun(v_s, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), ())
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_b, x_f, tau)

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), rqr_funcs)
    def test_rqr_lstsq_flexible_signature_with_vectors_mv(self, arrays, fun):
        m_sb, m_bs = hn.core_only(*arrays[:-2])
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape[-2:] for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        hy.assume(la.norm(v_s) > 0.)

        _, x_f, tau = fun(m_bs, v_s)
        expect = utn.array_return_shape('(m,n),(n,p)->(m,p)', m_bs, m_sb)[:-1]
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), expect)
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), tall[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), tall[:-2])

        m_sb, m_bs = arrays[:-2]
        wide, tall = [arr.shape for arr in arrays[:-2]]

        _, x_f, tau = fun(m_bs, v_s)
        x_f, tau = unbroadcast_factors(v_s, x_f, tau)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), utn.drop(wide))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), tall[:-1])

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), rqr_funcs)
    def test_rqr_lstsq_flexible_signature_with_vectors_vm(self, arrays, fun):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_bs, m_sb))
        off_b, y_one = utn.make_off_by_one(m_sb, m_bs)

        _, x_f, tau = fun(v_s, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), utn.drop(tall))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), tall[:-1])

        _, x_f, tau = fun(v_b, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), utn.drop(wide))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), wide[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_s, x_f, tau)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as Mv:
            gfl.rqr_lstsq(m_bs[y_one], x_f[off_b], tau[off_b])

    @hy.given(hn.vectors, hn.vectors, qr_funcs)
    def test_rqr_lstsq_flexible_signature_with_vectors_vv(self, v_s, v_b, fun):
        hy.assume(len(v_s) != len(v_b))
        hy.assume(la.norm(v_s) > 0.)

        _, x_f, tau = fun(v_s, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), ())
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_b, x_f, tau)


class TestLstsqVal(TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None))
    def test_lstsq_returns_expected_values(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        cond_bs = np.linalg.cond(m_bs).max()
        cond_sb = np.linalg.cond(m_sb).max()

        # overconstrained
        x_sb = gfl.lstsq(m_bs, m_bb)
        m_bst = la.dagger(m_bs)
        # with self.subTest(msg='lstsq(over)'):
        self.assertArrayAllClose(m_bst @ m_bs @ x_sb, m_bst @ m_bb,
                                 cond=cond_bs)
        x_bs = gfl.rlstsq(m_bb, m_sb)
        m_sbt = la.dagger(m_sb)
        # with self.subTest(msg='rlstsq(over)'):
        self.assertArrayAllClose(x_bs @ m_sb @ m_sbt, m_bb @ m_sbt,
                                 cond=cond_sb)
        # underconstrained
        x_bs = gfl.lstsq(m_sb, m_ss)
        # with self.subTest(msg='lstsq(under)'):
        self.assertArrayAllClose(m_sb @ x_bs, m_ss, cond=cond_sb)
        x_sb = gfl.rlstsq(m_ss, m_bs)
        # with self.subTest(msg='rlstsq(under)'):
        self.assertArrayAllClose(x_sb @ m_bs, m_ss, cond=cond_bs)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None), qr_funcs)
    def test_lstsq_qr_returns_expected_values_with_tall(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_bs))
        cond = np.linalg.cond(m_bs).max()

        # overconstrained
        x0_sb = gfl.lstsq(m_bs, m_bb)
        # overconstrained
        x_sb, x_f, tau = fun(m_bs, m_bb)
        # with self.subTest('lstsq_qr(over,' + suffix):
        self.assertArrayAllClose(x_sb, x0_sb, cond=cond)
        # overconstrained
        xx_sb = gfl.qr_lstsq(x_f, tau, m_bb)
        # with self.subTest('qr_lstsq(over,' + suffix):
        self.assertArrayAllClose(xx_sb, x0_sb, cond=cond)
        # underconstrained
        y_sb = gfl.rqr_lstsq(m_ss, x_f, tau)
        # with self.subTest('rqr_lstsq(under,' + suffix):
        self.assertArrayAllClose(y_sb @ m_bs, m_ss, cond=cond)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None), rqr_funcs)
    def test_rlstsq_qr_returns_expected_values_with_wide(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        cond = np.linalg.cond(m_sb).max()

        # overconstrained
        x0_bs = gfl.rlstsq(m_bb, m_sb)
        # overconstrained
        x_bs, x_f, tau = fun(m_bb, m_sb)
        # with self.subTest('rlstsq_qr(under,' + suffix):
        self.assertArrayAllClose(x_bs, x0_bs, cond=cond)
        # overconstrained
        xx_bs = gfl.rqr_lstsq(m_bb, x_f, tau)
        # with self.subTest('rqr_rlstsq(under,' + suffix):
        self.assertArrayAllClose(xx_bs, x0_bs, cond=cond)
        # underconstrained
        y_bs = gfl.qr_lstsq(x_f, tau, m_ss)
        # with self.subTest('qr_rlstsq(over,' + suffix):
        self.assertArrayAllClose(m_sb @ y_bs, m_ss, cond=cond)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None), qr_funcs)
    def test_lstsq_qr_returns_expected_values_with_wide(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_sb))
        cond = np.linalg.cond(m_sb).max()

        # underconstrained
        x0_bs = gfl.lstsq(m_sb, m_ss)
        # underconstrained
        x_bs, x_f, tau = fun(m_sb, m_ss)
        # with self.subTest('lstsq_qr(under,' + suffix):
        self.assertArrayAllClose(x_bs, x0_bs, cond=cond)
        # underconstrained
        xx_bs = gfl.qr_lstsq(x_f, tau, m_ss)
        # with self.subTest('qr_lstsq(under,' + suffix):
        self.assertArrayAllClose(xx_bs, x0_bs, cond=cond)
        # overconstrained
        y_bs = gfl.rqr_lstsq(m_bb, x_f, tau)
        m_sbt = la.dagger(m_sb)
        # with self.subTest('rqr_lstsq(over,' + suffix):
        self.assertArrayAllClose(y_bs @ m_sb @ m_sbt, m_bb @ m_sbt, cond=cond)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None), rqr_funcs)
    def test_rlstsq_qr_returns_expected_values_with_tall(self, arrays, fun):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_well_behaved(m_bs))
        cond = np.linalg.cond(m_bs).max()

        # underconstrained
        x0_sb = gfl.rlstsq(m_ss, m_bs)
        # underconstrained
        x_sb, x_f, tau = fun(m_ss, m_bs)
        # with self.subTest('rlstsq_qr(over,' + suffix):
        self.assertArrayAllClose(x_sb, x0_sb, cond=cond)
        # underconstrained
        xx_sb = gfl.rqr_lstsq(m_ss, x_f, tau)
        # with self.subTest('rqr_rlstsq(over,' + suffix):
        self.assertArrayAllClose(xx_sb, x0_sb, cond=cond)
        # overconstrained
        y_sb = gfl.qr_lstsq(x_f, tau, m_bb)
        m_bst = la.dagger(m_bs)
        # with self.subTest('qr_rlstsq(under,' + suffix):
        self.assertArrayAllClose(m_bst @ m_bs @ y_sb, m_bst @ m_bb, cond=cond)

    @expectedFailure
    @errstate
    @hy.given(hn.constant('(a,b)', None, min_side=2))
    def test_lstsq_qr_raises_with_low_rank(self, ones_bs):
        hy.assume(hn.tall(ones_bs))

        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lstsq_qrn(ones_bs, ones_bs[..., :2])


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
