# -*- coding: utf-8 -*-
"""Test qr & lstsq families of gufuncs
"""
import unittest
import hypothesis as hy
import hypothesis.extra.numpy as hyn
import numpy as np
import numpy_linalg.gufuncs._gufuncs_qr_lstsq as gfl
import numpy_linalg.gufuncs._gufuncs_blas as gfb
from numpy_linalg import transpose, dagger, row, col, scalar
from numpy_linalg.gufuncs import array_return_shape
if 'tests.' in __name__:
    from .test_gufunc import utn, hn, main, vectors, drop, make_bad_broadcast, make_off_by_one
    from .test_linalg import trnsp
else:
    # pylint: disable=import-error
    from test_gufunc import utn, hn, main, vectors, drop, make_bad_broadcast, make_off_by_one
    from test_linalg import trnsp
# pylint: disable=missing-function-docstring
# =============================================================================
errstate = np.errstate(invalid='raise')
hy.settings.register_profile("debug",
                             suppress_health_check=(hy.HealthCheck.too_slow,))
hy.settings.load_profile('debug')
qr_funcs = hy.strategies.sampled_from([gfl.lstsq_qrm, gfl.lstsq_qrn])
rqr_funcs = hy.strategies.sampled_from([gfl.rlstsq_qrm, gfl.rlstsq_qrn])
# =============================================================================
__all__ = ['TestLstsqShape', 'TestLstsqVectors', 'TestLstsqVal']
# =============================================================================


def tau_len(array: np.ndarray, func: np.ufunc) -> int:
    cases = {'l': {'m': array.shape[-2], 'n': array.shape[-1]},
             'r': {'n': array.shape[-2], 'm': array.shape[-1]}}
    return (cases[func.__name__[0]][func.__name__[-1]],)


def tau_len_vec(vec: np.ndarray, func: np.ufunc) -> int:
    cases = {'l': {'m': (len(vec),), 'n': ()},
             'r': {'n': (), 'm': (len(vec),)}}
    return cases[func.__name__[0]][func.__name__[-1]]


# =============================================================================
# Test lstsq
# =============================================================================


class TestLstsqShape(utn.TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @errstate
    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_lstsq_returns_expected_shape(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))

        # with self.subTest('overconstrained'):
        expect = array_return_shape('(m,n),(m,p)->(n,p)', m_bs, m_bb)
        self.assertArrayShape(gfl.lstsq(m_bs, m_bb), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.lstsq(*make_bad_broadcast(m_bs, m_bb))
        # with self.subTest('underconstrained'):
        expect = array_return_shape('(m,n),(m,p)->(n,p)', m_sb, m_ss)
        self.assertArrayShape(gfl.lstsq(m_sb, m_ss), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.lstsq(*make_bad_broadcast(m_sb, m_ss))

    @errstate
    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_rlstsq_returns_expected_shape(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))

        # with self.subTest('underconstrained'):
        expect = array_return_shape('(m,n),(p,n)->(m,p)', m_ss, m_bs)
        self.assertArrayShape(gfl.rlstsq(m_ss, m_bs), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rlstsq(*make_bad_broadcast(m_ss, transpose(m_sb)))
        # with self.subTest('overconstrained'):
        expect = array_return_shape('(m,n),(p,n)->(m,p)', m_bb, m_sb)
        self.assertArrayShape(gfl.rlstsq(m_bb, m_sb), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rlstsq(*make_bad_broadcast(m_bb, transpose(m_bs)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), qr_funcs)
    def test_lstsq_qr_returns_expected_shape_tall(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))

        expect = array_return_shape('(m,n),(m,p)->(n,p),(n,m),()', m_bs, m_bb)
        tau = expect[2] + tau_len(m_bs, func)
        self.assertArrayShapesAre(func(m_bs, m_bb), expect[:2] + (tau,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(*make_bad_broadcast(m_bs, m_bb))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), qr_funcs)
    def test_qr_lstsq_returns_expected_shape_tall(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))

        _, x_f, tau = func(m_bs, m_bb)
        expect = array_return_shape('(n,m),(m,p)->(n,p)', x_f, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), expect)
        expect = array_return_shape('(m,n),(n,p)->(m,p)', m_ss, x_f)
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, *make_bad_broadcast(tau, m_bb, (1, 2)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), qr_funcs)
    def test_lstsq_qr_returns_expected_shape_wide(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(m_sb @ transpose(m_sb)))

        expect = array_return_shape('(m,n),(m,p)->(n,p),(n,m),()', m_sb, m_ss)
        tau = expect[2] + tau_len(m_sb, func)
        self.assertArrayShapesAre(func(m_sb, m_ss), expect[:2] + (tau,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(*make_bad_broadcast(m_sb, m_ss))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), qr_funcs)
    def test_qr_lstsq_returns_expected_shape_wide(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(m_sb @ transpose(m_sb)))

        _, x_f, tau = func(m_sb, m_ss)
        expect = array_return_shape('(n,m),(m,p)->(n,p)', x_f, m_ss)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_ss), expect)
        expect = array_return_shape('(m,n),(n,p)->(m,p)', m_bb, x_f)
        self.assertArrayShape(gfl.rqr_lstsq(m_bb, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_ss, x_f, tau)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, *make_bad_broadcast(tau, m_ss, (1, 2)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), rqr_funcs)
    def test_rlstsq_qr_returns_expected_shape_tall(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))

        expect = array_return_shape('(m,n),(p,n)->(m,p),(n,p),()', m_ss, m_bs)
        tau = expect[2] + tau_len(m_bs, func)
        self.assertArrayShapesAre(func(m_ss, m_bs), expect[:2] + (tau,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(*make_bad_broadcast(m_ss, transpose(m_sb)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), rqr_funcs)
    def test_rqr_lstsq_returns_expected_shape_tall(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.tall(m_bs))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))

        _, x_f, tau = func(m_ss, m_bs)
        expect = array_return_shape('(n,m),(m,p)->(n,p)', x_f, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), expect)
        expect = array_return_shape('(m,n),(n,p)->(m,p)', m_ss, x_f)
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, *make_bad_broadcast(tau, m_bb, (1, 2)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), rqr_funcs)
    def test_rlstsq_qr_returns_expected_shape_wide(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(m_sb @ transpose(m_sb)))

        expect = array_return_shape('(m,n),(p,n)->(m,p),(n,p),()', m_bb, m_sb)
        tau = expect[2] + tau_len(m_sb, func)
        self.assertArrayShapesAre(func(m_bb, m_sb), expect[:2] + (tau,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(*make_bad_broadcast(m_bb, transpose(m_bs)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'), rqr_funcs)
    def test_rqr_lstsq_returns_expected_shape_wide(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(m_sb @ transpose(m_sb)))

        _, x_f, tau = func(m_bb, m_sb)
        expect = array_return_shape('(n,m),(m,p)->(n,p)', x_f, m_ss)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_ss), expect)
        expect = array_return_shape('(m,n),(n,p)->(m,p)', m_bb, x_f)
        self.assertArrayShape(gfl.rqr_lstsq(m_bb, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_bs, x_f, tau)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, *make_bad_broadcast(tau, m_ss, (1, 2)))


class TestLstsqVectors(utn.TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'))
    def test_lstsq_flexible_signature_with_vectors_mv(self, arrays):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        off_b, y_one = make_off_by_one(m_sb, m_bs)

        self.assertArrayShape(gfl.lstsq(m_bs, v_b), drop(tall))
        self.assertArrayShape(gfl.lstsq(m_sb, v_s), drop(wide))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_sb, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as Mv:
            gfl.lstsq(m_sb[off_b], m_bs[y_one])

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'))
    def test_lstsq_flexible_signature_with_vectors_vm(self, arrays):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.nonsquare(m_sb))

        self.assertArrayShape(gfl.lstsq(v_s, m_sb), drop(wide))
        self.assertArrayShape(gfl.lstsq(v_b, m_bs), drop(tall))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(v_s, m_bs)

    @hy.given(vectors, vectors)
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
        off_b, y_one = make_off_by_one(m_sb, m_bs)

        self.assertArrayShape(gfl.rlstsq(v_b, m_sb), wide[:-1])
        self.assertArrayShape(gfl.rlstsq(v_s, m_bs), tall[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(v_b, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as vM: (3)(7)\(3)(7,7)
            gfl.rlstsq(m_bs[y_one], m_sb[off_b])

    @hy.given(vectors, vectors)
    def test_rlstsq_flexible_signature_with_vectors_vv(self, v_s, v_b):
        hy.assume(len(v_s) != len(v_b))

        self.assertArrayShape(gfl.rlstsq(v_s, v_s), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(v_s, v_b)


class TestLstsqQRVectors(utn.TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), qr_funcs)
    def test_lstsq_qr_flexible_signature_with_vectors_mv(self, arrays, func):
        m_sb, m_bs = arrays[:-2]
        wide, tall = [arr.shape for arr in arrays[:-2]]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))
        hy.assume(hn.all_non_singular(m_sb @ transpose(m_sb)))
        off_b, y_one = make_off_by_one(m_sb, m_bs)

        tau = m_bs.shape[:-2] + tau_len(m_bs, func)
        self.assertArrayShapesAre(func(m_bs, v_b),
                                    (drop(tall), trnsp(tall), tau))
        tau = m_sb.shape[:-2] + tau_len(m_sb, func)
        self.assertArrayShapesAre(func(m_sb, v_s),
                                    (drop(wide), trnsp(wide), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_sb, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as Mv:
            func(m_sb[off_b], m_bs[y_one])

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), qr_funcs)
    def test_lstsq_qr_flexible_signature_with_vectors_vm(self, arrays, func):
        m_sb, m_bs = arrays[:-2]
        wide, tall = [arr.shape for arr in arrays[:-2]]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        hy.assume(hn.wide(m_sb))

        tau = m_sb.shape[:-2] + tau_len_vec(v_s, func)
        self.assertArrayShapesAre(func(v_s, m_sb), (drop(wide), wide[:-1], tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_s, m_bs)

    @hy.given(vectors, vectors, qr_funcs)
    def test_lstsq_qr_flexible_signature_with_vectors_vv(self, v_s, v_b, func):
        hy.assume(len(v_s) != len(v_b))

        tau = tau_len_vec(v_s, func)
        self.assertArrayShapesAre(func(v_s, v_s), ((), v_s.shape, tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_s, v_b)

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), rqr_funcs)
    def test_rlstsq_qr_flexible_signature_with_vectors_mv(self, arrays, func):
        m_sb, m_bs = arrays[:-2]
        wide, tall = [arr.shape for arr in arrays[:-2]]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        hy.assume(hn.wide(m_sb))

        tau = m_sb.shape[:-2] + tau_len_vec(v_b, func)
        self.assertArrayShapesAre(func(m_sb, v_b), (wide[:-1], drop(wide), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_bs, v_b)

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), rqr_funcs)
    def test_rlstsq_qr_flexible_signature_with_vectors_vm(self, arrays, func):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(m_sb @ transpose(m_sb)))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))
        off_b, y_one = make_off_by_one(m_sb, m_bs)

        tau = m_sb.shape[:-2] + tau_len(m_sb, func)
        self.assertArrayShapesAre(func(v_b, m_sb),
                                  (wide[:-1], trnsp(wide), tau))
        tau = m_bs.shape[:-2] + tau_len(m_bs, func)
        self.assertArrayShapesAre(func(v_s, m_bs),
                                  (tall[:-1], trnsp(tall), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_b, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as vM:
            func(m_bs[y_one], m_sb[off_b])

    @hy.given(vectors, vectors, rqr_funcs)
    def test_rlstsq_qr_flexible_signature_with_vectors_vv(self, v_s, v_b, func):
        hy.assume(len(v_s) != len(v_b))

        tau = tau_len_vec(v_s, func)
        self.assertArrayShapesAre(func(v_s, v_s), ((), v_s.shape, tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_s, v_b)


class TestQRLstsqVectors(utn.TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), qr_funcs)
    def test_qr_lstsq_flexible_signature_with_vectors_mv(self, arrays, func):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))
        hy.assume(hn.all_non_singular(m_sb @ transpose(m_sb)))
        off_b, y_one = make_off_by_one(m_sb, m_bs)

        _, x_f, tau = func(m_bs, v_b)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), drop(tall))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), tall[:-1])

        _, x_f, tau = func(m_sb, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), drop(wide))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), wide[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as Mv:
            gfl.qr_lstsq(x_f[off_b], tau[off_b], m_bs[y_one])


    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), qr_funcs)
    def test_qr_lstsq_flexible_signature_with_vectors_vm(self, arrays, func):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        off_b, y_one = make_off_by_one(m_sb, m_bs)


        _, x_f, tau = func(v_s, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), drop(wide))
        expect = array_return_shape('(m,n),(n,p)->(m,p)', m_bs, m_sb)[:-1]
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), wide[:-2])

    @hy.given(vectors, vectors, qr_funcs)
    def test_qr_lstsq_flexible_signature_with_vectors_vv(self, v_s, v_b, func):
        v_s, v_b = hn.core_only(*arrays, dims=1)
        hy.assume(len(v_s) != len(v_b))

        _, x_f, tau = func(v_s, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), ())
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_b, x_f, tau)

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), rqr_funcs)
    def test_rqr_lstsq_flexible_signature_with_vectors_mv(self, arrays, func):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_bs, v_s)
        expect = array_return_shape('(m,n),(n,p)->(m,p)', m_bs, m_sb)[:-1]
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), expect)
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), tall[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), tall[:-2])

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'), rqr_funcs)
    def test_rqr_lstsq_flexible_signature_with_vectors_vm(self, arrays, func):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))
        hy.assume(hn.all_non_singular(m_sb @ transpose(m_sb)))
        off_b, y_one = make_off_by_one(m_sb, m_bs)

        _, x_f, tau = func(v_s, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), drop(tall))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), tall[:-1])

        _, x_f, tau = func(v_b, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), drop(wide))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), wide[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_s, x_f, tau)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as Mv: (4,1)(3,7)\(7)(3)
            gfl.rqr_lstsq(m_bs[y_one], x_f[off_b], tau[off_b])

    @hy.given(vectors, vectors, qr_funcs)
    def test_rqr_lstsq_flexible_signature_with_vectors_vv(self, v_s, v_b, func):
        v_s, v_b = hn.core_only(*arrays, dims=1)
        hy.assume(len(v_s) != len(v_b))

        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), ())
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_b, x_f, tau)


class TestLstsqVal(utn.TestCaseNumpy):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None))
    def test_lstsq_returns_expected_values(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))

        # overconstrained
        x_sb = gfl.lstsq(m_bs, m_bb)
        m_bst = dagger(m_bs)
        # with self.subTest(msg='lstsq(over)'):
        self.assertArrayAllClose(m_bst @ m_bs @ x_sb, m_bst @ m_bb)
        x_bs = gfl.rlstsq(m_bb, m_sb)
        m_sbt = dagger(m_sb)
        # with self.subTest(msg='rlstsq(over)'):
        self.assertArrayAllClose(x_bs @ m_sb @ m_sbt, m_bb @ m_sbt)
        # underconstrained
        x_bs = gfl.lstsq(m_sb, m_ss)
        # with self.subTest(msg='lstsq(under)'):
        self.assertArrayAllClose(m_sb @ x_bs, m_ss)
        x_sb = gfl.rlstsq(m_ss, m_bs)
        # with self.subTest(msg='rlstsq(under)'):
        self.assertArrayAllClose(x_sb @ m_bs, m_ss)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None), qr_funcs)
    def test_lstsq_qr_returns_expected_values_with_tall(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))

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

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None), rqr_funcs)
    def test_rlstsq_qr_returns_expected_values_with_wide(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(m_sb @ transpose(m_sb)))

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

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None), qr_funcs)
    def test_lstsq_qr_returns_expected_values_with_wide(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
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
        m_sbt = dagger(m_sb)
        # with self.subTest('rqr_lstsq(over,' + suffix):
        self.assertArrayAllClose(y_bs @ m_sb @ m_sbt, m_bb @ m_sbt)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None), rqr_funcs)
    def test_rlstsq_qr_returns_expected_values_with_tall(self, arrays, func):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.wide(m_sb))
        hy.assume(hn.all_non_singular(transpose(m_bs) @ m_bs))

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
        m_bst = dagger(m_bs)
        # with self.subTest('qr_rlstsq(under,' + suffix):
        self.assertArrayAllClose(m_bst @ m_bs @ y_sb, m_bst @ m_bb)

    @unittest.expectedFailure
    @errstate
    @hy.given(hn.constant('(a,b)', None))
    def test_lstsq_qr_raises_with_low_rank(self, arrays):
        hy.assume(hn.tall(ones_bs))
        hy.assume(ones_bs.shape[-1] > 1)

        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lstsq_qrn(ones_bs, ones_bs[..., :2])


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
