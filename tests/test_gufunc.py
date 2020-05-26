# -*- coding: utf-8 -*-
"""Test C-loop and BLAS ufuncs
"""
import hypothesis as hy
import numpy as np
import numpy_linalg.gufuncs._gufuncs_cloop as gfc
import numpy_linalg.gufuncs._gufuncs_blas as gfb
import numpy_linalg.testing.unittest_numpy as utn
import numpy_linalg.testing.hypothesis_numpy as hn
from numpy_linalg.testing import main, TestCaseNumpy
# =============================================================================
# pylint: disable=missing-function-docstring
errstate = np.errstate(invalid='raise')
hy.settings.register_profile("slow",
                             suppress_health_check=(hy.HealthCheck.too_slow,))
hy.settings.load_profile('slow')
np.set_printoptions(precision=2, threshold=10, edgeitems=2)
# =============================================================================
__all__ = ['TestBlas', 'TestBlasVectors', 'TestCloop']
# =============================================================================
# =============================================================================
# Test BLAS ufuncs
# =============================================================================


class TestBlas(TestCaseNumpy):
    """Testing norm, matmul and rmatmul"""

    def setUp(self):
        super().setUp()
        self.gfm = gfb

    @hy.given(hn.matrices_b)
    def test_norm_returns_expected_shapes(self, m_bs):
        v_s = m_bs[(0,) * (m_bs.ndim - 1)]
        tall = m_bs.shape

        self.assertArrayShape(self.gfm.norm(m_bs), tall[:-1])
        self.assertArrayShape(self.gfm.norm(m_bs, axis=-2), utn.drop(tall))
        self.assertArrayShape(self.gfm.norm(v_s, keepdims=True), (1,))
        self.assertArrayShape(self.gfm.norm(v_s), ())

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_norm_returns_expected_values(self, m_bs: np.ndarray):
        nout = np.empty_like(m_bs[..., 0].real)
        nrms = self.gfm.norm(m_bs, out=nout)
        norms = np.sqrt((np.abs(m_bs)**2).sum(-1))
        self.assertArrayAllClose(nrms, norms)
        self.assertArrayAllClose(nout, norms)

    @hy.given(hn.broadcastable('(a,b),(b,c)', 'd'))
    def test_matmul_returns_expected_shapes(self, arrays):
        m_sb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.nonsquare(m_bs))

        expect = utn.array_return_shape('(a,b),(b,c)->(a,c)', m_sb, m_bs)
        self.assertArrayShape(self.gfm.matmul(m_sb, m_bs), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gfm.matmul(m_bs, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gfm.matmul(*utn.make_bad_broadcast(m_sb, m_bs))

    @hy.given(hn.broadcastable('(a,b),(b,c)', None))
    def test_matmul_returns_expected_values(self, arrays):
        m_sb, m_bs = arrays
        expect = utn.array_return_shape('(a,b),(b,c)->(a,c)', m_sb, m_bs)

        pout = np.empty(expect, m_sb.dtype)
        pres = self.gfm.matmul(m_sb, m_bs, out=pout)
        prod = np.matmul(m_sb, m_bs)
        self.assertArrayAllClose(pres, prod)
        self.assertArrayAllClose(pout, prod)

    @hy.given(hn.broadcastable('(a,b),(b,c)', 'd'))
    def test_rmatmul_returns_expected_shapes(self, arrays):
        m_sb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.nonsquare(m_bs))

        expect = utn.array_return_shape('(a,b),(b,c)->(a,c)', m_sb, m_bs)
        self.assertArrayShape(self.gfm.rmatmul(m_bs, m_sb), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gfm.rmatmul(m_bs, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gfm.rmatmul(*utn.make_bad_broadcast(m_bs, m_sb))

    @hy.given(hn.broadcastable('(a,b),(b,c)', None))
    def test_rmatmul_returns_expected_values(self, arrays):
        m_sb, m_bs = arrays
        expect = utn.array_return_shape('(a,b),(b,c)->(a,c)', m_sb, m_bs)

        pout = np.empty(expect, m_sb.dtype)
        pres = self.gfm.rmatmul(m_bs, m_sb, out=pout)
        prod = np.matmul(m_sb, m_bs)
        self.assertArrayAllClose(pres, prod)
        self.assertArrayAllClose(pout, prod)


class TestBlasVectors(TestCaseNumpy):
    """Testing matmul and rmatmul"""

    def setUp(self):
        super().setUp()
        self.gfm = gfb

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_matmul_flexible_signature_with_vectors(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.nonsquare(m_sb))
        off_b, y_one = utn.make_off_by_one(m_sb, m_sb)

        # with self.subTest('matrix-vector'):
        self.assertArrayShape(self.gfm.matmul(m_sb, v_b), wide[:-1])
        self.assertArrayShape(self.gfm.matmul(m_bs, v_s), tall[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gfm.matmul(m_sb, v_s)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as Mv:
            self.gfm.matmul(m_sb[off_b], m_sb[y_one])
        # with self.subTest('vector-matrix'):
        self.assertArrayShape(self.gfm.matmul(v_s, m_ss), smol[:-1])
        self.assertArrayShape(self.gfm.matmul(v_b, m_bb), big[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gfm.matmul(v_b, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as vM:
            self.gfm.matmul(m_sb[y_one], m_sb[off_b])
        # with self.subTest('vector-vector'):
        self.assertArrayShape(self.gfm.matmul(v_s, v_s), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gfm.matmul(v_s, v_b)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_rmatmul_flexible_signature_with_vectors(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.nonsquare(m_sb))
        off_b, y_one = utn.make_off_by_one(m_sb, m_sb)

        # with self.subTest('matrix-vector'):
        self.assertArrayShape(self.gfm.rmatmul(v_s, m_bs), tall[:-1])
        self.assertArrayShape(self.gfm.rmatmul(v_b, m_sb), wide[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gfm.rmatmul(v_b, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as Mv:
            self.gfm.rmatmul(m_sb[y_one], m_sb[off_b])
        # w\ith self.subTest('vector-matrix'):
        self.assertArrayShape(self.gfm.rmatmul(m_ss, v_s), smol[:-1])
        self.assertArrayShape(self.gfm.rmatmul(m_bb, v_b), big[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gfm.rmatmul(m_bs, v_s)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succeed/broadcast error if interpreted as vM:
            self.gfm.rmatmul(m_sb[off_b], m_sb[y_one])
        # with self.subTest('vector-vector'):
        self.assertArrayShape(self.gfm.rmatmul(v_b, v_b), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gfm.rmatmul(v_b, v_s)


# =============================================================================
# Test cloop ufuncs
# =============================================================================


class TestCloop(TestBlas):
    """Testing norm, matmul, rmatmul and rtrue_tdivide
    """

    def setUp(self):
        super().setUp()
        self.gfm = gfc

    @hy.given(hn.broadcastable('(a,b),(a,b)', 'd'))
    def test_rtrue_divide_returns_expected_shapes(self, arrays):
        a_bs, m_bs = arrays

        a_bs[np.abs(a_bs) < 1e-5] += 1.
        expect = utn.array_return_shape('(),()->()', a_bs, m_bs)
        self.assertArrayShape(self.gfm.rtrue_divide(a_bs, m_bs), expect)
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gfm.rtrue_divide(*utn.make_bad_broadcast(m_bs, a_bs))

    @hy.given(hn.broadcastable('(a,b),(a,b)', None))
    def test_rtrue_divide_returns_expected_values(self, arrays):
        a_bs, m_bs = arrays

        expect = utn.array_return_shape('(),()->()', a_bs, m_bs)
        zout = np.empty(expect, m_bs.dtype)
        a_bs[np.abs(a_bs) < 1e-5] += 1.
        zres = self.gfm.rtrue_divide(a_bs, m_bs, out=zout)
        zzz = m_bs / a_bs
        self.assertArrayAllClose(zres, zzz)
        self.assertArrayAllClose(zout, zzz)


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
