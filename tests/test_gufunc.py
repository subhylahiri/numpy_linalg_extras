# -*- coding: utf-8 -*-
"""Test C-loop and BLAS ufuncs + qr
"""
import hypothesis as hy
import hypothesis.extra.numpy as hyn
import numpy as np
import numpy_linalg.gufuncs._gufuncs_cloop as gfc
import numpy_linalg.gufuncs._gufuncs_blas as gfb
from numpy_linalg.gufuncs import array_return_shape
if 'tests.' in __name__:
    from . import unittest_numpy as utn
    from . import hypothesis_numpy as hn
    from .unittest_tweaks import main
else:
    # pylint: disable=import-error
    import unittest_numpy as utn
    import hypothesis_numpy as hn
    from unittest_tweaks import main
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
# =============================================================================
errstate = np.errstate(invalid='raise')
hy.settings.register_profile("debug",
                             suppress_health_check=(hy.HealthCheck.too_slow,))
hy.settings.load_profile('debug')
vectors = hyn.arrays(dtype=np.float64,
                     shape=hyn.array_shapes(min_dims=1, max_dims=1),
                     elements=hn.real_numbers())
matrices = hyn.arrays(dtype=np.float64,
                      shape=hyn.array_shapes(min_dims=2),
                      elements=hn.real_numbers())
# =============================================================================
__all__ = ['TestBlas', 'TestBlasVectors', 'TestCloop']
# =============================================================================


def drop(shape, axis=-2):
    """Shape -> shape with one axis dropped"""
    return shape[:axis] + shape[axis+1:]


def make_bad_broadcast(left, right, cores=(2, 2)):
    """Stack arrays so they no longer broadcast"""
    axis = (left.ndim - cores[0]) - (right.ndim - cores[1])
    new_left = np.stack((left,) * 3)[np.s_[:,] + (None,) * (-axis)]
    new_right = np.stack((right,) * 2)[np.s_[:,] + (None,) * axis]
    return new_left, new_right


def make_off_by_one(matrices, vectors):
    """Arrange so that matrices.ndim = vectors.ndim + 1"""
    off_by_one = matrices.ndim - vectors.ndim - 1
    return (None,)*(-off_by_one), (None,)*off_by_one


# =============================================================================
# Test BLAS ufuncs
# =============================================================================


class TestBlas(utn.TestCaseNumpy):
    """Testing norm, matmul and rmatmul"""

    def setUp(self):
        self.gf = gfb
        super().setUp()

    @hy.given(matrices)
    def test_norm_returns_expected_shapes(self, m_bs):
        v_s = m_bs[(0,) * (m_bs.ndim - 1)]
        tall = m_bs.shape

        self.assertArrayShape(self.gf.norm(m_bs), tall[:-1])
        self.assertArrayShape(self.gf.norm(m_bs, axis=-2), drop(tall))
        self.assertArrayShape(self.gf.norm(v_s, keepdims=True), (1,))
        self.assertArrayShape(self.gf.norm(v_s), ())

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_norm_returns_expected_values(self, m_bs: np.ndarray):
        nout = np.empty_like(m_bs[..., 0].real)
        nrms = self.gf.norm(m_bs, out=nout)
        norms = np.sqrt((np.abs(m_bs)**2).sum(-1))
        self.assertArrayAllClose(nrms, norms)
        self.assertArrayAllClose(nout, norms)

    @hy.given(hn.broadcastable('(a,b),(b,c)', 'd'))
    def test_matmul_returns_expected_shapes(self, arrays):
        m_sb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.nonsquare(m_bs))

        expect = array_return_shape('(a,b),(b,c)->(a,c)', m_sb, m_bs)
        self.assertArrayShape(self.gf.matmul(m_sb, m_bs), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(m_bs, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gf.matmul(*make_bad_broadcast(m_sb, m_bs))

    @hy.given(hn.broadcastable('(a,b),(b,c)', None))
    def test_matmul_returns_expected_values(self, arrays):
        m_sb, m_bs = arrays
        expect = array_return_shape('(a,b),(b,c)->(a,c)', m_sb, m_bs)

        pout = np.empty(expect, m_sb.dtype)
        pres = self.gf.matmul(m_sb, m_bs, out=pout)
        prod = np.matmul(m_sb, m_bs)
        self.assertArrayAllClose(pres, prod)
        self.assertArrayAllClose(pout, prod)

    @hy.given(hn.broadcastable('(a,b),(b,c)', 'd'))
    def test_rmatmul_returns_expected_shapes(self, arrays):
        m_sb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.nonsquare(m_bs))

        expect = array_return_shape('(a,b),(b,c)->(a,c)', m_sb, m_bs)
        self.assertArrayShape(self.gf.rmatmul(m_bs, m_sb), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(m_bs, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gf.rmatmul(*make_bad_broadcast(m_bs, m_sb))

    @hy.given(hn.broadcastable('(a,b),(b,c)', None))
    def test_rmatmul_returns_expected_values(self, arrays):
        m_sb, m_bs = arrays
        expect = array_return_shape('(a,b),(b,c)->(a,c)', m_sb, m_bs)

        pout = np.empty(expect, m_sb.dtype)
        pres = self.gf.rmatmul(m_bs, m_sb, out=pout)
        prod = np.matmul(m_sb, m_bs)
        self.assertArrayAllClose(pres, prod)
        self.assertArrayAllClose(pout, prod)


class TestBlasVectors(utn.TestCaseNumpy):
    """Testing matmul and rmatmul"""

    def setUp(self):
        self.gf = gfb
        self.sctype = ['i']
        super().setUp()

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_matmul_flexible_signature_with_vectors(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.nonsquare(m_sb))
        off_b, y_one = make_off_by_one(m_sb, m_sb)

        # with self.subTest('matrix-vector'):
        self.assertArrayShape(self.gf.matmul(m_sb, v_b), wide[:-1])
        self.assertArrayShape(self.gf.matmul(m_bs, v_s), tall[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(m_sb, v_s)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as Mv:
            self.gf.matmul(m_sb[off_b], m_sb[y_one])
        # with self.subTest('vector-matrix'):
        self.assertArrayShape(self.gf.matmul(v_s, m_ss), smol[:-1])
        self.assertArrayShape(self.gf.matmul(v_b, m_bb), big[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(v_b, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as vM:
            self.gf.matmul(m_sb[y_one], m_sb[off_b])
        # with self.subTest('vector-vector'):
        self.assertArrayShape(self.gf.matmul(v_s, v_s), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(v_s, v_b)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_rmatmul_flexible_signature_with_vectors(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.nonsquare(m_sb))
        off_b, y_one = make_off_by_one(m_sb, m_sb)

        # with self.subTest('matrix-vector'):
        self.assertArrayShape(self.gf.rmatmul(v_s, m_bs), tall[:-1])
        self.assertArrayShape(self.gf.rmatmul(v_b, m_sb), wide[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(v_b, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as Mv:
            self.gf.rmatmul(m_sb[y_one], m_sb[off_b])
        # w\ith self.subTest('vector-matrix'):
        self.assertArrayShape(self.gf.rmatmul(m_ss, v_s), smol[:-1])
        self.assertArrayShape(self.gf.rmatmul(m_bb, v_b), big[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(m_bs, v_s)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as vM:
            self.gf.rmatmul(m_sb[off_b], m_sb[y_one])
        # with self.subTest('vector-vector'):
        self.assertArrayShape(self.gf.rmatmul(v_b, v_b), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(v_b, v_s)


# =============================================================================
# Test cloop ufuncs
# =============================================================================


class TestCloop(TestBlas):
    """Testing norm, matmul, rmatmul and rtrue_tdivide
    """

    def setUp(self):
        super().setUp()
        self.gf = gfc

    @hy.given(hn.broadcastable('(a,b),(a,b)', 'd'))
    def test_rtrue_divide_returns_expected_shapes(self, arrays):
        a_bs, m_bs = arrays

        expect = array_return_shape('(),()->()', a_bs, m_bs)
        self.assertArrayShape(self.gf.rtrue_divide(a_bs, m_bs), expect)
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gf.rtrue_divide(*make_bad_broadcast(m_bs, a_bs))

    @hy.given(hn.broadcastable('(a,b),(a,b)', None))
    def test_rtrue_divide_returns_expected_values(self, arrays):
        a_bs, m_bs = arrays

        expect = array_return_shape('(),()->()', a_bs, m_bs)
        zout = np.empty(expect, m_bs.dtype)
        a_bs[np.abs(a_bs) < 1e-5] += 1.
        zres = self.gf.rtrue_divide(a_bs, m_bs, out=zout)
        zzz = m_bs / a_bs
        self.assertArrayAllClose(zres, zzz)
        self.assertArrayAllClose(zout, zzz)


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
