# -*- coding: utf-8 -*-
"""Test solve & lu families of gufuncs
"""
from unittest import expectedFailure
import hypothesis as hy
import numpy as np
import numpy_linalg as la
import numpy_linalg.gufuncs._gufuncs_lu_solve as gfl
import numpy_linalg.testing.unittest_numpy as utn
import numpy_linalg.testing.hypothesis_numpy as hn
from numpy_linalg.testing import main, TestCaseNumpy
if 'tests.' in __name__:
    from .test_gufunc import make_bad_broadcast, make_off_by_one
else:
    from test_gufunc import make_bad_broadcast, make_off_by_one
# =============================================================================
# pylint: disable=missing-function-docstring
errstate = np.errstate(invalid='raise')
hy.settings.register_profile("slow",
                             suppress_health_check=(hy.HealthCheck.too_slow,))
hy.settings.load_profile('slow')
# =============================================================================
__all__ = ['TestLU', 'TestSolveShape', 'TestSolveVectors', 'TestSolveVal']
# =============================================================================
# Test LU
# =============================================================================


class TestLU(TestCaseNumpy):
    """Testing LU decomposition"""

    @hy.given(hn.broadcastable('(a,b),(b,b),(b,a)', 'd'))
    def test_lu_basic_returns_expected_shapes(self, arrays):
        m_sb, m_bb, m_bs = arrays
        wide, big, tall = [arr.shape for arr in arrays]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg="square"):
        self.assertArrayShapesAre(gfl.lu_m(m_bb), (big, big, big[:-1]))
        self.assertArrayShapesAre(gfl.lu_n(m_bb), (big, big, big[:-1]))
        # with self.subTest(msg="wide"):
        self.assertArrayShapesAre(gfl.lu_m(m_sb),
                                  (utn.chop(wide), wide, wide[:-1]))
        self.assertArrayShapesAre(gfl.lu_n(m_sb),
                                  (wide, utn.grow(wide), utn.drop(wide)))
        # with self.subTest(msg="tall"):
        self.assertArrayShapesAre(gfl.lu_m(m_bs),
                                  (utn.grow(tall), tall, tall[:-1]))
        self.assertArrayShapesAre(gfl.lu_n(m_bs),
                                  (tall, utn.chop(tall), utn.drop(tall)))

    @hy.given(hn.broadcastable('(a,b),(b,b),(b,a)', 'd'))
    def test_lu_raw_returns_expected_shapes(self, arrays):
        m_sb, m_bb, m_bs = arrays
        wide, big, tall = [arr.shape for arr in arrays]
        hy.assume(hn.wide(m_sb))

        # with self.subTest(msg="square"):
        self.assertArrayShapesAre(gfl.lu_rawm(m_bb), (big, big[:-1]))
        # with self.subTest(msg="wide"):
        self.assertArrayShapesAre(gfl.lu_rawm(m_sb),
                                  (utn.trnsp(wide), wide[:-1]))
        # with self.subTest(msg="tall"):
        self.assertArrayShapesAre(gfl.lu_rawn(m_bs),
                                  (utn.trnsp(tall), utn.drop(tall)))

    @hy.given(hn.broadcastable('(a,a)', None))
    def test_lu_basic_returns_expected_values_square(self, m_bb):
        big = m_bb.shape

        sq_l, sq_u, sq_ip = gfl.lu_m(m_bb)
        sq = gfl.rpivot(sq_l @ sq_u, sq_ip)
        sqp = gfl.pivot(m_bb, sq_ip)
        dinds = (...,) + np.diag_indices(big[-1], 2)  # to check l
        uinds = (...,) + np.triu_indices(big[-1], 1)  # to check l
        linds = (...,) + np.tril_indices(big[-1], -1)  # to check u
        # with self.subTest(msg="square"):
        self.assertArrayAllClose(sq_l[dinds], 1.)
        self.assertArrayAllClose(sq_l[uinds], 0.)
        self.assertArrayAllClose(sq_u[linds], 0.)
        self.assertArrayAllClose(sq_l @ sq_u, sqp)
        self.assertArrayAllClose(sq, m_bb)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_lu_basic_returns_expected_values_wide(self, m_sb):
        wide = m_sb.shape
        hy.assume(hn.wide(m_sb))

        wd_l, wd_u, wd_ip = gfl.lu_m(m_sb)
        wd = gfl.rpivot(wd_l @ wd_u, wd_ip)
        wdp = gfl.pivot(m_sb, wd_ip)
        dinds = (...,) + np.diag_indices(wide[-2], 2)  # to check l
        uinds = (...,) + np.triu_indices(wide[-2], 1, wide[-2])  # to check l
        linds = (...,) + np.tril_indices(wide[-2], -1, wide[-1])  # to check u
        # with self.subTest(msg="wide"):
        self.assertArrayAllClose(wd_l[dinds], 1.)
        self.assertArrayAllClose(wd_l[uinds], 0.)
        self.assertArrayAllClose(wd_u[linds], 0.)
        self.assertArrayAllClose(wd_l @ wd_u, wdp)
        self.assertArrayAllClose(wd, m_sb)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_lu_basic_returns_expected_values_tall(self, m_bs):
        tall = m_bs.shape
        hy.assume(hn.tall(m_bs))

        tl_l, tl_u, tl_ip = gfl.lu_n(m_bs)
        tl = gfl.rpivot(tl_l @ tl_u, tl_ip)
        tlp = gfl.pivot(m_bs, tl_ip)
        dinds = (...,) + np.diag_indices(tall[-1], 2)  # to check l
        uinds = (...,) + np.triu_indices(tall[-2], 1, tall[-1])  # to check l
        linds = (...,) + np.tril_indices(tall[-1], -1, tall[-1])  # to check u
        # with self.subTest(msg="tall"):
        self.assertArrayAllClose(tl_l[dinds], 1.)
        self.assertArrayAllClose(tl_l[uinds], 0.)
        self.assertArrayAllClose(tl_u[linds], 0.)
        self.assertArrayAllClose(tl_l @ tl_u, tlp)
        self.assertArrayAllClose(tl, m_bs)

    @hy.given(hn.broadcastable('(a,a)', None))
    def test_lu_raw_returns_expected_values_square(self, m_bb):
        sq_l, sq_u, sq_ip0 = gfl.lu_m(m_bb)
        sq_f, sq_ip = gfl.lu_rawm(m_bb)
        sq_f = la.transpose(sq_f)
        linds = (...,) + np.tril_indices(m_bb.shape[-1], -1)
        uinds = (...,) + np.triu_indices(m_bb.shape[-1], 0)
        # with self.subTest(msg="square"):
        self.assertArrayAllClose(sq_f[linds], sq_l[linds])
        self.assertArrayAllClose(sq_f[uinds], sq_u[uinds])
        self.assertEqual(sq_ip, sq_ip0)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_lu_raw_returns_expected_values_wide(self, m_sb):
        wide = m_sb.shape
        hy.assume(hn.wide(m_sb))

        wd_l, wd_u, wd_ip0 = gfl.lu_m(m_sb)
        wd_f, wd_ip = gfl.lu_rawm(m_sb)
        wd_f = la.transpose(wd_f)
        linds = (...,) + np.tril_indices(wide[-2], -1, wide[-1])
        uinds = (...,) + np.triu_indices(wide[-2], 0, wide[-1])
        # with self.subTest(msg="wide"):
        self.assertArrayAllClose(wd_f[linds], wd_l[linds])
        self.assertArrayAllClose(wd_f[uinds], wd_u[uinds])
        self.assertEqual(wd_ip, wd_ip0)

    @hy.given(hn.broadcastable('(a,b)', None))
    def test_lu_raw_returns_expected_values_tall(self, m_bs):
        tall = m_bs.shape
        hy.assume(hn.tall(m_bs))

        tl_l, tl_u, tl_ip0 = gfl.lu_n(m_bs)
        tl_f, tl_ip = gfl.lu_rawn(m_bs)
        tl_f = la.transpose(tl_f)
        linds = (...,) + np.tril_indices(tall[-2], -1, tall[-1])
        uinds = (...,) + np.triu_indices(tall[-2], 0, tall[-1])
        # with self.subTest(msg="tall"):
        self.assertArrayAllClose(tl_f[linds], tl_l[linds])
        self.assertArrayAllClose(tl_f[uinds], tl_u[uinds])
        self.assertEqual(tl_ip, tl_ip0)

    @hy.given(hn.broadcastable('(a,a)', 'd'))
    def test_inv_returns_expected_shapes(self, m_bb):
        big = m_bb.shape
        hy.assume(hn.all_non_singular(m_bb))

        # with self.subTest(msg='inv'):
        self.assertArrayShape(gfl.inv(m_bb), big)
        # with self.subTest(msg='inv,+lu'):
        _, square_f, square_ip = gfl.inv_lu(m_bb)
        self.assertArrayShapesAre(gfl.inv_lu(m_bb), (big, big, big[:-1]))
        # with self.subTest(msg='inv,-lu'):
        self.assertArrayShape(gfl.lu_inv(square_f, square_ip), big)

    @hy.given(hn.broadcastable('(a,a)', None))
    def test_inv_returns_expected_values(self, m_bb):
        hy.assume(hn.all_non_singular(m_bb))

        # with self.subTest(msg='inv'):
        id_b = np.identity(m_bb.shape[-1], m_bb.dtype)
        square_i = gfl.inv(m_bb)
        self.assertArrayAllClose(m_bb @ square_i, id_b)
        self.assertArrayAllClose(square_i @ m_bb, id_b)
        # with self.subTest(msg='inv,+lu'):
        square_if, square_f, square_ip = gfl.inv_lu(m_bb)
        luf, ipr = gfl.lu_rawn(m_bb)
        self.assertArrayAllClose(square_if, square_i)
        self.assertArrayAllClose(square_f, luf)
        self.assertEqual(square_ip, ipr)
        # with self.subTest(msg='inv,-lu'):
        square_fi = gfl.lu_inv(square_f, square_ip)
        self.assertArrayAllClose(square_fi, square_i)


# =============================================================================
# Test solve
# =============================================================================


class TestSolveShape(TestCaseNumpy):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_solve_returns_expected_shapes(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_ss))

        expect = utn.array_return_shape('(a,a),(a,b)->(a,b)', m_ss, m_sb)
        self.assertArrayShape(gfl.solve(m_ss, m_sb), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(m_bb, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.solve(*make_bad_broadcast(m_ss, m_sb))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_rsolve_returns_expected_shapes(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_bb))

        expect = utn.array_return_shape('(a,b),(b,b)->(a,b)', m_sb, m_bb)
        self.assertArrayShape(gfl.rsolve(m_sb, m_bb), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve(m_bs, m_bb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rsolve(*make_bad_broadcast(m_bs, m_ss))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_solve_lu_returns_expected_shapes(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_ss))

        expect = utn.array_return_shape('(a,a),(a,b)->(a,b)', m_ss, m_sb)
        expect_f = expect[:-2] + 2 * m_ss.shape[-1:]
        self.assertArrayShapesAre(gfl.solve_lu(m_ss, m_sb),
                                  (expect, expect_f, expect_f[:-1]))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve_lu(m_bb, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve_lu(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.solve_lu(*make_bad_broadcast(m_ss, m_sb))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_lu_solve_returns_expected_shapes(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_ss))

        _, x_f, i_p = gfl.solve_lu(m_ss, m_sb)
        expect = utn.array_return_shape('(a,a),(a,b)->(a,b)', m_ss, m_sb)
        self.assertArrayShape(gfl.lu_solve(x_f, i_p, m_sb), expect)
        expect = utn.array_return_shape('(a,b),(b,b)->(a,b)', m_bs, x_f)
        self.assertArrayShape(gfl.rlu_solve(m_bs, x_f, i_p), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lu_solve(x_f, i_p, m_bb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlu_solve(m_sb, x_f, i_p)
        _, x_f, i_p = gfl.solve_lu(m_ss, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.lu_solve(x_f, *make_bad_broadcast(i_p, m_sb, (1, 2)))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_rsolve_lu_returns_expected_shapes(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_bb))

        expect = utn.array_return_shape('(a,b),(b,b)->(a,b)', m_sb, m_bb)
        expect_f = expect[:-2] + m_bb.shape[-2:]
        self.assertArrayShapesAre(gfl.rsolve_lu(m_sb, m_bb),
                                  (expect, expect_f, expect_f[:-1]))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve_lu(m_bs, m_bb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve_lu(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rsolve_lu(*make_bad_broadcast(m_bs, m_ss))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', 'd'))
    def test_rlu_solve_returns_expected_shapes(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_bb))

        _, x_f, i_p = gfl.rsolve_lu(m_sb, m_bb)
        expect = utn.array_return_shape('(a,b),(b,b)->(a,b)', m_sb, x_f)
        self.assertArrayShape(gfl.rlu_solve(m_sb, x_f, i_p), expect)
        expect = utn.array_return_shape('(a,a),(a,b)->(a,b)', x_f, m_bs)
        self.assertArrayShape(gfl.lu_solve(x_f, i_p, m_bs), expect)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlu_solve(m_ss, x_f, i_p)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lu_solve(x_f, i_p, m_sb)
        _, x_f, i_p = gfl.rsolve_lu(m_sb, m_bb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rlu_solve(*make_bad_broadcast(m_sb, x_f), i_p)


class TestSolveVectors(TestCaseNumpy):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve with vectors"""

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a)', 'd'))
    def test_solve_flexible_signature_with_vectors(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-1]
        v_s = hn.core_only(arrays[-1], dims=1)
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_ss))
        off_b, y_one = make_off_by_one(m_bb, m_sb)

        # with self.subTest('solve'):
        self.assertArrayShape(gfl.solve(m_ss, v_s), m_ss.shape[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(m_bb, v_s)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(m_bs, v_s)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as Mv:
            gfl.solve(m_bb[off_b], m_sb[y_one])

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a)', 'd'))
    def test_solve_lu_flexible_signature_with_vectors(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-1]
        v_s = hn.core_only(arrays[-1], dims=1)
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_ss))
        off_b, y_one = make_off_by_one(m_bb, m_sb)

        # with self.subTest('solve_lu'):
        self.assertArrayShapesAre(
            gfl.solve_lu(m_ss, v_s),
            (m_ss.shape[:-1], m_ss.shape, m_ss.shape[:-1]))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve_lu(m_bb, v_s)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve_lu(m_bs, v_s)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as Mv:
            gfl.solve_lu(m_bb[off_b], m_sb[y_one])

    @hy.given(hn.broadcastable('(a,a),(a),(b)', 'd'))
    def test_lu_solve_flexible_signature_with_vectors(self, arrays):
        m_ss = arrays[0]
        v_s, v_b = hn.core_only(*arrays[1:], dims=1)
        hy.assume(len(v_s) != len(v_b))
        hy.assume(hn.all_non_singular(m_ss))

        # with self.subTest('lu_solve'):
        _, x_f, i_p = gfl.solve_lu(m_ss, v_s)
        self.assertArrayShape(gfl.lu_solve(x_f, i_p, v_s), m_ss.shape[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lu_solve(x_f, i_p, v_b)
        # with self.subTest('rlu_solve'):
        self.assertArrayShape(gfl.rlu_solve(v_s, x_f, i_p), m_ss.shape[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlu_solve(v_b, x_f, i_p)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a)', 'd'))
    def test_rsolve_flexible_signature_with_vectors(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-1]
        v_s = hn.core_only(arrays[-1], dims=1)
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_ss))
        off_b, y_one = make_off_by_one(m_ss, m_sb)

        # with self.subTest('rsolve'):
        self.assertArrayShape(gfl.rsolve(v_s, m_ss), m_ss.shape[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve(v_s, m_bb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve(v_s, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as vM:
            gfl.rsolve(m_sb[y_one], m_ss[off_b])

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a)', 'd'))
    def test_rsolve_lu_flexible_signature_with_vectors(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-1]
        v_s = hn.core_only(arrays[-1], dims=1)
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_ss))
        off_b, y_one = make_off_by_one(m_ss, m_sb)

        # with self.subTest('rsolve_lu'):
        self.assertArrayShapesAre(
            gfl.rsolve_lu(v_s, m_ss),
            (m_ss.shape[:-1], m_ss.shape, m_ss.shape[:-1]))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve_lu(v_s, m_bb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve_lu(v_s, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as vM:
            gfl.rsolve_lu(m_sb, m_ss)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a),(a),(b)', 'd'))
    def test_rlu_solve_flexible_signature_with_vectors(self, arrays):
        m_ss, m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        hy.assume(hn.nonsquare(m_sb))
        hy.assume(hn.all_non_singular(m_ss))
        off_b, y_one = make_off_by_one(m_ss, m_sb)

        # with self.subTest('rlu_solve'):
        _, x_f, i_p = gfl.rsolve_lu(v_s, m_ss)
        self.assertArrayShape(gfl.rlu_solve(v_s, x_f, i_p), m_ss.shape[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlu_solve(v_b, x_f, i_p)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would succed/broadcast error if interpreted as vM:
            gfl.rlu_solve(m_sb[y_one], x_f[off_b], i_p[off_b])
        self.assertArrayShape(gfl.lu_solve(x_f, i_p, v_s),  m_ss.shape[:-1])
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lu_solve(x_f, i_p, v_b)


class TestSolveVal(TestCaseNumpy):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a)', None))
    def test_solve_returns_expected_values(self, arrays):
        m_ss, m_sb, m_bs = arrays
        hy.assume(hn.all_non_singular(m_ss))

        x_sb = gfl.solve(m_ss, m_sb)
        # with self.subTest(msg='solve'):
        self.assertArrayAllClose(m_ss @ x_sb, m_sb)
        x_bs = gfl.rsolve(m_bs, m_ss)
        # with self.subTest(msg='rsolve'):
        self.assertArrayAllClose(x_bs @ m_ss, m_bs)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a)', None))
    def test_solve_lu_returns_expected_values(self, arrays):
        m_ss, m_sb, m_bs = arrays
        hy.assume(hn.all_non_singular(m_ss))

        x0_sb = gfl.solve(m_ss, m_sb)
        x_sb, x_f, i_p = gfl.solve_lu(m_ss, m_sb)
        # with self.subTest('solve0'):
        self.assertArrayAllClose(x_sb, x0_sb)
        xx_sb = gfl.lu_solve(x_f, i_p, m_sb)
        # with self.subTest('solve(lu)'):
        self.assertArrayAllClose(xx_sb, x0_sb)
        x_bs = gfl.rlu_solve(m_bs, x_f, i_p)
        # with self.subTest('rsolve(lu)'):
        self.assertArrayAllClose(x_bs @ m_ss, m_bs)

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a)', None))
    def test_rsolve_lu_returns_expected_values(self, arrays):
        m_ss, m_sb, m_bs = arrays
        hy.assume(hn.all_non_singular(m_ss))

        x0_bs = gfl.rsolve(m_bs, m_ss)
        x_bs, x_f, i_p = gfl.rsolve_lu(m_bs, m_ss)
        # with self.subTest('rsolve0'):
        self.assertArrayAllClose(x_bs, x0_bs)
        xx_bs = gfl.rlu_solve(m_bs, x_f, i_p)
        # with self.subTest('rsolve(rlu)'):
        self.assertArrayAllClose(xx_bs, x0_bs)
        x_sb = gfl.lu_solve(x_f, i_p, m_sb)
        # with self.subTest('solve(rlu)'):
        self.assertArrayAllClose(m_ss @ x_sb, m_sb)

    @expectedFailure
    @errstate
    @hy.given(hn.constant('(a,a)', None, min_side=2))
    def test_solve_raises_with_low_rank(self, ones_ss):
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.solve(ones_ss, ones_ss[...,:2])


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
