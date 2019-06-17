# -*- coding: utf-8 -*-
"""Test solve & lu families of gufuncs
"""
import numpy as np
import numpy_linalg.gufuncs._gufuncs_lu_solve as gfl
from numpy_linalg import transpose
if __name__.find('tests.') < 0:
    import unittest_numpy as utn
    from test_gufunc import TestMatsVecs
else:
    from . import unittest_numpy as utn
    from .test_gufunc import TestMatsVecs

errstate = utn.errstate(invalid='raise')
# =============================================================================
__all__ = ['TestLU', 'TestSolveShape', 'TestSolveVectors', 'TestSolveVal']
# =============================================================================
# %% Test LU
# =============================================================================


class TestLU(TestMatsVecs):
    """Testing LU decomposition"""

    def test_lu_basic_returns_expected_shapes(self):
        """Test shape of basic LU"""
        self.pick_var_type('d')
        with self.subTest(msg="square"):
            self.assertArrayShapesAre(gfl.lu_m(self.a_bb),
                                      ((3, 7, 7), (3, 7, 7), (3, 7)))
        with self.subTest(msg="wide"):
            self.assertArrayShapesAre(gfl.lu_m(self.a_sb),
                                      ((4, 1, 3, 3), (4, 1, 3, 7), (4, 1, 3)))
        with self.subTest(msg="tall"):
            self.assertArrayShapesAre(gfl.lu_n(self.m_bs),
                                      ((7, 3), (3, 3), (3,)))

    def test_lu_raw_returns_expected_shapes(self):
        """Test shape of raw LU"""
        self.pick_var_type('d')
        with self.subTest(msg="square"):
            self.assertArrayShapesAre(gfl.lu_rawm(self.a_bb),
                                      ((3, 7, 7), (3, 7)))
        with self.subTest(msg="wide"):
            self.assertArrayShapesAre(gfl.lu_rawm(self.a_sb),
                                      ((4, 1, 7, 3), (4, 1, 3)))
        with self.subTest(msg="tall"):
            self.assertArrayShapesAre(gfl.lu_rawn(self.m_bs), ((3, 7), (3,)))

    @utn.loop_test()
    def test_lu_basic_returns_expected_values(self, sctype):
        """Test values of basic LU"""
        self.pick_var_type(sctype)
        sq_l, sq_u, sq_ip = gfl.lu_m(self.a_bb)
        sq = gfl.rpivot(sq_l @ sq_u, sq_ip)
        sqp = gfl.pivot(self.a_bb, sq_ip)
        dinds = (...,) + np.diag_indices(7, 2)  # to check l
        uinds = (...,) + np.triu_indices(7, 1)  # to check l
        linds = (...,) + np.tril_indices(7, -1)  # to check u
        with self.subTest(msg="square"):
            self.assertArrayAllClose(sq_l[dinds], 1.)
            self.assertArrayAllClose(sq_l[uinds], 0.)
            self.assertArrayAllClose(sq_u[linds], 0.)
            self.assertArrayAllClose(sq_l @ sq_u, sqp)
            self.assertArrayAllClose(sq, self.a_bb)
        wd_l, wd_u, wd_ip = gfl.lu_m(self.a_sb)
        wd = gfl.rpivot(wd_l @ wd_u, wd_ip)
        wdp = gfl.pivot(self.a_sb, wd_ip)
        dinds = (...,) + np.diag_indices(3, 2)  # to check l
        uinds = (...,) + np.triu_indices(3, 1, 3)  # to check l
        linds = (...,) + np.tril_indices(3, -1, 7)  # to check u
        with self.subTest(msg="wide"):
            self.assertArrayAllClose(wd_l[dinds], 1.)
            self.assertArrayAllClose(wd_l[uinds], 0.)
            self.assertArrayAllClose(wd_u[linds], 0.)
            self.assertArrayAllClose(wd_l @ wd_u, wdp)
            self.assertArrayAllClose(wd, self.a_sb)
        tl_l, tl_u, tl_ip = gfl.lu_n(self.m_bs)
        tl = gfl.rpivot(tl_l @ tl_u, tl_ip)
        tlp = gfl.pivot(self.m_bs, tl_ip)
        dinds = (...,) + np.diag_indices(3, 2)  # to check l
        uinds = (...,) + np.triu_indices(7, 1, 3)  # to check l
        linds = (...,) + np.tril_indices(3, -1, 3)  # to check u
        with self.subTest(msg="tall"):
            self.assertArrayAllClose(tl_l[dinds], 1.)
            self.assertArrayAllClose(tl_l[uinds], 0.)
            self.assertArrayAllClose(tl_u[linds], 0.)
            self.assertArrayAllClose(tl_l @ tl_u, tlp)
            self.assertArrayAllClose(tl, self.m_bs)

    @utn.loop_test()
    def test_lu_raw_returns_expected_values(self, sctype):
        """Test values of raw LU"""
        self.pick_var_type(sctype)
        sq_l, sq_u, sq_ip0 = gfl.lu_m(self.a_bb)
        sq_f, sq_ip = gfl.lu_rawm(self.a_bb)
        sq_f = transpose(sq_f)
        linds = (...,) + np.tril_indices(7, -1)
        uinds = (...,) + np.triu_indices(7, 0)
        with self.subTest(msg="square"):
            self.assertArrayAllClose(sq_f[linds], sq_l[linds])
            self.assertArrayAllClose(sq_f[uinds], sq_u[uinds])
            self.assertEqual(sq_ip, sq_ip0)
        wd_l, wd_u, wd_ip0 = gfl.lu_m(self.a_sb)
        wd_f, wd_ip = gfl.lu_rawm(self.a_sb)
        wd_f = transpose(wd_f)
        linds = (...,) + np.tril_indices(3, -1, 7)
        uinds = (...,) + np.triu_indices(3, 0, 7)
        with self.subTest(msg="wide"):
            self.assertArrayAllClose(wd_f[linds], wd_l[linds])
            self.assertArrayAllClose(wd_f[uinds], wd_u[uinds])
            self.assertEqual(wd_ip, wd_ip0)
        tl_l, tl_u, tl_ip0 = gfl.lu_n(self.m_bs)
        tl_f, tl_ip = gfl.lu_rawn(self.m_bs)
        tl_f = transpose(tl_f)
        linds = (...,) + np.tril_indices(7, -1, 3)
        uinds = (...,) + np.triu_indices(7, 0, 3)
        with self.subTest(msg="tall"):
            self.assertArrayAllClose(tl_f[linds], tl_l[linds])
            self.assertArrayAllClose(tl_f[uinds], tl_u[uinds])
            self.assertEqual(tl_ip, tl_ip0)

    def test_inv_returns_expected_shapes(self):
        """Check that inv gufuncs all return arrays with the expected shape
        """
        self.pick_var_type('d')
        with self.subTest(msg='inv'):
            self.assertArrayShape(gfl.inv(self.a_bb), (3, 7, 7))
        with self.subTest(msg='inv,+lu'):
            _, square_f, square_ip = gfl.inv_lu(self.a_bb)
            self.assertArrayShapesAre(gfl.inv_lu(self.a_bb),
                                      ((3, 7, 7), (3, 7, 7), (3, 7)))
        with self.subTest(msg='inv,-lu'):
            self.assertArrayShape(gfl.lu_inv(square_f, square_ip), (3, 7, 7))

    @utn.loop_test()
    def test_inv_returns_expected_values(self, sctype):
        """Check that inv gufuncs all return arrays with the expected values
        """
        self.pick_var_type(sctype)
        with self.subTest(msg='inv'):
            square_i = gfl.inv(self.a_bb)
            self.assertArrayAllClose(square_i @ self.a_bb, self.id_b)
            self.assertArrayAllClose(self.a_bb @ square_i, self.id_b)
        with self.subTest(msg='inv,+lu'):
            square_if, square_f, square_ip = gfl.inv_lu(self.a_bb)
            luf, ip = gfl.lu_rawn(self.a_bb)
            self.assertArrayAllClose(square_if, square_i)
            self.assertArrayAllClose(square_f, luf)
            self.assertEqual(square_ip, ip)
        with self.subTest(msg='inv,-lu'):
            square_fi = gfl.lu_inv(square_f, square_ip)
            self.assertArrayAllClose(square_fi, square_i)


# =============================================================================
# %% Test solve
# =============================================================================


class TestSolveShape(TestMatsVecs):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    def test_solve_returns_expected_shapes(self):
        self.pick_var_type('d')
        self.assertArrayShape(gfl.solve(self.m_ss, self.m_sb), (3, 7))
        self.assertArrayShape(gfl.solve(self.a_ss, self.m_sb), (5, 1, 3, 7))
        self.assertArrayShape(gfl.solve(self.m_ss, self.a_sb), (4, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(self.m_bb, self.m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(self.m_bs, self.m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.solve(self.a_ss, self.a_sb)

    def test_rsolve_returns_expected_shapes(self):
        self.pick_var_type('d')
        self.assertArrayShape(gfl.rsolve(self.m_sb, self.m_bb), (3, 7))
        self.assertArrayShape(gfl.rsolve(self.a_bs, self.a_ss), (5, 2, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve(self.m_bs, self.m_bb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve(self.m_bs, self.m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rsolve(transpose(self.a_sb), self.a_ss)

    def test_solve_lu_returns_expected_shapes(self):
        self.pick_var_type('d')
        with self.subTest('solve_lu'):
            self.assertArrayShapesAre(gfl.solve_lu(self.m_ss, self.m_sb),
                                      ((3, 7), (3, 3), (3,)))
            self.assertArrayShapesAre(gfl.solve_lu(self.a_ss, self.m_sb),
                                      ((5, 1, 3, 7), (5, 1, 3, 3), (5, 1, 3)))
            self.assertArrayShapesAre(gfl.solve_lu(self.m_ss, self.a_sb),
                                      ((4, 1, 3, 7), (4, 1, 3, 3), (4, 1, 3)))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve_lu(self.m_bb, self.m_sb)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve_lu(self.m_bs, self.m_sb)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.solve_lu(self.a_ss, self.a_sb)
        with self.subTest('(r)lu_solve'):
            _, xf, p = gfl.solve_lu(self.m_ss, self.m_sb)
            self.assertArrayShape(gfl.lu_solve(xf, p, self.m_sb), (3, 7))
            self.assertArrayShape(gfl.lu_solve(xf, p, self.a_sb), (4, 1, 3, 7))
            self.assertArrayShape(gfl.rlu_solve(self.m_bs, xf, p), (7, 3))
            self.assertArrayShape(gfl.rlu_solve(self.a_bs, xf, p), (2, 7, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lu_solve(xf, p, self.m_bs)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlu_solve(self.m_sb, xf, p)
            _, xf, p = gfl.solve_lu(self.a_ss, self.m_sb)
            self.assertArrayShape(gfl.lu_solve(xf, p, self.m_sb), (5, 1, 3, 7))
            self.assertArrayShape(gfl.lu_solve(xf, p, transpose(self.a_bs)),
                                  (5, 2, 3, 7))
            self.assertArrayShape(gfl.rlu_solve(
                self.m_bs, xf, p), (5, 1, 7, 3))
            self.assertArrayShape(gfl.rlu_solve(
                self.a_bs, xf, p), (5, 2, 7, 3))
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.lu_solve(xf, p, self.a_sb)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.rlu_solve(transpose(self.a_sb), xf, p)

    def test_rsolve_lu_returns_expected_shapes(self):
        self.pick_var_type('d')
        with self.subTest('solve_lu'):
            self.assertArrayShapesAre(gfl.rsolve_lu(self.m_sb, self.m_bb),
                                      ((3, 7), (7, 7), (7,)))
            self.assertArrayShapesAre(gfl.rsolve_lu(self.a_bs, self.a_ss),
                                      ((5, 2, 7, 3), (5, 2, 3, 3), (5, 2, 3)))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve_lu(self.m_bs, self.m_bb)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve_lu(self.m_bs, self.m_sb)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.rsolve_lu(transpose(self.a_sb), self.a_ss)
        with self.subTest('(r)lu_solve'):
            _, xf, p = gfl.rsolve_lu(self.m_sb, self.m_bb)
            self.assertArrayShape(gfl.rlu_solve(self.m_sb, xf, p), (3, 7))
            self.assertArrayShape(gfl.rlu_solve(self.a_sb, xf, p),
                                  (4, 1, 3, 7))
            self.assertArrayShape(gfl.lu_solve(xf, p, self.m_bs), (7, 3))
            self.assertArrayShape(gfl.lu_solve(xf, p, self.a_bs), (2, 7, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlu_solve(self.m_bs, xf, p)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lu_solve(xf, p, self.m_sb)
            _, xf, p = gfl.rsolve_lu(self.m_sb, self.a_bb)
            self.assertArrayShape(gfl.rlu_solve(self.m_sb, xf, p), (3, 3, 7))
            self.assertArrayShape(gfl.rlu_solve(self.a_sb, xf, p),
                                  (4, 3, 3, 7))
            self.assertArrayShape(gfl.lu_solve(xf, p, self.m_bs), (3, 7, 3))
            self.assertArrayShape(gfl.lu_solve(xf, p, transpose(self.a_sb)),
                                  (4, 3, 7, 3))
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.lu_solve(xf, p, self.a_bs)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.rlu_solve(transpose(self.a_bs), xf, p)


class TestSolveVectors(TestMatsVecs):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve with vectors"""

    def test_solve_flexible_signature_with_vectors(self):
        self.pick_var_type('d')
        with self.subTest('solve'):
            self.assertArrayShape(gfl.solve(self.m_ss, self.v_s), (3,))
            self.assertArrayShape(gfl.solve(self.a_ss, self.v_s), (5, 1, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve(self.m_bb, self.v_s)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve(self.m_bs, self.v_s)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (3)(7,7)\(3)(7)
                gfl.solve(self.a_bb, self.m_sb)
        with self.subTest('solve_lu'):
            self.assertArrayShapesAre(gfl.solve_lu(self.m_ss, self.v_s),
                                      ((3,), (3, 3), (3,)))
            self.assertArrayShapesAre(gfl.solve_lu(self.a_ss, self.v_s),
                                      ((5, 1, 3), (5, 1, 3, 3), (5, 1, 3)))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve_lu(self.m_bb, self.v_s)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve_lu(self.m_bs, self.v_s)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (3)(7,7)\(3)(7)
                gfl.solve_lu(self.a_bb, self.m_sb)
        with self.subTest('lu_solve'):
            _, xf, p = gfl.solve_lu(self.m_ss, self.v_s)
            self.assertArrayShape(gfl.lu_solve(xf, p, self.v_s), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lu_solve(xf, p, self.v_b)
        with self.subTest('rlu_solve'):
            self.assertArrayShape(gfl.rlu_solve(self.v_s, xf, p), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlu_solve(self.v_b, xf, p)
        with self.subTest('lu_solve'):
            _, xf, p = gfl.solve_lu(self.a_bb, self.v_b)
            self.assertArrayShape(gfl.lu_solve(xf, p, self.v_b), (3, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (3)(7,7)\(3)(7)
                gfl.lu_solve(xf, p, self.m_sb)
        with self.subTest('rlu_solve'):
            self.assertArrayShape(gfl.rlu_solve(self.v_b, xf, p), (3, 7))
            # This would differ if interpreted as vM: (3)(7)/(3)(7,7) -> (3)(3)
            self.assertArrayShape(gfl.rlu_solve(self.m_sb, xf, p), (3, 3, 7))

    def test_rsolve_flexible_signature_with_vectors(self):
        self.pick_var_type('d')
        with self.subTest('rsolve'):
            self.assertArrayShape(gfl.rsolve(self.v_s, self.m_ss), (3,))
            self.assertArrayShape(gfl.rsolve(self.v_s, self.a_ss), (5, 1, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve(self.v_s, self.m_bb)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve(self.v_s, self.m_bs)
            # This would differ if interpreted as vM: (3)(7)/(3)(7,7)->(3)(7)
            self.assertArrayShape(gfl.rsolve(self.m_sb, self.a_bb), (3, 3, 7))
        with self.subTest('rsolve_lu'):
            self.assertArrayShapesAre(gfl.rsolve_lu(self.v_s, self.m_ss),
                                      ((3,), (3, 3), (3,)))
            self.assertArrayShapesAre(gfl.rsolve_lu(self.v_s, self.a_ss),
                                      ((5, 1, 3), (5, 1, 3, 3), (5, 1, 3)))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve_lu(self.v_s, self.m_bb)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve_lu(self.v_s, self.m_bs)
            # This would differ if interpreted as vM: (3)(7)/(3)(7,7)->(3)(7)
            self.assertArrayShapesAre(gfl.rsolve_lu(self.m_sb, self.a_bb),
                                      ((3, 3, 7), (3, 7, 7), (3, 7)))
        with self.subTest('rlu_solve'):
            _, xf, p = gfl.rsolve_lu(self.v_s, self.m_ss)
            self.assertArrayShape(gfl.rlu_solve(self.v_s, xf, p), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlu_solve(self.v_b, xf, p)
        with self.subTest('lu_solve'):
            self.assertArrayShape(gfl.lu_solve(xf, p, self.v_s), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lu_solve(xf, p, self.v_b)
        with self.subTest('rlu_solve'):
            _, xf, p = gfl.rsolve_lu(self.v_b, self.a_bb)
            self.assertArrayShape(gfl.rlu_solve(self.v_b, xf, p), (3, 7))
            # This would differ if interpreted as vM: (3)(7)/(3)(7,7)->(3)(7)
            self.assertArrayShape(gfl.rlu_solve(self.m_sb, xf, p), (3, 3, 7))
        with self.subTest('lu_solve'):
            self.assertArrayShape(gfl.lu_solve(xf, p, self.v_b), (3, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (3)(7,7)\(3)(7)
                gfl.lu_solve(xf, p, self.m_sb)


class TestSolveVal(TestMatsVecs):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    @utn.loop_test()
    def test_solve_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        a = gfl.solve(self.a_ss, self.m_sb)
        with self.subTest(msg='solve'):
            self.assertArrayAllClose(self.a_ss @ a, self.m_sb)
        b = gfl.rsolve(self.a_bs, self.a_ss)
        with self.subTest(msg='rsolve'):
            self.assertArrayAllClose(b @ self.a_ss, self.a_bs)

    @utn.loop_test()
    def test_solve_lu_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        a0 = gfl.solve(self.a_ss, self.m_sb)
        a, xf, p = gfl.solve_lu(self.a_ss, self.m_sb)
        with self.subTest('solve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.lu_solve(xf, p, self.m_sb)
        with self.subTest('solve(lu)'):
            self.assertArrayAllClose(aa, a0)
        b = gfl.rlu_solve(self.a_bs, xf, p)
        with self.subTest('rsolve(lu)'):
            self.assertArrayAllClose(b @ self.a_ss, self.a_bs)

    @utn.loop_test()
    def test_rsolve_lu_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        a0 = gfl.rsolve(self.a_bs, self.a_ss)
        a, xf, p = gfl.rsolve_lu(self.a_bs, self.a_ss)
        with self.subTest('rsolve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.rlu_solve(self.a_bs, xf, p)
        with self.subTest('rsolve(rlu)'):
            self.assertArrayAllClose(aa, a0)
        b = gfl.lu_solve(xf, p, self.m_sb)
        with self.subTest('solve(rlu)'):
            self.assertArrayAllClose(self.a_ss @ b, self.m_sb)

    @errstate
    @utn.loop_test(msg='rank')
    def test_solve_raises_with_low_rank(self, sctype):
        self.pick_var_type(sctype)
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.solve(self.ones_ss, self.m_sb)


# =============================================================================
if __name__ == '__main__':
    utn.main(verbosity=2)
