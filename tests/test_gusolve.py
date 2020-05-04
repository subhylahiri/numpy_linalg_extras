# -*- coding: utf-8 -*-
"""Test solve & lu families of gufuncs
"""
import hypothesis as hy
import numpy as np
import numpy_linalg.gufuncs._gufuncs_lu_solve as gfl
from numpy_linalg import transpose
if __name__.find('tests.') < 0:
    import unittest_numpy as utn
    from test_gufunc import TestMatsVecs
else:
    from . import unittest_numpy as utn
    from .test_gufunc import TestMatsVecs
# pylint: disable=invalid-name
# pylint: disable=missing-function-docstring
errstate = np.errstate(invalid='raise')
# =============================================================================
__all__ = ['TestLU', 'TestSolveShape', 'TestSolveVectors', 'TestSolveVal']
# =============================================================================
# Test LU
# =============================================================================


class TestLU(TestMatsVecs):
    """Testing LU decomposition"""

    def test_lu_basic_returns_expected_shapes(self):
        self.pick_var_type('d')
        arrays = [self.a_sb, self.a_bb, self.m_bs]
        m_sb, m_bb, m_bs = arrays
        wide, big, tall = [arr.shape for arr in arrays]
        widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        hy.assume(wide[-2] < wide[-1])

        with self.subTest(msg="square"):
            self.assertArrayShapesAre(gfl.lu_m(m_bb),
                                      ((3, 7, 7), (3, 7, 7), (3, 7)))
        with self.subTest(msg="wide"):
            self.assertArrayShapesAre(gfl.lu_m(m_sb),
                                      ((4, 1, 3, 3), (4, 1, 3, 7), (4, 1, 3)))
        with self.subTest(msg="tall"):
            self.assertArrayShapesAre(gfl.lu_n(m_bs),
                                      ((7, 3), (3, 3), (3,)))

    def test_lu_raw_returns_expected_shapes(self):
        self.pick_var_type('d')
        arrays = [self.a_sb, self.a_bb, self.m_bs]
        m_sb, m_bb, m_bs = arrays
        wide, big, tall = [arr.shape for arr in arrays]
        widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        hy.assume(wide[-2] < wide[-1])

        with self.subTest(msg="square"):
            self.assertArrayShapesAre(gfl.lu_rawm(m_bb),
                                      ((3, 7, 7), (3, 7)))
        with self.subTest(msg="wide"):
            self.assertArrayShapesAre(gfl.lu_rawm(m_sb),
                                      ((4, 1, 7, 3), (4, 1, 3)))
        with self.subTest(msg="tall"):
            self.assertArrayShapesAre(gfl.lu_rawn(m_bs), ((3, 7), (3,)))

    @utn.loop_test()
    def test_lu_basic_returns_expected_values_square(self, sctype):
        self.pick_var_type(sctype)
        m_bb = self.a_bb
        big = m_bb.shape

        sq_l, sq_u, sq_ip = gfl.lu_m(m_bb)
        sq = gfl.rpivot(sq_l @ sq_u, sq_ip)
        sqp = gfl.pivot(m_bb, sq_ip)
        dinds = (...,) + np.diag_indices(7, 2)  # to check l
        uinds = (...,) + np.triu_indices(7, 1)  # to check l
        linds = (...,) + np.tril_indices(7, -1)  # to check u
        with self.subTest(msg="square"):
            self.assertArrayAllClose(sq_l[dinds], 1.)
            self.assertArrayAllClose(sq_l[uinds], 0.)
            self.assertArrayAllClose(sq_u[linds], 0.)
            self.assertArrayAllClose(sq_l @ sq_u, sqp)
            self.assertArrayAllClose(sq, m_bb)

    @utn.loop_test()
    def test_lu_basic_returns_expected_values_wide(self, sctype):
        self.pick_var_type(sctype)
        m_sb = self.a_sb
        wide = m_sb.shape
        hy.assume(wide[-2] < wide[-1])

        wd_l, wd_u, wd_ip = gfl.lu_m(m_sb)
        wd = gfl.rpivot(wd_l @ wd_u, wd_ip)
        wdp = gfl.pivot(m_sb, wd_ip)
        dinds = (...,) + np.diag_indices(3, 2)  # to check l
        uinds = (...,) + np.triu_indices(3, 1, 3)  # to check l
        linds = (...,) + np.tril_indices(3, -1, 7)  # to check u
        with self.subTest(msg="wide"):
            self.assertArrayAllClose(wd_l[dinds], 1.)
            self.assertArrayAllClose(wd_l[uinds], 0.)
            self.assertArrayAllClose(wd_u[linds], 0.)
            self.assertArrayAllClose(wd_l @ wd_u, wdp)
            self.assertArrayAllClose(wd, m_sb)

    @utn.loop_test()
    def test_lu_basic_returns_expected_values_tall(self, sctype):
        self.pick_var_type(sctype)
        m_bs = self.a_bs
        tall = m_bs.shape
        hy.assume(tall[-2] > tall[-1])

        tl_l, tl_u, tl_ip = gfl.lu_n(m_bs)
        tl = gfl.rpivot(tl_l @ tl_u, tl_ip)
        tlp = gfl.pivot(m_bs, tl_ip)
        dinds = (...,) + np.diag_indices(3, 2)  # to check l
        uinds = (...,) + np.triu_indices(7, 1, 3)  # to check l
        linds = (...,) + np.tril_indices(3, -1, 3)  # to check u
        with self.subTest(msg="tall"):
            self.assertArrayAllClose(tl_l[dinds], 1.)
            self.assertArrayAllClose(tl_l[uinds], 0.)
            self.assertArrayAllClose(tl_u[linds], 0.)
            self.assertArrayAllClose(tl_l @ tl_u, tlp)
            self.assertArrayAllClose(tl, m_bs)

    @utn.loop_test()
    def test_lu_raw_returns_expected_values_square(self, sctype):
        self.pick_var_type(sctype)
        m_bb = self.a_bb
        big = m_bb.shape

        sq_l, sq_u, sq_ip0 = gfl.lu_m(m_bb)
        sq_f, sq_ip = gfl.lu_rawm(m_bb)
        sq_f = transpose(sq_f)
        linds = (...,) + np.tril_indices(7, -1)
        uinds = (...,) + np.triu_indices(7, 0)
        with self.subTest(msg="square"):
            self.assertArrayAllClose(sq_f[linds], sq_l[linds])
            self.assertArrayAllClose(sq_f[uinds], sq_u[uinds])
            self.assertEqual(sq_ip, sq_ip0)

    @utn.loop_test()
    def test_lu_raw_returns_expected_values_wide(self, sctype):
        self.pick_var_type(sctype)
        m_sb = self.a_sb
        wide = m_sb.shape
        hy.assume(wide[-2] < wide[-1])

        wd_l, wd_u, wd_ip0 = gfl.lu_m(m_sb)
        wd_f, wd_ip = gfl.lu_rawm(m_sb)
        wd_f = transpose(wd_f)
        linds = (...,) + np.tril_indices(3, -1, 7)
        uinds = (...,) + np.triu_indices(3, 0, 7)
        with self.subTest(msg="wide"):
            self.assertArrayAllClose(wd_f[linds], wd_l[linds])
            self.assertArrayAllClose(wd_f[uinds], wd_u[uinds])
            self.assertEqual(wd_ip, wd_ip0)

    @utn.loop_test()
    def test_lu_raw_returns_expected_values_tall(self, sctype):
        self.pick_var_type(sctype)
        m_bs = self.a_bs
        tall = m_bs.shape
        hy.assume(tall[-2] > tall[-1])

        tl_l, tl_u, tl_ip0 = gfl.lu_n(m_bs)
        tl_f, tl_ip = gfl.lu_rawn(m_bs)
        tl_f = transpose(tl_f)
        linds = (...,) + np.tril_indices(7, -1, 3)
        uinds = (...,) + np.triu_indices(7, 0, 3)
        with self.subTest(msg="tall"):
            self.assertArrayAllClose(tl_f[linds], tl_l[linds])
            self.assertArrayAllClose(tl_f[uinds], tl_u[uinds])
            self.assertEqual(tl_ip, tl_ip0)

    def test_inv_returns_expected_shapes(self):
        self.pick_var_type('d')
        m_bb = self.a_bb
        big = m_bb.shape
        hy.assume(np.all(utn.non_singular(m_bb)))

        with self.subTest(msg='inv'):
            self.assertArrayShape(gfl.inv(m_bb), (3, 7, 7))
        with self.subTest(msg='inv,+lu'):
            _, square_f, square_ip = gfl.inv_lu(m_bb)
            self.assertArrayShapesAre(gfl.inv_lu(m_bb),
                                      ((3, 7, 7), (3, 7, 7), (3, 7)))
        with self.subTest(msg='inv,-lu'):
            self.assertArrayShape(gfl.lu_inv(square_f, square_ip), (3, 7, 7))

    @utn.loop_test()
    def test_inv_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        m_bb = self.a_bb
        big = m_bb.shape
        hy.assume(np.all(utn.non_singular(m_bb)))

        with self.subTest(msg='inv'):
            square_i = gfl.inv(m_bb)
            self.assertArrayAllClose(square_i @ m_bb, self.id_b)
            self.assertArrayAllClose(m_bb @ square_i, self.id_b)
        with self.subTest(msg='inv,+lu'):
            square_if, square_f, square_ip = gfl.inv_lu(m_bb)
            luf, ip = gfl.lu_rawn(m_bb)
            self.assertArrayAllClose(square_if, square_i)
            self.assertArrayAllClose(square_f, luf)
            self.assertEqual(square_ip, ip)
        with self.subTest(msg='inv,-lu'):
            square_fi = gfl.lu_inv(square_f, square_ip)
            self.assertArrayAllClose(square_fi, square_i)


# =============================================================================
# Test solve
# =============================================================================


class TestSolveShape(TestMatsVecs):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    def test_solve_returns_expected_shapes(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        hy.assume(wide[-2] < wide[-1])
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))

        self.assertArrayShape(gfl.solve(m_ss, m_sb), (3, 7))
        self.assertArrayShape(gfl.solve(m_ss, m_sb), (5, 1, 3, 7))
        self.assertArrayShape(gfl.solve(m_ss, m_sb), (4, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(m_bb, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.solve(m_ss, m_sb)

    def test_rsolve_returns_expected_shapes(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        hy.assume(wide[-2] < wide[-1])
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))

        self.assertArrayShape(gfl.rsolve(m_sb, m_bb), (3, 7))
        self.assertArrayShape(gfl.rsolve(m_bs, m_ss), (5, 2, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve(m_bs, m_bb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rsolve(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rsolve(transpose(m_sb), m_ss)

    def test_solve_lu_returns_expected_shapes(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        hy.assume(wide[-2] < wide[-1])
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))

        with self.subTest('solve_lu'):
            self.assertArrayShapesAre(gfl.solve_lu(m_ss, m_sb),
                                      ((3, 7), (3, 3), (3,)))
            self.assertArrayShapesAre(gfl.solve_lu(m_ss, m_sb),
                                      ((5, 1, 3, 7), (5, 1, 3, 3), (5, 1, 3)))
            self.assertArrayShapesAre(gfl.solve_lu(m_ss, m_sb),
                                      ((4, 1, 3, 7), (4, 1, 3, 3), (4, 1, 3)))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve_lu(m_bb, m_sb)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve_lu(m_bs, m_sb)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.solve_lu(m_ss, m_sb)
        with self.subTest('(r)lu_solve'):
            _, xf, p = gfl.solve_lu(m_ss, m_sb)
            self.assertArrayShape(gfl.lu_solve(xf, p, m_sb), (3, 7))
            self.assertArrayShape(gfl.lu_solve(xf, p, m_sb), (4, 1, 3, 7))
            self.assertArrayShape(gfl.rlu_solve(m_bs, xf, p), (7, 3))
            self.assertArrayShape(gfl.rlu_solve(m_bs, xf, p), (2, 7, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lu_solve(xf, p, m_bs)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlu_solve(m_sb, xf, p)
            _, xf, p = gfl.solve_lu(m_ss, m_sb)
            self.assertArrayShape(gfl.lu_solve(xf, p, m_sb), (5, 1, 3, 7))
            self.assertArrayShape(gfl.lu_solve(xf, p, transpose(m_bs)),
                                  (5, 2, 3, 7))
            self.assertArrayShape(gfl.rlu_solve(
                m_bs, xf, p), (5, 1, 7, 3))
            self.assertArrayShape(gfl.rlu_solve(
                m_bs, xf, p), (5, 2, 7, 3))
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.lu_solve(xf, p, m_sb)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.rlu_solve(transpose(m_sb), xf, p)

    def test_rsolve_lu_returns_expected_shapes(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        hy.assume(wide[-2] < wide[-1])
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))

        with self.subTest('solve_lu'):
            self.assertArrayShapesAre(gfl.rsolve_lu(m_sb, m_bb),
                                      ((3, 7), (7, 7), (7,)))
            self.assertArrayShapesAre(gfl.rsolve_lu(m_bs, m_ss),
                                      ((5, 2, 7, 3), (5, 2, 3, 3), (5, 2, 3)))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve_lu(m_bs, m_bb)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve_lu(m_bs, m_sb)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.rsolve_lu(transpose(m_sb), m_ss)
        with self.subTest('(r)lu_solve'):
            _, xf, p = gfl.rsolve_lu(m_sb, m_bb)
            self.assertArrayShape(gfl.rlu_solve(m_sb, xf, p), (3, 7))
            self.assertArrayShape(gfl.rlu_solve(m_sb, xf, p),
                                  (4, 1, 3, 7))
            self.assertArrayShape(gfl.lu_solve(xf, p, m_bs), (7, 3))
            self.assertArrayShape(gfl.lu_solve(xf, p, m_bs), (2, 7, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlu_solve(m_bs, xf, p)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lu_solve(xf, p, m_sb)
            _, xf, p = gfl.rsolve_lu(m_sb, m_bb)
            self.assertArrayShape(gfl.rlu_solve(m_sb, xf, p), (3, 3, 7))
            self.assertArrayShape(gfl.rlu_solve(m_sb, xf, p),
                                  (4, 3, 3, 7))
            self.assertArrayShape(gfl.lu_solve(xf, p, m_bs), (3, 7, 3))
            self.assertArrayShape(gfl.lu_solve(xf, p, transpose(m_sb)),
                                  (4, 3, 7, 3))
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.lu_solve(xf, p, m_bs)
            with self.assertRaisesRegex(*utn.broadcast_err):
                gfl.rlu_solve(transpose(m_bs), xf, p)


class TestSolveVectors(TestMatsVecs):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve with vectors"""

    def test_solve_flexible_signature_with_vectors(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))
        hy.assume(wide[-2] < wide[-1])

        with self.subTest('solve'):
            self.assertArrayShape(gfl.solve(m_ss, v_s), (3,))
            self.assertArrayShape(gfl.solve(m_ss, v_s), (5, 1, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve(m_bb, v_s)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve(m_bs, v_s)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (3)(7,7)\(3)(7)
                gfl.solve(m_bb, m_sb)
        with self.subTest('solve_lu'):
            self.assertArrayShapesAre(gfl.solve_lu(m_ss, v_s),
                                      ((3,), (3, 3), (3,)))
            self.assertArrayShapesAre(gfl.solve_lu(m_ss, v_s),
                                      ((5, 1, 3), (5, 1, 3, 3), (5, 1, 3)))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve_lu(m_bb, v_s)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.solve_lu(m_bs, v_s)
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (3)(7,7)\(3)(7)
                gfl.solve_lu(m_bb, m_sb)
        with self.subTest('lu_solve'):
            _, xf, p = gfl.solve_lu(m_ss, v_s)
            self.assertArrayShape(gfl.lu_solve(xf, p, v_s), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lu_solve(xf, p, v_b)
        with self.subTest('rlu_solve'):
            self.assertArrayShape(gfl.rlu_solve(v_s, xf, p), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlu_solve(v_b, xf, p)
        with self.subTest('lu_solve'):
            _, xf, p = gfl.solve_lu(m_bb, v_b)
            self.assertArrayShape(gfl.lu_solve(xf, p, v_b), (3, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (3)(7,7)\(3)(7)
                gfl.lu_solve(xf, p, m_sb)
        with self.subTest('rlu_solve'):
            self.assertArrayShape(gfl.rlu_solve(v_b, xf, p), (3, 7))
            # This would differ if interpreted as vM: (3)(7)/(3)(7,7) -> (3)(3)
            self.assertArrayShape(gfl.rlu_solve(m_sb, xf, p), (3, 3, 7))

    def test_rsolve_flexible_signature_with_vectors(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))
        hy.assume(wide[-2] < wide[-1])

        with self.subTest('rsolve'):
            self.assertArrayShape(gfl.rsolve(v_s, m_ss), (3,))
            self.assertArrayShape(gfl.rsolve(v_s, m_ss), (5, 1, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve(v_s, m_bb)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve(v_s, m_bs)
            # This would differ if interpreted as vM: (3)(7)/(3)(7,7)->(3)(7)
            self.assertArrayShape(gfl.rsolve(m_sb, m_bb), (3, 3, 7))
        with self.subTest('rsolve_lu'):
            self.assertArrayShapesAre(gfl.rsolve_lu(v_s, m_ss),
                                      ((3,), (3, 3), (3,)))
            self.assertArrayShapesAre(gfl.rsolve_lu(v_s, m_ss),
                                      ((5, 1, 3), (5, 1, 3, 3), (5, 1, 3)))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve_lu(v_s, m_bb)
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rsolve_lu(v_s, m_bs)
            # This would differ if interpreted as vM: (3)(7)/(3)(7,7)->(3)(7)
            self.assertArrayShapesAre(gfl.rsolve_lu(m_sb, m_bb),
                                      ((3, 3, 7), (3, 7, 7), (3, 7)))
        with self.subTest('rlu_solve'):
            _, xf, p = gfl.rsolve_lu(v_s, m_ss)
            self.assertArrayShape(gfl.rlu_solve(v_s, xf, p), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.rlu_solve(v_b, xf, p)
        with self.subTest('lu_solve'):
            self.assertArrayShape(gfl.lu_solve(xf, p, v_s), (3,))
            with self.assertRaisesRegex(*utn.core_dim_err):
                gfl.lu_solve(xf, p, v_b)
        with self.subTest('rlu_solve'):
            _, xf, p = gfl.rsolve_lu(v_b, m_bb)
            self.assertArrayShape(gfl.rlu_solve(v_b, xf, p), (3, 7))
            # This would differ if interpreted as vM: (3)(7)/(3)(7,7)->(3)(7)
            self.assertArrayShape(gfl.rlu_solve(m_sb, xf, p), (3, 3, 7))
        with self.subTest('lu_solve'):
            self.assertArrayShape(gfl.lu_solve(xf, p, v_b), (3, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                # This would work if interpreted as Mv: (3)(7,7)\(3)(7)
                gfl.lu_solve(xf, p, m_sb)


class TestSolveVal(TestMatsVecs):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    @utn.loop_test()
    def test_solve_returns_expected_values(self, sctype):
        arrays = [self.m_ss, self.m_sb, self.m_bs]
        m_ss, m_sb, m_bs = arrays
        hy.assume(np.all(utn.non_singular(m_ss)))

        self.pick_var_type(sctype)
        a = gfl.solve(m_ss, m_sb)
        with self.subTest(msg='solve'):
            self.assertArrayAllClose(m_ss @ a, m_sb)
        b = gfl.rsolve(m_bs, m_ss)
        with self.subTest(msg='rsolve'):
            self.assertArrayAllClose(b @ m_ss, m_bs)

    @utn.loop_test()
    def test_solve_lu_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bs]
        m_ss, m_sb, m_bs = arrays
        hy.assume(np.all(utn.non_singular(m_ss)))

        a0 = gfl.solve(m_ss, m_sb)
        a, xf, p = gfl.solve_lu(m_ss, m_sb)
        with self.subTest('solve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.lu_solve(xf, p, m_sb)
        with self.subTest('solve(lu)'):
            self.assertArrayAllClose(aa, a0)
        b = gfl.rlu_solve(m_bs, xf, p)
        with self.subTest('rsolve(lu)'):
            self.assertArrayAllClose(b @ m_ss, m_bs)

    @utn.loop_test()
    def test_rsolve_lu_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bs]
        m_ss, m_sb, m_bs = arrays
        hy.assume(np.all(utn.non_singular(m_ss)))

        a0 = gfl.rsolve(m_bs, m_ss)
        a, xf, p = gfl.rsolve_lu(m_bs, m_ss)
        with self.subTest('rsolve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.rlu_solve(m_bs, xf, p)
        with self.subTest('rsolve(rlu)'):
            self.assertArrayAllClose(aa, a0)
        b = gfl.lu_solve(xf, p, m_sb)
        with self.subTest('solve(rlu)'):
            self.assertArrayAllClose(m_ss @ b, m_sb)

    @errstate
    @utn.loop_test(msg='rank')
    def test_solve_raises_with_low_rank(self, sctype):
        self.pick_var_type(sctype)
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.solve(self.ones_ss, self.ones_ss[...,:2])


# =============================================================================
if __name__ == '__main__':
    utn.main(verbosity=2)
