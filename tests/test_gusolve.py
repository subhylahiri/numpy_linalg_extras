# -*- coding: utf-8 -*-
"""Test solve & lu families of gufuncs
"""
import unittest
import numpy as np
import unittest_numpy as utn
import numpy_linalg.gufuncs._gufuncs_lu_solve as gfl
from numpy_linalg import transpose

errstate = utn.errstate(invalid='raise')
# =============================================================================
# %% Test LU
# =============================================================================


class TestLU(utn.TestCaseNumpy):
    """Testing LU decomposition"""

    def setUp(self):
        super().setUp()
        self.varnames = ['square', 'tall', 'wide', 'id_small']
        self._square = {}
        self._tall = {}
        self._wide = {}
        self._id_small = {}
        for sctype in self.sctype:
            self._square[sctype] = utn.randn_asa((2, 5, 5), sctype)
            self._wide[sctype] = utn.randn_asa((3, 1, 3, 6), sctype)
            self._tall[sctype] = utn.randn_asa((5, 2), sctype)
            self._id_small[sctype] = np.eye(5, dtype=sctype)

    def test_lu_basic_shape(self):
        """Test shape of basic LU"""
        self.pick_var_type('d')
        sq_l, sq_u, sq_ip = gfl.lu_m(self.square)
        with self.subTest(msg="square"):
            self.assertEqual(sq_l.shape, (2, 5, 5))
            self.assertEqual(sq_u.shape, (2, 5, 5))
            self.assertEqual(sq_ip.shape, (2, 5))
        wd_l, wd_u, wd_ip = gfl.lu_m(self.wide)
        with self.subTest(msg="wide"):
            self.assertEqual(wd_l.shape, (3, 1, 3, 3))
            self.assertEqual(wd_u.shape, (3, 1, 3, 6))
            self.assertEqual(wd_ip.shape, (3, 1, 3))
        tl_l, tl_u, tl_ip = gfl.lu_n(self.tall)
        with self.subTest(msg="tall"):
            self.assertEqual(tl_l.shape, (5, 2))
            self.assertEqual(tl_u.shape, (2, 2))
            self.assertEqual(tl_ip.shape, (2,))

    def test_lu_raw_shape(self):
        """Test shape of raw LU"""
        self.pick_var_type('d')
        sq_f, sq_ip = gfl.lu_rawm(self.square)
        with self.subTest(msg="square"):
            self.assertEqual(sq_f.shape, (2, 5, 5))
            self.assertEqual(sq_ip.shape, (2, 5))
        wd_f, wd_ip = gfl.lu_rawm(self.wide)
        with self.subTest(msg="wide"):
            self.assertEqual(wd_f.shape, (3, 1, 6, 3))
            self.assertEqual(wd_ip.shape, (3, 1, 3))
        tl_f, tl_ip = gfl.lu_rawn(self.tall)
        with self.subTest(msg="tall"):
            self.assertEqual(tl_f.shape, (2, 5))
            self.assertEqual(tl_ip.shape, (2,))

    @utn.loop_test()
    def test_lu_basic_val(self, sctype):
        """Test values of basic LU"""
        self.pick_var_type(sctype)
        sq_l, sq_u, sq_ip = gfl.lu_m(self.square)
        sq = gfl.rpivot(sq_l @ sq_u, sq_ip)
        sqp = gfl.pivot(self.square, sq_ip)
        dinds = (...,) + np.diag_indices(5, 2)
        linds = (...,) + np.tril_indices(5, -1)
        uinds = (...,) + np.triu_indices(5, 1)
        with self.subTest(msg="square"):
            self.assertArrayAllClose(sq_l[dinds], 1.)
            self.assertArrayAllClose(sq_l[uinds], 0.)
            self.assertArrayAllClose(sq_u[linds], 0.)
            self.assertArrayAllClose(sq_l @ sq_u, sqp)
            self.assertArrayAllClose(sq, self.square)
        wd_l, wd_u, wd_ip = gfl.lu_m(self.wide)
        wd = gfl.rpivot(wd_l @ wd_u, wd_ip)
        wdp = gfl.pivot(self.wide, wd_ip)
        dinds = (...,) + np.diag_indices(3, 2)
        linds = (...,) + np.tril_indices(3, -1)
        uinds = (...,) + np.triu_indices(3, 1)
        with self.subTest(msg="wide"):
            self.assertArrayAllClose(wd_l[dinds], 1.)
            self.assertArrayAllClose(wd_l[uinds], 0.)
            self.assertArrayAllClose(wd_u[linds], 0.)
            self.assertArrayAllClose(wd_l @ wd_u, wdp)
            self.assertArrayAllClose(wd, self.wide)
        tl_l, tl_u, tl_ip = gfl.lu_n(self.tall)
        tl = gfl.rpivot(tl_l @ tl_u, tl_ip)
        tlp = gfl.pivot(self.tall, tl_ip)
        dinds = (...,) + np.diag_indices(2, 2)
        linds = (...,) + np.tril_indices(2, -1)
        uinds = (...,) + np.triu_indices(2, 1)
        with self.subTest(msg="tall"):
            self.assertArrayAllClose(tl_l[dinds], 1.)
            self.assertArrayAllClose(tl_l[uinds], 0.)
            self.assertArrayAllClose(tl_u[linds], 0.)
            self.assertArrayAllClose(tl_l @ tl_u, tlp)
            self.assertArrayAllClose(tl, self.tall)

    @utn.loop_test()
    def test_lu_raw_val(self, sctype):
        """Test values of raw LU"""
        self.pick_var_type(sctype)
        sq_l, sq_u, sq_ip0 = gfl.lu_m(self.square)
        sq_f, sq_ip = gfl.lu_rawm(self.square)
        sq_f = transpose(sq_f)
        linds = (...,) + np.tril_indices(5, -1)
        uinds = (...,) + np.triu_indices(5, 0)
        with self.subTest(msg="square"):
            self.assertArrayAllClose(sq_f[linds], sq_l[linds])
            self.assertArrayAllClose(sq_f[uinds], sq_u[uinds])
            self.assertEqual(sq_ip, sq_ip0)
        wd_l, wd_u, wd_ip0 = gfl.lu_m(self.wide)
        wd_f, wd_ip = gfl.lu_rawm(self.wide)
        wd_f = transpose(wd_f)
        linds = (...,) + np.tril_indices(3, -1, 6)
        uinds = (...,) + np.triu_indices(3, 0, 6)
        with self.subTest(msg="wide"):
            self.assertArrayAllClose(wd_f[linds], wd_l[linds])
            self.assertArrayAllClose(wd_f[uinds], wd_u[uinds])
            self.assertEqual(wd_ip, wd_ip0)
        tl_l, tl_u, tl_ip0 = gfl.lu_n(self.tall)
        tl_f, tl_ip = gfl.lu_rawn(self.tall)
        tl_f = transpose(tl_f)
        linds = (...,) + np.tril_indices(5, -1, 2)
        uinds = (...,) + np.triu_indices(5, 0, 2)
        with self.subTest(msg="tall"):
            self.assertArrayAllClose(tl_f[linds], tl_l[linds])
            self.assertArrayAllClose(tl_f[uinds], tl_u[uinds])
            self.assertEqual(tl_ip, tl_ip0)

    def test_inv_shape(self):
        """Check that inv gufuncs all return arrays with the expected shape
        """
        self.pick_var_type('d')
        with self.subTest(msg='inv'):
            square_i = gfl.inv(self.square)
            self.assertEqual(square_i.shape, (2, 5, 5))
        with self.subTest(msg='inv,+lu'):
            square_i, square_f, square_ip = gfl.inv_lu(self.square)
            self.assertEqual(square_i.shape, (2, 5, 5))
            self.assertEqual(square_f.shape, (2, 5, 5))
            self.assertEqual(square_ip.shape, (2, 5))
        with self.subTest(msg='inv,-lu'):
            square_i = gfl.lu_inv(square_f, square_ip)
            self.assertEqual(square_i.shape, (2, 5, 5))

    @utn.loop_test()
    def test_inv_val(self, sctype):
        """Check that inv gufuncs all return arrays with the expected values
        """
        self.pick_var_type(sctype)
        with self.subTest(msg='inv'):
            square_i = gfl.inv(self.square)
            self.assertArrayAllClose(square_i @ self.square,
                                     self.id_small)
            self.assertArrayAllClose(self.square @ square_i,
                                     self.id_small)
        with self.subTest(msg='inv,+lu'):
            square_if, square_f, square_ip = gfl.inv_lu(self.square)
            luf, ip = gfl.lu_rawn(self.square)
            self.assertArrayAllClose(square_if, square_i)
            self.assertArrayAllClose(square_f, luf)
            self.assertArrayAllClose(square_ip, ip)
        with self.subTest(msg='inv,-lu'):
            square_fi = gfl.lu_inv(square_f, square_ip)
            self.assertArrayAllClose(square_fi, square_i)


# =============================================================================
# %% Test solve
# =============================================================================


class TestSolve(utn.TestCaseNumpy):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    def setUp(self):
        super().setUp()
        self.varnames = ['x', 'y', 'z', 'w', 'v', 'yt', 'ones']
        self._x = {}
        self._y = {}
        self._z = {}
        self._w = {}
        self._v = {}
        self._yt = {}
        self._ones = {}
        for sctype in self.sctype:
            self._x[sctype] = utn.randn_asa((2, 5, 5), sctype)
            self._y[sctype] = utn.randn_asa((5, 2), sctype)
            self._z[sctype] = utn.randn_asa((3, 1, 5, 4), sctype)
            self._w[sctype] = utn.randn_asa((3, 1, 1, 5), sctype)
            self._v[sctype] = utn.randn_asa((4, 5), sctype)
            self._yt[sctype] = transpose(self._y[sctype])
            self._ones[sctype] = utn.ones_asa((5, 3), sctype)


class TestSolveShape(TestSolve):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    def test_solve_shape(self):
        """Check if solve, rsolve return arrays with the expected shape
        """
        self.pick_var_type('d')
        a = gfl.solve(self.x, self.y)
        self.assertEqual(a.shape, (2, 5, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(self.x, self.yt)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.solve(self.yt, self.x)
        b = gfl.rsolve(self.yt, self.x)
        self.assertEqual(b.shape, (2, 2, 5))

    def test_solvelu_shape(self):
        """Check if solve_lu, lu_solve return arrays with the expected shape
        """
        self.pick_var_type('d')
        a, xf, p = gfl.solve_lu(self.x, self.y)
        self.assertArrayEqual(a.shape, (2, 5, 2))
        self.assertArrayEqual(xf.shape, (2, 5, 5))
        self.assertArrayEqual(p.shape, (2, 5))
        b = gfl.lu_solve(xf, p, self.z)
        self.assertArrayEqual(b.shape, (3, 2, 5, 4))
        c = gfl.rlu_solve(self.w, xf, p)
        self.assertArrayEqual(c.shape, (3, 2, 1, 5))

    def test_rsolvelu_shape(self):
        """Check if rsolve_lu, rlu_solve return arrays with the expected shape
        """
        self.pick_var_type('d')
        a, xf, p = gfl.rsolve_lu(self.w, self.x)
        self.assertArrayEqual(a.shape, (3, 2, 1, 5))
        self.assertArrayEqual(xf.shape, (3, 2, 5, 5))
        self.assertArrayEqual(p.shape, (3, 2, 5))
        b = gfl.rlu_solve(self.v, xf, p)
        self.assertArrayEqual(b.shape, (3, 2, 4, 5))


class TestSolveVal(TestSolve):
    """Testing (r)solve, (r)solve_lu and (r)lu_solve"""

    @utn.loop_test()
    def test_solve_val(self, sctype):
        """Check if solve, rsolve return the expected values
        """
        self.pick_var_type(sctype)
        a = gfl.solve(self.x, self.y)
        with self.subTest(msg='solve'):
            self.assertArrayAllClose(self.x @ a, self.y)
        b = gfl.rsolve(self.v, self.x)
        with self.subTest(msg='rsolve'):
            self.assertArrayAllClose(b @ self.x, self.v)

    @utn.loop_test()
    def test_solvelu_val(self, sctype):
        """Check if solve_lu, lu_solve, rlu_solve return the expected values
        """
        self.pick_var_type(sctype)
        a0 = gfl.solve(self.x, self.y)
        a, xf, p = gfl.solve_lu(self.x, self.y)
        with self.subTest('solve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.lu_solve(xf, p, self.y)
        with self.subTest('solve(lu)'):
            self.assertArrayAllClose(aa, a0)
        b = gfl.rlu_solve(self.v, xf, p)
        with self.subTest('rsolve(lu)'):
            self.assertArrayAllClose(b @ self.x, self.v)

    @utn.loop_test()
    def test_rsolvelu_val(self, sctype):
        """Check if rsolve_lu, lu_solve, rlu_solve return the expected values
        """
        self.pick_var_type(sctype)
        a0 = gfl.rsolve(self.w, self.x)
        a, xf, p = gfl.rsolve_lu(self.w, self.x)
        with self.subTest('rsolve0'):
            self.assertArrayAllClose(a, a0)
        aa = gfl.rlu_solve(self.w, xf, p)
        with self.subTest('rsolve(rlu)'):
            self.assertArrayAllClose(aa, a0)
        b = gfl.lu_solve(xf, p, self.z)
        with self.subTest('solve(rlu)'):
            self.assertArrayAllClose(self.x @ b, self.z)

    @unittest.expectedFailure
    @errstate
    @utn.loop_test(msg='rank')
    def test_rank(self, sctype):
        """Check if solve raises an exception when divisor is rank deficient
        """
        self.pick_var_type(sctype)
        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.solve(self.ones, self.z)


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
