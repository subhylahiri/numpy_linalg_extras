# -*- coding: utf-8 -*-
"""Test C-loop and BLAS ufuncs + qr
"""
import unittest
import numpy as np
import unittest_numpy as utn
import numpy_linalg.gufuncs._gufuncs_cloop as gfc
import numpy_linalg.gufuncs._gufuncs_blas as gfb

errstate = utn.errstate(invalid='raise')
# =============================================================================

# =============================================================================
# %% Test BLAS ufuncs
# =============================================================================


class TestBlas(utn.TestCaseNumpy):
    """Testing norm, matmul and rmatmul"""

    def setUp(self):
        super().setUp()
        self.gf = gfb
        self.sctype.append('i')
        self.varnames = ['n', 'u', 'v', 'w', 'x', 'y', 'z']
        self._n = {}
        self._w = {}
        self._u = {}
        self._v = {}
        self._x = {}
        self._y = {}
        self._z = {}
        for sctype in self.sctype:
            self._u[sctype] = utn.randn_asa((3,), sctype)
            self._v[sctype] = utn.randn_asa((5,), sctype)
            self._x[sctype] = utn.randn_asa((2, 3, 5), sctype)
            self._y[sctype] = utn.randn_asa((5, 2), sctype)
            self._z[sctype] = self._x[sctype] @ self._y[sctype]
            self._w[sctype] = utn.asa(np.arange(24).reshape((2, 3, 4)),
                                      np.arange(8).reshape((2, 1, 4)), sctype)
            nsq = utn.asa(np.array([[14, 126, 366], [734, 1230, 1854]]),
                          np.array([[14], [126]]), sctype)
            self._n[sctype] = np.sqrt(nsq.real + nsq.imag)

    def test_norm_shape(self):
        """Check that norm gufunc returns arrays with the expected shape
        """
        self.pick_var_type('d')
        self.assertArrayShaped(self.gf.norm(self.x), (2, 3))
        self.assertArrayShaped(self.gf.norm(self.x, axis=1), (2, 5))
        self.assertArrayShaped(self.gf.norm(self.x, keepdims=True), (2, 3, 1))

    @utn.loop_test(msg='norm val', attr_inds=slice(-1))
    def test_norm_val(self, sctype: str):
        """Check that norm gufunc returns the expected values
        """
        self.pick_var_type('d')
        nout = np.empty((2, 3), dtype=sctype.lower())
        n = self.gf.norm(self.w, out=nout)
        self.assertArrayAllClose(n, self.n)
        self.assertArrayAllClose(n, nout)

    def test_matmul_shape(self):
        """Check that matmul gufunc returns arrays with the expected shape
        """
        self.pick_var_type('d')
        self.assertArrayShaped(self.gf.matmul(self.x, self.y), (2, 3, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(self.y, self.x)

    @unittest.expectedFailure
    def test_matmul_shape_flexible_signature(self):
        """Check that matmul gufunc deals with vectors correctly
        """
        self.assertArrayShaped(self.gf.matmul(self.x, self.v), (2, 3))
        self.assertArrayShaped(self.gf.matmul(self.u, self.x), (2, 5))
        self.assertArrayShaped(self.gf.matmul(self.u, self.u), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(self.x, self.u)
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(self.v, self.x)
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(self.v, self.u)

    @utn.loop_test(msg='matmul val')
    def test_matmul_val(self, sctype):
        """Check that matmul gufunc returns the expected values
        """
        self.pick_var_type(sctype)
        zout = np.empty((2, 3, 2), sctype)
        z = self.gf.matmul(self.x, self.y, out=zout)
        self.assertArrayAllClose(z, self.z)

    def test_rmatmul_shape(self):
        """Check that rmatmul gufunc returns arrays with the expected shape
        """
        self.pick_var_type('d')
        self.assertArrayShaped(self.gf.rmatmul(self.y, self.x), (2, 3, 2))
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(self.x, self.y)

    @unittest.expectedFailure
    def test_rmatmul_shape_flexible_signature(self):
        """Check that rmatmul gufunc deals with vectors correctly
        """
        self.assertArrayShaped(self.gf.rmatmul(self.x, self.u), (2, 3))
        self.assertArrayShaped(self.gf.rmatmul(self.v, self.x), (2, 5))
        self.assertArrayShaped(self.gf.rmatmul(self.u, self.u), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(self.x, self.v)
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(self.u, self.x)
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(self.v, self.u)

    @utn.loop_test(msg='rmatmul val')
    def test_rmatmul_val(self, sctype):
        """Check that rmatmul gufunc returns the expected values
        """
        self.pick_var_type(sctype)
        zout = np.empty((2, 3, 2), sctype)
        z = self.gf.rmatmul(self.y, self.x, out=zout)
        self.assertArrayAllClose(z, self.z)


# =============================================================================
# %% Test cloop ufuncs
# =============================================================================


class TestCloop(TestBlas):
    """Testing norm, matmul, rmatmul and rtrue_tdivide
    """

    def setUp(self):
        super().setUp()
        self.gf = gfc

    def test_rdiv_shape(self):
        """Check that rtrue_divide returns arrays with the expected shape
        """
        self.pick_var_type('d')
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gf.rtrue_divide(self.x, self.z)

    @utn.loop_test(msg="x \\ y == y / x. ", attr_inds=slice(-1,))
    def test_rdiv_val(self, sctype):
        """Check that rtrue_divide returns the expected values
        """
        self.pick_var_type(sctype)
        zout = np.empty((2, 3, 5), sctype)
        x = self.x
        x[np.abs(x) < 1e-5] += 1.
        y = self.y.T[:, None]
        z = self.gf.rtrue_divide(x, y, out=zout)
        zz = y / x
        self.assertArrayAllClose(z, zz)
        self.assertArrayNotAllClose(z, x / y, msg='x \\ y != x / y')


# =============================================================================
if __name__ == '__main__':
    utn.main(verbosity=2)
