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
        self.x = {}
        self.y = {}
        self.z = {}
        self.w = {}
        self.n = {}
        for sctype in self.sctype:
            self.x[sctype] = utn.randn_asa((2, 3, 5), sctype)
            self.y[sctype] = utn.randn_asa((5, 2), sctype)
            self.z[sctype] = self.x[sctype] @ self.y[sctype]
            self.w[sctype] = utn.asa(np.arange(24).reshape((2, 3, 4)),
                                     np.arange(8).reshape((2, 1, 4)), sctype)
            nsq = utn.asa(np.array([[14, 126, 366], [734, 1230, 1854]]),
                          np.array([[14], [126]]), sctype)
            self.n[sctype] = np.sqrt(nsq.real + nsq.imag)

    def test_norm_shape(self):
        """Check that norm returns arrays with the expected shape
        """
        self.assertEqual(self.gf.norm(self.x['d']).shape, (2, 3))
        self.assertEqual(self.gf.norm(self.x['d'], axis=1).shape, (2, 5))
        self.assertEqual(self.gf.norm(self.x['d'], keepdims=True).shape,
                         (2, 3, 1))

    @utn.loop_test(msg='norm val', attr_inds=slice(-1))
    def test_norm_val(self, sctype: str):
        """Check that norm returns the expected values
        """
        nout = np.empty((2, 3), dtype=sctype.lower())
        n = self.gf.norm(self.w[sctype], out=nout)
        self.assertArrayAllClose(n, self.n[sctype])
        self.assertArrayAllClose(n, nout)

    def test_matmul_shape(self):
        """Check that matmul returns arrays with the expected shape
        """
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(self.y['d'], self.x['d'])

    @utn.loop_test(msg='matmul val')
    def test_matmul_val(self, sctype):
        """Check that matmul returns the expected values
        """
        zout = np.empty((2, 3, 2), sctype)
        z = self.gf.matmul(self.x[sctype], self.y[sctype], out=zout)
        self.assertArrayAllClose(z, self.z[sctype])

    def test_rmatmul_shape(self):
        """Check that rmatmul returns arrays with the expected shape
        """
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(self.x['d'], self.y['d'])

    @utn.loop_test(msg='rmatmul val')
    def test_rmatmul_val(self, sctype):
        """Check that rmatmul returns the expected values
        """
        zout = np.empty((2, 3, 2), sctype)
        z = self.gf.rmatmul(self.y[sctype], self.x[sctype], out=zout)
        self.assertArrayAllClose(z, self.z[sctype])


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
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gf.rtrue_divide(self.x['d'], self.z['d'])

    @utn.loop_test(msg="x \\ y == y / x. ", attr_inds=slice(-1,))
    def test_rdiv_val(self, sctype):
        """Check that rtrue_divide returns the expected values
        """
        zout = np.empty((2, 3, 5), sctype)
        x = self.x[sctype]
        x[np.abs(x) < 1e-5] += 1.
        y = self.y[sctype].T[:, None]
        z = self.gf.rtrue_divide(x, y, out=zout)
        zz = y / x
        self.assertArrayAllClose(z, zz)
        self.assertArrayNotAllClose(z, x / y, msg='x \\ y != x / y')


# =============================================================================
if __name__ == '__main__':
    unittest.main(verbosity=2)
#    ut.main(testRunner=ut.TextTestRunner(resultclass=utn.TestResultNumpy))
