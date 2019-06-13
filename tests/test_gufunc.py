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


class TestMatsVecs(utn.TestCaseNumpy):
    """Collection of vectors and matrices for tests"""

    def setUp(self):
        extra_sctypes = getattr(self, 'sctype', [])
        super().setUp()
        self.sctype.extend(extra_sctypes)
        # prefixes: v(ector), m(atrix, a(rray of matrices)
        # suffixes: s(mall), b(ig)
        self.varnames = ['v_s', 'v_b',
                         'm_bs', 'm_sb', 'm_ss', 'm_bb',
                         'a_bs', 'a_sb', 'a_ss', 'a_bb']
        self._v_s = {}
        self._v_b = {}
        self._m_bs = {}
        self._m_sb = {}
        self._m_ss = {}
        self._m_bb = {}
        self._a_bs = {}
        self._a_sb = {}
        self._a_ss = {}
        self._a_bb = {}
        for sctype in self.sctype:
            self._v_s[sctype] = utn.randn_asa((3,), sctype)
            self._v_b[sctype] = utn.randn_asa((7,), sctype)
            self._m_bs[sctype] = utn.randn_asa((7, 3), sctype)
            self._m_sb[sctype] = utn.randn_asa((3, 7), sctype)
            self._m_ss[sctype] = utn.randn_asa((3, 3), sctype)
            self._m_bb[sctype] = utn.randn_asa((7, 7), sctype)
            self._a_bs[sctype] = utn.randn_asa((2, 7, 3), sctype)
            self._a_sb[sctype] = utn.randn_asa((4, 1, 3, 7), sctype)
            self._a_ss[sctype] = utn.randn_asa((5, 1, 3, 3), sctype)
            self._a_bb[sctype] = utn.randn_asa((3, 7, 7), sctype)


class TestBlas(TestMatsVecs):
    """Testing norm, matmul and rmatmul"""

    def setUp(self):
        self.gf = gfb
        self.sctype = ['i']
        super().setUp()
        self.varnames += ['nrms', 'vecs', 'prod']
        self._nrms = {}
        self._vecs = {}
        self._prod = {}
        for sctype in self.sctype:
            self._prod[sctype] = self._a_sb[sctype] @ self._a_bs[sctype]
            self._vecs[sctype] = utn.asa(np.arange(24).reshape((2, 3, 4)),
                                         np.arange(8).reshape((2, 1, 4)),
                                         sctype)
            nsq = utn.asa(np.array([[14, 126, 366], [734, 1230, 1854]]),
                          np.array([[14], [126]]), sctype)
            self._nrms[sctype] = np.sqrt(nsq.real + nsq.imag)

    def test_norm_shape(self):
        """Check that norm gufunc returns arrays with the expected shape
        """
        self.pick_var_type('d')
        self.assertArrayShape(self.gf.norm(self.a_bs), (2, 7))
        self.assertArrayShape(self.gf.norm(self.a_bs, axis=1), (2, 3))
        self.assertArrayShape(self.gf.norm(self.v_b, keepdims=True), (1,))
        self.assertArrayShape(self.gf.norm(self.v_b), ())

    @utn.loop_test(msg='norm val', attr_inds=slice(-1))
    def test_norm_val(self, sctype: str):
        """Check that norm gufunc returns the expected values
        """
        self.pick_var_type('d')
        nout = np.empty((2, 3), dtype=sctype.lower())
        nrms = self.gf.norm(self.vecs, out=nout)
        self.assertArrayAllClose(nrms, self.nrms)
        self.assertArrayAllClose(nout, self.nrms)

    def test_matmul_shape(self):
        """Check that matmul gufunc returns arrays with the expected shape
        """
        self.pick_var_type('d')
        self.assertArrayShape(self.gf.matmul(self.m_sb, self.m_bs), (3, 3))
        self.assertArrayShape(self.gf.matmul(self.a_bs, self.m_ss), (2, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.matmul(self.m_bs, self.m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gf.matmul(self.a_ss, self.a_sb)

    @unittest.expectedFailure
    def test_matmul_shape_flexible_signature(self):
        """Check that matmul gufunc deals with vectors correctly
        """
        with self.subTest('matrix-vector'):
            self.assertArrayShape(self.gf.matmul(self.m_sb, self.v_b), (3,))
            self.assertArrayShape(self.gf.matmul(self.a_bs, self.v_s), (2, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                self.gf.matmul(self.m_sb, self.v_s)
            with self.assertRaisesRegex(*utn.core_dim_err):
                self.gf.matmul(self.a_bb, self.m_sb)  # Mv would work, but MM
        with self.subTest('vector-matrix'):
            self.assertArrayShape(self.gf.matmul(self.v_s, self.m_ss), (3,))
            self.assertArrayShape(self.gf.matmul(self.v_b, self.a_bb), (3, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                self.gf.matmul(self.v_b, self.m_sb)
            self.assertArrayShape(self.gf.matmul(self.m_sb, self.a_bs),
                                  (2, 3, 3))  # Mv wouldn't work, but MM ok
        with self.subTest('vector-vector'):
            self.assertArrayShape(self.gf.matmul(self.v_s, self.v_s), ())
            with self.assertRaisesRegex(*utn.core_dim_err):
                self.gf.matmul(self.v_s, self.v_b)

    @utn.loop_test(msg='matmul val')
    def test_matmul_val(self, sctype):
        """Check that matmul gufunc returns the expected values
        """
        self.pick_var_type(sctype)
        pout = np.empty((4, 2, 3, 3), sctype)
        prod = self.gf.matmul(self.a_sb, self.a_bs, out=pout)
        self.assertArrayAllClose(prod, self.prod)
        self.assertArrayAllClose(pout, self.prod)

    def test_rmatmul_shape(self):
        """Check that rmatmul gufunc returns arrays with the expected shape
        """
        self.pick_var_type('d')
        self.assertArrayShape(self.gf.rmatmul(self.m_sb, self.m_bs), (7, 7))
        self.assertArrayShape(self.gf.rmatmul(self.a_bs, self.m_bb), (2, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            self.gf.rmatmul(self.m_bs, self.m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gf.rmatmul(self.a_sb, self.a_ss)

    @unittest.expectedFailure
    def test_rmatmul_shape_flexible_signature(self):
        """Check that rmatmul gufunc deals with vectors correctly
        """
        with self.subTest('matrix-vector'):
            self.assertArrayShape(self.gf.rmatmul(self.v_s, self.m_bs), (7,))
            self.assertArrayShape(self.gf.rmatmul(self.v_b, self.a_sb),
                                  (4, 1, 3))
            with self.assertRaisesRegex(*utn.core_dim_err):
                self.gf.rmatmul(self.v_b, self.m_bs)
            with self.assertRaisesRegex(*utn.core_dim_err):
                self.gf.rmatmul(self.m_sb, self.a_bb)  # Mv would work, but MM
        with self.subTest('vector-matrix'):
            self.assertArrayShape(self.gf.rmatmul(self.m_sb, self.v_s), (7,))
            self.assertArrayShape(self.gf.rmatmul(self.a_sb, self.v_s),
                                  (4, 1, 7))
            with self.assertRaisesRegex(*utn.core_dim_err):
                self.gf.rmatmul(self.m_bs, self.v_s)
            self.assertArrayShape(self.gf.rmatmul(self.a_bs, self.m_bb),
                                  (2, 7, 3))  # Mv wouldn't work, but MM ok
        with self.subTest('vector-vector'):
            self.assertArrayShape(self.gf.rmatmul(self.v_b, self.v_b), ())
            with self.assertRaisesRegex(*utn.core_dim_err):
                self.gf.rmatmul(self.v_b, self.v_s)

    @utn.loop_test(msg='rmatmul val')
    def test_rmatmul_val(self, sctype):
        """Check that rmatmul gufunc returns the expected values
        """
        self.pick_var_type(sctype)
        pout = np.empty((4, 2, 3, 3), sctype)
        prod = self.gf.rmatmul(self.a_bs, self.a_sb, out=pout)
        self.assertArrayAllClose(prod, self.prod)
        self.assertArrayAllClose(pout, self.prod)


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
        self.assertArrayShape(self.gf.rtrue_divide(self.m_sb, self.a_sb),
                              (4, 1, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            self.gf.rtrue_divide(self.a_sb, self.prod)

    @utn.loop_test(msg="x \\ y == y / x. ", attr_inds=slice(-1,))
    def test_rdiv_val(self, sctype):
        """Check that rtrue_divide returns the expected values
        """
        self.pick_var_type(sctype)
        zout = np.empty((4, 1, 3, 7), sctype)
        x = self.m_bs.T
        x[np.abs(x) < 1e-5] += 1.
        y = self.a_sb
        z = self.gf.rtrue_divide(x, y, out=zout)
        zz = y / x
        self.assertArrayAllClose(z, zz)
        self.assertArrayNotAllClose(z, x / y, msg='x \\ y != x / y')


# =============================================================================
if __name__ == '__main__':
    utn.main(verbosity=2)
