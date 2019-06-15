# -*- coding: utf-8 -*-
"""Test lnarray class
"""
import numpy as np
import numpy_linalg as la
import numpy_linalg.gufuncs as gf
import unittest_numpy as utn
from test_gufunc import TestMatsVecs

# =============================================================================
__all__ = ['TestArray', 'TestPinvarray']
# =============================================================================
# %% Test python classes
# =============================================================================


class TestNewClasses(TestMatsVecs):
    """Testing lnarray, pinvarray, etc"""

    def setUp(self):
        super().setUp()
        self.varnames = ['u', 'v', 'w', 'x', 'y', 'z']
        self._u = {}
        self._v = {}
        self._w = {}
        self._x = {}
        self._y = {}
        self._z = {}
        for sctype in self.sctype:
            self._u[sctype] = utn.randn_asa((7, 5), sctype).view(la.lnarray)
            self._v[sctype] = utn.randn_asa((5, 2), sctype)
            self._w[sctype] = utn.randn_asa((3, 3), sctype).view(la.lnarray)
            self._x[sctype] = utn.randn_asa((2, 5, 3), sctype).view(la.lnarray)
            self._y[sctype] = utn.randn_asa((3, 5), sctype)
            self._z[sctype] = utn.randn_asa((3,), sctype)


class TestArray(TestNewClasses):
    """Testing lnarray"""

    def setUp(self):
        self.sctype = ['i']
        super().setUp()

    def test_array_type(self):
        """Check that functions & operators return the correct type
        """
        self.pick_var_type('d')
        v, w, x, y = self.v, self.w, self.x, self.y
        self.assertIsInstance(x @ y, la.lnarray)
        self.assertIsInstance(y @ x, la.lnarray)
        xyout = np.empty((2, 5, 5), 'd')
        xy = la.matmul(x, y, xyout)
        self.assertIsInstance(xy, np.ndarray)
        self.assertIsInstance(xyout, np.ndarray)
        self.assertIsInstance(np.matmul(x, y), np.ndarray)
        self.assertIsInstance(la.solve(w, y), la.lnarray)
        self.assertIsInstance(np.linalg.solve(w, y), np.ndarray)
        self.assertIsInstance(la.lstsq(x, v), la.lnarray)
        self.assertIsInstance(np.linalg.lstsq(x[0], v, rcond=None)[0],
                              np.ndarray)
        self.assertIsInstance(la.lu(w)[0], la.lnarray)
        self.assertIsInstance(la.lu(v)[0], np.ndarray)
        self.assertIsInstance(la.qr(w)[0], la.lnarray)
        self.assertIsInstance(la.qr(v)[0], np.ndarray)
        self.assertIsInstance(la.lq(w)[0], la.lnarray)
        self.assertIsInstance(la.lq(v)[0], np.ndarray)
        self.assertIsInstance(la.lqr(w)[0], la.lnarray)
        self.assertIsInstance(la.lqr(v)[0], np.ndarray)
        self.assertIsInstance(np.linalg.qr(w)[0], np.ndarray)

    def test_array_shape(self):
        """Check that shape manipulation properties & methods work
        """
        self.pick_var_type('D')
        w, x = self.w, self.x
        self.assertArrayShape(x.t, (2, 3, 5))
        self.assertArrayShape(x.h, (2, 3, 5))
        self.assertArrayNotAllClose(x.t, x.h)
        self.assertArrayShape(w.c, (3, 3, 1))
        self.assertArrayShape(x.c.uc, (2, 5, 3))
        self.assertArrayShape(w.r, (3, 1, 3))
        self.assertArrayShape(x.r.ur, (2, 5, 3))
        self.assertArrayShape(w.s, (3, 3, 1, 1))
        self.assertArrayShape(x.s.us, (2, 5, 3))
        self.assertArrayShape(w.expand_dims(1, 3), (3, 1, 3, 1))
        self.assertArrayShape((x.s * w).flattish(1, 4), (2, 45, 3))
        with self.assertRaisesRegex(ValueError, "repeated axes"):
            x.expand_dims(2, -3)
        with self.assertRaises(ValueError):
            (x.s * w).flattish(3, -3)

    @utn.loop_test(attr_inds=slice(4))
    def test_array_value(self, sctype):
        """Check that operators and functions return the correct value
        """
        self.pick_var_type(sctype)
        w, x, z = self.w, self.x, self.z
        xwout = np.empty((2, 5, 3), sctype)
        xw = la.matmul(x, w, xwout)
        self.assertArrayAllClose(xw, xwout)
        self.assertArrayAllClose(x @ w, np.matmul(x, w))
        self.assertArrayAllClose(x @ w, np.matmul(x, w))
        self.assertArrayAllClose(x @ z, np.matmul(x, z))
        self.assertArrayAllClose(gf.solve(w, z), np.linalg.solve(w, z))
        self.assertArrayAllClose(gf.lstsq(x.t[0], z),
                                 np.linalg.lstsq(x[0].t, z, rcond=None)[0])
        self.assertArrayAllClose(gf.rmatmul(w, x), np.matmul(x, w))
        x @= w
        self.assertArrayAllClose(xw, x)


class TestPinvarray(TestNewClasses):
    """test pinvarray & invarray classes
    """

    def test_pinv_type(self):
        """test type attributes
        """
        self.pick_var_type('D')
        w = self.w
        self.assertIsInstance(w.pinv, la.pinvarray)
        self.assertIsInstance(w.inv, la.invarray)
        self.assertIs(w.pinv.dtype, np.dtype('D'))
        self.assertIsInstance(w.pinv.pinv, la.lnarray)
        self.assertIsInstance(w.inv.inv, la.lnarray)
        self.assertIsInstance(w.pinv(), la.lnarray)
        self.assertIsInstance(w.inv(), la.lnarray)
        self.pick_var_type('d')
        p = la.pinvarray(self.v)
        self.assertIsInstance(p, la.pinvarray)
        self.assertIsInstance(p.pinv, la.lnarray)
        self.assertIsInstance(2 * p, la.pinvarray)
        self.assertIsInstance((2 * p).pinv, la.lnarray)
        pout = la.pinvarray(la.empty_like(self.v))
        np.multiply(2, p, pout)
        self.assertIsInstance(pout, la.pinvarray)
        self.assertIsInstance(pout.pinv, la.lnarray)
        with self.assertRaises(AttributeError):
            p.inv
        with self.assertRaises(TypeError):
            w.inv.pinv

    def test_pinv_shape(self):
        """test shape attributes
        """
        self.pick_var_type('d')
        xp = self.x.pinv
        self.assertEqual(xp.ndim, 3)
        self.assertEqual(xp.shape, (2, 3, 5))
        self.assertEqual(xp.size, 30)
        self.assertArrayShape(xp(), (2, 3, 5))
        with self.assertRaises(ValueError):
            self.x.inv
        xp = self.x.c.pinv
        self.assertArrayShape(xp.swapaxes(0, 1), (5, 2, 1, 3))
        self.assertArrayShape(xp.swapaxes(0, 2), (1, 5, 2, 3))
        self.assertArrayShape(xp.swapaxes(-1, -2), (2, 5, 3, 1))

    @utn.loop_test()
    def test_pinv_funcs(self, sctype):
        """test pinvarray behaviour in gufuncs
        """
        self.pick_var_type(sctype)
        u, v, x = self.u, self.v, self.x
        self.assertArrayAllClose(gf.matmul(x.pinv, v), gf.lstsq(x, v))
        self.assertArrayAllClose(gf.matmul(u, x.pinv.t), gf.rlstsq(u, x.t))
        xpout = la.pinvarray(la.empty_like(x))
        xp = np.multiply(x.pinv, 2, out=xpout)
        self.assertArrayAllClose(xp.pinv, xpout.pinv)
        self.assertArrayAllClose(xp.pinv, x / 2)
        with self.assertRaises(TypeError):
            gf.matmul(x.pinv, u.pinv)
        self.assertArrayAllClose(gf.lstsq(u.pinv, v), gf.matmul(u, v))
        with self.assertRaises(TypeError):
            gf.lstsq(v, u.pinv)
        self.assertArrayAllClose(gf.rlstsq(v.T, u.t.pinv), gf.matmul(v.T, u.t))
        with self.assertRaises(TypeError):
            gf.rlstsq(u.t.pinv, v.T)
        with self.assertRaises(TypeError):
            gf.rmatmul(u.pinv, x.pinv)
        with self.assertRaises(TypeError):
            gf.solve(u.pinv, v)
        with self.assertRaises(TypeError):
            gf.rsolve(u, x.pinv)

    @utn.loop_test()
    def test_inv_funcs(self, sctype):
        """test invarray behaviour in gufuncs
        """
        self.pick_var_type(sctype)
        w, x, y = self.w, self.x, self.y
        xw = x[:, :3]
        self.assertArrayAllClose(gf.matmul(w.inv, y), gf.solve(w, y))
        self.assertArrayAllClose(gf.matmul(x, w.inv), gf.rsolve(x, w))
        self.assertArrayAllClose(gf.matmul(w.inv, xw.inv).inv, xw @ w)
        self.assertArrayAllClose(gf.solve(w.inv, y), gf.matmul(w, y))
        self.assertArrayAllClose(gf.solve(xw, w.inv).inv, gf.matmul(w, xw))
        self.assertArrayAllClose(gf.solve(xw.inv, w.inv), gf.rsolve(xw, w))
        self.assertArrayAllClose(gf.rsolve(w, xw.inv), gf.matmul(w, xw))
        self.assertArrayAllClose(gf.rsolve(xw.inv, w).inv, gf.matmul(w, xw))
        self.assertArrayAllClose(gf.rsolve(xw.inv, w.inv), gf.solve(xw, w))
        self.assertArrayAllClose(gf.rmatmul(w, xw.inv), gf.solve(xw, w))
        self.assertArrayAllClose(gf.rmatmul(xw.inv, w), gf.rsolve(w, xw))
        self.assertArrayAllClose(gf.rmatmul(xw.inv, w.inv).inv, xw @ w)
        with self.assertRaises(TypeError):
            gf.lstsq(w.inv, y)
        with self.assertRaises(TypeError):
            gf.rlstsq(x, w.inv)

    @utn.loop_test()
    def test_pinv_ops(self, sctype):
        """test pinvarray behaviour in operators
        """
        self.pick_var_type(sctype)
        u, v, x = self.u, self.v, self.x
        y, vs = self.y, v.view(la.lnarray).s
        self.assertArrayAllClose(x.pinv @ v, gf.lstsq(x, v))
        self.assertArrayAllClose(x.pinv() @ v, gf.lstsq(x, v))
        self.assertArrayAllClose(u @ x.pinv.t, gf.rlstsq(u, x.t))
        with self.assertRaises(TypeError):
            x.pinv @ u.pinv
        self.assertArrayAllClose((x.pinv * 3.5).pinv, x / 3.5)
        self.assertArrayAllClose((2.4 * x.pinv).pinv, x / 2.4)
        self.assertArrayAllClose((x.pinv / 3.564).pinv, x * 3.564)
        with self.assertRaises(TypeError):
            65 / x.pinv
        self.assertArrayAllClose((x.pinv * vs).pinv, x / vs)
        self.assertArrayAllClose((vs * x.pinv).pinv, x / vs)
        self.assertArrayAllClose((x.pinv / vs).pinv, x * vs)
        with self.assertRaises(TypeError):
            vs / x.pinv
        with self.assertRaises(TypeError):
            vs.pinv * x.pinv
        with self.assertRaises(TypeError):
            x.pinv + y
        xold = 1. * x
        xp = x.pinv
        xp *= 2
        self.assertArrayAllClose(x, xold / 2)

    @utn.loop_test()
    def test_inv_ops(self, sctype):
        """test invarray behaviour in operators
        """
        self.pick_var_type(sctype)
        vs, w = self.v.view(la.lnarray).s, self.w
        x, y = self.x, self.y
        xw = x[:, :3]
        self.assertArrayAllClose(w.inv @ y, gf.solve(w, y))
        self.assertArrayAllClose(w.inv() @ y, gf.solve(w, y))
        self.assertArrayAllClose(x @ w.inv, gf.rsolve(x, w))
        self.assertArrayAllClose((w.inv @ xw.inv).inv, xw @ w)
        self.assertArrayAllClose((w.inv * 3.5).inv, w / 3.5)
        self.assertArrayAllClose((2.4 * w.inv).inv, w / 2.4)
        self.assertArrayAllClose((w.inv / 3.564).inv, w * 3.564)
        with self.assertRaises(TypeError):
            45.564 / w.inv
        self.assertArrayAllClose((xw.inv * vs).inv, xw / vs)
        self.assertArrayAllClose((vs * xw.inv).inv, xw / vs)
        self.assertArrayAllClose((xw.inv / vs).inv, xw * vs)
        with self.assertRaises(TypeError):
            vs / xw.inv
        with self.assertRaises(TypeError):
            vs.inv * xw.inv
        with self.assertRaises(TypeError):
            w + xw.inv
        xwold = 1. * xw
        xwi = xw.inv
        xwi @= w.inv
        self.assertArrayAllClose(xw, w @ xwold)


if __name__ == '__main__':
    utn.main(verbosity=2)
