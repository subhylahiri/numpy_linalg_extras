# -*- coding: utf-8 -*-
"""Test qr & lstsq families of gufuncs
"""
import unittest
import numpy as np
import numpy_linalg.gufuncs._gufuncs_qr_lstsq as gfl
import numpy_linalg.gufuncs._gufuncs_blas as gfb
from numpy_linalg import transpose, dagger, row, col, scalar
if __name__.find('tests.') < 0:
    # pylint: disable=import-error
    from test_gufunc import utn, hn, main, TestMatsVecs
    from test_linalg import trnsp, drop, chop, grow
    from test_gusolve import make_off_by_one, make_bad_broadcast
else:
    from .test_gufunc import utn, hn, main, TestMatsVecs
    from .test_linalg import trnsp, drop, chop, grow
# pylint: disable=missing-function-docstring
errstate = np.errstate(invalid='raise')
# =============================================================================
__all__ = ['TestLstsqShape', 'TestLstsqVectors', 'TestLstsqVal']

# =============================================================================
# Test lstsq
# =============================================================================


class TestLstsq(TestMatsVecs):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    def setUp(self):
        self.func = [gfl.lstsq_qrm, gfl.lstsq_qrn,
                     gfl.rlstsq_qrm, gfl.rlstsq_qrn]
        self.tau_len = [{'bs': (7,), 'sb': (3,), 'vs': (3,), 'vb': (7,)},
                        {'bs': (3,), 'sb': (7,), 'vs': (), 'vb': ()},
                        {'bs': (3,), 'sb': (7,), 'vs': (3,), 'vb': (7,)},
                        {'bs': (7,), 'sb': (3,), 'vs': (), 'vb': ()}]
        super().setUp()


class TestLstsqShape(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @errstate
    def test_lstsq_returns_expected_shape(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest('overconstrained'):
        self.assertArrayShape(gfl.lstsq(m_bs, m_bb), (3, 7))
        self.assertArrayShape(gfl.lstsq(m_bs, m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.lstsq(m_bs, m_bb), (3, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.lstsq(m_bs, m_bb)
        # with self.subTest('underconstrained'):
        self.assertArrayShape(gfl.lstsq(m_sb, m_ss), (7, 3))
        self.assertArrayShape(
                            gfl.lstsq(m_sb, m_ss), (4, 1, 7, 3))
        self.assertArrayShape(
                            gfl.lstsq(m_sb, m_ss), (5, 1, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.lstsq(m_sb, m_ss)

    @errstate
    def test_rlstsq_returns_expected_shape(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # with self.subTest('underconstrained'):
        self.assertArrayShape(gfl.rlstsq(m_ss, m_bs), (3, 7))
        self.assertArrayShape(
                            gfl.rlstsq(m_ss, m_bs), (5, 1, 3, 7))
        self.assertArrayShape(gfl.rlstsq(m_ss, m_bs), (2, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rlstsq(m_ss, transpose(m_sb))
        # with self.subTest('overconstrained'):
        self.assertArrayShape(gfl.rlstsq(m_bb, m_sb), (7, 3))
        self.assertArrayShape(gfl.rlstsq(m_bb, m_sb), (3, 7, 3))
        self.assertArrayShape(
                            gfl.rlstsq(m_bb, m_sb), (4, 1, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.rlstsq(m_bb, transpose(m_bs))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_shape_tall(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['bs']
        self.assertArrayShapesAre(func(m_bs, m_bb),
                                  ((3, 7), (3, 7), tau))
        self.assertArrayShapesAre(func(m_bs, m_bb),
                                  ((2, 3, 7), (2, 3, 7), (2,) + tau))
        self.assertArrayShapesAre(func(m_bs, m_bb),
                                  ((3, 3, 7), (3, 3, 7), (3,) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(m_bs, m_bb)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_returns_expected_shape_tall(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_bs, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 7))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (3, 7))
        self.assertArrayShape(
            gfl.rqr_lstsq(m_ss, x_f, tau), (5, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        _, x_f, tau = func(m_bs, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (2, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, tau, m_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_shape_wide(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['sb']
        self.assertArrayShapesAre(func(m_sb, m_ss),
                                  ((7, 3), (7, 3), tau))
        self.assertArrayShapesAre(func(m_sb, m_ss), (
            (4, 1, 7, 3), (4, 1, 7, 3), (4, 1) + tau))
        self.assertArrayShapesAre(func(m_sb, m_ss), (
            (5, 1, 7, 3), (5, 1, 7, 3), (5, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(m_sb, m_ss)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_returns_expected_shape_wide(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_bs, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 7))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (3, 7))
        self.assertArrayShape(
                        gfl.rqr_lstsq(m_ss, x_f, tau), (5, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        _, x_f, tau = func(m_bs, m_bb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (2, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, tau, m_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_shape_tall(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['bs']
        self.assertArrayShapesAre(func(m_ss, m_bs),
                                  ((3, 7), (3, 7), tau))
        self.assertArrayShapesAre(func(m_ss, m_bs),
                                  ((2, 3, 7), (2, 3, 7), (2,) + tau))
        self.assertArrayShapesAre(func(m_ss, m_bs), (
            (5, 1, 3, 7), (5, 1, 3, 7), (5, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_sb, m_bs)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(m_ss, transpose(m_sb))

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_returns_expected_shape_tall(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_ss, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 7))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (3, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (3, 7))
        self.assertArrayShape(
            gfl.rqr_lstsq(m_ss, x_f, tau), (5, 1, 3, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_sb)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_sb, x_f, tau)
        _, x_f, tau = func(m_ss, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_bb), (2, 3, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_ss, x_f, tau), (2, 3, 7))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, tau, m_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_shape_wide(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['sb']
        self.assertArrayShapesAre(func(m_bb, m_sb),
                                  ((7, 3), (7, 3), tau))
        self.assertArrayShapesAre(func(m_bb, m_sb),
                                  ((3, 7, 3), (3, 7, 3), (3,) + tau))
        self.assertArrayShapesAre(func(m_bb, m_sb),
                                  ((4, 1, 7, 3), (4, 1, 7, 3), (4, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_bs, m_sb)
        with self.assertRaisesRegex(*utn.broadcast_err):
            func(m_bb, transpose(m_bs))

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_returns_expected_shape_wide(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_bb, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_ss), (7, 3))
        self.assertArrayShape(
            gfl.qr_lstsq(x_f, tau, m_ss), (5, 1, 7, 3))
        self.assertArrayShape(gfl.rqr_lstsq(m_bb, x_f, tau), (7, 3))
        self.assertArrayShape(
            gfl.rqr_lstsq(m_bb, x_f, tau), (3, 7, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(m_bs, x_f, tau)
        _, x_f, tau = func(m_bb, m_sb)
        self.assertArrayShape(
                        gfl.qr_lstsq(x_f, tau, m_ss), (4, 1, 7, 3))
        self.assertArrayShape(
                        gfl.rqr_lstsq(m_bb, x_f, tau), (4, 1, 7, 3))
        with self.assertRaisesRegex(*utn.broadcast_err):
            gfl.qr_lstsq(x_f, tau, m_ss)


class TestLstsqVectors(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    def test_lstsq_flexible_signature_with_vectors_mv(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.lstsq(m_bs, v_b), (3,))
        self.assertArrayShape(gfl.lstsq(m_sb, v_s), (7,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(m_sb, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
            gfl.lstsq(m_sb, m_bs)

    def test_lstsq_flexible_signature_with_vectors_vm(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.lstsq(v_s, m_sb), (7,))
        self.assertArrayShape(gfl.lstsq(v_s, m_sb), (4, 1, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(v_s, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vM: (3)(7)\(3)(7,7)
            gfl.lstsq(m_sb, m_bb)

    def test_lstsq_flexible_signature_with_vectors_vv(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.lstsq(v_s, v_s), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.lstsq(v_s, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: ()(3)\(7)(3)
            gfl.lstsq(v_s, m_bs)

    def test_rlstsq_flexible_signature_with_vectors_mv(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.rlstsq(m_sb, v_b), (3,))
        self.assertArrayShape(gfl.rlstsq(m_sb, v_b), (4, 1, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(m_bs, v_b)
        # If interpreted as Mv, this would be: (5,1)(3,3)/(7)(3)->(5,7)(3)
        self.assertArrayShape(gfl.rlstsq(m_ss, m_bs),
                                (5, 1, 3, 7))

    def test_rlstsq_flexible_signature_with_vectors_vm(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.rlstsq(v_b, m_sb), (3,))
        self.assertArrayShape(gfl.rlstsq(v_s, m_bs), (7,))
        self.assertArrayShape(gfl.rlstsq(v_b, m_sb), (4, 1, 3))
        self.assertArrayShape(gfl.rlstsq(v_s, m_bs), (2, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(v_b, m_bs)
        # If interpreted as vM, this would be: (3)(7)/(4,1)(3,7)->(4,3)(3)
        self.assertArrayShape(gfl.rlstsq(m_sb, m_sb),
                                (4, 1, 3, 3))

    def test_rlstsq_flexible_signature_with_vectors_vv(self):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.rlstsq(v_s, v_s), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rlstsq(v_s, v_b)
        # If interpreted as vvv, this would be: (7)(3)/(2,7)(3)->(2,7)
        self.assertArrayShape(gfl.rlstsq(m_bs, m_bs), (2, 7, 7))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_flexible_signature_with_vectors_mv(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['bs']
        self.assertArrayShapesAre(func(m_bs, v_b),
                                    ((3,), (3, 7), tau))
        self.assertArrayShapesAre(func(m_bs, v_b),
                                    ((2, 3), (2, 3, 7), (2,) + tau))
        tau = tau_len['sb']
        self.assertArrayShapesAre(func(m_sb, v_s),
                                    ((7,), (7, 3), tau))
        self.assertArrayShapesAre(func(m_sb, v_s),
                                    ((4, 1, 7), (4, 1, 7, 3), (4, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_sb, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
            func(m_sb, m_bs)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_flexible_signature_with_vectors_vm(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['vs']
        self.assertArrayShapesAre(func(v_s, m_sb),
                                    ((7,), (3,), tau))
        self.assertArrayShapesAre(func(v_s, m_sb),
                                    ((4, 1, 7), (4, 1, 3), (4, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_s, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vM: (3)(7)\(3)(7,7)
            func(m_sb, m_bb)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[:2])
    def test_lstsq_qr_flexible_signature_with_vectors_vv(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShapesAre(func(v_s, v_s),
                                    ((), (3,), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_s, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: ()(3)\(7)(3)
            func(v_s, m_bs)

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_flexible_signature_with_vectors_mv(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['vb']
        self.assertArrayShapesAre(func(m_sb, v_b),
                                    ((3,), (7,), tau))
        self.assertArrayShapesAre(func(m_sb, v_b),
                                    ((4, 1, 3), (4, 1, 7), (4, 1) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(m_bs, v_b)
        tau = tau_len['bs']
        # If interpreted as Mv, this would be: (5,1)(3,3)/(7)(3)->(5,7)(3)
        self.assertArrayShapesAre(func(m_ss, m_bs), (
            (5, 1, 3, 7), (5, 1, 3, 7), (5, 1) + tau))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_flexible_signature_with_vectors_vm(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['sb']
        self.assertArrayShapesAre(func(v_b, m_sb),
                                    ((3,), (7, 3), tau))
        self.assertArrayShapesAre(func(v_b, m_sb),
                                    ((4, 1, 3), (4, 1, 7, 3), (4, 1) + tau))
        tau = tau_len['bs']
        self.assertArrayShapesAre(func(v_s, m_bs),
                                    ((7,), (3, 7), tau))
        self.assertArrayShapesAre(func(v_s, m_bs),
                                    ((2, 7), (2, 3, 7), (2,) + tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_b, m_bs)
        tau = tau_len['sb']
        # If interpreted as vM, this would be: (3)(7)/(4,1)(3,7)->(4,3)(3)
        self.assertArrayShapesAre(func(m_sb, m_sb), (
            (4, 1, 3, 3), (4, 1, 7, 3), (4, 1) + tau))

    @utn.loop_test(attr_name=('func', 'tau_len'), attr_inds=np.s_[2:])
    def test_rlstsq_qr_flexible_signature_with_vectors_vv(self, func, tau_len):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        tau = tau_len['vs']
        self.assertArrayShapesAre(func(v_s, v_s),
                                  ((), (3,), tau))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(v_s, v_b)
        tau = tau_len['bs']
        # If interpreted as vvv, this would be: (7)(3)/(2,7)(3)->(2,7)
        self.assertArrayShapesAre(func(m_bs, m_bs),
                                    ((2, 7, 7), (2, 3, 7), (2,) + tau))

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_flexible_signature_with_vectors_mv(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(m_bs, v_b)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), (2, 3))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), (2, 7))

        _, x_f, tau = func(m_bs, v_b)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), (3,))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), (7,))

        _, x_f, tau = func(m_sb, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), (4, 1, 7))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), (4, 1, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
            gfl.qr_lstsq(x_f, tau, m_bs)

        _, x_f, tau = func(m_sb, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), (7,))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), (3,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            func(x_f, tau, v_b)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_flexible_signature_with_vectors_vm(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vM: (3)(7)\(3)(7,7)
            gfl.qr_lstsq(x_f, tau, m_bb)

        _, x_f, tau = func(v_s, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), (7,))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), (7,))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), (4, 1, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), (2, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())

    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_qr_lstsq_flexible_signature_with_vectors_vv(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), ())
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_b, x_f, tau)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: ()(3)\(7)(3)
            gfl.qr_lstsq(x_f, tau, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: (7)(3)/()(3)
            gfl.rqr_lstsq(m_sb, x_f, tau)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_flexible_signature_with_vectors_mv(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        _, x_f, tau = func(v_s, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), (2, 3))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), (2, 7))

        _, x_f, tau = func(v_s, m_bs)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_b), (3,))
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), (7,))

        _, x_f, tau = func(v_b, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), (4, 1, 7))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), (4, 1, 3))
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as Mv: (4,1)(3,7)\(7)(3)
            gfl.qr_lstsq(x_f, tau, m_bs)

        _, x_f, tau = func(v_b, m_sb)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), (7,))
        self.assertArrayShape(gfl.rqr_lstsq(v_b, x_f, tau), (3,))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_flexible_signature_with_vectors_vm(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vM: (3)(7)\(3)(7,7)
            gfl.qr_lstsq(x_f, tau, m_bb)

        _, x_f, tau = func(m_bs, v_s)
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), (7,))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), (7,))
        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, m_sb), (4, 1, 7))
        self.assertArrayShape(gfl.rqr_lstsq(m_bs, x_f, tau), (2, 7))
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, m_bs)
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())

    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rqr_lstsq_flexible_signature_with_vectors_vv(self, func):
        self.pick_var_type('d')
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs, self.v_s, self.v_b]
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays[:-2]]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        self.assertArrayShape(gfl.qr_lstsq(x_f, tau, v_s), ())
        self.assertArrayShape(gfl.rqr_lstsq(v_s, x_f, tau), ())
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.qr_lstsq(x_f, tau, v_b)
        with self.assertRaisesRegex(*utn.core_dim_err):
            gfl.rqr_lstsq(v_b, x_f, tau)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: ()(3)\(7)(3)
            gfl.qr_lstsq(x_f, tau, m_bs)
        with self.assertRaisesRegex(*utn.core_dim_err):
            # This would work if interpreted as vvv: (7)(3)/()(3)
            gfl.rqr_lstsq(m_sb, x_f, tau)


class TestLstsqVal(TestLstsq):
    """Testing (r)lstsq, (r)lstsq_qr? and (r)qr_lstsq"""

    @utn.loop_test()
    def test_lstsq_returns_expected_values(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        # overconstrained
        x_sb = gfl.lstsq(m_bs, m_bb)
        a_bst = dagger(m_bs)
        # with self.subTest(msg='lstsq(over)'):
        self.assertArrayAllClose(a_bst @ m_bs @ x_sb, a_bst @ m_bb)
        x_bs = gfl.rlstsq(m_bb, m_sb)
        a_sbt = dagger(m_sb)
        # with self.subTest(msg='rlstsq(over)'):
        self.assertArrayAllClose(x_bs @ m_sb @ a_sbt, m_bb @ a_sbt)
        # underconstrained
        x_bs = gfl.lstsq(m_sb, m_ss)
        # with self.subTest(msg='lstsq(under)'):
        self.assertArrayAllClose(m_sb @ x_bs, m_ss)
        x_sb = gfl.rlstsq(m_ss, m_bs)
        # with self.subTest(msg='rlstsq(under)'):
        self.assertArrayAllClose(x_sb @ m_bs, m_ss)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_values_with_tall(self, sctype, func):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        suffix = func.__name__[-1] + ')'
        # overconstrained
        x0_sb = gfl.lstsq(m_bs, m_bb)
        # overconstrained
        x_sb, x_f, tau = func(m_bs, m_bb)
        # with self.subTest('lstsq_qr(over,' + suffix):
        self.assertArrayAllClose(x_sb, x0_sb)
        # overconstrained
        xx_sb = gfl.qr_lstsq(x_f, tau, m_bb)
        # with self.subTest('qr_lstsq(over,' + suffix):
        self.assertArrayAllClose(xx_sb, x0_sb)
        # underconstrained
        y_sb = gfl.rqr_lstsq(m_ss, x_f, tau)
        # with self.subTest('rqr_lstsq(under,' + suffix):
        self.assertArrayAllClose(y_sb @ m_bs, m_ss)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_values_with_wide(self, sctype, func):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        suffix = func.__name__[-1] + ')'
        # overconstrained
        x0_bs = gfl.rlstsq(m_bb, m_sb)
        # overconstrained
        x_bs, x_f, tau = func(m_bb, m_sb)
        # with self.subTest('rlstsq_qr(under,' + suffix):
        self.assertArrayAllClose(x_bs, x0_bs)
        # overconstrained
        xx_bs = gfl.rqr_lstsq(m_bb, x_f, tau)
        # with self.subTest('rqr_rlstsq(under,' + suffix):
        self.assertArrayAllClose(xx_bs, x0_bs)
        # underconstrained
        y_bs = gfl.qr_lstsq(x_f, tau, m_ss)
        # with self.subTest('qr_rlstsq(over,' + suffix):
        self.assertArrayAllClose(m_sb @ y_bs, m_ss)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[:2])
    def test_lstsq_qr_returns_expected_values_with_wide(self, sctype, func):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        suffix = func.__name__[-1] + ')'
        # underconstrained
        x0_bs = gfl.lstsq(m_sb, m_ss)
        # underconstrained
        x_bs, x_f, tau = func(m_sb, m_ss)
        # with self.subTest('lstsq_qr(under,' + suffix):
        self.assertArrayAllClose(x_bs, x0_bs)
        # underconstrained
        xx_bs = gfl.qr_lstsq(x_f, tau, m_ss)
        # with self.subTest('qr_lstsq(under,' + suffix):
        self.assertArrayAllClose(xx_bs, x0_bs)
        # overconstrained
        y_bs = gfl.rqr_lstsq(m_bb, x_f, tau)
        a_sbt = dagger(m_sb)
        # with self.subTest('rqr_lstsq(over,' + suffix):
        self.assertArrayAllClose(y_bs @ m_sb @ a_sbt, m_bb @ a_sbt)

    @utn.loop_test()
    @utn.loop_test(attr_name='func', attr_inds=np.s_[2:])
    def test_rlstsq_qr_returns_expected_values_with_tall(self, sctype, func):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        suffix = func.__name__[-1] + ')'
        # underconstrained
        x0_sb = gfl.rlstsq(m_ss, m_bs)
        # underconstrained
        x_sb, x_f, tau = func(m_ss, m_bs)
        # with self.subTest('rlstsq_qr(over,' + suffix):
        self.assertArrayAllClose(x_sb, x0_sb)
        # underconstrained
        xx_sb = gfl.rqr_lstsq(m_ss, x_f, tau)
        # with self.subTest('rqr_rlstsq(over,' + suffix):
        self.assertArrayAllClose(xx_sb, x0_sb)
        # overconstrained
        y_sb = gfl.qr_lstsq(x_f, tau, m_bb)
        a_bst = dagger(m_bs)
        # with self.subTest('qr_rlstsq(under,' + suffix):
        self.assertArrayAllClose(a_bst @ m_bs @ y_sb, a_bst @ m_bb)

    @unittest.expectedFailure
    @errstate
    @utn.loop_test(msg='rank')
    def test_lstsq_qr_raises_with_low_rank(self, sctype):
        self.pick_var_type(sctype)
        arrays = [self.m_ss, self.m_sb, self.m_bb, self.m_bs]
        m_ss, m_sb, m_bb, m_bs = arrays
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        smob, widb, bib, talb = [arr.shape[:-2] for arr in arrays]
        mini, maxi = wide[-2:-1], wide[-1:]
        hy.assume(hn.wide(m_sb))

        with self.assertRaisesRegex(*utn.invalid_err):
            gfl.lstsq_qrn(self.ones_bs, m_bs)


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
