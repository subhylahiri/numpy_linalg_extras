# -*- coding: utf-8 -*-
"""Test python wrappers of gufuncs.
It has been established, in test_gu*.py, that the gufuncs return the correct
values. We just check that the python wrappers call the correct ones.
"""
from unittest import expectedFailure
import hypothesis as hy
import numpy as np
import numpy_linalg as la
import numpy_linalg.gufuncs as gf
import numpy_linalg.testing.unittest_numpy as utn
import numpy_linalg.testing.hypothesis_numpy as hn
from numpy_linalg.testing import main, TestCaseNumpy
# =============================================================================
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-sequence-index
errstate = np.errstate(invalid='raise')
hy.settings.register_profile("slow",
                             suppress_health_check=(hy.HealthCheck.too_slow,))
hy.settings.load_profile('slow')
# =============================================================================
__all__ = ['TestShape', 'TestValue']
# =============================================================================
# Shape helper
# =============================================================================


def tril(matrix):
    """Lower triangle of matrix, excluding diagonal"""
    return np.tril(matrix, -1)


# =============================================================================
# Test python funcs
# =============================================================================


class TestShape(TestCaseNumpy):
    """Testing shapes returned by linalg functions"""

    @hy.given(hn.matrices_b)
    def test_functions_shape(self, array):
        shape = array.shape
        self.assertArrayShape(la.transpose(array), utn.trnsp(shape))
        self.assertArrayShape(la.row(array), shape[:-1] + (1,) + shape[-1:])
        self.assertArrayShape(la.col(array), shape + (1,))
        self.assertArrayShape(la.scalar(array), shape + (1, 1))

    @hy.given(hn.broadcastable('(a,b),(b,a),(a),(b)', 'd'))
    def test_functions_matmul(self, arrays):
        m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        wide, tall = [arr.shape for arr in arrays[:-2]]

        # with self.subTest('matmul'):
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', tall, wide)
        self.assertArrayShape(la.matmul(m_bs, m_sb), expect)
        self.assertArrayShape(la.matmul(m_bs, v_s), tall[:-1])
        self.assertArrayShape(la.matmul(v_b, m_bs), utn.drop(tall))
        self.assertArrayShape(la.matmul(v_s, v_s), ())
        # with self.subTest('rmatmul'):
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', wide, tall)
        self.assertArrayShape(gf.rmatmul(m_bs, m_sb), expect)
        self.assertArrayShape(gf.rmatmul(m_sb, v_s), utn.drop(wide))
        self.assertArrayShape(gf.rmatmul(v_b, m_sb), wide[:-1])
        self.assertArrayShape(gf.rmatmul(v_b, v_b), ())

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a),(a),(b)', 'd'))
    def test_functions_solve(self, arrays):
        m_ss, m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.all_non_singular(m_ss))

        # with self.subTest('solve'):
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', smol, wide)
        self.assertArrayShape(la.solve(m_ss, m_sb),expect)
        self.assertArrayShape(la.solve(m_ss, v_s), smol[:-1])
        # with self.subTest('rsolve'):
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', tall, smol)
        self.assertArrayShape(la.rsolve(m_bs, m_ss), expect)
        self.assertArrayShape(la.rsolve(v_s, m_ss), smol[:-1])

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_functions_lstsq(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]

        # with self.subTest('lstsq'):
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', tall, big)
        self.assertArrayShape(la.lstsq(m_bs, m_bb), expect)
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', wide, smol)
        self.assertArrayShape(la.lstsq(m_sb, m_ss), expect)
        self.assertArrayShape(la.lstsq(m_sb, v_s), utn.drop(wide))
        self.assertArrayShape(la.lstsq(v_s, m_ss), smol[:-1])
        # with self.subTest('rlstsq'):
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', smol, tall)
        self.assertArrayShape(la.rlstsq(m_ss, m_bs), expect)
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', big, wide)
        self.assertArrayShape(la.rlstsq(m_bb, m_sb), expect)
        self.assertArrayShape(la.rlstsq(v_b, m_sb), wide[:-1])
        self.assertArrayShape(la.rlstsq(m_ss, v_s), smol[:-1])
        self.assertArrayShape(la.rlstsq(v_s, m_bs), tall[:-1])

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a)', 'd'))
    def test_functions_matldiv(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-1]
        v_s = hn.core_only(arrays[-1], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-1]]
        hy.assume(hn.all_non_singular(m_ss))
        hy.assume(hn.wide(m_sb))

        # with self.subTest('solve'):
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', smol, wide)
        self.assertArrayShape(la.matldiv(m_ss, m_sb), expect)
        self.assertArrayShape(la.matldiv(m_ss, v_s), smol[:-1])
        # with self.subTest('lstsq'):
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', tall, big)
        self.assertArrayShape(la.matldiv(m_bs, m_bb), expect)
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', wide, smol)
        self.assertArrayShape(la.matldiv(m_sb, m_ss), expect)
        self.assertArrayShape(la.matldiv(m_sb, v_s), utn.drop(wide))
        self.assertArrayShape(la.matldiv(v_s, m_ss), smol[:-1])

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a),(a),(b)', 'd'))
    def test_functions_matrdiv(self, arrays):
        m_ss, m_sb, m_bs = arrays[:-2]
        v_s, v_b = hn.core_only(*arrays[-2:], dims=1)
        smol, wide, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(hn.all_non_singular(m_ss))
        hy.assume(hn.wide(m_sb))

        # with self.subTest('rsolve'):
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', tall, smol)
        self.assertArrayShape(la.matrdiv(m_bs, m_ss), expect)
        self.assertArrayShape(la.matrdiv(v_s, m_ss), smol[:-1])
        # with self.subTest('rlstsq'):
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', smol, tall)
        self.assertArrayShape(la.matrdiv(m_ss, m_bs), expect)
        self.assertArrayShape(la.matrdiv(v_b, m_sb), wide[:-1])
        self.assertArrayShape(la.matrdiv(m_ss, v_s), smol[:-1])
        self.assertArrayShape(la.matrdiv(v_s, m_bs), tall[:-1])

    @hy.given(hn.broadcastable('(a,b),(b,a)', 'd'))
    def test_qr(self, arrays):
        m_sb, m_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        hy.assume(hn.wide(m_sb))

        # with self.subTest("reduced"):
        self.assertArrayShapesAre(la.qr(m_bs, 'reduced'),
                                  (tall, utn.chop(tall)))
        self.assertArrayShapesAre(la.qr(m_sb, 'reduced'),
                                  (utn.chop(wide), wide))
        # with self.subTest("complete"):
        self.assertArrayShapesAre(la.qr(m_bs, 'complete'),
                                  (utn.grow(tall), tall))
        self.assertArrayShapesAre(la.qr(m_sb, 'complete'),
                                  (utn.chop(wide), wide))
        # with self.subTest("r/raw"):
        self.assertArrayShape(la.qr(m_bs, 'r'), utn.chop(tall))
        self.assertArrayShape(la.qr(m_sb, 'r'), wide)
        self.assertArrayShapesAre(la.qr(m_bs, 'raw'),
                                  (utn.trnsp(tall), utn.drop(tall)))
        self.assertArrayShapesAre(la.qr(m_sb, 'raw'),
                                  (utn.trnsp(wide), wide[:-1]))

    @hy.given(hn.broadcastable('(a,b),(b,a)', 'd'))
    def test_lq(self, arrays):
        m_sb, m_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        hy.assume(hn.wide(m_sb))

        # with self.subTest("reduced"):
        self.assertArrayShapesAre(la.lq(m_bs, 'reduced'),
                                  (tall, utn.chop(tall)))
        self.assertArrayShapesAre(la.lq(m_sb, 'reduced'),
                                  (utn.chop(wide), wide))
        # with self.subTest("complete"):
        self.assertArrayShapesAre(la.lq(m_bs, 'complete'),
                                  (tall, utn.chop(tall)))
        self.assertArrayShapesAre(la.lq(m_sb, 'complete'),
                                  (wide, utn.grow(wide)))
        # with self.subTest("l/raw"):
        self.assertArrayShape(la.lq(m_bs, 'l'), tall)
        self.assertArrayShape(la.lq(m_sb, 'l'), utn.chop(wide))
        self.assertArrayShapesAre(la.lq(m_bs, 'raw'),
                                  (utn.trnsp(tall), utn.drop(tall)))
        self.assertArrayShapesAre(la.lq(m_sb, 'raw'),
                                  (utn.trnsp(wide), wide[:-1]))

    @hy.given(hn.broadcastable('(a,b),(b,a)', 'd'))
    def test_lqr(self, arrays):
        m_sb, m_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        hy.assume(hn.wide(m_sb))

        # with self.subTest("reduced"):
        self.assertArrayShapesAre(la.lqr(m_bs, 'reduced'),
                                  (tall, utn.chop(tall)))
        self.assertArrayShapesAre(la.lqr(m_sb, 'reduced'),
                                  (utn.chop(wide), wide))
        # with self.subTest("complete"):
        self.assertArrayShapesAre(la.lqr(m_bs, 'complete'),
                                  (utn.grow(tall), tall))
        self.assertArrayShapesAre(la.lqr(m_sb, 'complete'),
                                  (wide, utn.grow(wide)))
        # with self.subTest("r/l/raw"):
        self.assertArrayShape(la.lqr(m_bs, 'l'), utn.chop(tall))
        self.assertArrayShape(la.lqr(m_sb, 'l'), utn.chop(wide))
        self.assertArrayShape(la.lqr(m_bs, 'r'), utn.chop(tall))
        self.assertArrayShape(la.lqr(m_sb, 'r'), utn.chop(wide))
        self.assertArrayShapesAre(la.lqr(m_bs, 'raw'),
                                  (utn.trnsp(tall), utn.drop(tall)))
        self.assertArrayShapesAre(la.lqr(m_sb, 'raw'),
                                  (utn.trnsp(wide), wide[:-1]))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a)', 'd'))
    def test_lu(self, arrays):
        m_ss, m_sb, m_bs = arrays
        smol, wide, tall = [arr.shape for arr in arrays]
        hy.assume(hn.wide(m_sb))

        # with self.subTest("separate"):
        self.assertArrayShapesAre(la.lu(m_ss, 'separate'),
                                  (smol, smol, smol[:-1]))
        self.assertArrayShapesAre(la.lu(m_bs, 'separate'),
                                  (tall, utn.chop(tall), utn.drop(tall)))
        self.assertArrayShapesAre(la.lu(m_sb, 'separate'),
                                  (utn.chop(wide), wide, wide[:-1]))
        # with self.subTest("raw"):
        self.assertArrayShapesAre(la.lu(m_ss, 'raw'), (smol, smol[:-1]))
        self.assertArrayShapesAre(la.lu(m_bs, 'raw'),
                                  (utn.trnsp(tall), utn.drop(tall)))
        self.assertArrayShapesAre(la.lu(m_sb, 'raw'),
                                  (utn.trnsp(wide), wide[:-1]))


class TestValue(TestCaseNumpy):
    """Testing values returned by linalg functions"""

    @hy.given(hn.broadcastable('(a,b),(b,a)', None))
    def test_functions_matmul(self, arrays):
        m_sb, m_bs = arrays

        # with self.subTest('matmul'):
        self.assertArrayAllClose(la.matmul(m_bs, m_sb), gf.matmul(m_bs, m_sb))
        # with self.subTest('rmatmul'):
        self.assertArrayAllClose(gf.rmatmul(m_bs, m_sb), gf.rmatmul(m_bs, m_sb))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a)', None))
    def test_functions_solve(self, arrays):
        m_ss, m_sb, m_bs = arrays
        hy.assume(hn.all_non_singular(m_ss))

        # with self.subTest('solve'):
        self.assertArrayAllClose(la.solve(m_ss, m_sb), gf.solve(m_ss, m_sb))
        slv_sh = utn.array_return_shape('(a,a),(a,b)->(a,b)', m_ss, m_sb)
        slv_out = np.empty(slv_sh, m_ss.dtype)
        slv_r = la.solve(m_ss, m_sb, out=slv_out)
        self.assertArrayAllClose(slv_out, slv_r)
        # with self.subTest('rsolve'):
        self.assertArrayAllClose(la.rsolve(m_bs, m_ss), gf.rsolve(m_bs, m_ss))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a),(b)', None))
    def test_functions_lstsq(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-1]
        v_b = hn.core_only(arrays[-1], dims=1)
        hy.assume(hn.wide(m_sb))

        # with self.subTest('lstsq'):
        self.assertArrayAllClose(la.lstsq(m_bs, m_bb), gf.lstsq(m_bs, m_bb))
        self.assertArrayAllClose(la.lstsq(m_sb, m_ss), gf.lstsq(m_sb, m_ss))
        lsq_sh = utn.array_return_shape('(a,b),(a,c)->(b,c)', m_bs, m_bb)
        lsq_out = np.empty(lsq_sh, m_bs.dtype)
        lsq_r = la.lstsq(m_bs, m_bb, out=lsq_out)
        self.assertArrayAllClose(lsq_out, lsq_r)
        # with self.subTest('rlstsq'):
        self.assertArrayAllClose(la.rlstsq(m_ss, m_bs), gf.rlstsq(m_ss, m_bs))
        self.assertArrayAllClose(la.rlstsq(v_b, m_sb), gf.rlstsq(v_b, m_sb))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None))
    def test_functions_matldiv(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(hn.all_non_singular(m_ss))
        hy.assume(hn.wide(m_sb))

        # with self.subTest('solve'):
        self.assertArrayAllClose(la.matldiv(m_ss, m_sb), gf.solve(m_ss, m_sb))
        slv_sh = utn.array_return_shape('(a,a),(a,b)->(a,b)', m_ss, m_sb)
        slv_out = np.empty(slv_sh, m_ss.dtype)
        slv_r = la.matldiv(m_ss, m_sb, out=slv_out)
        self.assertArrayAllClose(slv_out, slv_r)
        # with self.subTest('lstsq'):
        self.assertArrayAllClose(la.matldiv(m_bs, m_bb), gf.lstsq(m_bs, m_bb))
        self.assertArrayAllClose(la.matldiv(m_sb, m_ss), gf.lstsq(m_sb, m_ss))

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a),(b)', None))
    def test_functions_matrdiv(self, arrays):
        m_ss, m_sb, m_bs = arrays[:-1]
        v_b = hn.core_only(arrays[-1], dims=1)
        hy.assume(hn.all_non_singular(m_ss))
        hy.assume(hn.wide(m_sb))

        # with self.subTest('rsolve'):
        self.assertArrayAllClose(la.matrdiv(m_bs, m_ss), gf.rsolve(m_bs, m_ss))
        # with self.subTest('rlstsq'):
        self.assertArrayAllClose(la.matrdiv(m_ss, m_bs), gf.rlstsq(m_ss, m_bs))
        self.assertArrayAllClose(la.matrdiv(v_b, m_sb), gf.rlstsq(v_b, m_sb))

    @hy.given(hn.broadcastable('(a,b),(b,a)', None))
    def test_qr(self, arrays):
        m_sb, m_bs = arrays
        box = np.s_[..., :m_sb.shape[-2], :]
        hy.assume(hn.wide(m_sb))

        # with self.subTest("reduced"):
        unitary, right = la.qr(m_bs, 'reduced')
        self.assertArrayAllClose(unitary @ right, m_bs)
        unitary, right = la.qr(m_sb, 'reduced')
        self.assertArrayAllClose(unitary @ right, m_sb)
        # with self.subTest("complete"):
        unitary, right = la.qr(m_bs, 'complete')
        self.assertArrayAllClose(unitary @ right, m_bs)
        unitary, right = la.qr(m_sb, 'complete')
        self.assertArrayAllClose(unitary @ right, m_sb)
        # with self.subTest("r/raw"):
        right = la.qr(m_bs, 'r')
        hhold, _ = la.qr(m_bs, 'raw')
        self.assertArrayAllClose(right, np.triu(la.transpose(hhold))[box])
        right = la.qr(m_sb, 'r')
        hhold, _ = la.qr(m_sb, 'raw')
        self.assertArrayAllClose(right, np.triu(la.transpose(hhold)))

    @hy.given(hn.broadcastable('(a,b),(b,a)', None))
    def test_lq(self, arrays):
        m_sb, m_bs = arrays
        box = np.s_[..., :m_sb.shape[-2]]
        hy.assume(hn.wide(m_sb))

        # with self.subTest("reduced"):
        left, unitary = la.lq(m_bs, 'reduced')
        self.assertArrayAllClose(left @ unitary, m_bs)
        left, unitary = la.lq(m_sb, 'reduced')
        self.assertArrayAllClose(left @ unitary, m_sb)
        # with self.subTest("complete"):
        left, unitary = la.lq(m_bs, 'complete')
        self.assertArrayAllClose(left @ unitary, m_bs)
        left, unitary = la.lq(m_sb, 'complete')
        self.assertArrayAllClose(left @ unitary, m_sb)
        # with self.subTest("l/raw"):
        left = la.lq(m_bs, 'l')
        hhold, _ = la.lq(m_bs, 'raw')
        self.assertArrayAllClose(left, np.tril(la.transpose(hhold)))
        left = la.lq(m_sb, 'l')
        hhold, _ = la.lq(m_sb, 'raw')
        self.assertArrayAllClose(left, np.tril(la.transpose(hhold))[box])

    @hy.given(hn.broadcastable('(a,b),(b,a)', None))
    def test_lqr(self, arrays):
        m_sb, m_bs = arrays
        box = np.s_[..., :m_sb.shape[-2], :]
        hy.assume(hn.wide(m_sb))

        # with self.subTest("reduced"):
        unitary, right = la.lqr(m_bs, 'reduced')
        self.assertArrayAllClose(unitary @ right, m_bs)
        left, unitary = la.lqr(m_sb, 'reduced')
        self.assertArrayAllClose(left @ unitary, m_sb)
        # with self.subTest("complete"):
        unitary, right = la.lqr(m_bs, 'complete')
        self.assertArrayAllClose(unitary @ right, m_bs)
        left, unitary = la.lqr(m_sb, 'complete')
        self.assertArrayAllClose(left @ unitary, m_sb)
        # with self.subTest("r/l/raw"):
        right = la.lqr(m_bs, 'r')
        hhold, _ = la.lqr(m_bs, 'raw')
        self.assertArrayAllClose(right, np.triu(la.transpose(hhold))[box])
        left = la.lqr(m_sb, 'r')
        hhold, _ = la.lqr(m_sb, 'raw')
        self.assertArrayAllClose(left, np.tril(la.transpose(hhold))[box[:-1]])

    @hy.given(hn.broadcastable('(a,a),(a,b),(b,a)', None))
    def test_lu(self, arrays):
        m_ss, m_sb, m_bs = arrays
        box = np.s_[..., :m_sb.shape[-2], :]
        hy.assume(hn.wide(m_sb))

        # with self.subTest("square"):
        lower, upper, piv = la.lu(m_ss, 'separate')
        luf, piv = la.lu(m_ss, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(lower @ upper, gf.pivot(m_ss, piv))
        self.assertArrayAllClose(tril(lower), tril(luf))
        self.assertArrayAllClose(upper, np.triu(luf))
        # with self.subTest("wide"):
        lower, upper, piv = la.lu(m_bs, 'separate')
        luf, piv = la.lu(m_bs, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(tril(lower), tril(luf))
        self.assertArrayAllClose(upper, np.triu(luf)[box])
        # with self.subTest("wide"):
        lower, upper, piv = la.lu(m_sb, 'separate')
        luf, piv = la.lu(m_sb, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(tril(lower), tril(luf)[box[:-1]])
        self.assertArrayAllClose(upper, np.triu(luf))

    @expectedFailure
    @errstate
    @hy.given(hn.constant('(a,a)', ['d', 'D'], min_side=2))
    def test_low_rank(self, ones_ss):
        unitary, right = la.qr(ones_ss)
        self.assertArrayAllClose(unitary @ right, ones_ss)
        lower, upper, _ = la.lu(ones_ss)
        self.assertArrayAllClose(lower @ upper, ones_ss)
        with self.assertRaisesRegex(*utn.invalid_err):
            la.solve(ones_ss, ones_ss[...,:2])


# =============================================================================
if __name__ == '__main__':
    main(verbosity=2)
