# -*- coding: utf-8 -*-
"""Test python wrappers of gufuncs.
It has been established, in test_gu*.py, that the gufuncs return the correct
values. We just check that the python wrappers call the correct ones.
"""
import unittest
import hypothesis as hy
import hypothesis.extra.numpy as hyn
import numpy as np
import numpy_linalg as la
import numpy_linalg._linalg as lr
import numpy_linalg.gufuncs as gf
if __name__.find('tests.') < 0:
    import unittest_numpy as utn
else:
    from . import unittest_numpy as utn
# pylint: disable=missing-function-docstring
# pylint: disable=invalid-sequence-index
errstate = np.errstate(invalid='raise')
# =============================================================================
__all__ = ['TestShape', 'TestValue']
# =============================================================================
# Shape helper
# =============================================================================


def trnsp(shape):
    """Shape -> shape of transposed array"""
    return shape[:-2] + shape[:-3:-1]


def drop(shape, axis=-2):
    """Shape -> shape with one axis dropped"""
    return shape[:axis] + shape[axis+1:]


def insert(shape, axis=-1):
    """Shape -> shape with one axis inserted"""
    return shape[:axis] + (1,) + shape[axis:]


def tril(matrix):
    """Lower triangle of matrix, excluding diagonal"""
    return np.tril(matrix, -1)


# =============================================================================
# Test python funcs
# =============================================================================


class TestShape(utn.TestCaseNumpy):
    """Testing shapes returned by linalg functions"""

    @hy.given(hyn.arrays('d', hyn.array_shapes(min_dims=2)))
    def test_functions_shape(self, array):
        shape = array.shape
        self.assertArrayShape(la.transpose(array), trnsp(shape))
        self.assertArrayShape(la.row(array), insert(shape))
        self.assertArrayShape(la.col(array), shape + (1,))
        self.assertArrayShape(la.scalar(array), shape + (1, 1))

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_functions_matmul(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))

        # with self.subTest('matmul'):
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', tall, wide)
        self.assertArrayShape(la.matmul(m_bs, m_sb), expect)
        self.assertArrayShape(la.matmul(m_bs, v_s), tall[:-1])
        self.assertArrayShape(la.matmul(v_b, m_bs), drop(tall))
        self.assertArrayShape(la.matmul(v_s, v_s), ())
        # with self.subTest('rmatmul'):
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', wide, tall)
        self.assertArrayShape(lr.rmatmul(m_bs, m_sb), expect)
        self.assertArrayShape(lr.rmatmul(m_sb, v_s), drop(wide))
        self.assertArrayShape(lr.rmatmul(v_b, m_sb), wide[:-1])
        self.assertArrayShape(lr.rmatmul(v_b, v_b), ())

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_functions_solve(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))

        # with self.subTest('solve'):
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', smol, wide)
        self.assertArrayShape(la.solve(m_ss, m_sb),expect)
        self.assertArrayShape(la.solve(m_ss, v_s), smol[:-1])
        # with self.subTest('rsolve'):
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', tall, smol)
        self.assertArrayShape(la.rsolve(m_bs, m_ss), expect)
        self.assertArrayShape(la.rsolve(v_s, m_ss), smol[:-1])

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_functions_lstsq(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))

        # with self.subTest('lstsq'):
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', tall, big)
        self.assertArrayShape(la.lstsq(m_bs, m_bb), expect)
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', wide, smol)
        self.assertArrayShape(la.lstsq(m_sb, m_ss), expect)
        self.assertArrayShape(la.lstsq(m_sb, v_s), drop(wide))
        self.assertArrayShape(la.lstsq(v_s, m_ss), smol[:-1])
        # with self.subTest('rlstsq'):
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', smol, tall)
        self.assertArrayShape(la.rlstsq(m_ss, m_bs), expect)
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', big, wide)
        self.assertArrayShape(la.rlstsq(m_bb, m_sb), expect)
        self.assertArrayShape(la.rlstsq(v_b, m_sb), wide[:-1])
        self.assertArrayShape(la.rlstsq(m_ss, v_s), smol[:-1])
        self.assertArrayShape(la.rlstsq(v_s, m_bs), tall[:-1])

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_functions_matldiv(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))
        hy.assume(wide[-2] < wide[-1])

        # with self.subTest('solve'):
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', smol, wide)
        self.assertArrayShape(la.matldiv(m_ss, m_sb), expect)
        self.assertArrayShape(la.matldiv(m_ss, v_s), smol[:-1])
        # with self.subTest('lstsq'):
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', tall, big)
        self.assertArrayShape(la.matldiv(m_bs, m_bb), expect)
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', wide, smol)
        self.assertArrayShape(la.matldiv(m_sb, m_ss), expect)
        self.assertArrayShape(la.matldiv(m_sb, v_s), drop(wide))
        self.assertArrayShape(la.matldiv(v_s, m_ss), smol[:-1])

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_functions_matrdiv(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(np.all(utn.non_singular(m_bb)))
        hy.assume(wide[-2] < wide[-1])

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

    @hy.given(utn.broadcastable('(a,b),(b,a)', 'd'))
    def test_qr(self, arrays):
        m_sb, m_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        widb, talb = wide[:-2], tall[:-2]  # broadcasting shape
        mini, maxi = (wide[-2],), (wide[-1],)
        hy.assume(wide[-2] < wide[-1])

        # with self.subTest("reduced"):
        self.assertArrayShapesAre(la.qr(m_bs, 'reduced'),
                                    (tall, talb + 2*mini))
        self.assertArrayShapesAre(la.qr(m_sb, 'reduced'),
                                    (widb + 2*mini, wide))
        # with self.subTest("complete"):
        self.assertArrayShapesAre(la.qr(m_bs, 'complete'),
                                    (talb + 2*maxi, tall))
        self.assertArrayShapesAre(la.qr(m_sb, 'complete'),
                                    (widb + 2*mini, wide))
        # with self.subTest("r/raw"):
        self.assertArrayShape(la.qr(m_bs, 'r'), talb + 2*mini)
        self.assertArrayShape(la.qr(m_sb, 'r'), wide)
        self.assertArrayShapesAre(la.qr(m_bs, 'raw'), (trnsp(tall),
                                                        talb + mini))
        self.assertArrayShapesAre(la.qr(m_sb, 'raw'), (trnsp(wide),
                                                        widb + mini))

    @hy.given(utn.broadcastable('(a,b),(b,a)', 'd'))
    def test_lq(self, arrays):
        m_sb, m_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        widb, talb = wide[:-2], tall[:-2]  # broadcasting shape
        mini, maxi = (wide[-2],), (wide[-1],)
        hy.assume(wide[-2] < wide[-1])

        # with self.subTest("reduced"):
        self.assertArrayShapesAre(la.lq(m_bs, 'reduced'),
                                    (tall, talb + 2*mini))
        self.assertArrayShapesAre(la.lq(m_sb, 'reduced'),
                                    (widb + 2*mini, wide))
        # with self.subTest("complete"):
        self.assertArrayShapesAre(la.lq(m_bs, 'complete'),
                                    (tall, talb + 2*mini))
        self.assertArrayShapesAre(la.lq(m_sb, 'complete'),
                                    (wide, widb + 2*maxi))
        # with self.subTest("l/raw"):
        self.assertArrayShape(la.lq(m_bs, 'l'), tall)
        self.assertArrayShape(la.lq(m_sb, 'l'), widb + 2*mini)
        self.assertArrayShapesAre(la.lq(m_bs, 'raw'),
                                    (trnsp(tall), talb + mini))
        self.assertArrayShapesAre(la.lq(m_sb, 'raw'),
                                    (trnsp(wide), widb + mini))

    @hy.given(utn.broadcastable('(a,b),(b,a)', 'd'))
    def test_lqr(self, arrays):
        m_sb, m_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        widb, talb = wide[:-2], tall[:-2]  # broadcasting shape
        mini, maxi = (wide[-2],), (wide[-1],)
        hy.assume(wide[-2] < wide[-1])

        # with self.subTest("reduced"):
        self.assertArrayShapesAre(la.lqr(m_bs, 'reduced'),
                                    (tall, talb + 2*mini))
        self.assertArrayShapesAre(la.lqr(m_sb, 'reduced'),
                                    (widb + 2*mini, wide))
        # with self.subTest("complete"):
        self.assertArrayShapesAre(la.lqr(m_bs, 'complete'),
                                    (talb + 2*maxi, tall))
        self.assertArrayShapesAre(la.lqr(m_sb, 'complete'),
                                    (wide, widb + 2*maxi))
        # with self.subTest("r/l/raw"):
        self.assertArrayShape(la.lqr(m_bs, 'l'), talb + 2*mini)
        self.assertArrayShape(la.lqr(m_sb, 'l'), widb + 2*mini)
        self.assertArrayShape(la.lqr(m_bs, 'r'), talb + 2*mini)
        self.assertArrayShape(la.lqr(m_sb, 'r'), widb + 2*mini)
        self.assertArrayShapesAre(la.lqr(m_bs, 'raw'),
                                    (trnsp(tall), talb + mini))
        self.assertArrayShapesAre(la.lqr(m_sb, 'raw'),
                                    (trnsp(wide), widb + mini))

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,a)', 'd'))
    def test_lu(self, arrays):
        m_ss, m_sb, m_bs = arrays
        smol, wide, tall = [arr.shape for arr in arrays]
        smob, widb, talb = smol[:-2], wide[:-2], tall[:-2]  # broadcasting shape
        mini, maxi = (wide[-2],), (wide[-1],)
        hy.assume(wide[-2] < wide[-1])

        # with self.subTest("separate"):
        self.assertArrayShapesAre(la.lu(m_ss, 'separate'),
                                (smol, smol, smol[:-1]))
        self.assertArrayShapesAre(la.lu(m_bs, 'separate'),
                                (tall, talb + 2*mini, talb + mini))
        self.assertArrayShapesAre(la.lu(m_sb, 'separate'),
                                (widb + 2*mini, wide, widb + mini))
        # with self.subTest("raw"):
        self.assertArrayShapesAre(la.lu(m_ss, 'raw'), (smol, smob+mini))
        self.assertArrayShapesAre(la.lu(m_bs, 'raw'),
                                    (trnsp(tall), talb+mini))
        self.assertArrayShapesAre(la.lu(m_sb, 'raw'),
                                    (trnsp(wide), widb+mini))


class TestValue(utn.TestCaseNumpy):
    """Testing values returned by linalg functions"""

    @hy.given(utn.broadcastable('(a,b),(b,a)', None))
    def test_functions_matmul(self, arrays):
        m_sb, m_bs = arrays

        # with self.subTest('matmul'):
        self.assertArrayAllClose(la.matmul(m_bs, m_sb),
                                 gf.matmul(m_bs, m_sb))
        # with self.subTest('rmatmul'):
        self.assertArrayAllClose(lr.rmatmul(m_bs, m_sb),
                                 gf.rmatmul(m_bs, m_sb))

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,a)', None))
    def test_functions_solve(self, arrays):
        m_ss, m_sb, m_bs = arrays
        hy.assume(np.all(utn.non_singular(m_ss)))

        # with self.subTest('solve'):
        self.assertArrayAllClose(la.solve(m_ss, m_sb), gf.solve(m_ss, m_sb))
        slv_sh = gf.array_return_shape('(a,a),(a,b)->(a,b)', m_ss, m_sb)
        slv_out = np.empty(slv_sh, m_ss.dtype)
        slv_r = la.solve(m_ss, m_sb, out=slv_out)
        self.assertArrayAllClose(slv_out, slv_r)
        # with self.subTest('rsolve'):
        self.assertArrayAllClose(la.rsolve(m_bs, m_ss),
                                    gf.rsolve(m_bs, m_ss))

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(b)', None))
    def test_functions_lstsq(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays[:-1]
        v_b = utn.core_only(arrays[-1], dims=1)
        hy.assume(m_sb.shape[-2] < m_sb.shape[-1])

        # with self.subTest('lstsq'):
        self.assertArrayAllClose(la.lstsq(m_bs, m_bb), gf.lstsq(m_bs, m_bb))
        self.assertArrayAllClose(la.lstsq(m_sb, m_ss), gf.lstsq(m_sb, m_ss))
        lsq_sh = gf.array_return_shape('(a,b),(a,c)->(b,c)', m_bs, m_bb)
        lsq_out = np.empty(lsq_sh, m_bs.dtype)
        lsq_r = la.lstsq(m_bs, m_bb, out=lsq_out)
        self.assertArrayAllClose(lsq_out, lsq_r)
        # with self.subTest('rlstsq'):
        self.assertArrayAllClose(la.rlstsq(m_ss, m_bs),
                                    gf.rlstsq(m_ss, m_bs))
        self.assertArrayAllClose(la.rlstsq(v_b, m_sb), gf.rlstsq(v_b, m_sb))

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a)', None))
    def test_functions_matldiv(self, arrays):
        m_ss, m_sb, m_bb, m_bs = arrays
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(m_sb.shape[-2] < m_sb.shape[-1])

        # with self.subTest('solve'):
        self.assertArrayAllClose(la.matldiv(m_ss, m_sb),
                                    gf.solve(m_ss, m_sb))
        slv_sh = gf.array_return_shape('(a,a),(a,b)->(a,b)', m_ss, m_sb)
        slv_out = np.empty(slv_sh, m_ss.dtype)
        slv_r = la.matldiv(m_ss, m_sb, out=slv_out)
        self.assertArrayAllClose(slv_out, slv_r)
        # with self.subTest('lstsq'):
        self.assertArrayAllClose(la.matldiv(m_bs, m_bb),
                                    gf.lstsq(m_bs, m_bb))
        self.assertArrayAllClose(la.matldiv(m_sb, m_ss),
                                    gf.lstsq(m_sb, m_ss))

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,a),(b)', None))
    def test_functions_matrdiv(self, arrays):
        m_ss, m_sb, m_bs = arrays[:-1]
        v_b = utn.core_only(arrays[-1], dims=1)
        hy.assume(np.all(utn.non_singular(m_ss)))
        hy.assume(m_sb.shape[-2] < m_sb.shape[-1])

        # with self.subTest('rsolve'):
        self.assertArrayAllClose(la.matrdiv(m_bs, m_ss),
                                     gf.rsolve(m_bs, m_ss))
        # with self.subTest('rlstsq'):
        self.assertArrayAllClose(la.matrdiv(m_ss, m_bs),
                                    gf.rlstsq(m_ss, m_bs))
        self.assertArrayAllClose(la.matrdiv(v_b, m_sb),
                                    gf.rlstsq(v_b, m_sb))

    @hy.given(utn.broadcastable('(a,b),(b,a)', None))
    def test_qr(self, arrays):
        m_sb, m_bs = arrays
        box = np.s_[..., :m_sb.shape[-2], :]
        hy.assume(m_sb.shape[-2] < m_sb.shape[-1])

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

    @hy.given(utn.broadcastable('(a,b),(b,a)', None))
    def test_lq(self, arrays):
        m_sb, m_bs = arrays
        box = np.s_[..., :m_sb.shape[-2]]
        hy.assume(m_sb.shape[-2] < m_sb.shape[-1])

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

    @hy.given(utn.broadcastable('(a,b),(b,a)', None))
    def test_lqr(self, arrays):
        m_sb, m_bs = arrays
        box = np.s_[..., :m_sb.shape[-2], :]
        hy.assume(m_sb.shape[-2] < m_sb.shape[-1])

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

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,a)', None))
    def test_lu(self, arrays):
        m_ss, m_sb, m_bs = arrays
        box = np.s_[..., :m_sb.shape[-2], :]
        hy.assume(m_sb.shape[-2] < m_sb.shape[-1])

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

    @unittest.expectedFailure
    @errstate
    @hy.given(utn.constant('(a,a)', ['d', 'D'], min_side=2))
    def test_low_rank(self, ones_ss):
        unitary, right = la.qr(ones_ss)
        self.assertArrayAllClose(unitary @ right, ones_ss)
        lower, upper, _ = la.lu(ones_ss)
        self.assertArrayAllClose(lower @ upper, ones_ss)
        with self.assertRaisesRegex(*utn.invalid_err):
            la.solve(ones_ss, ones_ss[...,:2])


# =============================================================================
if __name__ == '__main__':
    utn.main(verbosity=2)
