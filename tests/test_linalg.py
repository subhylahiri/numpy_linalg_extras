# -*- coding: utf-8 -*-
"""Test python wrappers of gufuncs.
It has been established, in test_gu*.py, that the gufuncs return the correct
values. We just check that the python wrappers call the correct ones.
"""
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


# =============================================================================
# Test python funcs
# =============================================================================


class TestShape(utn.TestCaseNumpy):
    """Testing shapes returned by linalg functions"""

    @hy.given(hyn.arrays('d', hyn.array_shapes(min_dims=2)))
    def test_shape_functions(self, array):
        # shape
        shape = array.shape
        self.assertArrayShape(la.transpose(array), trnsp(shape))
        self.assertArrayShape(la.row(array), insert(shape))
        self.assertArrayShape(la.col(array), shape + (1,))
        self.assertArrayShape(la.scalar(array), shape + (1, 1))

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_linalg_functions(self, arrays):
        a_ss, a_sb, a_bb, a_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(np.all(utn.non_singular(a_ss)))
        hy.assume(np.all(utn.non_singular(a_bb)))
        # matmul
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', tall, wide)
        self.assertArrayShape(la.matmul(a_bs, a_sb), expect)
        self.assertArrayShape(la.matmul(a_bs, v_s), tall[:-1])
        self.assertArrayShape(la.matmul(v_b, a_bs), drop(tall))
        self.assertArrayShape(la.matmul(v_s, v_s), ())
        # rmatmul
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', wide, tall)
        self.assertArrayShape(lr.rmatmul(a_bs, a_sb), expect)
        self.assertArrayShape(lr.rmatmul(a_sb, v_s), drop(wide))
        self.assertArrayShape(lr.rmatmul(v_b, a_sb), wide[:-1])
        self.assertArrayShape(lr.rmatmul(v_b, v_b), ())
        # solve
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', smol, wide)
        self.assertArrayShape(la.solve(a_ss, a_sb),expect)
        self.assertArrayShape(la.solve(a_ss, v_s), smol[:-1])
        # rsolve
        expect = gf.return_shape('(a,b),(b,c)->(a,c)', tall, smol)
        self.assertArrayShape(la.rsolve(a_bs, a_ss), expect)
        self.assertArrayShape(la.rsolve(v_s, a_ss), smol[:-1])
        # lstsq
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', tall, big)
        self.assertArrayShape(la.lstsq(a_bs, a_bb), expect)
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', wide, smol)
        self.assertArrayShape(la.lstsq(a_sb, a_ss), expect)
        self.assertArrayShape(la.lstsq(a_sb, v_s), drop(wide))
        self.assertArrayShape(la.lstsq(v_s, a_ss), smol[:-1])
        # rlstsq
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', smol, tall)
        self.assertArrayShape(la.rlstsq(a_ss, a_bs), expect)
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', big, wide)
        self.assertArrayShape(la.rlstsq(a_bb, a_sb), expect)
        self.assertArrayShape(la.rlstsq(v_b, a_sb), wide[:-1])
        self.assertArrayShape(la.rlstsq(a_ss, v_s), smol[:-1])
        self.assertArrayShape(la.rlstsq(v_s, a_bs), tall[:-1])

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', 'd'))
    def test_matdiv_functions(self, arrays):
        a_ss, a_sb, a_bb, a_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        smol, wide, big, tall = [arr.shape for arr in arrays[:-2]]
        hy.assume(np.all(utn.non_singular(a_ss)))
        hy.assume(np.all(utn.non_singular(a_bb)))
        hy.assume(wide[-2] < wide[-1])
        # solve
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', smol, wide)
        self.assertArrayShape(la.matldiv(a_ss, a_sb), expect)
        self.assertArrayShape(la.matldiv(a_ss, v_s), smol[:-1])
        # rsolve
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', tall, smol)
        self.assertArrayShape(la.matrdiv(a_bs, a_ss), expect)
        self.assertArrayShape(la.matrdiv(v_s, a_ss), smol[:-1])
        # lstsq
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', tall, big)
        self.assertArrayShape(la.matldiv(a_bs, a_bb), expect)
        expect = gf.return_shape('(a,b),(a,c)->(b,c)', wide, smol)
        self.assertArrayShape(la.matldiv(a_sb, a_ss), expect)
        self.assertArrayShape(la.matldiv(a_sb, v_s), drop(wide))
        self.assertArrayShape(la.matldiv(v_s, a_ss), smol[:-1])
        # rlstsq
        expect = gf.return_shape('(a,b),(c,b)->(a,c)', smol, tall)
        self.assertArrayShape(la.matrdiv(a_ss, a_bs), expect)
        self.assertArrayShape(la.matrdiv(v_b, a_sb), wide[:-1])
        self.assertArrayShape(la.matrdiv(a_ss, v_s), smol[:-1])
        self.assertArrayShape(la.matrdiv(v_s, a_bs), tall[:-1])

    @hy.given(utn.broadcastable('(a,b),(b,a)', 'd'))
    def test_qr(self, arrays):
        a_sb, a_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        widb, talb = wide[:-2], tall[:-2]  # broadcasting shape
        mini, maxi = (wide[-2],), (wide[-1],)
        hy.assume(wide[-2] < wide[-1])

        self.assertArrayShapesAre(la.qr(a_bs, 'reduced'), (tall, talb + 2*mini))
        self.assertArrayShapesAre(la.qr(a_sb, 'reduced'), (widb + 2*mini, wide))
        self.assertArrayShapesAre(la.qr(a_bs, 'complete'), (talb+2*maxi, tall))
        self.assertArrayShapesAre(la.qr(a_sb, 'complete'), (widb+2*mini, wide))
        self.assertArrayShape(la.qr(a_bs, 'r'), talb + 2*mini)
        self.assertArrayShape(la.qr(a_sb, 'r'), wide)
        self.assertArrayShapesAre(la.qr(a_bs, 'raw'), (trnsp(tall), talb+mini))
        self.assertArrayShapesAre(la.qr(a_sb, 'raw'), (trnsp(wide), widb+mini))

    @hy.given(utn.broadcastable('(a,b),(b,a)', 'd'))
    def test_lq(self, arrays):
        a_sb, a_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        widb, talb = wide[:-2], tall[:-2]  # broadcasting shape
        mini, maxi = (wide[-2],), (wide[-1],)
        hy.assume(wide[-2] < wide[-1])

        self.assertArrayShapesAre(la.lq(a_bs, 'reduced'), (tall, talb + 2*mini))
        self.assertArrayShapesAre(la.lq(a_sb, 'reduced'), (widb + 2*mini, wide))
        self.assertArrayShapesAre(la.lq(a_bs, 'complete'), (tall, talb+2*mini))
        self.assertArrayShapesAre(la.lq(a_sb, 'complete'), (wide, widb+2*maxi))
        self.assertArrayShape(la.lq(a_bs, 'l'), tall)
        self.assertArrayShape(la.lq(a_sb, 'l'), widb + 2*mini)
        self.assertArrayShapesAre(la.lq(a_bs, 'raw'), (trnsp(tall), talb+mini))
        self.assertArrayShapesAre(la.lq(a_sb, 'raw'), (trnsp(wide), widb+mini))

    @hy.given(utn.broadcastable('(a,b),(b,a)', 'd'))
    def test_lqr(self, arrays):
        a_sb, a_bs = arrays
        wide, tall = [arr.shape for arr in arrays]
        widb, talb = wide[:-2], tall[:-2]  # broadcasting shape
        mini, maxi = (wide[-2],), (wide[-1],)
        hy.assume(wide[-2] < wide[-1])

        self.assertArrayShapesAre(la.lqr(a_bs, 'reduced'), (tall, talb+2*mini))
        self.assertArrayShapesAre(la.lqr(a_sb, 'reduced'), (widb+2*mini, wide))
        self.assertArrayShapesAre(la.lqr(a_bs, 'complete'), (talb+2*maxi, tall))
        self.assertArrayShapesAre(la.lqr(a_sb, 'complete'), (wide, widb+2*maxi))
        self.assertArrayShape(la.lqr(a_bs, 'l'), talb+2*mini)
        self.assertArrayShape(la.lqr(a_sb, 'l'), widb+2*mini)
        self.assertArrayShape(la.lqr(a_bs, 'r'), talb+2*mini)
        self.assertArrayShape(la.lqr(a_sb, 'r'), widb+2*mini)
        self.assertArrayShapesAre(la.lqr(a_bs, 'raw'), (trnsp(tall), talb+mini))
        self.assertArrayShapesAre(la.lqr(a_sb, 'raw'), (trnsp(wide), widb+mini))

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,a)', 'd'))
    def test_lu(self, arrays):
        a_ss, a_sb, a_bs = arrays
        smol, wide, tall = [arr.shape for arr in arrays]
        smob, widb, talb = smol[:-2], wide[:-2], tall[:-2]  # broadcasting shape
        mini, maxi = (wide[-2],), (wide[-1],)
        hy.assume(wide[-2] < wide[-1])

        self.assertArrayShapesAre(la.lu(a_ss, 'separate'),
                                  (smol, smol, smol[:-1]))
        self.assertArrayShapesAre(la.lu(a_bs, 'separate'),
                                  (tall, talb + 2*mini, talb + mini))
        self.assertArrayShapesAre(la.lu(a_sb, 'separate'),
                                  (widb + 2*mini, wide, widb + mini))
        self.assertArrayShapesAre(la.lu(a_ss, 'raw'), (smol, smob+mini))
        self.assertArrayShapesAre(la.lu(a_bs, 'raw'), (trnsp(tall), talb+mini))
        self.assertArrayShapesAre(la.lu(a_sb, 'raw'), (trnsp(wide), widb+mini))


class TestValue(utn.TestCaseNumpy):
    """Testing values returned by linalg functions"""

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', None))
    def test_linalg_functions(self, arrays):
        a_ss, a_sb, a_bb, a_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        hy.assume(np.all(utn.non_singular(a_ss)))
        hy.assume(np.all(utn.non_singular(a_bb)))
        hy.assume(a_sb.shape[-2] < a_sb.shape[-1])
        # matmul
        self.assertArrayAllClose(la.matmul(a_bs, a_sb), gf.matmul(a_bs, a_sb))
        # rmatmul
        self.assertArrayAllClose(lr.rmatmul(a_bs, a_sb), gf.rmatmul(a_bs, a_sb))
        # solve
        self.assertArrayAllClose(la.solve(a_ss, a_sb), gf.solve(a_ss, a_sb))
        slv_sh = gf.array_return_shape('(a,a),(a,b)->(a,b)', a_ss, a_sb)
        slv_out = np.empty(slv_sh, a_ss.dtype)
        slv_r = la.solve(a_ss, a_sb, out=slv_out)
        self.assertArrayAllClose(slv_out, slv_r)
        # rsolve
        self.assertArrayAllClose(la.rsolve(a_bs, a_ss), gf.rsolve(a_bs, a_ss))
        # lstsq
        self.assertArrayAllClose(la.lstsq(a_bs, a_bb), gf.lstsq(a_bs, a_bb))
        self.assertArrayAllClose(la.lstsq(a_sb, a_ss), gf.lstsq(a_sb, a_ss))
        lsq_sh = gf.array_return_shape('(a,b),(a,c)->(b,c)', a_bs, a_bb)
        lsq_out = np.empty(lsq_sh, a_bs.dtype)
        lsq_r = la.lstsq(a_bs, a_bb, out=lsq_out)
        self.assertArrayAllClose(lsq_out, lsq_r)
        # rlstsq
        self.assertArrayAllClose(la.rlstsq(a_ss, a_bs), gf.rlstsq(a_ss, a_bs))
        self.assertArrayAllClose(la.rlstsq(v_b, a_sb), gf.rlstsq(v_b, a_sb))

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,b),(b,a),(a),(b)', None))
    def test_matdiv_functions(self, arrays):
        a_ss, a_sb, a_bb, a_bs = arrays[:-2]
        v_s, v_b = utn.core_only(*arrays[-2:], dims=1)
        hy.assume(np.all(utn.non_singular(a_ss)))
        hy.assume(np.all(utn.non_singular(a_bb)))
        hy.assume(a_sb.shape[-2] < a_sb.shape[-1])
        # solve
        self.assertArrayAllClose(la.matldiv(a_ss, a_sb), gf.solve(a_ss, a_sb))
        slv_sh = gf.array_return_shape('(a,a),(a,b)->(a,b)', a_ss, a_sb)
        slv_out = np.empty(slv_sh, a_ss.dtype)
        slv_r = la.matldiv(a_ss, a_sb, out=slv_out)
        self.assertArrayAllClose(slv_out, slv_r)
        # rsolve
        self.assertArrayAllClose(la.matrdiv(a_bs, a_ss), gf.rsolve(a_bs, a_ss))
        # lstsq
        self.assertArrayAllClose(la.matldiv(a_bs, a_bb), gf.lstsq(a_bs, a_bb))
        self.assertArrayAllClose(la.matldiv(a_sb, a_ss), gf.lstsq(a_sb, a_ss))
        # rlstsq
        self.assertArrayAllClose(la.matrdiv(a_ss, a_bs), gf.rlstsq(a_ss, a_bs))
        self.assertArrayAllClose(la.matrdiv(v_b, a_sb), gf.rlstsq(v_b, a_sb))

    @hy.given(utn.broadcastable('(a,b),(b,a)', None))
    def test_qr(self, arrays):
        a_sb, a_bs = arrays
        box = np.s_[..., :a_sb.shape[-2], :]
        hy.assume(a_sb.shape[-2] < a_sb.shape[-1])

        unitary, right = la.qr(a_bs, 'reduced')
        self.assertArrayAllClose(unitary @ right, a_bs)
        unitary, right = la.qr(a_sb, 'reduced')
        self.assertArrayAllClose(unitary @ right, a_sb)
        unitary, right = la.qr(a_bs, 'complete')
        self.assertArrayAllClose(unitary @ right, a_bs)
        unitary, right = la.qr(a_sb, 'complete')
        self.assertArrayAllClose(unitary @ right, a_sb)
        right = la.qr(a_bs, 'r')
        hhold, _ = la.qr(a_bs, 'raw')
        self.assertArrayAllClose(right, np.triu(la.transpose(hhold))[box])
        right = la.qr(a_sb, 'r')
        hhold, _ = la.qr(a_sb, 'raw')
        self.assertArrayAllClose(right, np.triu(la.transpose(hhold)))

    @hy.given(utn.broadcastable('(a,b),(b,a)', None))
    def test_lq(self, arrays):
        a_sb, a_bs = arrays
        box = np.s_[..., :a_sb.shape[-2]]
        hy.assume(a_sb.shape[-2] < a_sb.shape[-1])

        left, unitary = la.lq(a_bs, 'reduced')
        self.assertArrayAllClose(left @ unitary, a_bs)
        left, unitary = la.lq(a_sb, 'reduced')
        self.assertArrayAllClose(left @ unitary, a_sb)
        left, unitary = la.lq(a_bs, 'complete')
        self.assertArrayAllClose(left @ unitary, a_bs)
        left, unitary = la.lq(a_sb, 'complete')
        self.assertArrayAllClose(left @ unitary, a_sb)
        left = la.lq(a_bs, 'l')
        hhold, _ = la.lq(a_bs, 'raw')
        self.assertArrayAllClose(left, np.tril(la.transpose(hhold)))
        left = la.lq(a_sb, 'l')
        hhold, _ = la.lq(a_sb, 'raw')
        self.assertArrayAllClose(left, np.tril(la.transpose(hhold))[box])

    @hy.given(utn.broadcastable('(a,b),(b,a)', None))
    def test_lqr(self, arrays):
        a_sb, a_bs = arrays
        box = np.s_[..., :a_sb.shape[-2], :]
        hy.assume(a_sb.shape[-2] < a_sb.shape[-1])

        unitary, right = la.lqr(a_bs, 'reduced')
        self.assertArrayAllClose(unitary @ right, a_bs)
        left, unitary = la.lqr(a_sb, 'reduced')
        self.assertArrayAllClose(left @ unitary, a_sb)
        unitary, right = la.lqr(a_bs, 'complete')
        self.assertArrayAllClose(unitary @ right, a_bs)
        left, unitary = la.lqr(a_sb, 'complete')
        self.assertArrayAllClose(left @ unitary, a_sb)
        right = la.lqr(a_bs, 'r')
        hhold, _ = la.lqr(a_bs, 'raw')
        self.assertArrayAllClose(right, np.triu(la.transpose(hhold))[box])
        left = la.lqr(a_sb, 'r')
        hhold, _ = la.lqr(a_sb, 'raw')
        self.assertArrayAllClose(left, np.tril(la.transpose(hhold))[box[:-1]])

    @hy.given(utn.broadcastable('(a,a),(a,b),(b,a)', None))
    def test_lu(self, arrays):
        a_ss, a_sb, a_bs = arrays
        box = np.s_[..., :a_sb.shape[-2], :]
        hy.assume(a_sb.shape[-2] < a_sb.shape[-1])

        lower, upper, piv = la.lu(a_ss, 'separate')
        luf, piv = la.lu(a_ss, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(lower @ upper, gf.pivot(a_ss, piv))
        self.assertArrayAllClose(np.tril(lower, -1), np.tril(luf, -1))
        self.assertArrayAllClose(upper, np.triu(luf))
        lower, upper, piv = la.lu(a_bs, 'separate')
        luf, piv = la.lu(a_bs, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(np.tril(lower, -1), np.tril(luf, -1))
        self.assertArrayAllClose(upper, np.triu(luf)[box])
        lower, upper, piv = la.lu(a_sb, 'separate')
        luf, piv = la.lu(a_sb, 'raw')
        luf = la.transpose(luf)
        self.assertArrayAllClose(np.tril(lower, -1), np.tril(luf, -1)[box[:-1]])
        self.assertArrayAllClose(upper, np.triu(luf))

    @errstate
    @hy.given(utn.constant('(a,a)', ['d', 'D'], min_side=2))
    def test_low_rank(self, ones_ss):
        unitary, right = la.qr(ones_ss)
        self.assertArrayAllClose(unitary @ right, ones_ss)
        lower, upper, _ = la.lu(ones_ss)
        self.assertArrayAllClose(lower @ upper, ones_ss)
        with self.assertRaisesRegex(*utn.invalid_err):
            la.solve(ones_ss, ones_ss)


# =============================================================================
if __name__ == '__main__':
    utn.main(verbosity=2)
