# Linear algebra enhancements for numpy

This package contains classes and functions that make the syntax for linear
algebra in `numpy` cleaner, particularly with respect to broadcasting and
matrix division. 

[Instructions for building the C modules below](#building-the-cpython-modules)

<!-- MDTOC maxdepth:6 firsth1:1 numbering:0 flatten:0 bullets:1 updateOnSave:1 -->

- [Linear algebra enhancements for numpy](#linear-algebra-enhancements-for-numpy)   
   - [Rationale](#rationale)   
   - [Requirements](#requirements)   
   - [Classes](#classes)   
   - [Functions](#functions)   
   - [GUfuncs](#gufuncs)   
   - [Wrappers](#wrappers)   
   - [Examples](#examples)   
   - [Building the C modules](#building-the-c-modules)   
   - [Running unit tests](#running-unit-tests)   
   - [To dos](#to-dos)   

<!-- /MDTOC -->

The main way of using this is package is via the `lnarray` class
(the `lstsq` and `qr` functions are the only other things I find useful here).
All of the functions will work with `numpy.ndarray` objects as well.
Furthermore, `matmul` is implemented with a `gufunc` which should make
broadcasting faster.

The `lnarray` class has properties `t` for transposing, `h` for
conjugate-transposing, `r` for row vectors, `c` for column vectors and `s` for
scalars in a way that fits with `numpy.linalg` broadcasting rules (`t,h` only
transpose the last two indices, `r,c,s` add singleton axes so that linear
algebra routines treat them as arrays of vectors/scalars rather than matrices,
and `ur,uc,us` undo the effects of `r,c,s`).

The `lnarray` class also has properties for delayed matrix division:
```python
>>> z = x.inv @ y
>>> z = x @ y.inv
>>> z = x.pinv @ y
>>> z = x @ y.pinv
```
None of the above actually invert the matrices. They return `invarray/pinvarray`
objects that call `solve/lstsq` behind the scenes, which is [faster and more
accurate](https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/).
To get the actual inverse matrices you can explicitly call the objects:
```python
>>> x = y.inv()
>>> x = y.pinv()
```

## Rationale

Consider the following expression that appeared in my work:
$A = \xi Z Q Z w$, where $Z = (e \xi - W)^{-1}$

In Matlab, I can compute this with:
```matlab
Zi = ev * xi - W
A = (xi / Zi) * Q * (Zi \ w);
```
Prior to Python 3.5 and Numpy 1.10, avoiding `inv`, I'd have to write it like this:
```python
Zi = np.dot(ev, xi) - W
A = np.dot(np.dot(np.linalg.solve(Zi.T, xi).T, Q), np.linalg.solve(Zi, w))
# or
A = np.linalg.solve(Zi.T, xi).T.dot(Q).dot(np.linalg.solve(Zi, w))
```
Things do get better in Python 3.5 and Numpy 1.10:
```python
A = np.linalg.solve(Zi.T, xi).T @ Q @ np.linalg.solve(Zi, w)
```
If I want it to broadcast I'd also have to replace `T` with `swapaxes(-2, -1)`.
Using this package, I can write it as
```python
A = xi @ Zi.inv @ Q @ (Zi.inv @ w)
# or even
Z = (ev @ xi - W).inv
A = xi @ Z @ Q @ (Z @ w)
```
The parentheses in the last expression are only there for efficiency,
it would still work without them.
Note that the `.inv` appears in the same place as the $^{-1}$ in the
mathematical expression above.

To do this, I reimplemented several `numpy` functions, some of them in `C`.
In each case, this wheel reinvention was done for one of the following reasons:
1. The `numpy` version doesn't broadcast (e.g. `lstsq`, `qr`).
1. The `numpy` version doesn't work well with subclasses (e.g. `matmul`).
1. The underlying `gufunc` is not part of the public API, so I didn't want to
rely on it.
1. I needed a `gufunc` version of `lstsq` neither requires an `rcond` input
nor returns any diagnostic information.
1. Completeness (e.g. `inv`).

I did not reimplement several `numpy.linalg` functions for one of the following
reasons:
1. There's no way to make it fit with the standard broadcasting rules
(e.g. `dot`, `tensordot`).
1. The `numpy` version already does everything I need (e.g. `eig`, `svd`).

## Requirements

* Python 3.7
* Numpy 1.16
* C compiler or prebuilt binaries in `numpy_linalg.gufuncs`
(see [below](#building-the-cpython-modules))
* BLAS/Lapack distribution that was present when the binaries were built

## Classes

* `lnarray`:
    Subclass of `numpy.ndarray` with properties such as `pinv/inv` for matrix
    division, `t` and `h` for transposing stacks of matrices, `r`, `c` and `s`
    for dealing with stacks of vectors and scalars.
* `invarray`:
    Performs exact matrix division when it is matrix multiplied (@).
    Returned by `lnarray.inv`. It calls `solve` behind the scenes.
    Does not actually invert the matrix unless it is explicitly called.
    Other operations, such as addition are not defined. This object contains a
    reference to the original array, so in place modifications of an `invarray`
    object will affect the original `lnarray` object.
    I think it is best not to store these objects in variables, and call on
    `lnarray.inv` on the rhs instead.
* `pinvarray`:
    Performs least-squares matrix division when it is matrix multiplied (@).
    Returned by `lnarray.pinv`. It calls `lstsq` behind the scenes.
    Does not actually pseudoinvert the matrix unless it is explicitly called.
    Other operations, such as addition are not defined. This object contains a
    reference to the original array, so in place modifications of a `pinvarray`
    object will affect the original `lnarray` object.
    I think it is best not to store these objects in variables, and call on
    `lnarray.pinv` on the rhs instead.

## Functions

The following implement by operators/properties of the classes above.
* `matmul`:
    Matrix multiplication with broadcasting and BLAS acceleration.
* `solve`:
    Linear equation solving (matrix left-division) with broadcasting and Lapack
    acceleration.
* `rsolve`:
    Reversed Linear equation solving (matrix right-division) with broadcasting.
* `lstsq`:
    Linear least squares solutions (matrix left-division) with broadcasting.
    Unlike `numnpy.linalg.lstsq`, this does not take an `rcond` parameter, or
    return diagnostic information (which is better suited to binary operators).
    However, it does broadcast and pass through subclasses.
* `rlstsq`:
    Reversed linear least squares problems (matrix right-division) with
    broadcasting.
* `transpose`:
    Transpose last two indices.
* `dagger`:
    Complex conjugate and transpose last two indices.
* `col`:
    Treat multi-dim array as a stack of column vectors.
* `row`:
    Treat multi-dim array as a stack of row vectors.
* `scalar`:
    Treat multi-dim array as a stack of scalars.
* `flattish`:
    Flatten a subset of axes.
* `expand_dims`:
    Add new singleton axes.    

The following do not implement operators/properties of the classes above.
* `matldiv`:
    Matrix division from left (exact or least-squares).
* `matrdiv`:
    Matrix division from right (exact or least-squares).
* `norm`:
    Vector 2-norm. Broadcasts and passes through subclasses.
* `lu`:
    LU decomposition with broadcasting and subclass passing.
* `qr`:
    QR decomposition with broadcasting and subclass passing. Does not implement
    the deprecated modes of `numpy.linalg.qr`.
* `lq`:
    LQ decomposition with broadcasting and subclass passing.
* `lqr`:
    For wide matrices LQ decomposition, otherwise QR decomposition.

The following operations will do the right thing, but may be better avoided:
```python
>>> matmul(invarray, invarray)
>>> solve(invarray, lnarray)
>>> solve(lnarray, invarray)
>>> solve(invarray, invarray)
>>> rsolve(invarray, lnarray)
>>> rsolve(lnarray, invarray)
>>> rsolve(invarray, invarray)
>>> lstsq(pinvarray, lnarray)
>>> lstsq(pinvarray, pinvarray)
>>> rlstsq(lnarray, pinvarray)
>>> rlstsq(pinvarray, pinvarray)
```
The following are not defined:
```python
>>> matmul(pinvarray, pinvarray)
>>> lstsq(lnarray, pinvarray)
>>> rlstsq(pinvarray, lnarray)
```
Combining `invarray` and `pinvarray` will either fail or produce weird results.

## GUfuncs

The following can be found in `numpy_linalg.gufuncs`:
* `gufuncs.matmul`:
    These are literally the same as the function above.
* `gufuncs.solve`:
* `gufuncs.rsolve`:
* `gufuncs.lstsq`:
* `gufuncs.rlstsq`:
* `gufuncs.norm`:
* `gufuncs.lu_m`:
    Implements `lu` for wide matrices.
* `gufuncs.lu_n`:
    Implements `lu` for narrow matrices.
* `gufuncs.lu_rawm`:
    Implements `lu` in `raw` mode.
* `gufuncs.lu_rawn`:
* `gufuncs.qr_m`:
    Implements `qr` for wide matrices in `reduced` mode, and all matrices in
    `complete` mode.
* `gufuncs.qr_n`:
    Implements `qr` for narrow matrices in `reduced` mode.
* `gufuncs.qr_rm`:
    Implement `qr` in `r` mode.
* `gufuncs.qr_rn`:
* `gufuncs.qr_rawm`:
    Implement `qr` in `raw` mode.
* `gufuncs.qr_rawn`:
* `gufuncs.lq_m`:
    Implement `lq`.
* `gufuncs.lq_n`:
* `gufuncs.lq_rm`:
* `gufuncs.lq_rn`:
* `gufuncs.lq_rawm`:
* `gufuncs.lq_rawn`:
* `gufuncs.pivot`:
    Perform row pivots with the output of `lu_*`.
* `gufuncs.rpivot`:
* `gufuncs.solve_lu`:
    Also return LU decomposition in `raw` form for future use.
* `gufuncs.rsolve_lu`:
* `gufuncs.lu_solve`:
    Use LU decomposition in `raw` form from previous use.
* `gufuncs.rlu_solve`:
* `gufuncs.lstsq_qrm`:
    Also return QR/LQ decomposition in `raw` form for future use.
* `gufuncs.lstsq_qrn`:
* `gufuncs.rlstsq_qrm`:
* `gufuncs.rlstsq_qrn`:
* `gufuncs.qr_lstsq`:
    Use QR/LQ decomposition in `raw` form from previous use.
* `gufuncs.rqr_lstsq`:
* `gufuncs.inv`:
    Matrix inverse.
* `gufuncs.inv_lu`:
    Also return LU decomposition in `raw` form for future use.
* `gufuncs.lu_inv`:
    Use LU decomposition in `raw` form from previous use.
* `gufuncs.pinv`:
    Moore-Penrose pseudoinverse.
* `gufuncs.pinv_qrm`:
    Also return QR/LQ decomposition in `raw` form for future use.
* `gufuncs.pinv_qrn`:
* `gufuncs.qr_pinv`:
    Use QR/LQ decomposition in `raw` form from previous use.
* `gufuncs.rmatmul`
    Reversed version of `matmul`. Used by `invarray`, otherwise not useful.
* `gufuncs.rtrue_tivide`:
    Reversed version of `np.true_divide`. Used by `pinvarray` and `invarray`, 
    otherwise not useful.

## Wrappers

* `wrappers.wrap_one`:
    Create version of `numpy` function with single `lnarray` output.
* `wrappers.wrap_several`:
    Create version of `numpy` function with multiple `lnarray` outputs.
* `wrappers.wrap_some`:
    Create version of `numpy` function with some `lnarray` outputs, some
    non-array outputs.
* `wrappers.wrap_sub`:
    Create version of `numpy` function with single `lnarray` output, passing
    through subclasses.
* `wrappers.wrap_subseveral`:
    Create version of `numpy` function with multiple `lnarray` outputs, passing
    through subclasses.
* `wrappers.wrap_subsome`:
    Create version of `numpy` function with some `lnarray` outputs, some
    non-array outputs, passing through subclasses.

## Examples

```python
>>> import numpy as np
>>> import numpy_linalg as la
>>> x = la.lnarray(np.random.rand(2, 3, 4))
>>> y = la.lnarray(np.random.rand(2, 3, 4))
>>> z = x.pinv @ y
>>> w = x @ y.pinv
>>> u = x @ y.t
>>> v = (x.r @ y[:, None, ...].t).ur
```

## Building the C modules

You will need to have the appropriate C compilers.
On Linux, you should already have them.
On Windows, [see here](https://wiki.python.org/moin/WindowsCompilers).

You will need a BLAS/Lapack distribution. Anaconda usually uses MKL, but they
recently moved the headers to a different package. You can install them with:
```
> conda install mkl-devel
```
Another option is [OpenBLAS](https://www.openblas.net/)
```
> conda install openblas -c conda-forge
```
([see here](https://docs.continuum.io/mkl-optimizations/#uninstalling-mkl)
under Uninstalling MKL).

If your BLAS/Lapack distribution is somewhere `numpy` isn't expecting, you can
provide directions in a
[site.cfg file](https://github.com/numpy/numpy/blob/master/site.cfg.example).

Once you have all of the above, you can build the C modules in-place:
```
> python setup.py build_ext
```
or you can install the package system-wide:
```
> python setup.py install
```
If you have `setuptools`, you can also do:
```
> python setup.py develop
```
this builds it in-place and creates an `.egg-link` file to make it available
system-wide.

Note: if you update to a new version of `numpy`, you might need to rebuild
the C modules.

## Running unit tests

You can test the build process and installation by running the unit tests.
```
> python -m unittest
```
You can expect occassional failures when using single precision floats.
The failure messages would have `sctype='f'` or `sctype='F'` in the titles and 
the mismatch displayed should be small, e.g. `Should be zero: 2.1e-5 at (2, 7)`.

You can customise which tests are run and how the results are displayed
using the command line options for the 
[unittest module](https://docs.python.org/3/library/unittest.html#command-line-interface).

## To dos

* SVD based versions of `lstsq_qr` and `qr_lstsq`
(and a QR based version of `lstsq` for completeness).
* Allow `invarray`/`pinvarray` to save/use LU/QR/SVD factors.
