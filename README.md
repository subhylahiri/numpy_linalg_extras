# Numpy linear algebra enhancements

[Instructions for building the CPython modules below](#building-the-cpython-modules)

This package contains classes and functions that make the syntax for linear
algebra in `numpy` cleaner, particularly with respect to broadcasting and
matrix division. The main way of using this is via the `lnarray` class
(the `qr` function is the only other thing I find useful here). All of the
functions will work with `numpy.ndarray` objects as well. Furthermore,
`matmul` is implemented with a `gufunc` which should make broadcasting faster.

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
To get the actual inverse matrices you can call the objects:
```python
>>> x = y.inv()
>>> x = y.pinv()
```

## Classes

* `lnarray`:  
    Subclass of `numpy.ndarray` with properties such as `pinv/inv` for matrix
    division, `t` and `h` for transposing stacks of matrices, `r`, `c` and `s`
    for dealing with stacks of vectors and scalars.
* `pinvarray`:  
    Provides interface for matrix division when it is matrix multiplied (@).
    Returned by `lnarray.pinv`. It calls `lstsq` behind the scenes.
    Does not actually pseudoinvert the matrix unless it is explicitly called.
    Other operations, such as addition are not defined. This object contains a
    reference to the original array, so in place modifications of a `pinvarray`
    object will affect the original `lnarray` object.
    I think it is best not to store these objects in variables, and call on
    `lnarray.pinv` on the rhs instead.
* `invarray`:  
    Provides interface for matrix division when it is matrix multiplied (@).
    Returned by `lnarray.inv`. It calls `solve` behind the scenes.
    Does not actually invert the matrix unless it is explicitly called.
    Other operations, such as addition are not defined. This object contains a
    reference to the original array, so in place modifications of a `pinvarray`
    object will affect the original `lnarray` object.
    I think it is best not to store these objects in variables, and call on
    `lnarray.inv` on the rhs instead.

## Functions

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
* `norm`
    Vector 2-norm. Broadcasts and passes through subclasses.
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
* `matldiv`:  
    Matrix division from left (exact or least-squares).
* `matrdiv`:  
    Matrix division from right (exact or least-squares).
* `lu`:  
    LU decomposition with broadcasting and subclass passing.
* `qr`:  
    QR decomposition with broadcasting and subclass passing. Does not implement
    the deprecated modes of `numpy.linalg.qr`.

## GUfuncs

These implement the functions above.
* `gufuncs.matmul`:  
* `gufuncs.solve`:  
* `gufuncs.rsolve`:  
* `gufuncs.lstsq`:  
* `gufuncs.rlstsq`:  
* `gufuncs.norm`:  
    This is literally the same as the functions above.
* `gufuncs.lu_m`:  
    Implements `lu` for wide matrices.
* `gufuncs.lu_n`:  
    Implements `lu` for narrow matrices.
* `gufuncs.lu_rawm`:  
* `gufuncs.lu_rawn`:  
    Implements `lu` in `raw` mode.
* `gufuncs.qr_m`:  
    Implements `qr` for wide matrices in `reduced` mode, and all matrices in
    `complete` mode.
* `gufuncs.qr_n`:  
    Implements `qr` for narrow matrices in `reduced` mode.
* `gufuncs.qr_rm`:  
* `gufuncs.qr_rn`:  
    Implement `qr` in `r` mode.
* `gufuncs.qr_rawm`:  
* `gufuncs.qr_rawn`:  
    Implement `qr` in `raw` mode.
* `gufuncs.pivot`:  
* `gufuncs.rpivot`:  
    Perform row pivots with the output of `lu_*`.
* `gufuncs.solve_lu`:  
* `gufuncs.rsolve_lu`:  
    Also return LU decomposition in `raw` form for future use.
* `gufuncs.lu_solve`:  
* `gufuncs.rlu_solve`:  
    Use LU decomposition in `raw` form from previous use.
* `gufuncs.lstsq_qrm`:  
* `gufuncs.lstsq_qrn`:  
* `gufuncs.rlstsq_qrm`:  
* `gufuncs.rlstsq_qrn`:  
    Also return QR decomposition in `raw` form for future use.
* `gufuncs.qr_lstsq`:  
* `gufuncs.rqr_lstsq`:  
    Use QR decomposition in `raw` form from previous use.
* `gufuncs.rmatmul`
* `gufuncs.rtrue_tivide`:  
    Reversed versions of `matmul` and `np.true_divide`. Used by `pinvarray` and
    `invarray`.

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

## Requirements

* Python 3.6
* Numpy 1.15
* C compiler or prebuilt binaries in `numpy_linalg.gufuncs` (see below)
* BLAS/Lapack distribution that was present when the binaries were built

## Building the CPython modules

You will need to have the appropriate C compilers. On Linux, you should already have them.
On Windows, [see here](https://wiki.python.org/moin/WindowsCompilers).

You will need a BLAS/Lapack distribution. Anaconda usually uses MKL, but they
recently moved the headers to a different package. You can install them with:
```
> conda install mkl-devel
```
I found this wreaked havoc with its dependencies.
Alternatively, you can downgrade to a version that has the headers, e.g.
```
> conda install mkl=2018.0.3
```
Another option is [OpenBLAS](https://www.openblas.net/)
```
> conda install openblas -c conda-forge
```
([see here](https://docs.continuum.io/mkl-optimizations/#uninstalling-mkl) under
Uninstalling MKL).

If your BLAS/Lapack distribution is installed somewhere `numpy` isn't expecting,
you can provide directions in a [site.cfg file](https://github.com/numpy/numpy/blob/master/site.cfg.example).

Once you have all of the above, you can build the CPython modules in-place:
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


## To dos

* SVD based versions of `lstsq_qr` and `qr_lstsq`
(and a QR based version of `lstsq` for completeness).
* `Gufunc` version of `pinv`.
* Versions of `inv`/`pinv` that return/use LU/QR/SVD factors.
* Allow `invarray`/`pinvarray` to save/use LU/QR/SVD factors.
