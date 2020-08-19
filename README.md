# Linear algebra enhancements for numpy

This package contains classes and functions that make the syntax for linear
algebra in `numpy` cleaner, particularly with respect to broadcasting and
matrix division. 

[Instructions for building the C modules below](#building-the-c-modules)

<!-- MDTOC maxdepth:6 firsth1:1 numbering:0 flatten:0 bullets:1 updateOnSave:1 -->

- [Linear algebra enhancements for numpy](#linear-algebra-enhancements-for-numpy)   
   - [Rationale](#rationale)   
   - [Requirements](#requirements)   
   - [Classes](#classes)   
   - [Functions](#functions)   
   - [Array creation](#array-creation)   
   - [GUfuncs](#gufuncs)   
   - [Wrappers](#wrappers)   
   - [Examples](#examples)   
   - [Building the C modules](#building-the-c-modules)   
   - [Running unit tests](#running-unit-tests)   
   - [To dos](#to-dos)   

<!-- /MDTOC -->

The main way of using this is package is via the `lnarray` class
(`qr` functions are the only other things I find useful here).
All of the functions defined here will work with `numpy.ndarray` objects as well.
Most `numpy` functions will return an `lnarray` when any parameter is an `lnarray`.

The `lnarray` class has properties `t` for transposing, `h` for
conjugate-transposing, `r` for row vectors, `c` for column vectors and `s` for
scalars in a way that fits with `numpy.linalg` broadcasting rules (`t,h` only
transpose the last two indices, `r,c,s` add singleton axes so that linear
algebra routines treat them as arrays of vectors/scalars rather than matrices,
```python
>>> z = x @ y.inv
>>> z = x.pinv @ y
>>> z = x @ y.pinv
```
None of the above actually invert the matrices. 
They return `invarray/pinvarray` objects that call `solve/lstsq` behind the scenes, 
which is [faster and more accurate][dont-invert-matrix].
To get the actual inverse matrices you can explicitly call the objects:
```python
>>> x = y.inv()
>>> x = y.pinv()
```

You can create `lnarray`s by view casting from `ndarray`s.
This module also has versions of `NumPy`'s array creation routines that have been wrapped
to return `lnarray`s. [2](#footnotes)
```python
>>> x = np.arange(6).view(nl.lnarray)
>>> y = nl.arange(6)
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
Zi = ev @ xi - W
A = np.linalg.solve(Zi.T, xi).T @ Q @ np.linalg.solve(Zi, w)
```
If I want it to broadcast I'd also have to replace `T` with `swapaxes(-2, -1)`.

Using this package, however, I can write it as
```python
Zi = ev @ xi - W
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
1. The `numpy` version doesn't work well with subclasses (e.g. `norm`).
1. The underlying `gufunc` is not part of the public API, so I didn't want to
rely on it (e.g. `solve`).
1. I needed a `gufunc` version of `lstsq` neither requires an `rcond` input
nor returns any diagnostic information.
1. Completeness (e.g. `inv`).

I did not reimplement several `numpy.linalg` functions for one of the following
reasons:
1. There's no way to make it fit with the standard broadcasting rules
(e.g. `dot`, `tensordot`).
1. The `numpy` version already does everything I need (e.g. `eig`, `svd`).

## Requirements

* [Python 3.7](https://docs.python.org/3/)
* [Numpy 1.17](https://numpy.org/doc/stable/index.html)
* BLAS/Lapack distribution that was present when the binaries were built
* [to build] C compiler or prebuilt binaries in `numpy_linalg.gufuncs`
([see below](#building-the-cpython-modules))
* [to build] [Setuptools v41.0](https://setuptools.readthedocs.io) (recommended).
* [to test] [Hypothesis 5.8](https://hypothesis.readthedocs.io).

The version numbers above are minimum requirements only.
Checkout the branch `_v0.2.0` if you need NumPy 1.16 compatability.
Checkout the branch `_v0.1.0` if you need Python 3.6 or NumPy 1.15 compatability.

## Classes

* `lnarray`:
    Subclass of `numpy.ndarray` with properties such as `pinv/inv` for matrix
    division, `t` and `h` for transposing stacks of matrices, `r`, `c` and `s`
    for dealing with stacks of vectors and scalars.
* `invarray`:
    Performs exact matrix division when it is matrix multiplied (@).
    Returned by `lnarray.inv`.
    Does not actually invert the matrix unless it is explicitly called,
    it calls `solve` behind the scenes instead.
    Other operations, such as addition are not defined. This object contains a
    reference to the original array, so in place modifications of an `invarray`
    object will affect the original `lnarray` object, and *vice versa*.
    I think it is best not to store these objects in variables, and call on
    `lnarray.inv` on the rhs instead.
* `pinvarray`:
    Performs least-squares matrix division when it is matrix multiplied (@).
    Returned by `lnarray.pinv`. 
    Does not actually pseudoinvert the matrix unless it is explicitly called,
    it calls `lstsq` behind the scenes instead.
    Other operations, such as addition are not defined. This object contains a
    reference to the original array, so in place modifications of a `pinvarray`
    object will affect the original `lnarray` object, and *vice versa*.
    I think it is best not to store these objects in variables, and call on
    `lnarray.pinv` on the rhs instead.

## Functions

The following implement operators/properties of the classes above.
* `matmul`:
    Alias for `numpy.matmul` [1](#footnotes).
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
* `foldaxis`:
    Fold an axis.
* `expand_dims`:
    Add new singleton axes.    

The following are not operators/properties of the classes above.
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
>>> matmul(lhs_array.inv, rhs_array.inv)   -> matmul(rhs_array, lhs_array).inv
>>> solve(lhs_array.inv, rhs_array)        -> matmul(lhs_array, rhs_array)
>>> solve(lhs_array, rhs_array.inv)        -> matmul(rhs_array, lhs_array).inv
>>> solve(lhs_array.inv, rhs_array.inv)    -> rsolve(lhs_array, rhs_array)
>>> rsolve(lhs_array.inv, rhs_array)       -> matmul(rhs_array, lhs_array).inv
>>> rsolve(lhs_array, rhs_array.inv)       -> matmul(lhs_array, rhs_array)
>>> rsolve(lhs_array.inv, rhs_array.inv)   -> solve(lhs_array, rhs_array)
>>> lstsq(lhs_array.pinv, rhs_array)       -> matmul(lhs_array, rhs_array)
>>> lstsq(lhs_array.pinv, rhs_array.pinv)  -> rlstsq(lhs_array, rhs_array)
>>> rlstsq(lhs_array, rhs_array.pinv)      -> matmul(lhs_array, rhs_array)
>>> rlstsq(lhs_array.pinv, rhs_array.pinv) -> lstsq(lhs_array, rhs_array)
>>> lstsq(lhs_array.inv, rhs_array)        -> matmul(lhs_array, rhs_array)
>>> rlstsq(lhs_array, rhs_array.inv)       -> matmul(lhs_array, rhs_array)
>>> lstsq(lhs_array.inv, rhs_array.pinv)   -> rlstsq(lhs_array, rhs_array)
>>> rlstsq(lhs_array.inv, rhs_array.pinv)  -> solve(lhs_array, rhs_array)
>>> lstsq(lhs_array.inv, rhs_array.inv)    -> rsolve(lhs_array, rhs_array)
>>> rlstsq(lhs_array.inv, rhs_array.inv)   -> solve(lhs_array, rhs_array)
>>> lstsq(lhs_array.pinv, rhs_array.inv)   -> rsolve(lhs_array, rhs_array)
>>> rlstsq(lhs_array.pinv, rhs_array.inv)  -> lstsq(lhs_array, rhs_array)
>>> solve(lhs_array.inv, rhs_array.pinv)   -> rlstsq(lhs_array, rhs_array)
>>> rsolve(lhs_array.pinv, rhs_array.inv)  -> lstsq(lhs_array, rhs_array)
```
The following are not defined:
```python
>>> matmul(pinvarray, pinvarray)
>>> lstsq(lnarray, pinvarray)
>>> rlstsq(pinvarray, lnarray)
>>> lstsq(lnarray, invarray)
>>> rlstsq(invarray, lnarray)
>>> solve(pinvarray, lnarray)
>>> rsolve(pinvarray, lnarray)
>>> solve(lnarray, pinvarray)
>>> rsolve(lnarray, pinvarray)
>>> solve(pinvarray, pinvarray)
>>> rsolve(pinvarray, pinvarray)
>>> matmul(invarray, pinvarray)
>>> matmul(pinvarray, invarray)
>>> solve(pinvarray, invarray)
>>> rsolve(invarray, pinvarray)
```

## Array creation

This module has versions of `NumPy`'s array creation routines that have been wrapped
to return `lnarray`s [2](#footnotes). 
These are those functions that return arrays from non-array inputs, namely:  
`empty`, `eye`, `identity`, `ones`, `zeros`, `full`,
`array`, `asarray`, `asanyarray`, `ascontiguousarray`, `asfarray`, `asfortranarray`, `asarray_chkfinite`, 
`copy`, `require`, `load`, `loadtxt`, `genfromtxt`,
`fromfile`, `fromregex`, `frombuffer`, `fromstring`, `fromfunction`, `fromiter`,
`arange`, `linspace`, `logspace`, `geomspace`, `meshgrid`,
`ravel_multi_index`, `unravel_index`, `diag_indices`, `mask_indices`,
`tril_indices`, `triu_indices`, `indices`, `mgrid`, `ogrid`, `r_`, `c_`.
All except for `memmap`, which can still be view-cast to an `lnarray`.

The instances `mgrid`, `ogrid`, `r_`, `c_` return `lnarray`s when subscripted.
When `r_` and `c_` are used to concatenate, they will convert `ndarray`s to `lnarray`s.

In addition we have the following in submodules:
* `fft.fftfreq`, `fft.rfftfreq`:
    These are wrapped versions of the `numpy` functions.
* `random.default_rng`:
    This returns a version of `numpy.random.Generator` whose methods return `lnarray`s 
    instead of `ndarrays`.

The `random` module also has wrapped versions of the legacy functions in `numpy.random`, 
but not the `RandomState` class.

All of these functions and objects will probably confuse your IDE's autocomplete, etc.

## GUfuncs

The following can be found in `numpy_linalg.gufuncs`:
* `gufuncs.matmul`:
    This is an alias for `numpy.matmul` [1](#footnotes).
* `gufuncs.solve`:
    These are literally the same as the function above.
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

Tools for wrapping `numpy` functions to return `lnarrays`.
These classes can be found in `numpy_linalg.wrappers`:
* `wrappers.Wrappers`:
    Class with methods to wrap `numpy` functions to return `lnarray`s instead
    of `ndarray`s, e.g.:
    * `wrappers.Wrappers.one`:
        Create version of `numpy` function with single `lnarray` output.
    * `wrappers.Wrappers.several:`
        Create version of `numpy` function with multiple `lnarray` outputs.
    * `wrappers.Wrappers.some:`
        Create version of `numpy` function with some `lnarray` outputs, some
        non-array outputs.
* `wrappers.WrappedClass`:
    When this class is subclassed, the resulting class has the same methods as
    the object passed in the constructor, except they return `lnarray`s instead
    of `ndarray`s.
* `wrappers.WrappedSubscriptable`:
    When this class is subclassed, the resulting class's instances can be
    subscripted in the same manner as the object passed in the constructor,
    except it will return `lnarray`s instead of `ndarray`s.

## Testing

Tools for writing unit tests.
They can be found in `numpy_linalg.testing`:
* `testing.main`:
    Runs unit tests when called. By default it does not sort the tests.
* `testing.load_tests_helper`:
    Implements the `load_tests` protocol to load tests in order of definition.
* `testing.TestCaseNumpy`:
    A subclass of `unittest.TestCase` with methods for testing arrays.
    It can be used as a base class for your test cases.
* `testing.unittest_tweaks`:
    Module with classes related to `main` and `load_tests_helper`.
* `testing.unittest_numpy`:
    Module with tools for testing arrays.
* `testing.hypothesis_numpy`:
    Module with tools for `hypothesis` to generate examples for unit tests.

## Converters

Helpers for writing `__array_ufunc__` methods.
They can be found in `numpy_linalg.convert`:
* `convert.conv_in_attr`:
    Process inputs to an `__array_ufunc__` method using an attribute.
* `convert.conv_in_view`:
    Process inputs to an `__array_ufunc__` method using view method.
* `convert.conv_out_attr`:
    Process outputs to an `__array_ufunc__` method using an attribute.
* `convert.conv_out_init`:
    Process outputs from an `__array_ufunc__` method using a constructor.
* `convert.conv_out_view`:
    Process outputs from an `__array_ufunc__` method using a view method.


## Other utilities

These can be found in `numpy_linalg.gufuncs`:
* `gufuncs.unbroadcast_factors`:
    Undo broadcasting in factors returned by gufuncs.
* `gufuncs.make_errobj`:
    create an error handler object for use as ``extobj`` gufunc parameter.
* `gufuncs.return_shape_mat`:
    Shape of result of broadcasted matrix operation, from shapes.
* `gufuncs.array_return_shape_mat`:
    Shape of result of broadcasted matrix operation, from arrays.


## Examples

```python
>>> import numpy as np
>>> import numpy_linalg as la
>>> x = la.random.rand(2, 3, 4)
>>> y = la.random.rand(2, 3, 4)
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
([see here under "Uninstalling MKL"][uninstall-MKL]).

If your BLAS/Lapack distribution is somewhere `numpy` isn't expecting, you can
provide directions in a [site.cfg file].

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

Note: if you update to a new version of `python` or `numpy`, you might need to rebuild
the C modules.

## Running unit tests

You can test the build process and installation by running the unit tests 
(which require [the `hypothesis` package](https://hypothesis.readthedocs.io)).
Execute any of the following commands in the folder containing this file:
```
> python -m tests
> python -m unittest
> python -m unittest discover -s <folder/containing/README.md>
```
You can customise which tests are run and how the results are displayed using 
[the command line options for the `unittest` module][unittest-cli].

You can expect occassional failures when using single precision floats.
The 'falsifying example' produced by `hypothesis` would have `dtype=numpy.float32` 
or `dtype=numpy.complex64`. The mismatch displayed by `unittest` should be small, 
e.g. `Largest mismatch: 2.1e-5 at (2, 7) with dtype=float32`.

Because the underlying BLAS/LAPACK routines raise runtime warnings when passed 
`inf` or `nan`, these values are excluded from tests. 
Most of these functions return all `nan`s in such cases.
Low rank matrices are not properly tested yet.

## To dos

* SVD based versions of `lstsq_qr` and `qr_lstsq`
(and a QR based version of `lstsq` for completeness).
* Allow `invarray`/`pinvarray` to save/use LU/QR/SVD factors/inverse.
* Write other norms.
* Wrap `memmap` & test.
* Wrap `masked_array` & test.

#

### Footnotes

1. This package previously used a custom `gufunc` for `matmul`, 
    but as of v1.16 `NumPy` does this so we use that instead.
2. This package currently also has wrapped versions of `NumPy`'s array manipulation routines, 
    but as of `NumPy` v1.17, the `__array_function__` protocol has removed their need.
    When called with an `lnarray` parameter, the numpy functions now return `lnarray`s.
    Technically, they can be thought of as array creation routines when called with 
    array-like parameters, but we will rely on the `numpy` versions in the future. 
    The implementations in this package now issue deprecation warnings.

[dont-invert-matrix]: <https://www.johndcook.com/blog/2010/01/19/dont-invert-that-matrix/> "Blog post about matrix inversion."

[uninstall-mkl]: <https://docs.continuum.io/mkl-optimizations/#uninstalling-mkl> "Uninstall MKL to use OpenBLAS."

[site.cfg file]: <https://github.com/numpy/numpy/blob/master/site.cfg.example> "Example site.cfg file."

[unittest-cli]: <https://docs.python.org/3/library/unittest.html#command-line-interface> "Unittest command line options."
