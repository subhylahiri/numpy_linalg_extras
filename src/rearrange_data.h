/* -*- Mode: C -*- */
/* Common code for creating GUFuncs with BLAS/Lapack
*/
#ifndef GUF_REARRANGE
#define GUF_REARRANGE
/*
*****************************************************************************
**                            Includes                                     **
*****************************************************************************
*/
#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "gufunc_common_f.h"
/*
*****************************************************************************
**                            Factories                                    **
*****************************************************************************
*/
#define DECLARE_FUNC_LINEARIZE(SHAPE)                                                               \
    void linearize_FLOAT_## SHAPE(void *dst_in, const void *src_in, const LINEARIZE_DATA_t* data);  \
    void linearize_DOUBLE_## SHAPE(void *dst_in, const void *src_in, const LINEARIZE_DATA_t* data); \
    void linearize_CFLOAT_## SHAPE(void *dst_in, const void *src_in, const LINEARIZE_DATA_t* data); \
    void linearize_CDOUBLE_## SHAPE(void *dst_in, const void *src_in, const LINEARIZE_DATA_t* data);

#define DECLARE_FUNC_DELINEARIZE(SHAPE)                                                               \
    void delinearize_FLOAT_## SHAPE(void *dst_in, const void *src_in, const LINEARIZE_DATA_t* data);  \
    void delinearize_DOUBLE_## SHAPE(void *dst_in, const void *src_in, const LINEARIZE_DATA_t* data); \
    void delinearize_CFLOAT_## SHAPE(void *dst_in, const void *src_in, const LINEARIZE_DATA_t* data); \
    void delinearize_CDOUBLE_## SHAPE(void *dst_in, const void *src_in, const LINEARIZE_DATA_t* data);

#define DECLARE_FUNC_FILL(NAME, SHAPE)                                        \
    void NAME ##_FLOAT_## SHAPE(void *dst_in, const LINEARIZE_DATA_t* data);  \
    void NAME ##_DOUBLE_## SHAPE(void *dst_in, const LINEARIZE_DATA_t* data); \
    void NAME ##_CFLOAT_## SHAPE(void *dst_in, const LINEARIZE_DATA_t* data); \
    void NAME ##_CDOUBLE_## SHAPE(void *dst_in, const LINEARIZE_DATA_t* data);
/*
*****************************************************************************
**                           Declarations                                  **
*****************************************************************************
*/
DECLARE_FUNC_LINEARIZE(matrix)
DECLARE_FUNC_DELINEARIZE(matrix)
DECLARE_FUNC_DELINEARIZE(triu)
DECLARE_FUNC_DELINEARIZE(tril)
DECLARE_FUNC_FILL(nan, matrix)
DECLARE_FUNC_FILL(zero, matrix)
DECLARE_FUNC_FILL(zero, triu)
DECLARE_FUNC_FILL(zero, tril)
DECLARE_FUNC_FILL(eye, matrix)
DECLARE_FUNC_LINEARIZE(vec)
DECLARE_FUNC_DELINEARIZE(vec)
DECLARE_FUNC_FILL(nan, vec)

fortran_int FLOAT_real_int(fortran_real val);
fortran_int DOUBLE_real_int(fortran_doublereal val);
fortran_int CFLOAT_real_int(fortran_complex val);
fortran_int CDOUBLE_real_int(fortran_doublecomplex val);
/*
*****************************************************************************
**                          Integer versions                               **
*****************************************************************************
*/
static NPY_INLINE void
linearize_INT_vec(void *dst_in,
                const void *src_in,
                const LINEARIZE_DATA_t *data)
{
    fortran_int *dst = (fortran_int *) dst_in;
    npy_int *src = (npy_int *) src_in;

    if (dst) {
        fortran_int len = (fortran_int)data->columns;
        fortran_int strides = (fortran_int)(data->column_strides/sizeof(npy_int));
        int j;
        for (j = 0; j < len; ++j) {
            *dst = (fortran_int)*src;
            src += strides;
            dst += 1;
        }
    }
}

static NPY_INLINE void
delinearize_INT_vec(void *dst_in,
                    const void *src_in,
                    const LINEARIZE_DATA_t *data)
{
    fortran_int *src = (fortran_int *) src_in;
    npy_int *dst = (npy_int *) dst_in;

    if (dst) {
        fortran_int len = (fortran_int)data->columns;
        fortran_int strides = (fortran_int)(data->column_strides/sizeof(npy_int));
        int j;
        for (j = 0; j < len; ++j) {
            *dst = (npy_int)*src;
            src += 1;
            dst += strides;
        }
    }
}
/*
*****************************************************************************
*****************************************************************************
*/
#endif
