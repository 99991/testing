import numpy as np
import scipy.sparse
import os
from ctypes import c_int, c_double, pointer, POINTER, CDLL

err = os.system("gcc -O3 -o ichol.so -fPIC -shared ichol.c")
assert(err == 0)

c_int_p = POINTER(c_int)
c_double_p = POINTER(c_double)

library = CDLL("./ichol.so")

_ichol_c = library._ichol
_ichol_c.restype = c_int
_ichol_c.argtypes = [
    c_int,
    c_double_p,
    c_int_p,
    c_int_p,
    c_double_p,
    c_int_p,
    c_int_p,
    c_double,
    c_int]

_backsub_L_csc_inplace = library._backsub_L_csc_inplace
_backsub_L_csc_inplace.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_double_p,
    c_int]

_backsub_LT_csc_inplace = library._backsub_LT_csc_inplace
_backsub_LT_csc_inplace.argtypes = [
    c_double_p,
    c_int_p,
    c_int_p,
    c_double_p,
    c_int]

class CholeskyDecomposition(object):
    def __init__(self, Ltuple):
        self.Ltuple = Ltuple

    @property
    def L(self):
        Lv, Lr, Lp = self.Ltuple
        n = len(Lp) - 1
        return scipy.sparse.csc_matrix(self.Ltuple, (n, n))

    def __call__(self, b):
        Lv, Lr, Lp = self.Ltuple
        assert(Lv.dtype == np.float64)
        assert(Lr.dtype == np.int32)
        assert(Lp.dtype == np.int32)
        n = len(b)
        x = b.copy()
        _backsub_L_csc_inplace(Lv.ctypes.data_as(c_double_p), Lr.ctypes.data_as(c_int_p), Lp.ctypes.data_as(c_int_p), x.ctypes.data_as(c_double_p), n)
        _backsub_LT_csc_inplace(Lv.ctypes.data_as(c_double_p), Lr.ctypes.data_as(c_int_p), Lp.ctypes.data_as(c_int_p), x.ctypes.data_as(c_double_p), n)
        return x

def ichol_c(
    A,
    droptol,
    max_nnz=int(4e9 / 16),
):
    assert(A.has_canonical_format)

    m, n = A.shape

    assert m == n

    Lv = np.empty(max_nnz, dtype=np.float64)  # Values of non-zero elements of L
    Lr = np.empty(max_nnz, dtype=np.int32)  # Row indices of non-zero elements of L
    Lp = np.zeros(n + 1, dtype=np.int32)  # Start(Lp[i]) and end(Lp[i+1]) index of L[:, i] in Lv

    assert(A.data.dtype == np.float64)
    assert(A.indices.dtype == np.int32)
    assert(A.indptr.dtype == np.int32)
    assert(Lv.dtype == np.float64)
    assert(Lr.dtype == np.int32)
    assert(Lp.dtype == np.int32)

    nnz = _ichol_c(
        n,

        A.data.ctypes.data_as(c_double_p),
        A.indices.ctypes.data_as(c_int_p),
        A.indptr.ctypes.data_as(c_int_p),
        Lv.ctypes.data_as(c_double_p),
        Lr.ctypes.data_as(c_int_p),
        Lp.ctypes.data_as(c_int_p),

        droptol,
        max_nnz)

    if nnz == -2:
        raise ValueError("Thresholded incomplete Cholesky decomposition failed because more than max_nnz non-zero elements were created. Try increasing max_nnz or droptol.")

    if nnz == -1:
        raise ValueError("Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A.")

    return CholeskyDecomposition((Lv, Lr, Lp))
