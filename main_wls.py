from PIL import Image
import scipy.sparse.linalg
import numpy as np
import time, os
from skimage.color import rgb2lab, lab2rgb
from ctypes import c_int, c_double, c_void_p, pointer, POINTER, CDLL
import matplotlib.pyplot as plt

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
    c_double,
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

def conv_mat(shape, dx_dy_value):
    h, w = shape
    n = h * w

    x, y = np.mgrid[:w, :h]
    x = x.flatten()
    y = y.flatten()

    i = x + y * w

    i_inds = []
    j_inds = []
    values = []

    for dx, dy, value in dx_dy_value:
        x2 = np.clip(x + dx, 0, w - 1)
        y2 = np.clip(y + dy, 0, h - 1)

        j = x2 + y2 * w

        i_inds.append(i)
        j_inds.append(j)
        values.append(value * np.ones(n))

    i_inds = np.concatenate(i_inds)
    j_inds = np.concatenate(j_inds)
    values = np.concatenate(values)

    A = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), shape=(n, n))

    return A

def cg(
    A,
    b,
    x0=None,
    atol=0.0,
    rtol=1e-7,
    maxiter=10000,
    callback=None,
    M=None,
    reorthogonalize=False,
):
    if M is None:

        def precondition(x):
            return x

    elif callable(M):
        precondition = M
    else:

        def precondition(x):
            return M.dot(x)

    x = np.zeros_like(b) if x0 is None else x0.copy()

    norm_b = np.linalg.norm(b)

    if callable(A):
        r = b - A(x)
    else:
        r = b - A.dot(x)

    norm_r = np.linalg.norm(r)

    if norm_r < atol or norm_r < rtol * norm_b:
        return x

    z = precondition(r)
    p = z.copy()
    rz = np.inner(r, z)

    for iteration in range(maxiter):
        r_old = r.copy()

        if callable(A):
            Ap = A(p)
        else:
            Ap = A.dot(p)

        alpha = rz / np.inner(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        norm_r = np.linalg.norm(r)

        if callback is not None:
            callback(A, x, b, norm_b, r, norm_r)

        if norm_r < atol or norm_r < rtol * norm_b:
            return x

        z = precondition(r)

        if reorthogonalize:
            beta = np.inner(r - r_old, z) / rz
            rz = np.inner(r, z)
        else:
            beta = 1.0 / rz
            rz = np.inner(r, z)
            beta *= rz

        p *= beta
        p += z

    raise ValueError(
        "Conjugate gradient descent did not converge within %d iterations" % maxiter
    )

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
    discard_threshold=1e-4,
    shifts=[0.0, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 10.0, 100, 1e3, 1e4, 1e5],
    max_nnz=int(4e9 / 16),
    relative_threshold=0.0,
):
    if not A.has_canonical_format:
        A.sum_duplicates()

    m, n = A.shape

    assert m == n

    Lv = np.empty(max_nnz, dtype=np.float64)  # Values of non-zero elements of L
    Lr = np.empty(max_nnz, dtype=np.int32)  # Row indices of non-zero elements of L
    Lp = np.zeros(n + 1, dtype=np.int32)  # Start(Lp[i]) and end(Lp[i+1]) index of L[:, i] in Lv

    for shift in shifts:
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

            discard_threshold,
            relative_threshold,
            shift,
            max_nnz)

        print("nnz:", nnz * 1e-6, "million")

        if nnz >= 0:
            break

        if nnz == -1:
            print("PERFORMANCE WARNING:")
            print(
                "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A with parameters:"
            )
            print("    discard_threshold = %e" % discard_threshold)
            print("    shift = %e" % shift)
            print("Try decreasing discard_threshold or start with a larger shift")
            print("")

        if nnz == -2:
            raise ValueError(
                "Thresholded incomplete Cholesky decomposition failed because more than max_nnz non-zero elements were created. Try increasing max_nnz or discard_threshold."
            )

    if nnz < 0:
        raise ValueError(
            "Thresholded incomplete Cholesky decomposition failed due to insufficient positive-definiteness of matrix A and diagonal shifts did not help."
        )

    print("Copying")

    return CholeskyDecomposition((Lv, Lr, Lp))

def test():
    n = 100
    A = np.random.rand(n, n)
    A[np.random.rand(n, n) < 0.9] = 0
    A = A.T @ A

    A = scipy.sparse.csc_matrix(A)

    x_true = np.random.rand(n)

    decomposition = ichol_c(A, discard_threshold=0.0, relative_threshold=0.0)

    print("Decomposition computed")

    b = A @ x_true

    print("Applying decomposition")

    x = decomposition(b)

    print("error:", np.linalg.norm(x - x_true))
    print("")

def wls(image, lambd):

    h, w = image.shape
    n = h * w

    lambd = 1/lambd

    A = conv_mat(image.shape, [
        [0, 0, 1 + 4 * lambd],
        [-1, 0, -lambd],
        [+1, 0, -lambd],
        [0, -1, -lambd],
        [0, +1, -lambd],
    ])

    b = image.ravel()

    #import pyamg;M = pyamg.smoothed_aggregation_solver(A).aspreconditioner()

    t = time.perf_counter()

    M = ichol_c(A, discard_threshold=0, relative_threshold=1e-4)

    dt = time.perf_counter() - t

    print(f"{dt} seconds to compute preconditioner\n")

    if 1:
        print("Writing linear system to disk")
        i, j, v = scipy.sparse.find(A)
        with open("size.bin", "wb") as f: f.write(np.uint32([h, w]).tobytes())
        with open("i.bin", "wb") as f: f.write(i.astype(np.uint32).tobytes())
        with open("j.bin", "wb") as f: f.write(j.astype(np.uint32).tobytes())
        with open("v.bin", "wb") as f: f.write(v.astype(np.float64).tobytes())
        with open("b.bin", "wb") as f: f.write(b.astype(np.float64).tobytes())

        print(i.shape, j.shape, v.shape, b.shape)

    print(f"nnz = {M.L.nnz*1e-6} million ({M.L.nnz / len(b)} per row)")


    residuals = []
    def callback(A, x, b, norm_b, r, norm_r):
        residuals.append(norm_r)

        print(len(residuals), norm_r)

    t = time.perf_counter()

    result = cg(A, b, M=M, callback=callback, atol=0, rtol=1e-8)

    dt = time.perf_counter() - t
    print(f"\n{dt} seconds to solve")

    return result.reshape(h, w)

def main():
    test()

    image = Image.open("flower.png").convert("RGB")

    # Small size for testing
    if 0:
        size = (80, 53)
        image = image.resize(size)

    image = np.array(image) / 255.0

    lab = rgb2lab(image)

    L = lab[:, :, 0]

    val0 = 25/2
    val1 = 1
    val2 = 1
    saturation = 1.1

    print("starting")

    t = time.perf_counter()

    L0 = wls(L, 0.125)
    L1 = wls(L, 0.5)

    dt = time.perf_counter() - t

    print(dt, "seconds")

    def sigmoid(x, a):
        # Apply Sigmoid
        y = 1./(1+np.exp(-a*x)) - 0.5
        # Re-scale
        y05 = 1./(1+np.exp(-a*0.5)) - 0.5
        y = y*(0.5/y05)
        return y

    diff0 = sigmoid((L-L0)/100,val0)*100
    diff1 = sigmoid((L0-L1)/100,val1)*100
    base = (sigmoid((L1-56)/100,val2)*100)+56

    lab[:, :, 0] = base + diff1 + diff0

    rgb = lab2rgb(lab)

    Image.fromarray(np.clip(rgb*255, 0, 255).astype(np.uint8)).save("fine_python.png")

    for i, (img, title) in enumerate([
        (image, "input"),
        (rgb, "enhanced"),
        (L0, "L0"),
        (L1, "L1"),
    ]):
        plt.subplot(2, 2, 1 + i)
        plt.title(title)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

if __name__ == "__main__":
    main()
