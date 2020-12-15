import numpy as np
import scipy.sparse
from PIL import Image
import numpy as np
from pymatting import cg, cf_laplacian, ProgressCallback
import time

def make_P(shape):
    h, w = shape
    n = h * w
    h2 = h // 2
    w2 = w // 2
    n2 = w2 * h2

    weights = np.float64([1, 2, 1, 2, 4, 2, 1, 2, 1]) / 16

    x2 = np.repeat(np.tile(np.arange(w2), h2), 9)
    y2 = np.repeat(np.repeat(np.arange(h2), w2), 9)

    x = x2 * 2 + np.tile([-1, 0, 1, -1, 0, 1, -1, 0, 1], n2)
    y = y2 * 2 + np.tile([-1, -1, -1, 0, 0, 0, 1, 1, 1], n2)

    mask = (0 <= x) & (x < w) & (0 <= y) & (y <= h)

    i_inds = (x2 + y2 * w2)[mask]
    j_inds = (x + y * w)[mask]
    values = np.tile(weights, n2)[mask]

    downsample = scipy.sparse.csr_matrix((values, (i_inds, j_inds)), (n2, n))
    upsample = downsample.T

    return upsample, downsample, (h2, w2)

def spai0_step(L, C, b, x, m_diag, num_iter):
    # Optimize first iteration if x is not given, i.e. assumed to be zero
    if x is None:
        if num_iter > 0:
            x = m_diag * b
            num_iter -= 1
        else:
            x = np.zeros_like(b)

    for iteration in range(num_iter):
        x = x - m_diag * (L.dot(x) + C.dot(x) - b)

    return x

def jacobi_step(L, C, A_diag, b, x, num_iter, omega):
    if x is None:
        if num_iter > 0:
            x = omega * b / A_diag
            num_iter -= 1
        else:
            x = np.zeros_like(b)

    for _ in range(num_iter):
        x = x + omega * (b - L.dot(x) - C.dot(x)) / A_diag

    return x

def build_cache_L(L, shape, cache):
    if min(shape) <= 8: return

    upsample, downsample, coarse_shape = make_P(shape)

    coarse_L = downsample.dot(L).dot(upsample)

    L_diag = L.diagonal()

    L2 = L - scipy.sparse.diags(L_diag)
    L2 = L2.multiply(L2)
    L2_sum = np.array(L2.sum(axis=0)).ravel()

    cache[shape] = {
        "upsample": upsample,
        "downsample": downsample,
        "coarse_shape": coarse_shape,
        "coarse_L": coarse_L,
        "L2_sum": L2_sum,
        "L_diag": L_diag,
        "L": L,
    }

    build_cache_L(coarse_L, coarse_shape, cache)

def build_cache_C(C, shape, cache):
    if shape not in cache: return

    d = cache[shape]

    coarse_shape = d["coarse_shape"]
    downsample = d["downsample"]
    upsample = d["upsample"]

    coarse_C = downsample.dot(C).dot(upsample)

    C_diag = C.diagonal()

    d = cache[shape]

    d["coarse_C"] = coarse_C
    d["C_diag"] = C_diag
    d["C"] = C

    build_cache_C(coarse_C, coarse_shape, cache)

def vcycle_step(b, shape, cache):
    # Don't need to solve A x = b here, zero vector is good enough.
    if shape not in cache:
        return np.zeros_like(b)

    d = cache[shape]

    coarse_shape = d["coarse_shape"]
    downsample = d["downsample"]
    upsample = d["upsample"]
    L_diag = d["L_diag"]
    C_diag = d["C_diag"]
    L2_sum = d["L2_sum"]
    L = d["L"]
    C = d["C"]

    use_jacobi = False

    # compute M for spai0 preconditioner
    m_diag = (L_diag + C_diag) / (L2_sum + (L_diag + C_diag)**2)

    if use_jacobi:
        x = jacobi_step(L, C, L_diag + C_diag, b, None, 1, 0.8)
    else:
        x = spai0_step(L, C, b, None, m_diag, 1)

    # calculate residual error to perfect solution
    residual = b - (L.dot(x) + C.dot(x))

    # downsample residual error
    coarse_residual = downsample.dot(residual)

    # calculate coarse solution for residual
    coarse_x = vcycle_step(coarse_residual, coarse_shape, cache)

    # apply coarse correction
    x += upsample.dot(coarse_x)

    if use_jacobi:
        x = jacobi_step(L, C, L_diag + C_diag, b, x, 1, 0.8)
    else:
        x = spai0_step(L, C, b, x, m_diag, 1)

    return x

def main():
    print("loading images")

    size = (34, 22)
    size = (680, 440)
    image = np.array(Image.open("images/lemur.png").convert("RGB").resize(size, Image.BOX)) / 255.0
    trimap = np.array(Image.open("images/lemur_trimap.png").convert("L").resize(size, Image.NEAREST)) / 255.0

    is_fg = trimap == 1.0
    is_bg = trimap == 0.0
    is_known = is_fg | is_bg
    is_unknown = ~is_known

    b = 100.0 * is_fg.flatten()
    c = 100.0 * is_known.flatten()

    shape = trimap.shape
    h, w = shape

    L = cf_laplacian(image)

    cache = {}
    C = scipy.sparse.diags(c)
    build_cache_L(L, shape, cache)
    build_cache_C(C, shape, cache)
    M = lambda x: vcycle_step(x, shape, cache)

    x = cg(lambda x: L.dot(x) + c * x, b, M=M, callback=ProgressCallback())

    print("\nbaseline:")
    print("iteration      69 - 2.690681e-03 (0.00269068076571873623)")

    alpha = np.clip(x, 0, 1).reshape(h, w)

    import matplotlib.pyplot as plt
    for i, img in enumerate([image, trimap, alpha]):
        plt.subplot(1, 3, 1 + i)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
