from util import *
from ichol_c import ichol_c
import time
from numba import njit

w = 680
h = 440
n = w * h

def save_sparse_matrix(name, A):
    i, j, v = scipy.sparse.find(A)
    with open(f"{name}_i.bin", "wb") as f: f.write(i.astype(np.uint32).tobytes())
    with open(f"{name}_j.bin", "wb") as f: f.write(j.astype(np.uint32).tobytes())
    with open(f"{name}_v.bin", "wb") as f: f.write(v.astype(np.float64).tobytes())

def load_sparse_matrix(name):
    i = np.fromfile(f"{name}_i.bin", dtype=np.uint32)
    j = np.fromfile(f"{name}_j.bin", dtype=np.uint32)
    v = np.fromfile(f"{name}_v.bin", dtype=np.float64)
    return scipy.sparse.csc_matrix((v, (i, j)), shape=(n, n))

@njit("void(f8[:, :, :], f8, i8, f8[:, :, :], i8[:], i8[:])", cache=True)
def _cf(image, epsilon, r, values, indices, indptr):
    h, w, d = image.shape
    assert d == 3
    size = 2 * r + 1
    window_area = size * size

    for yi in range(h):
        for xi in range(w):
            i = xi + yi * w
            k = i * (4 * r + 1) ** 2
            for yj in range(yi - 2 * r, yi + 2 * r + 1):
                for xj in range(xi - 2 * r, xi + 2 * r + 1):
                    j = xj + yj * w

                    if 0 <= xj < w and 0 <= yj < h:
                        indices[k] = j

                    k += 1

            indptr[i + 1] = k

    # Centered and normalized window colors
    c = np.zeros((2 * r + 1, 2 * r + 1, 3))

    # For each pixel of image
    for y in range(r, h - r):
        for x in range(r, w - r):

            # For each color channel
            for dc in range(3):
                # Calculate sum of color channel in window
                s = 0.0
                for dy in range(size):
                    for dx in range(size):
                        s += image[y + dy - r, x + dx - r, dc]

                # Calculate centered window color
                for dy in range(2 * r + 1):
                    for dx in range(2 * r + 1):
                        c[dy, dx, dc] = (
                            image[y + dy - r, x + dx - r, dc] - s / window_area
                        )

            # Calculate covariance matrix over color channels with epsilon regularization
            a00 = epsilon
            a01 = 0.0
            a02 = 0.0
            a11 = epsilon
            a12 = 0.0
            a22 = epsilon

            for dy in range(size):
                for dx in range(size):
                    a00 += c[dy, dx, 0] * c[dy, dx, 0]
                    a01 += c[dy, dx, 0] * c[dy, dx, 1]
                    a02 += c[dy, dx, 0] * c[dy, dx, 2]
                    a11 += c[dy, dx, 1] * c[dy, dx, 1]
                    a12 += c[dy, dx, 1] * c[dy, dx, 2]
                    a22 += c[dy, dx, 2] * c[dy, dx, 2]

            a00 /= window_area
            a01 /= window_area
            a02 /= window_area
            a11 /= window_area
            a12 /= window_area
            a22 /= window_area

            det = (
                a00 * a12 * a12
                + a01 * a01 * a22
                + a02 * a02 * a11
                - a00 * a11 * a22
                - 2 * a01 * a02 * a12
            )

            inv_det = 1.0 / det

            # Calculate inverse covariance matrix
            m00 = (a12 * a12 - a11 * a22) * inv_det
            m01 = (a01 * a22 - a02 * a12) * inv_det
            m02 = (a02 * a11 - a01 * a12) * inv_det
            m11 = (a02 * a02 - a00 * a22) * inv_det
            m12 = (a00 * a12 - a01 * a02) * inv_det
            m22 = (a01 * a01 - a00 * a11) * inv_det

            # For each pair ((xi, yi), (xj, yj)) in a (2 r + 1)x(2 r + 1) window
            for dyi in range(2 * r + 1):
                for dxi in range(2 * r + 1):
                    s = c[dyi, dxi, 0]
                    t = c[dyi, dxi, 1]
                    u = c[dyi, dxi, 2]

                    c0 = m00 * s + m01 * t + m02 * u
                    c1 = m01 * s + m11 * t + m12 * u
                    c2 = m02 * s + m12 * t + m22 * u

                    for dyj in range(2 * r + 1):
                        for dxj in range(2 * r + 1):
                            xi = x + dxi - r
                            yi = y + dyi - r
                            xj = x + dxj - r
                            yj = y + dyj - r

                            i = xi + yi * w
                            j = xj + yj * w

                            # Calculate contribution of pixel pair to L_ij
                            temp = (
                                c0 * c[dyj, dxj, 0]
                                + c1 * c[dyj, dxj, 1]
                                + c2 * c[dyj, dxj, 2]
                            )

                            value = (1.0 if (i == j) else 0.0) - (
                                1 + temp
                            ) / window_area

                            dx = xj - xi + 2 * r
                            dy = yj - yi + 2 * r

                            values[i, dy, dx] += value


def cf_laplacian(image, epsilon=1e-7, radius=1):
    h, w, d = image.shape
    n = h * w

    # Data for matting laplacian in csr format
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = np.zeros(n * (4 * radius + 1) ** 2, dtype=np.int64)
    values = np.zeros((n, 4 * radius + 1, 4 * radius + 1), dtype=np.float64)

    _cf(image, epsilon, radius, values, indices, indptr)

    L = scipy.sparse.csr_matrix((values.ravel(), indices, indptr), (n, n))

    return L

def make_linear_system():
    from PIL import Image
    size = (w, h)
    image = np.array(Image.open("lemur.png").convert("RGB").resize(size, Image.BOX)) / 255.0
    trimap = np.array(Image.open("lemur_trimap.png").convert("L").resize(size, Image.NEAREST)) / 255.0

    is_fg = trimap == 1.0
    is_bg = trimap == 0.0
    is_known = np.logical_or(is_fg, is_bg)

    L = cf_laplacian(image)

    A = L + scipy.sparse.diags(100.0 * is_known.flatten())

    b = 100.0 * is_fg.flatten()

    return A, b

def main():
    print("Building linear system")

    A, b = make_linear_system()

    # Saving
    save_sparse_matrix("A", A)
    with open("b.bin", "wb") as f: f.write(b.astype(np.float64).tobytes())
    with open("shape.bin", "wb") as f: f.write(np.uint32([h, w]))

    A.sum_duplicates()

    droptol = 1e-4

    # Run multiple times to see if there is any variance
    for _ in range(5):
        t = time.perf_counter()

        decomposition = ichol_c(A, droptol)

        dt = time.perf_counter() - t

        print("Compute ichol C: %f seconds" % dt)

    t = time.perf_counter()

    x, n_iter = cg(A, b, M=decomposition)

    dt = time.perf_counter() - t

    print("Solve: %f seconds" % dt)
    print("%d iterations" % n_iter)

    if not os.path.isfile("L_i.bin"):
        print("Run ichol_matlab.m now and then rerun this script")
        return

    L_matlab = load_sparse_matrix("L")

    print("SAD:", abs(L_matlab - decomposition.L).sum())

    if 1:
        alpha = np.clip(x, 0, 1).reshape(h, w)

        import matplotlib.pyplot as plt
        plt.imshow(alpha, cmap='gray')
        plt.show()

if __name__ == "__main__":
    main()
