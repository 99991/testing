import os
import numpy as np
import scipy.sparse.linalg

def cg(
    A,
    b,
    x0=None,
    atol=0.0,
    rtol=1e-7,
    maxiter=10000,
    M=None,
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
        return x, 0

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

        print(f"Iteration {iteration + 1} - |r| = {norm_r}")

        if norm_r < atol or norm_r < rtol * norm_b:
            return x, iteration + 1

        z = precondition(r)

        beta = 1.0 / rz
        rz = np.inner(r, z)
        beta *= rz

        p *= beta
        p += z

    raise ValueError("Conjugate gradient descent did not converge within %d iterations" % maxiter)
