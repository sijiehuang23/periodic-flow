import numba as nb


@nb.njit
def cross(c, a, b):
    """
    Not that for 2D case, the cross product is designed in such a way that it works 
    for the nonlinear product u x w, where w the vorticity field is a scalar field.
    """
    ndim = a.ndim
    shape = a.shape

    if ndim == 3:
        for i in range(shape[1]):
            for j in range(shape[2]):
                c[0, i, j] = a[1, i, j] * b[i, j]
                c[1, i, j] = -a[0, i, j] * b[i, j]
    elif ndim == 4:
        for i in range(shape[1]):
            for j in range(shape[2]):
                for l in range(shape[3]):
                    a0, a1, a2 = a[0, i, j, l], a[1, i, j, l], a[2, i, j, l]
                    b0, b1, b2 = b[0, i, j, l], b[1, i, j, l], b[2, i, j, l]

                    c[0, i, j, l] = a1 * b2 - a2 * b1
                    c[1, i, j, l] = a2 * b0 - a0 * b2
                    c[2, i, j, l] = a0 * b1 - a1 * b0


@nb.njit
def curl(c, k, a):
    shape = a.shape

    if a.ndim == 3:
        for i in range(shape[1]):
            for j in range(shape[2]):
                c[i, j] = 1j * (k[0, i, j] * a[1, i, j] - k[1, i, j] * a[0, i, j])

    elif a.ndim == 4:
        for i in range(shape[1]):
            for j in range(shape[2]):
                for l in range(shape[3]):
                    k0, k1, k2 = k[0, i, j, l], k[1, i, j, l], k[2, i, j, l]
                    a0, a1, a2 = a[0, i, j, l], a[1, i, j, l], a[2, i, j, l]

                    c[0, i, j, l] = 1j * (k1 * a2 - k2 * a1)
                    c[1, i, j, l] = 1j * (k2 * a0 - k0 * a2)
                    c[2, i, j, l] = 1j * (k0 * a1 - k1 * a0)


@nb.njit
def apply_linear_operator(Lu, L, u):
    ndim = u.ndim
    shape = u.shape

    if ndim == 3:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    Lu[i, j, k] = L[j, k] * u[i, j, k]

    elif ndim == 4:
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(shape[2]):
                    for l in range(shape[3]):
                        Lu[i, j, k, l] = L[j, k, l] * u[i, j, k, l]


@nb.njit
def leray_projection(u, k, k_k2, p_hat):
    ndim = u.ndim
    shape = u.shape

    if ndim == 3:
        for i in range(shape[1]):
            for j in range(shape[2]):
                p_hat_ij = k_k2[0, i, j] * u[0, i, j] + k_k2[1, i, j] * u[1, i, j]

                p_hat[i, j] = p_hat_ij

                u[0, i, j] -= p_hat_ij * k[0, i, j]
                u[1, i, j] -= p_hat_ij * k[1, i, j]

    elif ndim == 4:
        for i in range(shape[1]):
            for j in range(shape[2]):
                for l in range(shape[3]):
                    p_hat_ijl = (
                        k_k2[0, i, j, l] * u[0, i, j, l]
                        + k_k2[1, i, j, l] * u[1, i, j, l]
                        + k_k2[2, i, j, l] * u[2, i, j, l]
                    )

                    p_hat[i, j, l] = p_hat_ijl

                    u[0, i, j, l] -= p_hat_ijl * k[0, i, j, l]
                    u[1, i, j, l] -= p_hat_ijl * k[1, i, j, l]
                    u[2, i, j, l] -= p_hat_ijl * k[2, i, j, l]
