import numpy as np
from numpy.linalg import solve, norm

def admm(w0, A, b, rhofull, tau):
    # problem size
    d, n = b.shape
    Id = np.eye(d)

    # initialize u0
    u0 = np.zeros((d, n))
    for i in range(n):
        u0[:, i] = w0

    # initialize lambda
    lmda = np.zeros((d, n))
    for i in range(n):
        lmda[:, i] = - (A[:, :, i] @ u0[:, i] + b[:, i])

    T = 10000

    # update the current iterates
    wc = w0
    uc = u0
    mod_uc = uc + lmda / rhofull
    tepst_full = np.zeros(n)

    for t in range(T + 1):
        # solve min varphi0
        Q0 = A[:, :, n-1] + np.sum(rhofull) * Id
        p0 = b[:, n-1] - mod_uc @ rhofull
        wc = solve(Q0, -p0)

        # solve min varphii
        up = uc

        tParrell = np.zeros(n)
        for i in range(n):
            Qi = A[:, :, i] + rhofull[i] * Id
            pi = b[:, i] + lmda[:, i] - rhofull[i] * wc
            uc[:, i] = solve(Qi, -pi)
            lmda[:, i] = lmda[:, i] + rhofull[i] * (uc[:, i] - wc)
            tepst_full[i] = norm(Qi @ wc + pi - rhofull[i] * (wc - up[:, i]), ord=np.inf)

        mod_uc = uc + lmda / rhofull

        # check the termination criterion
        if np.sum(tepst_full) <= tau:
            break

    w = wc

    initer = t + 1

    return w, initer