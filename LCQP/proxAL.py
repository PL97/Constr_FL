import numpy as np
from numpy.linalg import norm

def proxAL(w0, mu0full, A, b, C, dfull, admm, beta, rhofull, eps1, eps2, bars = 0.01):
    ttiniter = 0
    wc = w0  # set current iterate to be w0
    K = 10000  # maximum number of iterations
    mukfull = mu0full  # set current Lagrangian multipliers to be mu0
    

    d, n = b.shape
    Id = np.eye(d)


    Qal = np.zeros((d, d, n))
    pal = np.zeros((d, n))

    for k in range(K + 1):

        # tolerance
        tauk = bars / (k + 1) ** 2
        for i in range(n):
            Qal[:, :, i] = A[:, :, i] + beta * C[:, :, i].T @ C[:, :, i] + Id / ((n + 1) * beta)
            pal[:, i] = b[:, i] + beta * C[:, :, i].T @ (mukfull[:, i] / beta + dfull[:, i]) - wc / ((n + 1) * beta)

        wp = wc  # copy the current solution

        wc, initer = admm(wc, Qal, pal, rhofull, tauk)  # compute the next solution

        mupfull = mukfull.copy()  # copy the current Lagrangian multiplier

        for i in range(n):
            mukfull[:, i] = mukfull[:, i] + beta * (C[:, :, i] @ wc + dfull[:, i])  # update the Lagrangian multiplier

        # termination criterion
        if norm(wp - wc, ord=np.inf) + beta * tauk <= beta * eps1:
            if np.max(np.abs(mupfull - mukfull)) <= beta * eps2:
                break
        ttiniter += initer

    w = wc
    outiter = k + 1

    objval = 0
    for i in range(n):
        objval = objval + wc.T @ A[:, :, i] @ wc / 2 + b[:, i].T @ wc

    cnstrv = 0
    for i in range(n):
        cnstrv = max(cnstrv, norm(C[:, :, i] @ wc + dfull[:, i], ord=np.inf))


    ret_cproxal = {
        "w": w,
        "objpal": objval,
        "constrpal": cnstrv,
        "floutiter": int(outiter),
        "flttiniter": int(ttiniter)
    }


    return ret_cproxal
