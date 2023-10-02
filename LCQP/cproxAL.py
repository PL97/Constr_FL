import numpy as np
from numpy.linalg import solve, norm

def cproxAL(w0, mu0full, A, b, C, dfull, beta, eps1, eps2, bars=0.01):
    wc = w0  # set current iterate to be w0
    K = 1000  # maximum number of iterations
    mukfull = mu0full  # set current Lagrangian multipliers to be mu0

    d, n = b.shape
    Id = np.eye(d)

    for k in range(K + 1):

        Qal = np.zeros((d, d))
        pal = np.zeros((d, 1))

        # tolerance
        tauk = bars / (k + 1) ** 2
        for i in range(n):
            Qal = Qal + A[:, :, i] + beta * C[:, :, i].T @ C[:, :, i] + Id / ((n + 1) * beta)
            pal = pal + b[:, i].reshape(-1, 1) + beta * C[:, :, i].T @ (mukfull[:, i].reshape(-1, 1) / beta + dfull[:, i].reshape(-1, 1)) - wc.reshape(-1, 1) / ((n + 1) * beta)
        
        wp = wc  # copy the current solution
        wc = solve(Qal, -pal)  # compute the next solution

        mupfull = mukfull.copy()  # copy the current Lagrangian multiplier
        for i in range(n):
            mukfull[:, i] = mukfull[:, i] + beta * ((C[:, :, i] @ wc).flatten() + dfull[:, i])  # update the Lagrangian multiplier

        # termination criterion
        if norm(wp - wc, ord=np.inf) + beta * tauk <= beta * eps1:
            if np.max(np.abs(mupfull - mukfull)) <= beta * eps2:
                break
            
        
        

    w = wc
    outiter = k + 1

    objval = 0
    for i in range(n):
        objval = objval + wc.T @ A[:, :, i] @ wc / 2 + b[:, i].T @ wc

    cnstrv = 0
    for i in range(n):
        cnstrv = max(cnstrv, norm(C[:, :, i] @ wc.reshape(-1, ) + dfull[:, i], ord=np.inf))

    ret_cproxal = {
        "w": w,
        "objcpal": objval,
        "constrcpal": cnstrv,
        "coutiter": int(outiter)
    }

    return ret_cproxal
