import numpy as np
from admm import sigmoid

def proxAL(w0, mu0full, X0, X1, r, admm, beta, rhofull, eps1, eps2):
    wc = w0
    K = 10000
    mukfull = mu0full
    bars = 1

    in_iter = 0

    d, _, n = X1.shape

    for k in range(K + 1):
        print(f"----------------{k}/{K}---------------")
        tauk = bars / (k + 1)**2

        wp = wc

        wc, in_iter_k = admm(wc, X0, X1, mukfull, r, rhofull, beta, tauk)
        
        in_iter += in_iter_k
        

        mupfull = mukfull.copy()
        
        tmp1 = (wc@X1.reshape(X1.shape[0], -1)).reshape(X1.shape[1], -1, order='C')
        mukfull = np.maximum(mukfull + beta * (np.sum(-np.log(sigmoid(tmp1)), axis=0) - r), 0)
        
        tmp2 = (wc.T@X0.reshape(X0.shape[0], -1)).reshape(X0.shape[1], -1, order='C')
        obj_val = np.mean(np.log(1 + np.exp(tmp2)))
            
        
        stat_mea = np.zeros(d)
        for i in range(n):
            stat_mea += X0[:, :, i] @ (np.exp(wc.T @ X0[:, :, i]) / (1 + np.exp(wc.T @ X0[:, :, i]))) + (-X1[:, :, i] @ (1 / (1 + np.exp(wc.T @ X1[:, :, i])))) * mukfull[i]

        
        
        stat_val = np.linalg.norm(stat_mea, np.inf)
        
        feas_mea = np.zeros(n)
        for i in range(n):
            feas_mea[i] = max(np.sum(-wc.T @ X1[:, :, i] + np.log(1 + np.exp(wc.T @ X1[:, :, i]))) - r[i], 0)
        
        feas_val = max(feas_mea)

        if np.linalg.norm(wp - wc, np.inf) + beta * tauk <= beta * eps1:
            if max(np.abs(mupfull - mukfull)) <= beta * eps2:
                break
        
    out_iter = k + 1
    w = wc
    return w, out_iter, in_iter, stat_val, feas_val, obj_val
