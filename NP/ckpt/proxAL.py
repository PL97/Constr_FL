import numpy as np

def proxAL(w0, mu0full, X0, X1, r, admm, beta, rhofull, eps1, eps2):
    wc = w0
    K = 10000
    mukfull = mu0full
    bars = 1

    in_iter = 0

    d, _, n = X1.shape

    for k in range(K + 1):
        tauk = bars / (k + 1)**2

        wp = wc

        wc, in_iter_k = admm(wc, X0, X1, mukfull, r, rhofull, beta, tauk)
        
        in_iter += in_iter_k
        

        # mupfull = mukfull.copy()
        # tmp1 = (wc@X1.reshape(X1.shape[0], -1)).reshape(X1.shape[0], -1)
        # print(mukfull.shape, tmp1.shape, wc.shape, X1.shape)
        # mukfull = np.maximum(mukfull + beta * (np.sum(-tmp1 + np.log(1 + np.exp(tmp1)), axis=0) - r), 0)
        mupfull = mukfull.copy()
        for i in range(n):
            mukfull[i] = max(mukfull[i] + beta * (np.sum(-wc.T @ X1[:, :, i] + np.log(1 + np.exp(wc.T @ X1[:, :, i]))) - r[i]), 0)
        
        
        tmp2 = (wc@X0.reshape(X0.shape[0], -1)).reshape(X0.shape[0], -1)
        obj_val = np.sum(np.log(1 + np.exp(tmp2)))
        # obj_val = 0
        # for i in range(n):
        #     obj_val += np.sum(np.log(1 + np.exp(wc.T @ X0[:, :, i])))
            
        
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
