import numpy as np
from admm import sigmoid
from utils.eval import model_eval
import copy

def proxAL(w0, mu0full, X0, X1, r, admm, beta, rhofull, eps1, eps2):
    wc = w0
    K = 10000
    mukfull = mu0full
    bars = 1e-4

    in_iter = 0

    d, _, n = X1.shape

    for k in range(K + 1):
        print(f"----------------{k}/{K}---------------")
        tauk = bars / (k + 1)**2

        wp = wc
        wc, in_iter_k = admm(wc, X0, X1, mukfull, r, rhofull, beta, tauk)
        in_iter += in_iter_k
        

        mupfull = copy.deepcopy(mukfull)
        
        tmp1 = (wc@X1.reshape(X1.shape[0], -1)).reshape(X1.shape[1], -1, order='C')
        mukfull = np.maximum(mukfull + beta * (np.sum(-np.log(sigmoid(tmp1)), axis=0) - r), 0)
        
        tmp2 = (wc.T@X0.reshape(X0.shape[0], -1)).reshape(X0.shape[1], -1, order='C')
        obj_val = np.mean(np.log(1 + np.exp(tmp2)))

        
        stat_mea = np.zeros(d)
        for i in range(n):
            stat_mea += (1/X0.shape[1]) * X0[:, :, i] @ sigmoid(wc.T @ X0[:, :, i]) + (1/X1.shape[1]) * (-X1[:, :, i] @ sigmoid(-wc.T @ X1[:, :, i])) * mukfull[i]

        
        stat_val = np.linalg.norm(stat_mea, np.inf)
        
        feas_mea = np.zeros(n)
        obj_mea = np.zeros(n)
        for i in range(n):
            feas_mea[i] = max((np.sum(-np.log(sigmoid(wc.T @ X1[:, :, i]))) - r[i])/X1.shape[1], 0)

            eps = 1e-100
            inner_prod_X0 = np.dot(wc.T, X0[:, :, i])
            obj_mea[i] = np.mean(inner_prod_X0-np.log(sigmoid(inner_prod_X0)+eps)) 
        
        feas_val = max(feas_mea)
        # print(model_eval(wc, X0, X1))
        # print(obj_mea)
        print(feas_mea)
        print(np.linalg.norm(wp - wc, np.inf))
        print(max(np.abs(mupfull - mukfull)))
        

        if np.linalg.norm(wp - wc, np.inf) + beta * tauk <= beta * eps1:
            if max(np.abs(mupfull - mukfull)) <= beta * eps2:
                break
        
        
    out_iter = k + 1
    w = wc
    return w, out_iter, in_iter, stat_val, feas_val, obj_val
