import numpy as np
from scipy.optimize import minimize
from admm import admm
from utils.utils import sigmoid, loss_groups, loss_groups_g, c, eps, c_g
from utils.utils import constr_stat, obj_val, cnstr_val, sigmoid

def federa_proxAL(X0, X1, X0g, X1g, w0,mu0, beta, r, rho, eps1, eps2, bars = 0.001):
    n = X0.shape[2]
    N = (X0.shape[1]+X1.shape[1])*n
    wk = w0.copy()
    muk = mu0.copy()

    

    K=10000

    objs = []
    constrs = []
    
    for k in range(K):
        objs.append(obj_val(X0, X1, wk))
        constrs.append(cnstr_val(X0, X1, X0g, X1g, wk))
        print(f"----------------{k}/{K}---------------")

        tauk = bars / (k + 1)**2

        wk_copy = wk.copy()

        wk, _  = admm(X0=X0, X1=X1, X0g=X0g, X1g=X1g, w0=wk_copy, muk=muk, wk = wk_copy, tauk=tauk, rho=rho, beta=beta, r=r)


        muk_copy = muk.copy()

        for i in range(n+1):
            if i == n:
                ci01, ci10 = c_g(X0g, X1g, wk, r)
            else:
                ci01, ci10 = c(X0, X1, wk,i, r)

            muk[0,i] = np.maximum(muk_copy[0,i] + beta * ci01,0)
            muk[1,i] = np.maximum(muk_copy[1,i] + beta * ci10,0)
        print(np.linalg.norm(wk - wk_copy, np.inf)/beta + tauk, np.max(np.abs(muk - muk_copy))/beta)
        if np.linalg.norm(wk - wk_copy, np.inf) + beta * tauk <= beta * eps1:
            if np.max(np.abs(muk - muk_copy)) <= beta * eps2:
                break
    return wk, muk, objs, constrs


def central_proxAL(X0, X1, X0g, X1g, w0,mu0, beta, r, eps1, eps2, bars = 0.001):
    n = X0.shape[2]
    N = (X0.shape[1]+X1.shape[1])*n
    wk = w0.copy()
    muk = mu0.copy()


    K=100

    objs = []
    constrs = []

    for k in range(K):
        objs.append(obj_val(X0, X1, wk))
        constrs.append(cnstr_val(X0, X1, X0g, X1g, wk))
        print(f"----------------{k}/{K}---------------")

        tauk = bars / (k + 1)**2

        wk_copy = wk.copy()

        def proxAL_subpb(w):
            proxAL_val = 0
            for i in range(n):

                w_times_X0 = np.dot(w.T, X0[:,:,i])
                w_times_X1 = np.dot(w.T, X1[:,:,i])
                fi = (np.sum(w_times_X0-np.log(sigmoid(w_times_X0)+eps)) + np.sum(-np.log(sigmoid(w_times_X1)+eps)))/N

                loss_group0, loss_group1 = loss_groups(X0, X1, w,i)


                ci01 = loss_group0 - loss_group1 - r[i]   # group 0 - group 1
                ci10 = loss_group1 - loss_group0 - r[i]   # group 1 - group 0

                Pi = fi + np.maximum(muk[0, i] + beta * ci01, 0) ** 2/(2*beta) + np.maximum(muk[1, i] + beta * ci10, 0) ** 2/(2*beta)

                proxAL_val += Pi

            loss_group0, loss_group1 = loss_groups_g(X0g, X1g, w)

            ci01 = loss_group0 - loss_group1 - r[n]   # group 0 - group 1
            ci10 = loss_group1 - loss_group0 - r[n]   # group 1 - group 0

            Pg = np.maximum(muk[0, n] + beta * ci01, 0) ** 2/(2*beta) + np.maximum(muk[1, n] + beta * ci10, 0) ** 2/(2*beta)

            proxAL_val += Pg + np.linalg.norm(w - wk_copy,2)**2/(2 * beta)

            return proxAL_val


        wk = minimize(proxAL_subpb, wk_copy, tol=tauk*1e-4, method="L-BFGS-B")['x']





        muk_copy = muk.copy()


        for i in range(n+1):
            if i == n:
                ci01, ci10 = c_g(X0g, X1g, wk, r)
            else:
                ci01, ci10 = c(X0, X1, wk,i, r)
            muk[0,i] = np.maximum(muk_copy[0,i] + beta * ci01,0)
            muk[1,i] = np.maximum(muk_copy[1,i] + beta * ci10,0)
        print(np.linalg.norm(wk - wk_copy, np.inf)/beta + tauk, np.max(np.abs(muk - muk_copy))/beta)
        if np.linalg.norm(wk - wk_copy, np.inf) + beta * tauk <= beta * eps1:
            if np.max(np.abs(muk - muk_copy)) <= beta * eps2:
                break
    return wk, muk, objs, constrs
