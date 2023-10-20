import numpy as np
from scipy.optimize import minimize
from admm import admm, sigmoid, c
from admm import constr_stat, obj_val, cnstr_val

eps = 1e-50

def central_proxAL(X0,X1,w0,mu0, beta,r,eps1,eps2):
    Ni0, Ni1 = X0.shape[1], X1.shape[1]
    n = X0.shape[2]
    wk = w0.copy()
    muk = mu0.copy()

    bars = 0.001

    K=100

    for k in range(K):
        print(f"----------------{k}/{K}---------------")

        tauk = bars / (k + 1)**2

        wk_copy = wk.copy()

        def proxAL_subpb(w):
            proxAL_val = 0
            for i in range(n):
                w_times_X0 = np.dot(w.T, X0[:, :, i])
                w_times_X1 = np.dot(w.T, X1[:, :, i])
                fi = np.sum(w_times_X0-np.log(sigmoid(w_times_X0)+eps))/(n*Ni0)
                ci = np.sum(-np.log(sigmoid(w_times_X1)+eps))/Ni1 - r[i]
                Pi = fi + np.maximum(muk[i] + beta * ci, 0) ** 2/(2*beta)

                proxAL_val += Pi

            proxAL_val += np.linalg.norm(w - wk_copy,2)**2/(2 * beta)

            return proxAL_val


        wk = minimize(proxAL_subpb, wk_copy, tol=tauk, method="L-BFGS-B")['x']





        muk_copy = muk.copy()


        for i in range(n):
            muk[i] = np.maximum(muk_copy[i] + beta * c(X1, wk,i, r),0)

        print(np.linalg.norm(wk - wk_copy, np.inf)/beta + tauk, max(np.abs(muk - muk_copy))/beta)
        if np.linalg.norm(wk - wk_copy, np.inf) + beta * tauk <= beta * eps1:
            if max(np.abs(muk - muk_copy)) <= beta * eps2:
                break
    return wk, muk


def federa_proxAL(X0, X1, w0,mu0,eps1, eps2, beta, rho, r):
    n = X0.shape[2]
    wk = w0.copy()
    muk = mu0.copy()

    bars = 0.001

    K=100

    objs = []
    constrs = []

    for k in range(K):
        objs.append(obj_val(wk, X0))
        constrs.append(cnstr_val(wk, X1))
        print(f"----------------{k}/{K}---------------")

        tauk = bars / (k + 1)**2

        wk_copy = wk.copy()

        # wk, _  = admm(w0=wk_copy, muk=muk, wk = wk_copy, tauk=tauk)
        wk, _  = admm(X0=X0, X1=X1, w0=w0, muk=muk, wk=wk, tauk=tauk, rho=rho, beta=beta, r=r)

        muk_copy = muk.copy()

        for i in range(n):
            muk[i] = np.maximum(muk_copy[i] + beta * c(X1, wk,i, r),0)
        print(np.linalg.norm(wk - wk_copy, np.inf)/beta + tauk, max(np.abs(muk - muk_copy))/beta)
        if np.linalg.norm(wk - wk_copy, np.inf) + beta * tauk <= beta * eps1:
            if max(np.abs(muk - muk_copy)) <= beta * eps2:
                break
    return wk, muk, objs, constrs