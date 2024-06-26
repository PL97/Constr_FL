import numpy as np
from scipy.optimize import minimize
from utils.utils import P, DP, c, P0, DP0

def admm(X0, X1, X0g, X1g, w0,muk,wk,tauk, rho, beta, r):
    d, _, n = X0.shape
    N = (X0.shape[1]+X1.shape[1])*n
    u0 = np.zeros((d,n))
    for i in range(n):
        u0[:, i] = w0

    lmda = np.zeros((d,n))
    for i in range(n):
        lmda[:,i] = -DP(X0, X1, w0,i,muk,wk, beta, r)

    tu0 = np.zeros((d,n))
    for i in range(n):
        tu0[:,i] = w0  + lmda[:,i]/rho[i]


    T = 50
    q = 0.9
    wt = w0
    ut = u0
    tut = tu0
    tepst_vec = np.zeros(n)

    for t in range(T):
        vareps = max(1e-10, q**t)
        # perform aggregation
        def phi0(w):
            Pg = P0(X0, X1, X0g, X1g, w,muk,wk, beta, r)
            global_penalty = 0
            for i in range(n):
                global_penalty += rho[i] * np.linalg.norm(tut[:,i] - w, 2)**2 / 2

            phig = Pg + global_penalty

            return phig

        wt_copy = wt.copy()

        wt = minimize(phi0, wt_copy, tol=vareps*1e-4, method="L-BFGS-B")['x']

        # wt = (np.dot(tut, rho) + wk / ((n + 1) * beta)) / (np.sum(rho) + 1 / ((n + 1) * beta))


        DP_all = np.zeros(d)


        # perform local updates
        for i in range(n):
            def phi(w):
                Pi = P(X0, X1, w,i,muk,wk, beta, r)
                phii = Pi + np.dot(lmda[:, i], w - wt) + rho[i]*np.linalg.norm(w-wt, 2)**2/2
                return phii

            def Dphi(w):
                DPi = DP(X0, X1, w,i,muk,wk, beta, r)
                Dphii = DPi + lmda[:, i] + rho[i]*(w - wt)
                return Dphii

            uit_copy = ut[:, i].copy()

            ut[:, i] = minimize(phi, uit_copy, tol=vareps*1e-4, method="L-BFGS-B")['x']



            tepst_vec[i] = np.linalg.norm(Dphi(wt) - rho[i]*(wt - uit_copy), np.inf)

            lmda[:, i] = lmda[:, i] + rho[i] * (ut[:, i] - wt)


            tut = ut + np.dot(lmda, np.diag(1.0/rho))



            DP_all += DP(X0, X1, wt,i,muk,wk, beta, r)


        DP_all = DP_all + DP0(X0, X1, X0g, X1g, wt,muk,wk, beta, r)



        stat_mea = np.linalg.norm(DP_all,np.inf)
        if np.mod(t,10) == 0:
            print("---------------", t,"/",T ,"-------------------")
            print("stationary uperbound:", np.sum(tepst_vec)+vareps)
            print("true stationary measure:", stat_mea)


        if np.sum(tepst_vec) + vareps <= tauk:
            print("---------------", t,"/",T ,"-------------------")
            print("stationary uperbound:", np.sum(tepst_vec)+vareps)
            print("true stationary measure:", stat_mea)
            break
    return wt, t
