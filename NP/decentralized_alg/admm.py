import numpy as np
from scipy.optimize import minimize

eps = 1e-50

def sigmoid(x):
    positive_mask = x >= 0
    sigmoid_values = np.zeros_like(x, dtype=np.float64)

    sigmoid_values[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
    sigmoid_values[~positive_mask] = np.exp(x[~positive_mask]) / (1 + np.exp(x[~positive_mask]))

    return sigmoid_values


def P(X0, X1, w,i,mu,wk,r,beta):
    Ni0, Ni1 = X0.shape[1], X1.shape[1]
    n = X0.shape[2]
    
    w_times_X0 = np.dot(w.T, X0[:,:,i])
    w_times_X1 = np.dot(w.T, X1[:,:,i])
    fi = np.sum(w_times_X0-np.log(sigmoid(w_times_X0)+eps))/(n*Ni0)
    ci = np.sum(-np.log(sigmoid(w_times_X1)+eps))/Ni1 - r[i]
    Pi = fi + np.maximum(mu[i] + beta * ci, 0) ** 2/(2*beta) + np.linalg.norm(w-wk, 2)**2/(2*(n + 1)*beta)
    return Pi

def DP(X0, X1, w,i,mu,wk, beta, r):
    Ni0, Ni1 = X0.shape[1], X1.shape[1]
    n = X0.shape[2]
    
    w_times_X0 = np.dot(w.T, X0[:, :, i])
    w_times_X1 = np.dot(w.T, X1[:, :, i])

    Dfi = np.dot(X0[:, :, i], sigmoid(w_times_X0).T)/(n*Ni0)

    ci = np.sum(-np.log(sigmoid(w_times_X1)+eps))/Ni1 - r[i]
    Dci = -np.dot(X1[:, :, i], sigmoid(-w_times_X1).T)/Ni1

    DPi = Dfi + Dci * np.maximum(mu[i] + beta * ci, 0) + (w-wk)/((n+1)*beta)

    return DPi

def c(X1, w,i, r):
    Ni1 = X1.shape[1]
    w_times_X1 = np.dot(w.T, X1[:,:,i])
    ci = np.sum(-np.log(sigmoid(w_times_X1)+eps))/Ni1 - r[i]

    return ci

def constr_stat(X0,X1,w,mu):
    Ni0, Ni1 = X0.shape[1], X1.shape[1]
    n = X0.shape[2]
    cstat = 0

    for i in range(n):
        w_times_X0 = np.dot(w.T, X0[:,:,i])
        w_times_X1 = np.dot(w.T, X1[:,:,i])
        Dfi = np.dot(X0[:, :, i], sigmoid(w_times_X0).T)/(n*Ni0)
        Dci = -np.dot(X1[:, :, i], sigmoid(-w_times_X1).T)/Ni1

        cstat += Dfi + Dci*mu[i]

    return np.linalg.norm(cstat , np.inf)


def admm(X0, X1, w0,muk,wk,tauk, rho, beta, r):
    Ni0, Ni1 = X0.shape[1], X1.shape[1]
    d, _, n = X0.shape
    u0 = np.zeros((d,n))
    for i in range(n):
        u0[:, i] = w0

    lmda = np.zeros((d,n))
    for i in range(n):
        # lmda[:,i] = -DP(w0,i,muk,wk)
        lmda[:, i] = -DP(X0, X1, w0,i,muk,wk, beta, r)

    tu0 = np.zeros((d,n))
    for i in range(n):
        tu0[:,i] = w0  + lmda[:,i]/rho[i]




    T = 300
    q = 0.9


    wt = w0
    ut = u0
    tut = tu0



    tepst_vec = np.zeros(n)



    for t in range(T):
        vareps = max(1e-10, q**t)

        # perform aggregation
        wt = (np.dot(tut, rho) + wk / ((n + 1) * beta)) / (np.sum(rho) + 1 / ((n + 1) * beta))


        DP_all = np.zeros(d)

        # perform local updates
        for i in range(n):
            def phi(w):
                Pi = P(X0, X1, w,i,muk,wk,r,beta)
                phii = Pi + np.dot(lmda[:, i], w - wt) + rho[i]*np.linalg.norm(w-wt, 2)**2/2
                return phii

            def Dphi(w):
                # DPi = DP(w,i,muk,wk)
                DPi = DP(X0, X1, w,i,muk,wk, beta, r)
                Dphii = DPi + lmda[:, i] + rho[i]*(w - wt)
                return Dphii





            uit_copy = ut[:, i].copy()

            ut[:, i] = minimize(phi, uit_copy, tol=vareps*1e-2, method="L-BFGS-B")['x']

            lmda[:, i] = lmda[:, i] + rho[i] * (ut[:, i] - wt)

            tepst_vec[i] = np.linalg.norm(Dphi(wt) - rho[i]*(wt - uit_copy), np.inf)

            tut = ut + np.dot(lmda, np.diag(1.0/rho))

            DP_all = DP_all + DP(X0, X1, wt,i,muk,wk, beta, r)



        DP_all = DP_all + (wt-wk)/((n+1)*beta)

        if np.sum(tepst_vec) <= tauk:
            break

        stat_mea = np.linalg.norm(DP_all,np.inf)
        # if np.mod(t,10) == 0:
        #     print("---------------", t,"/",T ,"-------------------")
        #     print("stationary uperbound:", np.sum(tepst_vec))
        #     print("true stationary measure:", stat_mea)


    return wt, t



def obj_val(w,X0):
    _, Ni0, n = X0.shape
    class0_loss = 0
    obj_list = []
    for i in range(n):
        w_times_X0 = np.dot(w.T, X0[:,:,i])
        fi = np.sum(w_times_X0-np.log(sigmoid(w_times_X0)+eps))/(Ni0)
        obj_list.append(fi)
        
        
    return obj_list

def cnstr_val(w,X1):
    _, Ni1, n = X1.shape
    class1_loss = np.zeros(n)
    for i in range(n):
        w_times_X1 = np.dot(w.T, X1[:,:,i])
        ci = np.sum(-np.log(sigmoid(w_times_X1)+eps))/Ni1
        
        class1_loss[i] = ci
        
    return class1_loss
    