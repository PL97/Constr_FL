import numpy as np
# from SpaRSA import SpaRSA
from scipy.optimize import minimize

def admm(w0, X0, X1, mu, r, rhofull, beta, tau):
    d, _, n = X0.shape
    q = 0.99

    u0 = np.zeros((d, n))
    for i in range(n):
        u0[:, i] = w0

    lmda = np.zeros((d, n))
    for i in range(n):
        def DPi(w):
            exp_term = np.exp(np.dot(w.T, X0[:, :, i]))
            sigmoid_term = exp_term / (1 + exp_term)
            first_term = np.dot(X0[:, :, i], sigmoid_term.T)
            second_term = -np.dot(X1[:, :, i], 1 / (1 + np.exp(np.dot(w.T, X1[:, :, i]))))
            third_term = (w - w0) / ((n + 1) * beta)
            return first_term + second_term*np.maximum(mu[i] + beta * (np.sum(-np.dot(w.T, X1[:, :, i]) + np.log(1 + np.exp(np.dot(w.T, X1[:, :, i])))) - r[i]), 0) + third_term
        
        lmda[:, i] = -DPi(w0)

    T = 10000
    wc = w0
    uc = u0
    mod_uc = uc + np.dot(lmda, np.diag(1.0 / rhofull))
    tepst_full = np.zeros(n)

    for t in range(T + 1):
        vareps = max(1e-10, q ** t)

        # perform aggregation
        wc = (np.dot(mod_uc, rhofull) + w0 / ((n + 1) * beta)) / (np.sum(rhofull) + 1 / ((n + 1) * beta))

        # perform local updates
        for i in range(n):
            
            
            def Pi(w):
                first_term = np.sum(np.log(1 + np.exp(np.dot(w.T, X0[:, :, i]))))
                second_term = np.maximum(mu[i] + beta * (np.sum(-np.dot(w.T, X1[:, :, i]) + np.log(1 + np.exp(np.dot(w.T, X1[:, :, i])))) - r[i]), 0) ** 2 / (2 * beta)
                third_term = np.linalg.norm(w - w0, 2) ** 2 / (2 * (n + 1) * beta)
                fourth_term = np.dot(lmda[:, i], w - wc)
                fifth_term = np.dot(rhofull[i], np.linalg.norm(w - wc, 2) ** 2) / 2
                return first_term + second_term + third_term + fourth_term + fifth_term
            
            def DPi(w):
                exp_term = np.exp(np.dot(w.T, X0[:, :, i]))
                sigmoid_term = exp_term / (1 + exp_term)
                first_term = np.dot(X0[:, :, i], sigmoid_term.T)
                second_term = -np.dot(X1[:, :, i], 1 / (1 + np.exp(np.dot(w.T, X1[:, :, i]))))
                third_term = (w - w0) / ((n + 1) * beta)
                fourth_term = lmda[:, i] + rhofull[i] * (w - wc)
                return first_term + second_term * np.maximum(mu[i] + beta * (np.sum(-np.dot(w.T, X1[:, :, i]) + np.log(1 + np.exp(np.dot(w.T, X1[:, :, i])))) - r[i]), 0) + third_term + fourth_term
            
            DPic = DPi(wc)
            up = uc.copy()
            # uc[:, i] = SpaRSA(up[:, i], Pi, DPi, vareps)
            uc[:, i] = minimize(Pi, up[:, i], tol=vareps)['x']
            lmda[:, i] = lmda[:, i] + rhofull[i] * (uc[:, i] - wc)
            tepst_full[i] = np.linalg.norm(DPic - rhofull[i] * (wc - up[:, i]), np.inf)

        mod_uc = uc + np.dot(lmda, np.diag(1.0 / rhofull))

        # termination criterion
        if np.sum(tepst_full) <= tau:
            break

    in_iter = t + 1
    w = wc
    return w, in_iter