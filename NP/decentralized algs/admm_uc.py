import numpy as np
from admm import sigmoid
from scipy.optimize import minimize

def admm_uc(w0, X0, X1, rhofull, tau):
    d, _, n = X0.shape
    q = 0.9

    u0 = np.zeros((d, n))
    for i in range(n):
        u0[:, i] = w0

    lmda = np.zeros((d, n))
    for i in range(n):
        def DPi(w):
            inner_prod_X0 = np.dot(w.T, X0[:, :, i])
            inner_prod_X1 = np.dot(w.T, X1[:, :, i])
            sigmoid_term0 = sigmoid(inner_prod_X0)
            sigmoid_term1 = sigmoid(-inner_prod_X1)
            first_term = np.dot(X0[:, :, i], sigmoid_term0.T)
            second_term = -np.dot(X1[:, :, i], sigmoid_term1.T)
            return first_term + second_term

        lmda[:, i] = -DPi(w0)

    T = 10000
    eps = 1e-50
    wc = w0
    uc = u0
    mod_uc = uc + np.dot(lmda, np.diag(1.0 / rhofull))
    tepst_full = np.zeros(n)

    for t in range(T + 1):
        print(t)
        vareps = max(1e-20, q ** t)

        # perform aggregation
        wc = np.dot(mod_uc, rhofull) / np.sum(rhofull)

        # perform local updates
        for i in range(n):

            def Pi(w):
                inner_prod_X0 = np.dot(w.T, X0[:, :, i])
                inner_prod_X1 = np.dot(w.T, X1[:, :, i])
                first_term = np.sum(inner_prod_X0-np.log(sigmoid(inner_prod_X0)+eps))
                second_term = np.sum(-np.log(sigmoid(inner_prod_X1)+eps))
                third_term = np.dot(lmda[:, i], w - wc)
                fourth_term = np.dot(rhofull[i], np.linalg.norm(w - wc, 2) ** 2) / 2
                return (first_term + second_term) + third_term + fourth_term


            def DPi(w):
                inner_prod_X0 = np.dot(w.T, X0[:, :, i])
                inner_prod_X1 = np.dot(w.T, X1[:, :, i])
                sigmoid_term0 = sigmoid(inner_prod_X0)
                sigmoid_term1 = sigmoid(-inner_prod_X1)
                first_term = np.dot(X0[:, :, i], sigmoid_term0.T)
                second_term = -np.dot(X1[:, :, i], sigmoid_term1.T)
                third_term = lmda[:, i] + rhofull[i] * (w - wc)
                return (first_term + second_term) + third_term

            DPic = DPi(wc)
            up = uc.copy()
            # uc[:, i] = SpaRSA(up[:, i], Pi, DPi, vareps)
            uc[:, i] = minimize(Pi, up[:, i], tol=vareps, method="L-BFGS-B")['x']
            lmda[:, i] = lmda[:, i] + rhofull[i] * (uc[:, i] - wc)
            tepst_full[i] = np.linalg.norm(DPic - rhofull[i] * (wc - up[:, i]), np.inf)

        mod_uc = uc + np.dot(lmda, np.diag(1.0 / rhofull))

        print("tau: ", np.sum(tepst_full))
        # termination criterion
        if np.sum(tepst_full) <= tau:
            break

    in_iter = t + 1
    w = wc
    return w, in_iter