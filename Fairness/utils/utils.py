import numpy as np
from scipy.optimize import minimize

def sigmoid(x):
    positive_mask = x >= 0
    sigmoid_values = np.zeros_like(x, dtype=np.float64)

    sigmoid_values[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
    sigmoid_values[~positive_mask] = np.exp(x[~positive_mask]) / (1 + np.exp(x[~positive_mask]))

    return sigmoid_values


eps = 1e-50


def loss_groups(X0, X1, w,i):
    X0i = X0[:,:,i]
    X1i = X1[:,:,i]

    last_row_0i = X0i[-1]
    X00 = X0i[:, last_row_0i == 0]   # label 0 group 0
    X01 = X0i[:, last_row_0i == 1]   # label 0 group 1

    last_row_1i = X1i[-1]
    X10 = X1i[:, last_row_1i == 0]   # label 1 group 0
    X11 = X1i[:, last_row_1i == 1]   # label 1 group 1

    w_times_X00 = np.dot(w.T, X00)
    w_times_X01 = np.dot(w.T, X01)
    w_times_X10 = np.dot(w.T, X10)
    w_times_X11 = np.dot(w.T, X11)

    Ni00 = X00.shape[1]
    Ni01 = X01.shape[1]
    Ni10 = X10.shape[1]
    Ni11 = X11.shape[1]

    (np.sum(w_times_X00-np.log(sigmoid(w_times_X00)+eps)) + np.sum(-np.log(sigmoid(w_times_X10)+eps)))/(Ni00+Ni10)

    loss_group0 = (np.sum(w_times_X00-np.log(sigmoid(w_times_X00)+eps)) + np.sum(-np.log(sigmoid(w_times_X10)+eps)))/(Ni00+Ni10)

    loss_group1 = (np.sum(w_times_X01-np.log(sigmoid(w_times_X01)+eps)) + np.sum(-np.log(sigmoid(w_times_X11)+eps)))/(Ni01+Ni11)

    return loss_group0, loss_group1


def Dloss_groups(X0, X1, w,i):
    X0i = X0[:,:,i]
    X1i = X1[:,:,i]

    last_row_0i = X0i[-1]
    X00 = X0i[:, last_row_0i == 0]   # label 0 group 0
    X01 = X0i[:, last_row_0i == 1]   # label 0 group 1

    last_row_1i = X1i[-1]
    X10 = X1i[:, last_row_1i == 0]   # label 1 group 0
    X11 = X1i[:, last_row_1i == 1]   # label 1 group 1

    w_times_X00 = np.dot(w.T, X00)
    w_times_X01 = np.dot(w.T, X01)
    w_times_X10 = np.dot(w.T, X10)
    w_times_X11 = np.dot(w.T, X11)

    Ni00 = X00.shape[1]
    Ni01 = X01.shape[1]
    Ni10 = X10.shape[1]
    Ni11 = X11.shape[1]

    Dloss_group0 = (np.dot(X00, sigmoid(w_times_X00).T) -np.dot(X10, sigmoid(-w_times_X10).T))/(Ni00+Ni10)

    Dloss_group1 = (np.dot(X01, sigmoid(w_times_X01).T) -np.dot(X11, sigmoid(-w_times_X11).T))/(Ni01+Ni11)

    return Dloss_group0, Dloss_group1

def P(X0, X1, w,i,mu,wk, beta, r):
    n = X0.shape[2]
    N = (X0.shape[1]+X1.shape[1])*n
    w_times_X0 = np.dot(w.T, X0[:,:,i])
    w_times_X1 = np.dot(w.T, X1[:,:,i])
    fi = (np.sum(w_times_X0-np.log(sigmoid(w_times_X0)+eps)) + np.sum(-np.log(sigmoid(w_times_X1)+eps)))/N

    loss_group0, loss_group1 = loss_groups(X0, X1, w,i)


    ci01 = loss_group0 - loss_group1 - r[i]   # group 0 - group 1
    ci10 = loss_group1 - loss_group0 - r[i]   # group 1 - group 0
    Pi = fi + np.maximum(mu[0, i] + beta * ci01, 0) ** 2/(2*beta) + np.maximum(mu[1, i] + beta * ci10, 0) ** 2/(2*beta) + np.linalg.norm(w-wk, 2)**2/(2*(n + 1)*beta)

    return Pi

def DP(X0, X1, w,i,mu,wk, beta, r):
    n = X0.shape[2]
    N = (X0.shape[1]+X1.shape[1])*n

    w_times_X0 = np.dot(w.T, X0[:, :, i])
    w_times_X1 = np.dot(w.T, X1[:, :, i])

    Dfi = (np.dot(X0[:, :, i], sigmoid(w_times_X0).T) -np.dot(X1[:, :, i], sigmoid(-w_times_X1).T))/N

    loss_group0, loss_group1 = loss_groups(X0, X1, w,i)

    Dloss_group0, Dloss_group1 = Dloss_groups(X0, X1, w,i)

    ci01 = loss_group0 - loss_group1 - r[i]   # group 0 - group 1
    ci10 = loss_group1 - loss_group0 - r[i]   # group 1 - group 0

    Dci01 = Dloss_group0- Dloss_group1
    Dci10 = Dloss_group1- Dloss_group0

    DPi = Dfi + Dci01 * np.maximum(mu[0, i] + beta * ci01, 0) + Dci10 * np.maximum(mu[1, i] + beta * ci10, 0) + (w-wk)/((n+1)*beta)

    return DPi

def c(X0, X1, w,i, r):

    loss_group0, loss_group1 = loss_groups(X0, X1, w,i)

    ci01 = loss_group0 - loss_group1 - r[i]   # group 0 - group 1
    ci10 = loss_group1 - loss_group0 - r[i]   # group 1 - group 0

    return ci01, ci10

# constrained stationary
def constr_stat(X0, X1, w,mu, r):
    n = X0.shape[2]
    N = (X0.shape[1]+X1.shape[1])*n
    cstat = 0

    for i in range(n):

        w_times_X0 = np.dot(w.T, X0[:, :, i])
        w_times_X1 = np.dot(w.T, X1[:, :, i])

        Dfi = (np.dot(X0[:, :, i], sigmoid(w_times_X0).T) -np.dot(X1[:, :, i], sigmoid(-w_times_X1).T))/N

        loss_group0, loss_group1 = loss_groups(X0, X1, w,i)
        ci01 = loss_group0 - loss_group1 - r[i]
        ci10 = loss_group1 - loss_group0 - r[i]


        Dloss_group0, Dloss_group1 = Dloss_groups(X0, X1, w,i)
        Dci01 = Dloss_group0- Dloss_group1
        Dci10 = Dloss_group1- Dloss_group0

        cstat += Dfi + Dci01*mu[0, i] + Dci10*mu[1, i]

    return np.linalg.norm(cstat , np.inf)


def obj_val(X0, X1, w):
    n = X0.shape[2]
    N = (X0.shape[1]+X1.shape[1])*n
    logistic_loss = 0

    for i in range(n):
        w_times_X0 = np.dot(w.T, X0[:,:,i])
        w_times_X1 = np.dot(w.T, X1[:,:,i])
        fi = (np.sum(w_times_X0-np.log(sigmoid(w_times_X0)+eps)) + np.sum(-np.log(sigmoid(w_times_X1)+eps)))/N

        logistic_loss += fi

    return logistic_loss


def cnstr_val(X0, X1, w):
    n = X0.shape[2]

    cnstr_val = np.zeros(n)

    for i in range(n):
        loss_group0, loss_group1 = loss_groups(X0, X1, w,i)
        cnstr_val[i] = np.abs(loss_group0-loss_group1)

    return cnstr_val

