import numpy as np
from proxAL import proxAL
from admm import admm
from scipy.optimize import minimize

num_seed = 1
sum_table = np.zeros((5, num_seed))

for ii in range(1, num_seed + 1):
    print(ii)
    np.random.seed(ii)

    d = 5
    n = 2
    Ni0 = 100
    Ni1 = 10

    X0 = np.random.rand(d, Ni0, n) - 0.5
    X1 = np.random.rand(d, Ni1, n) + 0.5 

    X1full = np.zeros((d, Ni1 * n))
    for i in range(n):
        X1full[:, i * Ni1:(i+1) * Ni1] = X1[:, :, i]

    def Loss1(w):
        return np.sum(-w.T @ X1full + np.log(1 + np.exp(w.T @ X1full)))

    w0 = np.ones(d)
    eps = 1e-5
    wfeas = minimize(Loss1, w0, tol=eps)['x']
    
    rfull = np.zeros(n)
    delta_r = 1
    for i in range(n):
        rfull[i] = np.sum(-wfeas.T @ X1[:, :, i] + np.log(1 + np.exp(wfeas.T @ X1[:, :, i]))) + delta_r

    w0 = np.zeros(d)
    mu0full = np.zeros(n)
    rhofull = np.ones(n)
    beta = 100
    eps1 = 1e-3
    eps2 = 1e-3

    w, out_iter, in_iter, stat_val, feas_val, obj_val = proxAL(w0, mu0full, X0, X1, rfull, admm, beta, rhofull, eps1, eps2)
    
    # w, out_iter, in_iter, stat_val, feas_val, obj_val = proxAL(w0, mu0full, X0, X1, rfull, admm, beta, rhofull, eps1, eps2)

    sum_table[:, ii - 1] = [out_iter, in_iter, stat_val, feas_val, obj_val]

ave_out_iter = np.mean(sum_table[0, :])
ave_in_iter = np.mean(sum_table[1, :])
ave_stat_val = np.mean(sum_table[2, :])
ave_feas_val = np.mean(sum_table[3, :])
ave_obj_val = np.mean(sum_table[4, :])

print("Average Out Iterations:", ave_out_iter)
print("Average In Iterations:", ave_in_iter)
print("Average Stat Val:", ave_stat_val)
print("Average Feas Val:", ave_feas_val)
print("Average Obj Val:", ave_obj_val)
