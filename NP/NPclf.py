import numpy as np
from proxAL import proxAL
from admm import admm, sigmoid
from scipy.optimize import minimize

num_seed = 1
sum_table = np.zeros((5, num_seed))

for ii in range(1, num_seed + 1):
    print(ii)
    np.random.seed(ii)

    d = 3 #! 30 ~ 50
    n = 2
    Ni0 = 300 #! 30000
    Ni1 = 100

    X0 = (np.random.rand(d, Ni0, n) - 0.5) * 0.1
    X1 = (np.random.rand(d, Ni1, n) + 0.5)*0.1
    
    # for i in range(n):
    #     X0[-1, :, i] = 1
    #     X1[-1, :, i] = 1    
    
    
    X1full = np.zeros((d, Ni1 * n))
    for i in range(n):
        X1full[:, i * Ni1:(i+1) * Ni1] = X1[:, :, i]
        
    X0full = np.zeros((d, Ni0 * n))
    for i in range(n):
        X0full[:, i * Ni0:(i+1) * Ni0] = X0[:, :, i]
    

    X1full = np.zeros((d, Ni1 * n))
    for i in range(n):
        X1full[:, i * Ni1:(i+1) * Ni1] = X1[:, :, i]

    def Loss1(w):
        return np.mean(-np.log(sigmoid(w.T @ X1full)))

    w0 = np.ones(d)
    eps = 1e-3
    wfeas = minimize(Loss1, w0, tol=eps, method="L-BFGS-B")['x']
    
    
    import matplotlib.pyplot as plt
    w = wfeas
    plt.scatter(X0full[0, :], X0full[1, :], label="0")
    plt.scatter(X1full[0, :], X1full[1, :], label="1")
    plt.legend()
    x = np.array(range(-1, 1))
    y = -(w[0]/w[1]) * x - w[2]/w[1]
    plt.plot(x, y)
    plt.savefig("data.png")
    
    rfull = np.zeros(n)
    delta_r = 10 * Ni1
    for i in range(n):
        # rfull[i] = np.sum(-np.log(sigmoid(wfeas.T @ X1[:, :, i]))) + delta_r
        rfull[i] = delta_r

    w0 = np.zeros(d)
    mu0full = np.zeros(n)
    rhofull = np.ones(n) * 0.01 #! set a smaller number, check the constraints
    beta = 10
    eps1 = 1e-3
    eps2 = 1e-3

    w, out_iter, in_iter, stat_val, feas_val, obj_val = proxAL(w0, mu0full, X0, X1, rfull, admm, beta, rhofull, eps1, eps2)
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


import matplotlib.pyplot as plt
w = wfeas
plt.scatter(X0full[0, :], X0full[1, :], label="0")
plt.scatter(X1full[0, :], X1full[1, :], label="1")
plt.legend()
x = np.array(range(-1, 1))
y = (w[1]/w[0]) * x - w[2]
plt.plot(x, y)
plt.savefig("data_final.png")