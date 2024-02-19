
import numpy as np
from admm import admm
from proxal import federa_proxAL, central_proxAL
from utils.utils import constr_stat, obj_val, cnstr_val
from solve_uc import solve_uc

# https://ocw.mit.edu/courses/res-ec-001-exploring-fairness-in-machine-learning-for-international-development-spring-2020/pages/module-four-case-studies/case-study-mitigating-gender-bias/

# d = 10     # number of features
# n = 5     # number of clients
# Ni0 = 100 # number of class 0 in client i
# Ni1 = 100  # number of class 1 in client i


# N = (Ni0+Ni1) * n

# # generate the dataset on clients
# X0_ns = (np.random.rand(d-1, Ni0, n)- 0.3)
# X1_ns = (np.random.rand(d-1, Ni1, n) - 1 + 0.3)

# X0_s = np.random.randint(2, size=(Ni0, n))
# X1_s = np.random.randint(2, size=(Ni1, n))


# X0 = np.zeros((d,Ni0,n))
# X1 = np.zeros((d,Ni1,n))

# for i in range(n):
#     X0[:,:,i] = np.concatenate((X0_ns[:,:,i],X0_s[:,i].reshape((1,Ni0))), axis=0)
#     X1[:,:,i] = np.concatenate((X1_ns[:,:,i],X1_s[:,i].reshape((1,Ni1))), axis=0)


# Ni0g = 200 # number of class 0 in the central server
# Ni1g = 200 # number of class 1 in the central server

# # generate the dataset on server
# X0g_ns = (np.random.rand(d-1, Ni0g)- 0.3)
# X1g_ns = (np.random.rand(d-1, Ni1g) - 1 + 0.3)

# X0g_s = np.random.randint(2, size=(Ni0g))
# X1g_s = np.random.randint(2, size=(Ni1g))


# X0g = np.concatenate((X0g_ns,X0g_s.reshape((1,Ni0g))), axis=0)
# X1g = np.concatenate((X1g_ns,X1g_s.reshape((1,Ni1g))), axis=0)

from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["n", "obj_ours", "obj_cen", "obj_uncnstr", "disparity(ours) mean", "disparity(ours) max", "disparity(cen) mean", "disparity(cen) max", "disparity(uncnstr) mean", "disparity(uncnstr) max"]

for n in [1, 5]:
    from utils.prepare_data import load_uci

    X0, X1, d = load_uci(n, seed=10)
    X0g, X1g, _ = load_uci(1, seed=10, file_name="adult_test1")
    X0g = np.squeeze(X0g)
    X1g = np.squeeze(X1g)

    mu0 = np.zeros((2,n+1))
    beta = 100

    w0 = np.ones(d)

    r = np.ones(n+1) * 0.1

    rho = np.ones(n) * 1

    eps1 = 1e-2
    eps2 = 1e-2


    wk_u = solve_uc(w0, X0, X1, eps1)
    
    
    wk_c, muk_c = central_proxAL(X0, X1, X0g, X1g, w0,mu0, beta, r, eps1, eps2)


    w0 = np.ones(d)
    mu0 = np.zeros((2,n+1))
    wk, muk = federa_proxAL(X0, X1, X0g, X1g, w0,mu0, beta, r, rho, eps1, eps2)


    print(constr_stat(X0, X1, wk, muk))
    print(obj_val(X0, X1, wk),obj_val(X0, X1, wk_c))
    print(cnstr_val(X0, X1, wk),cnstr_val(X0, X1, wk_c))
    print(muk, muk_c)

    federa_obj = obj_val(X0, X1, wk)
    central_obj = obj_val(X0, X1, wk_c)
    unconstr_obj = obj_val(X0, X1, wk_u)
    
    federa_cnstr = cnstr_val(X0, X1, wk)
    central_cnstr = cnstr_val(X0, X1, wk_c)
    unconstr_cnstr = cnstr_val(X0, X1, wk_u)

    x.add_row([n, "{:.4f}".format(federa_obj), "{:.4f}".format(central_obj), "{:.4f}".format(unconstr_obj), "{:.4f}".format(np.mean(federa_cnstr)), "{:.4f}".format(np.amax(federa_cnstr)), "{:.4f}".format(np.mean(central_cnstr)), "{:.4f}".format(np.amax(central_cnstr)), "{:.4f}".format(np.mean(unconstr_cnstr)), "{:.4f}".format(np.amax(unconstr_cnstr))])


print(x)