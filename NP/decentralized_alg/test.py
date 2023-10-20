# -*- coding: utf-8 -*-
import numpy as np
import random
from scipy.optimize import minimize
import sys
sys.path.append("../")
from utils.prepare_data import load_uci



# d = 10    # number of features
# n = 5     # number of clients
# Ni0 = 100 # number of class 0 in client i
# Ni1 = 10  # number of class 1 in client i


# # generate the imbalanced dataset
# X0 = (np.random.rand(d, Ni0, n)- 0.3)
# X1 = (np.random.rand(d, Ni1, n) - 1 + 0.3)

from prettytable import PrettyTable

dataset_name = "adult"
n = 5

# for dataset_name in ["adult", "breast-cancer-wisc", "monks-1"]:
for dataset_name in ["monks-1"]:
    x = PrettyTable()
    x.field_names = ["n", "obj_ours", "obj_cen", "obj_uncnstr", "feas(ours) mean", "feas(ours) max", "feas(cen) mean", "feas(cen) max", "feas(uncnstr) mean", "feas(uncnstr) max"]
    for n in [1, 5, 10, 20]:
    
        X0, X1, d = load_uci(name=dataset_name, num_split=n)
        # X0, X1, d = load_uci(name="monks-1", num_split=n)
        # X0, X1, d = load_uci(name="breast-cancer-wisc", num_split=n)


        r = np.ones(n) * 0.2
        beta = 300
        rho = np.ones(n) * 0.01


        #wsol, in_iter = admm(w0,muk=mu,wk=wk,tauk=1e-5)


        #print(np.sum(np.zeros((2,3)),axis=0))

        from proxAL import federa_proxAL, central_proxAL
        from admm import constr_stat, obj_val, cnstr_val


        eps1 = 1e-3
        eps2 = 1e-3
        mu0 = np.zeros(n)
        w0 = np.ones(d)
        
        from solve_uc import solve_uc
        
        uncontr_w = solve_uc(w0, X0, X1, eps1)
        unconstr_obj = obj_val(uncontr_w, X0)
        unconstr_cnstr = cnstr_val(uncontr_w, X1)
        

        # constrained stationary
        # print(constr_stat(wk,muk))
        mu0 = np.zeros(n)
        w0 = np.ones(d)
        # wk_c, muk_c = central_proxAL(w0,mu0)
        central_w, central_mu = central_proxAL(X0=X0, X1=X1, w0=w0, mu0=mu0, beta=beta, r=r, eps1=eps1, eps2=eps2)
        central_obj = obj_val(central_w, X0)
        central_cnstr = cnstr_val(central_w, X1)

        
        # wk, muk = federa_proxAL(w0,mu0)
        federa_w, federa_mu, objs, constrs = federa_proxAL(X0, X1, w0=w0, mu0=mu0, eps1=eps1, eps2=eps2, beta=beta, rho=rho, r=r)
        federa_obj = obj_val(federa_w, X0)
        federa_cnstr = cnstr_val(federa_w, X1)
        np.save(f"files/obj_{dataset_name}_{n}.npy", objs)
        np.save(f"files/constr_{dataset_name}_{n}.npy", constrs)




        print("obj")
        print(central_obj)
        print(federa_obj)
        print("constr")
        print(central_cnstr)
        print(federa_cnstr)

        x.add_row([n, "{:.4f}".format(np.mean(federa_obj)), "{:.4f}".format(np.mean(central_obj)), "{:.4f}".format(np.mean(unconstr_obj)), "{:.4f}".format(np.mean(federa_cnstr)), "{:.4f}".format(np.amax(federa_cnstr)), "{:.4f}".format(np.mean(central_cnstr)), "{:.4f}".format(np.amax(central_cnstr)), "{:.4f}".format(np.mean(unconstr_cnstr)), "{:.4f}".format(np.amax(unconstr_cnstr))])

    print(x)