# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:02:10 2023

@author: chuan
"""
import sys

sys.path.append("../")

import numpy as np
from proxAL import proxAL
from admm import admm, sigmoid
# from admm_uc import admm_uc
from solve_uc import solve_uc
from solve_c import solve_c
from scipy.optimize import minimize

from utils.prepare_data import load_uci
from utils.eval import model_eval, obj_val, constr_val, uc_stat, c_stat
from utils.prepare_data import load_uci

np.random.seed(1)


from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["n", "obj(ours)", "obj(unconst)", "feas(ours)", "feas(unconst)"]

for n in [1]:
    print("==========================================================")
    # X0, X1, d = load_uci(name="breast-cancer-wisc", num_split=n)
    d = 10 #! 30 ~ 50
    n = 5
    Ni0 = 100 #! 30000
    Ni1 = 10

    X0 = (np.random.rand(d, Ni0, n)- 0.3)
    X1 = (np.random.rand(d, Ni1, n) - 0.3)


    w0 = np.ones(d)

    tau = 1e-3

    eps1 = tau 

    eps2 = tau

    beta = 1


    mu0 = np.zeros(n)
    constr_threshold = 0.2
    r = np.ones(n) * constr_threshold * Ni1

    wsol_uc = solve_uc(w0, X0, X1, tau)

    # print("objective value & constraint violation (constrained): ", [obj_val(wsol_c), constr_val(wsol_c)]) 


    wsol_c, mu_c, out_iter, objs, constrs = solve_c(w0, mu0, X0, X1, r, beta, eps1, eps2)

    # np.save("obj.npy", objs)
    # np.save("constr.npy", constrs)
    
    
    obj_o = obj_val(wsol_c, X0, X1)
    feas_o = np.amax(constr_val(wsol_c, X1)/Ni1-constr_threshold)
    obj_uc = obj_val(wsol_uc, X0, X1)
    feas_uc = np.amax(constr_val(wsol_uc, X1)/Ni1-constr_threshold)
    
    x.add_row([n, "{:.4f}".format(obj_o), "{:.4f}".format(obj_uc), "{:.4f}".format(feas_o), "{:.4f}".format(feas_uc)])
    
    
    print(wsol_c, wsol_uc)
    print(uc_stat(wsol_uc, X1, X0))
    print(c_stat(wsol_c, X1, X0, mu_c))
    
    print(constr_val(wsol_c, X1)/Ni1-constr_threshold)
    

    # print("objective value & constraint violation (unconstrained): ", [obj_val(wsol_uc, X0, X1), np.sum(constr_val(wsol_uc, X1))/(X1.shape[1]*X1.shape[2])])
    # print(constr_val(wsol_uc, X1)/X1.shape[1])
    # print("objective value & constraint violation (constrained): ", [obj_val(wsol_c, X0, X1), np.sum(constr_val(wsol_c, X1))/(X1.shape[1]*X1.shape[2])]) 
    # print(constr_val(wsol_c, X1)/X1.shape[1])
    # print("objective value & constraint violation (unconstrained): ", [obj_val(wsol_uc), constr_val(wsol_uc, X1)])
    # print("objective value & constraint violation (constrained): ", [obj_val(wsol_c), constr_val(wsol_c, X1)]) 

    # print("unconstrained: ", model_eval(wsol_uc, X0, X1))
    # print("constrained: ", model_eval(wsol_c, X0, X1))
    
print(x)



