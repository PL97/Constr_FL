# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:02:10 2023

@author: chuan
"""

import numpy as np
from proxAL import proxAL
from admm import admm, sigmoid
from admm_uc import admm_uc
from solve_uc import solve_uc
from solve_c import solve_c
from scipy.optimize import minimize

from utils.prepare_data import load_uci
from utils.eval import model_eval

np.random.seed(1)

d = 10 #! 30 ~ 50
n = 5
Ni0 = 100 #! 30000
Ni1 = 10

X0 = (np.random.rand(d, Ni0, n)- 0.3)
X1 = (np.random.rand(d, Ni1, n) - 1 + 0.3)

w0 = np.ones(d)

tau = 1e-3

eps1 = tau 

eps2 = tau

beta = 1


mu0 = np.zeros(n)

r = np.ones(n) *1

#wsol_uc = solve_uc(w0, X0, X1, tau)
wsol_c, mu_c, out_iter = solve_c(w0, mu0, X0, X1, r, beta, eps1, eps2)


rhofull = np.ones(n) * 0.1
#wsol_fluc, initer = admm_uc(w0, X0, X1, rhofull, tau)



wsol_flc, mu_flc, out_iter, in_iter = proxAL(w0, mu0, X0, X1, r, admm, beta, rhofull, eps1, eps2)

eps = 1e-50
def obj_val(w):
    objval = 0
    for i in range(n):
        inner_prod_X0 = np.dot(w.T, X0[:, :, i])
        inner_prod_X1 = np.dot(w.T, X1[:, :, i])
        first_term = np.sum(inner_prod_X0-np.log(sigmoid(inner_prod_X0)+eps))
        second_term = np.sum(-np.log(sigmoid(inner_prod_X1)+eps))
        objval = objval + first_term + second_term

    return objval

def constr_val(w):
    constrval = np.zeros(n)
    for i in range(n):
        inner_prod_X1 = np.dot(w.T, X1[:, :, i])
        constrval[i] = np.sum(-np.log(sigmoid(inner_prod_X1)+eps)) 
    return constrval
#wsol_feduc, in_iter = admm_uc(w0, X0, X1, rhofull, tau)


def uc_stat(w):
    ucstat = 0
    for i in range(n):
        inner_prod_X0 = np.dot(w.T, X0[:, :, i])
        inner_prod_X1 = np.dot(w.T, X1[:, :, i])
        sigmoid_term0 = sigmoid(inner_prod_X0)
        sigmoid_term1 = sigmoid(-inner_prod_X1)
        first_term = np.dot(X0[:, :, i], sigmoid_term0.T)
        second_term = -np.dot(X1[:, :, i], sigmoid_term1.T)
        
        ucstat = ucstat + first_term + second_term
    
    return np.linalg.norm(ucstat, np.inf)

def c_stat(w, mu):
    cstat = 0
    for i in range(n):
        inner_prod_X0 = np.dot(w.T, X0[:, :, i])
        inner_prod_X1 = np.dot(w.T, X1[:, :, i])
        sigmoid_term = sigmoid(inner_prod_X0)
        first_term = np.dot(X0[:, :, i], sigmoid_term.T)
        second_term = -np.dot(X1[:, :, i], sigmoid(-inner_prod_X1))
        
        cstat = cstat + first_term + second_term + second_term * mu[i]
    
    return np.linalg.norm(cstat , np.inf)


#print("wsol (unconstrained): ", wsol_uc)
#print("wsol (constrained): ", wsol_c)

#print("objective value & constraint violation (unconstrained): ", [obj_val(wsol_uc), constr_val(wsol_uc)])
#print("objective value & constraint violation (constrained): ", [obj_val(wsol_c), constr_val(wsol_c)]) 

#print("stationary measure (centralized, unconstrained): ", uc_stat(wsol_uc))
#print("stationary measure (decentralized, unconstrained): ", uc_stat(wsol_fluc))

print("stationary measure (centralized, constrained): ", [c_stat(wsol_c,mu_c),max(constr_val(wsol_c) - r)])
print("stationary measure (centralized, constrained): ", [c_stat(wsol_flc,mu_flc),max(constr_val(wsol_flc) - r)])


#print(wsol_c)
#print(wsol_flc)

