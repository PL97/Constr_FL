# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:10:52 2023

@author: chuan
"""

import numpy as np
from admm import sigmoid
from scipy.optimize import minimize

import sys
sys.path.append("../")
from utils.eval import model_eval, obj_val, constr_val, c_stat

def solve_c(w0, mu0, X0, X1, r, beta, eps1, eps2):
    wc = w0
    K = 10000
    muk = mu0
    bars = 0.001


    d, _, n = X1.shape
    
    eps = 1e-50
    
    objs, constrs = [], []
    
    for k in range(K + 1):
        objs.append(obj_val(wc, X0, X1))
        constrs.append(constr_val(wc, X1)/(X1.shape[1]))
        print(f"----------------{k}/{K}---------------")
        tauk = bars / (k + 1)**2

        wp = wc.copy()
        
        def AL(w):
            ALval = 0
            for i in range(n):
                inner_prod_X0 = np.dot(w.T, X0[:, :, i])
                inner_prod_X1 = np.dot(w.T, X1[:, :, i])
                first_term = np.sum(inner_prod_X0-np.log(sigmoid(inner_prod_X0)+eps)) + np.sum(-np.log(sigmoid(inner_prod_X1)+eps))   #! add a eps term to avoid log 0
                second_term = np.maximum(muk[i] + beta * (np.sum(-np.log(sigmoid(inner_prod_X1)+eps)) - r[i]), 0) ** 2 / (2 * beta)
                ALval = ALval + first_term + second_term
            
            third_term = np.linalg.norm(w - wc, 2) ** 2 / (2 * beta) #? change wc to wp
            return ALval + third_term
        
        wc = minimize(AL, wc, tol=tauk, method="L-BFGS-B")['x']
        

        mup = muk.copy()
       
        
        for i in range(n):
            inner_prod_X1 = np.dot(wc.T, X1[:, :, i])
            muk[i] = np.maximum(muk[i] + beta * (np.sum(-np.log(sigmoid(inner_prod_X1))) - r[i]), 0)
        
        
        print(np.linalg.norm(wp - wc, np.inf))
        print(max(np.abs(mup - muk)))
        
        if np.linalg.norm(wp - wc, np.inf) + beta * tauk <= beta * eps1:
            if max(np.abs(mup - muk)) <= beta * eps2:
                break
        # cstat = c_stat(wc, X1, X0, mu_c=muk)
        
        
        
        # print("diffmu: ", mup - muk)
        # print(constr_val(wc, X1) - r)
        # print("diffw: ", np.linalg.norm(wp - wc, np.inf) + beta * tauk) 
        # print("constrained stat: ", cstat)
        
        
        
        
    out_iter = k + 1
    w = wc
    
    return w, muk, out_iter, objs, constrs