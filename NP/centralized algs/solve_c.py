# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 14:10:52 2023

@author: chuan
"""

import numpy as np
from admm import sigmoid
from scipy.optimize import minimize

def solve_c(w0, mu0, X0, X1, r, beta, eps1, eps2):
    wc = w0
    K = 10000
    muk = mu0
    bars = 0.001


    d, _, n = X1.shape
    
    eps = 1e-50
    
    for k in range(K + 1):
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
            
            third_term = np.linalg.norm(w - wc, 2) ** 2 / (2 * beta)
            ALval = ALval + third_term
            
            
            return ALval
        
        wc = minimize(AL, wc, tol=tauk, method="L-BFGS-B")['x']
        

        mup = muk.copy()
        
        for i in range(n):
            inner_prod_X1 = np.dot(wc.T, X1[:, :, i])
            muk[i] = np.maximum(muk[i] + beta * (np.sum(-np.log(sigmoid(inner_prod_X1)+eps)) - r[i]), 0)
        
        if np.linalg.norm(wp - wc, np.inf) + beta * tauk <= beta * eps1:
            if max(np.abs(mup - muk)) <= beta * eps2:
                break
            
        print("diffmu: ", max(np.abs(mup - muk)))
        print("diffw: ", np.linalg.norm(wp - wc, np.inf) + beta * tauk) 
        
    out_iter = k + 1
    w = wc
    return w, muk, out_iter