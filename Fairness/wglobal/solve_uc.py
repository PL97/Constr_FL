# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:38:44 2023

@author: chuan
"""

import numpy as np
from utils.utils import sigmoid
from scipy.optimize import minimize



def solve_uc(w0, X0, X1, tau):
    
    d, _, n = X0.shape
    eps = 1e-50
    
    def F(w):
        Fval = 0;
        for i in range(n):
            inner_prod_X0 = np.dot(w.T, X0[:, :, i])
            inner_prod_X1 = np.dot(w.T, X1[:, :, i])
            first_term = np.sum(inner_prod_X0-np.log(sigmoid(inner_prod_X0)+eps))
            second_term = np.sum(-np.log(sigmoid(inner_prod_X1)+eps))
            
            Fval = Fval + first_term + second_term
        
        return Fval
    
    
    wsol = minimize(F, w0, tol=tau * 1e-3, method="L-BFGS-B")['x']
    
    return wsol