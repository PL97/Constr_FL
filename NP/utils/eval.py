import numpy as np
from admm import sigmoid

eps = 1e-50

def model_eval(w, X0, X1):
    ret = []
    for i in range(X1.shape[-1]):
        TN = np.sum(sigmoid(w.T @ X0[:, :, i]) < 0.5)
        TP = np.sum(sigmoid(w.T @ X1[:, :, i]) >= 0.5)
        ret.append({"Precall": TP/X1.shape[1]})
        ret.append({"Nrecall": TN/X0.shape[1]})
    return ret




eps = 1e-50
def obj_val(w, X0, X1):
    Ni0, Ni1 = X0.shape[1], X1.shape[1]
    objval = 0
    n = X1.shape[2]
    for i in range(n):
        inner_prod_X0 = np.dot(w.T, X0[:, :, i])
        inner_prod_X1 = np.dot(w.T, X1[:, :, i])
        first_term = np.sum(inner_prod_X0-np.log(sigmoid(inner_prod_X0)+eps))
        second_term = np.sum(-np.log(sigmoid(inner_prod_X1)+eps))
        objval = objval + first_term + second_term
    objval /= (n*Ni0+Ni1)
    return objval

def constr_val(w, X1):
    n = X1.shape[2]
    constrval = np.zeros(n)
    for i in range(n):
        inner_prod_X1 = np.dot(w.T, X1[:, :, i])
        constrval[i] = np.sum(-np.log(sigmoid(inner_prod_X1)+eps)) 
    return constrval
#wsol_feduc, in_iter = admm_uc(w0, X0, X1, rhofull, tau)


def uc_stat(w, X1, X0):
    n = X1.shape[2]
    ucstat = 0
    for i in range(n):
        inner_prod_X0 = np.dot(w.T, X0[:, :, i])
        inner_prod_X1 = np.dot(w.T, X1[:, :, i])
        sigmoid_term0 = sigmoid(inner_prod_X0)
        sigmoid_term1 = sigmoid(-inner_prod_X1)
        first_term = np.dot(X0[:, :, i], sigmoid_term0.T)
        second_term = -np.dot(X1[:, :, i], sigmoid_term1.T)
        
        ucstat = ucstat + first_term + second_term
    
    return np.linalg.norm(ucstat , np.inf)

def c_stat(w, X1, X0, mu_c):
    n = X1.shape[2]
    cstat1 = 0
    cstat2 = 0
    for i in range(n):
        inner_prod_X0 = np.dot(w.T, X0[:, :, i])
        inner_prod_X1 = np.dot(w.T, X1[:, :, i])
        sigmoid_term = sigmoid(inner_prod_X0)
        first_term = np.dot(X0[:, :, i], sigmoid_term.T)
        second_term = -np.dot(X1[:, :, i], sigmoid(-inner_prod_X1))
        
        cstat1 = cstat1 + first_term + second_term 
        cstat2 = cstat2 + second_term * mu_c[i]
        
    cstat = cstat1 + cstat2
    print("cstat")
    print(np.linalg.norm(cstat1, np.inf), np.linalg.norm(cstat2, np.inf))
    print(mu_c)
    
    return np.linalg.norm(cstat , np.inf)