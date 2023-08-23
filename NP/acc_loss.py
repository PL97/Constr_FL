import numpy as np
from proxAL import proxAL
from admm import admm, sigmoid
from scipy.optimize import minimize

from utils.prepare_data import load_uci
from utils.eval import model_eval

num_seed = 1
np.random.seed(0)

n = 10
X0, X1, d = load_uci(name="monks-1", num_split=n)
Ni0 = X0.shape[1]
Ni1 = X1.shape[1]


X1full = np.zeros((d, Ni1 * n))
for i in range(n):
    X1full[:, i * Ni1:(i+1) * Ni1] = X1[:, :, i]
    
X0full = np.zeros((d, Ni0 * n))
for i in range(n):
    X0full[:, i * Ni0:(i+1) * Ni0] = X0[:, :, i]
    
    
X1full_ex = np.expand_dims(X1full, -1)
X0full_ex = np.expand_dims(X0full, -1)

def Loss1(w):
    return np.mean(-np.log(sigmoid(w.T @ X1full)))

def Loss2(w):
    return np.mean(np.log(sigmoid(w.T @ X0full)))

def DLoss1(w):
    sig = sigmoid(w.T @ X1full)
    return (1/X1full.shape[1])*X1full@(sig - 1)

def DLoss2(w):
    sig = sigmoid(w.T @ X0full)
    return (1/X0full.shape[1])*X0full@(sig + 1)



w0 = np.ones(d)
lr = 1

total_e = 50

for i in range(total_e):
    print(Loss2(w0))
    w0 -= lr * DLoss2(w0)
    model_eval(w0, X0full_ex, X1full_ex)
    
# w0 = minimize(Loss1, w0, tol=0.001, method="L-BFGS-B")['x']
# model_eval(w0, X0full_ex, X1full_ex)
# asdf
    
print("------------------------------------")

lr = 10

for i in range(total_e):
    print(Loss1(w0))
    w0 -= lr * DLoss1(w0)
    model_eval(w0, X0full_ex, X1full_ex)
    
    
w0 = minimize(Loss1, w0, tol=0.0001, method="L-BFGS-B")['x']
model_eval(np.ones(d), X0full_ex, X1full_ex)
asdf
    