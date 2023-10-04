
import numpy as np
from admm import admm
from proxal import federa_proxAL, central_proxAL
from utils.utils import constr_stat, obj_val, cnstr_val


d = 5     # number of features
n = 5     # number of clients
Ni0 = 100 # number of class 0 in client i
Ni1 = 100  # number of class 1 in client i


N = (Ni0+Ni1) * n

# generate the imbalanced dataset
X0_ns = (np.random.rand(d-1, Ni0, n)- 0.3)
X1_ns = (np.random.rand(d-1, Ni1, n) - 1 + 0.3)

X0_s = np.random.randint(2, size=(Ni0, n))
X1_s = np.random.randint(2, size=(Ni1, n))


X0 = np.zeros((d,Ni0,n))
X1 = np.zeros((d,Ni1,n))

for i in range(n):
    X0[:,:,i] = np.concatenate((X0_ns[:,:,i],X0_s[:,i].reshape((1,Ni0))), axis=0)
    X1[:,:,i] = np.concatenate((X1_ns[:,:,i],X1_s[:,i].reshape((1,Ni1))), axis=0)


mu0 = np.zeros((2,n))
beta = 10



w0 = np.ones(d)

r = np.ones(n) * 0.2

rho = np.ones(n) * 1


eps1 = 1e-2
eps2 = 1e-2

w0 = np.ones(d)
mu0 = np.zeros((2,n))
wk, muk = federa_proxAL(X0, X1, w0,mu0, beta, rho, r, eps1, eps2)


print(wk)

print(constr_stat(X0, X1, wk, muk, r))

wk_c, muk_c = central_proxAL(X0, X1, w0,mu0, beta, r, eps1, eps2)

print(obj_val(X0, X1, wk),obj_val(X0, X1, wk_c))

print(cnstr_val(X0, X1, wk),cnstr_val(X0, X1, wk_c))