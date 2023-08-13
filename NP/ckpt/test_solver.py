from SpaRSA import SpaRSA
import numpy as np

# def fx(x):
#     return x**2

# def dfx(x):
#     return np.asarray([2*x])


d = 10
n = 5
Ni0 = 100
Ni1 = 10

X0 = np.random.rand(d, Ni0, n) - 0.5
X1 = np.random.rand(d, Ni1, n) - 0.5 

X1full = np.zeros((d, Ni1 * n))
for i in range(n):
    X1full[:, i * Ni1:(i+1) * Ni1] = X1[:, :, i]


def fx(w):
    return np.sum(-w.T @ X1full + np.log(1 + np.exp(w.T @ X1full)))

def dfx(w):
    return -X1full @ (1 / (1 + np.exp(w.T @ X1full))).T

def test_scipy(fx, x):
    from scipy.optimize import minimize
    
    return minimize(fx, x, tol=1e-10, method='CG')


def test_sparsa(fx, dfx, x):
    return SpaRSA(x, fx, dfx, 1e-5)

if __name__ == "__main__":
    # x = 10
    # print(x['x'])
    
    # x = 10
    w0 = np.ones(d)
    x = test_sparsa(fx, dfx, w0)
    print(fx(x))
    print(x)
    
    w0 = np.ones(d)
    w0 = test_scipy(fx, w0)
    print(w0)
    
    