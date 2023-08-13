import numpy as np
from proxAL import proxAL
from cproxAL import cproxAL
from admm import admm
from utils import format_results, generate_data

# Define the size table
d_n_m = np.array([
    [100, 10, 1],
    [100, 20, 1],
    [200, 10, 2],
    [200, 20, 2],
    [300, 10, 3],
    [300, 20, 3],
    [400, 10, 4],
    [400, 20, 4],
    [500, 10, 5],
    [500, 20, 5]
])

num_spl, _ = d_n_m.shape

# num_spl = 1

num_rdn = 1

seed=10


list_rec = np.zeros(7)



for ii in range(num_spl):

    # problem size
    d, n, m = d_n_m[ii]

    # random seed
    np.random.seed(seed)

    # generate the random matrices A, b, C, d
    A, b, C, dfull = generate_data(d, n, m)

    # hyperparameters
    eps1 = 1e-4
    eps2 = 1e-4
    beta = 1
    rhofull = np.ones(n) * 10
    w0 = np.ones(d)
    mu0full = np.zeros((m, n))

    # centralized proximal AL method
    # Assuming you've implemented the cproxAL function properly
    ret_cprox = cproxAL(w0, mu0full, A, b, C, dfull, beta, eps1, eps2)

    list_rec[1] = ret_cprox["objcpal"]
    list_rec[3] = ret_cprox["constrcpal"]
    list_rec[5] = ret_cprox["coutiter"]

    # proximal AL based FL algorithm
    # Assuming you've implemented the proxAL function properly
    ret_prox = proxAL(w0, mu0full, A, b, C, dfull, admm, beta, rhofull, eps1, eps2)

    list_rec[0] = ret_prox["objpal"]
    list_rec[2] = ret_prox["constrpal"]
    list_rec[4] = ret_prox["floutiter"]
    list_rec[6] = ret_prox["flttiniter"]
    
    format_results(list_rec)
