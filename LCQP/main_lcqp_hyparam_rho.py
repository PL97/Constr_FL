import numpy as np
from proxAL import proxAL
from cproxAL import cproxAL
from admm import admm
from utils import generate_data, format_results, measure_runtime

import cProfile, pstats, io
from pstats import SortKey
import re

# Define the size table
d_n_m = np.array([
    [500, 10, 1]
])

num_spl, _ = d_n_m.shape

num_rdn = 1

# ave_rep = np.zeros((num_spl, 9))

list_rec = np.zeros((9))

seed = 1

for rho in [0.1, 1, 10, 100, 1000]:

    # problem size
    d, n, m = d_n_m[0]

    # random seed
    np.random.seed(seed)

    A, b, C, dfull = generate_data(d, n, m)

    # hyperparameters
    eps1 = 1e-4
    eps2 = 1e-4
    beta = 1
    rhofull = np.ones(n) * rho
    bars=0.01
    w0 = np.ones(d)
    mu0full = np.zeros((m, n))
    
    ret_cproxal = measure_runtime(cproxAL, w0=w0, mu0full=mu0full, A=A, b=b, C=C, dfull=dfull, beta=beta, eps1=eps1, eps2=eps2, bars=bars)

    list_rec[1] = ret_cproxal['objcpal']
    list_rec[3] = ret_cproxal["constrcpal"]
    list_rec[5] = ret_cproxal["coutiter"]
    list_rec[7] = ret_cproxal["runtime"]

    # proximal AL based FL algorithm
    # Assuming you've implemented the proxAL function properly
    # wpal, objpal, constrpal, floutiter, flttiniter = proxAL(w0, mu0full, A, b, C, dfull, admm, beta, rhofull, eps1, eps2)
    ret_proxal = measure_runtime(proxAL, w0=w0, mu0full=mu0full, A=A, b=b, C=C, dfull=dfull, admm=admm, beta=beta, rhofull=rhofull, eps1=eps1, eps2=eps2, bars=bars)

    list_rec[0] = ret_proxal["objpal"]
    list_rec[2] = ret_proxal["constrpal"]
    list_rec[4] = ret_proxal["floutiter"]
    list_rec[6] = ret_proxal["flttiniter"]
    list_rec[8] = ret_proxal["runtime"]

    # ave_rep[ii, :] = np.mean(list_rec, axis=0)

    print(f"----------------rho={rho}------------------")
     
    format_results(list_rec)
