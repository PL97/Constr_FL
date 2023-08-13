import numpy as np
from proxAL import proxAL
from cproxAL import cproxAL
from admm import admm
from utils import generate_data, format_results, measure_runtime

# Define the size table
d_n_m = np.array([
    [100, 5, 1],
    [100, 10, 1],
    [200, 5, 2],
    [200, 10, 2],
    [300, 5, 3],
    [300, 10, 3],
    [400, 5, 4],
    [400, 10, 4],
    [500, 5, 5],
    [500, 10, 5],
])

num_spl, _ = d_n_m.shape

num_rdn = 1

repeat_num = 10

for ii in range(num_spl):

    # problem size
    d, n, m = d_n_m[ii]

    # random seed
    np.random.seed(210)

    A, b, C, dfull = generate_data(d, n, m)


    # hyperparameters
    eps1 = 1e-3
    eps2 = 1e-3
    beta = 1
    rhofull = np.ones(n) * 1
    w0 = np.ones(d)
    mu0full = np.zeros((m, n))

    list_rec = np.zeros((9, repeat_num))

    for i in range(repeat_num):
        np.random.seed(i)
        ret_cproxal = measure_runtime(cproxAL, w0=w0, mu0full=mu0full, A=A, b=b, C=C, dfull=dfull, beta=beta, eps1=eps1, eps2=eps2)

        list_rec[1, i] = ret_cproxal['objcpal']
        list_rec[3, i] = ret_cproxal["constrcpal"]
        list_rec[5, i] = ret_cproxal["coutiter"]
        list_rec[7, i] = ret_cproxal["runtime"]


        w0 = np.ones(d)
        
        # proximal AL based FL algorithm
        # Assuming you've implemented the proxAL function properly
        # wpal, objpal, constrpal, floutiter, flttiniter = proxAL(w0, mu0full, A, b, C, dfull, admm, beta, rhofull, eps1, eps2)
        ret_proxal = measure_runtime(proxAL, w0=w0, mu0full=mu0full, A=A, b=b, C=C, dfull=dfull, admm=admm, beta=beta, rhofull=rhofull, eps1=eps1, eps2=eps2)
        ret_cproxal = measure_runtime(cproxAL, w0=w0, mu0full=mu0full, A=A, b=b, C=C, dfull=dfull, beta=beta, eps1=eps1, eps2=eps2)

        list_rec[0, i] = ret_proxal["objpal"]
        list_rec[2, i] = ret_proxal["constrpal"]
        list_rec[4, i] = ret_proxal["floutiter"]
        list_rec[6, i] = ret_proxal["flttiniter"]
        list_rec[8, i] = ret_proxal["runtime"]

    # ave_rep[ii, :] = np.mean(list_rec, axis=0)

    list_rec_mean = np.mean(list_rec, axis=1)
    list_rec_std = np.std(list_rec, axis=1)
    
    results = []
    for i in range(9):
        results.append(f"{0:.4f} ± {0:.4f}".format(list_rec_mean[i], list_rec_std[i]))


    print(f"----------------{d_n_m[ii]}------------------")
    print("mean")
    format_results(list_rec_mean)
    print("std")
    format_results(list_rec_std)
