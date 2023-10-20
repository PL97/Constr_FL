
import numpy as np



from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["n", "obj_ours", "obj_cen", "obj_uncnstr", "disparity(ours) mean", "disparity(ours) max", "disparity(cen) mean", "disparity(cen) max","disparity(uncnstr) mean", "disparity(uncnstr) max"]



import sys
sys.path.append("wglobal")
from wglobal.proxal import federa_proxAL, central_proxAL
from wglobal.utils.utils import constr_stat, obj_val, cnstr_val, sigmoid
from wglobal.solve_uc import solve_uc
from wglobal.utils.prepare_data import load_uci
from wglobal.admm import admm



from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["n", "obj_ours", "obj_cen", "obj_uncnstr", "disparity(ours) mean", "disparity(ours) max", "disparity(ours) global", "disparity(cen) mean", "disparity(cen) max", "disparity(cen) global", "disparity(uncnstr) mean", "disparity(uncnstr) max", "disparity(uncnstr) global"]
dataset_name = "adult"

for n in [5, 10, 20]:
    X0, X1, d = load_uci(n, seed=10)
    X0g, X1g, _ = load_uci(1, seed=10, file_name="adult_test1")
    X0g = np.squeeze(X0g)
    X1g = np.squeeze(X1g)
    mu0 = np.zeros((2,n+1))
    beta = 10

    w0 = np.zeros(d)

    r = np.ones(n+1) * 0.1

    rho = np.ones(n) * 1e8

    eps1 = 1e-2
    eps2 = 1e-2

    eps = 1e-50

    wk_u = solve_uc(w0, X0, X1, eps1)
    
    
    wk_c, muk_c, _, _ = central_proxAL(X0, X1, X0g, X1g, w0,mu0, beta, r, eps1, eps2)
    
    w0 = np.zeros(d)
    mu0 = np.zeros((2,n+1))
    wk, muk, objs, constrs = federa_proxAL(X0, X1, X0g, X1g, w0,mu0, beta, r, rho, eps1, eps2)

    np.save(f"files_new/obj_{dataset_name}_{n}.npy", objs)
    np.save(f"files_new/constr_{dataset_name}_{n}.npy", constrs)
    

    federa_obj = obj_val(X0, X1, wk)
    central_obj = obj_val(X0, X1, wk_c)
    unconstr_obj = obj_val(X0, X1, wk_u)
    
    federa_cnstr = cnstr_val(X0, X1, X0g, X1g, wk)
    central_cnstr = cnstr_val(X0, X1, X0g, X1g, wk_c)
    unconstr_cnstr = cnstr_val(X0, X1, X0g, X1g, wk_u)

    x.add_row([n, "{:.4f}".format(federa_obj), "{:.4f}".format(central_obj), "{:.4f}".format(unconstr_obj), \
               "{:.4f}".format(np.mean(federa_cnstr)), "{:.4f}".format(np.amax(federa_cnstr)), "{:.4f}".format(federa_cnstr[-1]), \
                "{:.4f}".format(np.mean(central_cnstr)), "{:.4f}".format(np.amax(central_cnstr)), "{:.4f}".format(central_cnstr[-1]), \
                "{:.4f}".format(np.mean(unconstr_cnstr)), "{:.4f}".format(np.amax(unconstr_cnstr)), "{:.4f}".format(unconstr_cnstr[-1])])

print("experiment setup")
print(f"beta={beta}\tr={r}\trho={rho}\tn={n}")

print(x)
