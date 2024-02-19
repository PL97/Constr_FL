
import numpy as np
from admm import admm
from proxal import federa_proxAL, central_proxAL
from utils.utils import constr_stat, obj_val, cnstr_val, sigmoid
from solve_uc import solve_uc
import argparse
from copy import deepcopy
import json
from collections import defaultdict
from utils.prepare_data import load_uci


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["n", "obj_ours", "obj_cen", "obj_uncnstr", "disparity(ours) mean", "disparity(ours) max", "disparity(ours) global", "disparity(cen) mean", "disparity(cen) max", "disparity(cen) global", "disparity(uncnstr) mean", "disparity(uncnstr) max", "disparity(uncnstr) global"]
dataset_name = "adult"
workspace = "new_files_parallel"

parser = argparse.ArgumentParser()
parser.add_argument('--n_client', type=int, default=-1)
parser.add_argument('--repeat_idx', type=int, default=-1)
args = parser.parse_args()
n_clients = [5] if args.n_client == -1 else [args.n_client]
n_repeat = list(range(10)) if args.repeat_idx == -1 else [args.repeat_idx]





tmp_dict = defaultdict(lambda: defaultdict(lambda: {}))
for n in n_clients:
    X0, X1, d = load_uci(n, seed=10)
    X0g, X1g, _ = load_uci(1, seed=10, file_name="adult_test1")
    for random_idx in n_repeat:
        
        np.random.seed(random_idx)

        X0g = np.squeeze(X0g)
        X1g = np.squeeze(X1g)
        mu0 = np.zeros((2,n+1))
        beta = 10

        w0 = np.random.rand(d)
        w0 = w0/np.linalg.norm(w0, ord=2)

        r = np.ones(n+1) * 0.1

        rho = np.ones(n) * 1e8

        eps1 = 1e-3
        eps2 = 1e-3
        
        print(eps1, eps2, rho, beta)

        eps = 1e-50
        
        w0_copied = deepcopy(w0)
        w0_u = w0_copied
        # w0_u = np.zeros(d)
        wk_u = solve_uc(w0_u, X0, X1, eps1)
        
        w0_copied = deepcopy(w0)
        w0_c = w0_copied
        wk_c, muk_c, _, _ = central_proxAL(X0, X1, X0g, X1g, w0_c,mu0, beta, r, eps1, eps2)
        
        w0_copied = deepcopy(w0)
        w0_f = w0_copied
        mu0 = np.zeros((2,n+1))
        wk, muk, objs, constrs = federa_proxAL(X0, X1, X0g, X1g, w0_f, mu0, beta, r, rho, eps1, eps2)

        federa_obj = obj_val(X0, X1, wk)
        central_obj = obj_val(X0, X1, wk_c)
        unconstr_obj = obj_val(X0, X1, wk_u)
        
        federa_cnstr = cnstr_val(X0, X1, X0g, X1g, wk)
        central_cnstr = cnstr_val(X0, X1, X0g, X1g, wk_c)
        unconstr_cnstr = cnstr_val(X0, X1, X0g, X1g, wk_u)
        

print("experiment setup")
print(f"beta={beta}\tr={r}\trho={rho}\tn={n}")
print(x)
