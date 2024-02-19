
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

# https://ocw.mit.edu/courses/res-ec-001-exploring-fairness-in-machine-learning-for-international-development-spring-2020/pages/module-four-case-studies/case-study-mitigating-gender-bias/


# d = 10     # number of features
# n = 5     # number of clients
# Ni0 = 100 # number of class 0 in client i
# Ni1 = 100  # number of class 1 in client i


# N = (Ni0+Ni1) * n

# # # generate the dataset on clients
# X0_ns = (np.random.rand(d-1, Ni0, n)- 0.3)
# X1_ns = (np.random.rand(d-1, Ni1, n) - 1 + 0.3)

# X0_s = np.random.randint(2, size=(Ni0, n))
# X1_s = np.random.randint(2, size=(Ni1, n))


# X0 = np.zeros((d,Ni0,n))
# X1 = np.zeros((d,Ni1,n))

# for i in range(n):
#     X0[:,:,i] = np.concatenate((X0_ns[:,:,i],X0_s[:,i].reshape((1,Ni0))), axis=0)
#     X1[:,:,i] = np.concatenate((X1_ns[:,:,i],X1_s[:,i].reshape((1,Ni1))), axis=0)


# Ni0g = 2000 # number of class 0 in the central server
# Ni1g = 2000 # number of class 1 in the central server

# # # generate the dataset on server
# X0g_ns = (np.random.rand(d-1, Ni0g)- 0.3)
# X1g_ns = (np.random.rand(d-1, Ni1g) - 1 + 0.3)

# X0g_s = np.random.randint(2, size=(Ni0g))
# X1g_s = np.random.randint(2, size=(Ni1g))


# X0g = np.concatenate((X0g_ns,X0g_s.reshape((1,Ni0g))), axis=0)
# X1g = np.concatenate((X1g_ns,X1g_s.reshape((1,Ni1g))), axis=0)

# from prettytable import PrettyTable
# x = PrettyTable()
# x.field_names = ["n", "obj_ours", "obj_cen", "obj_uncnstr", "disparity(ours) mean", "disparity(ours) max", "disparity(cen) mean", "disparity(cen) max", "disparity(uncnstr) mean", "disparity(uncnstr) max"]

# if True:

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
n_clients = [20, 10, 5, 1] if args.n_client == -1 else [args.n_client]
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
        w0_u = np.zeros(d)
        wk_u = solve_uc(w0_u, X0, X1, eps1)

        unconstr_obj = obj_val(X0, X1, wk_u)
        
        unconstr_cnstr = cnstr_val(X0, X1, X0g, X1g, wk_u)
        
        tmp_dict[n][random_idx] = {
            "u_w": wk_u,
            "u_obj": unconstr_obj,
            "u_constr": unconstr_cnstr,
        }

json_object = json.dumps(tmp_dict, indent=4, cls=NumpyEncoder)
with open(f"{workspace}/{dataset_name}_repeat_{args.n_client}_{args.repeat_idx}_unconstr.json", "w") as outfile:
    outfile.write(json_object)

print("experiment setup")
print(f"beta={beta}\tr={r}\trho={rho}\tn={n}")
print(x)
