# -*- coding: utf-8 -*-
import numpy as np
import random
from scipy.optimize import minimize
import sys
from copy import deepcopy
sys.path.append("../")
from utils.prepare_data import load_uci
from solve_uc import solve_uc
from proxAL import federa_proxAL, central_proxAL
from admm import constr_stat, obj_val, cnstr_val
from collections import defaultdict
import json
import argparse


from prettytable import PrettyTable


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


n = 5
repeat_run = 1

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--n_client', type=int, default=-1)
args = parser.parse_args()
dataset_name = args.dataset

# for dataset_name in ["breast-cancer-wisc", "monks-1", "adult"]:
# for dataset_name in ["breast-cancer-wisc"]:
x = PrettyTable()
x.field_names = ["n", "obj_ours", "obj_cen", "obj_uncnstr", "feas(ours) mean", "feas(ours) max", "feas(cen) mean", "feas(cen) max", "feas(uncnstr) mean", "feas(uncnstr) max"]
tmp_dict = defaultdict(lambda: defaultdict(lambda: {}))
n_clients = [20, 10, 5, 1] if args.n_client == -1 else [args.n_client]
for n in n_clients:
    X0, X1, d = load_uci(name=dataset_name, num_split=n)
    for _ in range(repeat_run):
        np.random.seed(_)

        r = np.ones(n) * 0.2
        beta = 300
        rho = np.ones(n) * 0.01

        eps1 = 1e-3
        eps2 = 1e-3
        bars = 1e-3
        mu0 = np.zeros(n)
        w0 = np.random.rand(d)
        w0 = w0/np.linalg.norm(w0, ord=2)
        w0_init = deepcopy(w0)
        
        
        uncontr_w = solve_uc(w0, X0, X1, eps1)
        unconstr_obj = obj_val(uncontr_w, X0)
        unconstr_cnstr = cnstr_val(uncontr_w, X1)
        print("unconstrs finished!")
        

        # constrained stationary
        # print(constr_stat(wk,muk))
        mu0 = np.zeros(n)
        w0 = w0_init
        # wk_c, muk_c = central_proxAL(w0,mu0)
        central_w, central_mu = central_proxAL(X0=X0, X1=X1, w0=w0, mu0=mu0, beta=beta, r=r, eps1=eps1, eps2=eps2, bars=bars)
        central_obj = obj_val(central_w, X0)
        central_cnstr = cnstr_val(central_w, X1)
        print("central_proxAL finished!")

        
        # wk, muk = federa_proxAL(w0,mu0)
        federa_w, federa_mu, objs, constrs = federa_proxAL(X0, X1, w0=w0, mu0=mu0, eps1=eps1, eps2=eps2, beta=beta, rho=rho, r=r, bars=bars)
        federa_obj = obj_val(federa_w, X0)
        federa_cnstr = cnstr_val(federa_w, X1)
        print("federa_proxAL finished!")

        tmp_dict[n][_] = {
            "c_w": central_w,
            "c_obj": central_obj,
            "c_constr": central_cnstr,
            "u_w": uncontr_w,
            "u_obj": unconstr_obj,
            "u_constr": unconstr_cnstr,
            "f_w": federa_w,
            "f_obj": federa_obj,
            "f_constr": federa_cnstr
        }
        np.save(f"new_files_plot/obj_{dataset_name}_{n}_{_}.npy", objs)
        np.save(f"new_files_plot/constr_{dataset_name}_{n}_{_}.npy", constrs)
        
        
    json_object = json.dumps(tmp_dict, indent=4, cls=NumpyEncoder)
    
    with open(f"new_files_plot/{dataset_name}_repeat_{repeat_run}_{args.n_client}.json", "w") as outfile:
        outfile.write(json_object)
        
    ## aggregate results and print
    tmp_federa_obj, tmp_central_obj, tmp_unconstr_obj, tmp_central_cnstr, tmp_federa_cnstr, tmp_unconstr_cnstr = [], [], [], [], [], []
    for k, v in tmp_dict[n].items():
        tmp_central_obj.extend(v['c_obj'])
        tmp_federa_obj.extend(v['f_obj'])
        tmp_unconstr_obj.extend(v['u_obj'])
        
        tmp_central_cnstr.extend(v['c_constr'])
        tmp_federa_cnstr.extend(v['f_constr'])
        tmp_unconstr_cnstr.extend(v['u_constr'])
    # x.field_names = ["n", "obj_ours", "obj_cen", "obj_uncnstr", "feas(ours) mean", "feas(ours) max", "feas(cen) mean", "feas(cen) max", "feas(uncnstr) mean", "feas(uncnstr) max"]
    x.add_row([n, "{}±{}".format(np.mean(tmp_federa_obj), np.std(tmp_federa_obj)), \
                    "{}±{}".format(np.mean(tmp_central_obj), np.std(tmp_central_obj)), \
                    "{}±{}".format(np.mean(tmp_unconstr_obj), np.std(tmp_unconstr_obj)), \
                    "{}±{}".format(np.mean(tmp_federa_cnstr), np.std(tmp_federa_cnstr)), \
                    "{}".format(np.amax(tmp_federa_cnstr)), \
                    "{}±{}".format(np.mean(tmp_central_cnstr), np.std(tmp_central_cnstr)), \
                    "{}".format(np.amax(tmp_central_cnstr)), \
                    "{}±{}".format(np.mean(tmp_unconstr_cnstr), np.std(tmp_unconstr_cnstr)), \
                    "{}".format(np.amax(tmp_unconstr_cnstr))])
print(f"==================={dataset_name}=========================")
print(x)