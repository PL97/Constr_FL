import numpy as np
from proxAL import proxAL
from cproxAL import cproxAL
from admm import admm
from utils import format_results, generate_data
from copy import deepcopy
from collections import defaultdict
from prettytable import PrettyTable


def format_results(ave_rep):
    """format the output

    Args:
        ave_rep (np.array): output results where idx 0, 3, 5 is the centralized proxAL, whereas 0, 2, 4, 6 is the proxAL
            Notably, the order is obj value, constrain value, outlier iter, and inner iter (if exists)
    """ 
    myTable = PrettyTable(["d", "n", "m", "obj_cpal", "obj_fl", "diff", "constr_cpal", "constr_fl", "outeriter_cpal", "outeriter_fl"])
    tmp_dict = defaultdict(lambda: defaultdict(lambda: {}))
    for k, tmp_dict in ave_rep.items():
        d, n, m = k.split("_")
        objcpal = tmp_dict['objcpal']
        constrcpal = tmp_dict['constrcpal']
        coutiter = tmp_dict['coutiter']
        
        objpal = tmp_dict['objpal']
        constrpal = tmp_dict['constrpal']
        floutiter = tmp_dict['floutiter']
        
        diff = [abs(abs(x - y)/x) for x, y in zip(objcpal, objpal)]
        
        myTable.add_row([f"{d}", f"{n}", f"{m}", \
                            f"{np.mean(objcpal)}±{np.std(objcpal)}", \
                            f"{np.mean(objpal)}±{np.std(objpal)}", \
                            f"{np.mean(diff)}±{np.std(diff)}", \
                            f"{np.mean(constrcpal)}±{np.std(constrcpal)}", \
                            f"{np.mean(constrpal)}±{np.std(constrpal)}", \
                            f"{np.mean(coutiter)}±{np.std(coutiter)}", \
                            f"{np.mean(floutiter)}±{np.std(floutiter)}"])
    
    print(myTable)


# Define the size table

d_n_m = np.array([
    [100, 1, 1], 
    [100, 5, 1],
    [100, 10, 1], 
    [300, 1, 3], 
    [300, 5, 3], 
    [300, 10, 3], 
    [500, 1, 5], 
    [500, 5, 5], 
    [500, 10, 5]
])

num_spl, _ = d_n_m.shape

seed=10
repeat_run = 10

# list_rec = np.zeros(7)
results = {}



for ii in range(num_spl):

    # problem size
    d, n, m = d_n_m[ii]
    n = n+1

    # random seed
    np.random.seed(seed)

    # generate the random matrices A, b, C, d
    A, b, C, dfull = generate_data(d, n, m)
    
    tmp_results = defaultdict(lambda: [])
    for _ in range(repeat_run):
        np.random.seed(_)
        # hyperparameters
        eps1 = 1e-3
        eps2 = 1e-3
        beta = 10
        rhofull = np.ones(n) * 1
        w0 = np.random.rand(d)
        w0 = w0/np.linalg.norm(w0, ord=2)
        w0_init = deepcopy(w0)
        mu0full = np.zeros((m, n))
        bars=0.1

        # centralized proximal AL method
        # Assuming you've implemented the cproxAL function properly
        ret_cprox = cproxAL(w0, mu0full, A, b, C, dfull, beta, eps1, eps2, bars=bars)

        tmp_results['objcpal'].append(ret_cprox["objcpal"])
        tmp_results['constrcpal'].append(ret_cprox["constrcpal"])
        tmp_results['coutiter'].append(ret_cprox["coutiter"])
        
        w0 = w0_init
        # proximal AL based FL algorithm
        # Assuming you've implemented the proxAL function properly
        ret_prox = proxAL(w0, mu0full, A, b, C, dfull, admm, beta, rhofull, eps1, eps2, bars=bars)

        tmp_results['objpal'].append((ret_prox["objpal"]))
        tmp_results['constrpal'].append((ret_prox["constrpal"]))
        tmp_results['floutiter'].append(ret_prox["floutiter"])
        
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
        
        json_object = json.dumps(tmp_dict, indent=4, cls=NumpyEncoder)
        
        with open(f"new_files_plot/{dataset_name}_repeat_{repeat_run}_{args.n_client}.json", "w") as outfile:
            outfile.write(json_object)
    
    results[f"{d}_{n-1}_{m}"] = tmp_results
    format_results(results)
print("\n\n\n")
format_results(results)
