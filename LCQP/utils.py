import numpy as np
from prettytable import PrettyTable
import cProfile, pstats, io
from pstats import SortKey
import re


def generate_data(f_d: int, c_n: int, const_n: int):
    """generate constrained qudratic data

    Args:
        f_d (int): feature dimension
        c_n (int): number of clients
        const_n (int): number of constraints on each clients
        
    Returns:
        generate the random matrices A, b, C, d
    """
    A = np.zeros((f_d, f_d, c_n))
    
    for i in range(c_n):
        eigAi = np.random.rand(f_d) * 5 + 5 ## resaling
        # r = 0.4
        idx = list(range(eigAi.shape[0]))
        np.random.shuffle(idx)
        # eigAi[idx[:int(eigAi.shape[0]*r)]] = 0
        temp = np.random.randn(f_d, f_d)
        U, _, _ = np.linalg.svd(temp)
        A[:, :, i] = U.dot(np.diag(eigAi)).dot(U.T)

    A[:, :, c_n-1] = np.zeros((f_d, f_d))
    b = np.random.randn(f_d, c_n)
    b[:, c_n-1] = np.zeros(f_d)
    C = np.random.randn(const_n, f_d, c_n)
    dfull = np.random.randn(const_n, c_n)
    return A, b, C, dfull


def format_results(ave_rep: np.array):
    """format the output

    Args:
        ave_rep (np.array): output results where idx 0, 3, 5 is the centralized proxAL, whereas 0, 2, 4, 6 is the proxAL
            Notably, the order is obj value, constrain value, outlier iter, and inner iter (if exists)
    """ 
    obj_cproxal = ave_rep[1]
    constrcproxal = ave_rep[3]
    outitercproxal = ave_rep[5]
    wallclkcproxal = ave_rep[7] if len(ave_rep)>8 else -1
    
    obj_proxal = ave_rep[0]
    constrproxal = ave_rep[2]
    outiterproxal = ave_rep[4]
    inneriterproxal = ave_rep[6]
    wallclkproxal = ave_rep[8] if len(ave_rep)>8 else -1
    
    myTable = PrettyTable(["obj", "constr", "outeriter", "inneriter", "wallclock"])
    myTable.add_row([obj_cproxal, constrcproxal, outitercproxal, '-', wallclkcproxal])
    myTable.add_row([obj_proxal, constrproxal, outiterproxal, inneriterproxal, wallclkproxal])
    
    print(myTable)
    
    
def measure_runtime(func, **args):
    pr = cProfile.Profile()
    pr.enable()
    rets = func(**args)
    pr.disable()
    
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    substr = str(re.findall(r'function calls in [0-9]+.[0-9]+', s.getvalue()))
    ts = re.findall(r'[0-9]+.+[0-9]+', str(substr))[0]
    rets['runtime'] = float(ts)
    return rets
    
    
    
    
