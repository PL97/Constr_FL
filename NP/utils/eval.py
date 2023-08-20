import numpy as np
from admm import sigmoid

def model_eval(w, X0, X1):
    ret = []
    for i in range(X1.shape[-1]):
        TN = np.sum(sigmoid(w.T @ X0[:, :, i]) < 0.5)
        TP = np.sum(sigmoid(w.T @ X1[:, :, i]) > 0.5)
        ret.append({"Precall": TP/X1.shape[1]})
        ret.append({"Nrecall": TN/X0.shape[1]})
    print(ret)
    return ret