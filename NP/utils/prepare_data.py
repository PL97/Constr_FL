import pandas as pd
import numpy as np
import os
from collections import Counter
import math


def load_uci(name="adult", num_split=5, seed=10):
    np.random.seed(seed)
    path = "/home/le/Constr_FL/Cleaned_UCI_Datasets/binary_data/"
    df = pd.read_csv(os.path.join(path, name+".csv")).iloc[:300, :]
    # df = pd.read_csv(os.path.join(path, name+".csv"))

    n_total = df.shape[0]
    features = df.iloc[:, :-1].to_numpy()
    labels = df.iloc[:, -1].to_numpy()

    X0_all, X1_all = features[labels==0], features[labels==1]
    idx0 = list(range(X0_all.shape[0]))
    idx1 = list(range(X1_all.shape[0]))
    np.random.shuffle(idx0)
    np.random.shuffle(idx1)

    X0_split = math.floor(X0_all.shape[0]/num_split)
    X1_split = math.floor(X1_all.shape[0]/num_split)
    

    X0, X1 = [], []
    for i in range(num_split):
        tmp_idx0 = idx0[i*X0_split:(i+1)*X0_split]
        tmp_idx1 = idx1[i*X1_split:(i+1)*X1_split]

        X0.append(X0_all[tmp_idx0])
        X1.append(X1_all[tmp_idx1])
    X0, X1 = np.asarray(X0).transpose(2, 1, 0), np.asarray(X1).transpose(2, 1, 0)
    print(X0_all.shape,  X1_all.shape)
    print(X0.shape, X1.shape)
    return X0, X1, features.shape[1]



    


if __name__ == "__main__":
    X0, X1, _ = load_uci()
    print(_)
    for i in range(X0.shape[2]):
        print(len(X0[i]), len(X1[i]))