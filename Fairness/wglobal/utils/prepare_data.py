import pandas as pd
import numpy as np
import os
from collections import Counter
import math

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="name of the dataset", default="adult")
    args = parser.parse_args()
    return args


def load_uci(num_split=5, seed=10, file_name="adult_data"):
    np.random.seed(seed)
    path = f"/home/jusun/peng0347/Constr_FL/data/{file_name}.csv"
    df = pd.read_csv(path)
    sensitive = df.loc[:, "sex"]


    # features = df.iloc[:, 1:-2].to_numpy()
    # labels = df.iloc[:, -1].to_numpy()
    features, labels = df.drop(['salary', 'Unnamed: 0'], axis=1), df['salary'].to_numpy()
    
    features_name = features.columns.tolist()
    features_name.remove("sex")
    features_name.append("sex")
    features = features[features_name].to_numpy()


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
    return X0, X1, features.shape[1]


if __name__ == "__main__":
    load_uci()





# if __name__ == "__main__":
#     args = parse_args()
#     X0, X1, _ = load_uci(name=args.data)
#     print(_)
#     print(f"imbalance ratio: {X1.shape[1]*X1.shape[2]}/{X0.shape[1]*X0.shape[2]} -- {X1.shape[1]/X0.shape[1]}")
#     for i in range(X0.shape[2]):
#         print(len(X0[i]), len(X1[i]))