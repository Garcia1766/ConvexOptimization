import os, sys
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

DATA_DIR = 'random_data'
COST_SIZE_M = 10
COST_SIZE_N = 10

def generate_random_data(size_m: int, size_n: int):
    cost_mat = np.random.randint(1, 11, size=(size_m, size_n))
    mu_vec_ = np.random.rand(size_m)
    nu_vec_ = np.random.rand(size_n)
    mu_vec = np.array([[mu_vec_[i] / mu_vec_.sum()] for i in range(size_m)]).transpose()
    nu_vec = np.array([[nu_vec_[i] / nu_vec_.sum()] for i in range(size_n)]).transpose()
    print(mu_vec.shape)
    pd.DataFrame(data=cost_mat).to_csv(os.path.join(DATA_DIR, 'C.csv'), header=None, index=None)
    pd.DataFrame(data=mu_vec).to_csv(os.path.join(DATA_DIR, 'mu.csv'), header=None, index=None)
    pd.DataFrame(data=nu_vec).to_csv(os.path.join(DATA_DIR, 'nu.csv'), header=None, index=None)

def main():
    generate_random_data(COST_SIZE_M, COST_SIZE_N)

if __name__ == '__main__':
    main()
