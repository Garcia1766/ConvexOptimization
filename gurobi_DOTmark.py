import os
import time
from math import *
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import sys
import pickle

DATASET_DIR = 'GRFrough'
DATA_SIZE = 32
DATA_NUMs = [i for i in range(1001, 1011)]
OPTIMIZE_METHODs = [0, 1, 2]  # 0: primal simplex, 1: dual simplex, 2: barrier, 3: concurrent

data_filenames = [os.path.join('Data', DATASET_DIR, 'data'+str(DATA_SIZE)+'_'+str(DATA_NUM)+'.csv') for DATA_NUM in DATA_NUMs]
f = open('gurobi_' + DATASET_DIR + '_' + str(DATA_SIZE) + '.log', 'w')
sys.stdout = f
sys.stderr = f

def init_cost_mat(size: int):
    cost_mat = np.zeros((size**2, size**2))
    for i1 in range(size):
        for i2 in range(size):
            for j1 in range(size):
                for j2 in range(size):
                    cost_mat[i1*size+i2][j1*size+j2] = sqrt((i1 - j1)**2 + (i2 - j2)**2)
    np.save('cost_mat_size_' + str(size) + '.npy', cost_mat)

def init_cost_dict(size: int):
    cost_mat = np.load('cost_mat_size_' + str(size) + '.npy', allow_pickle=True)
    cost_dict = dict()
    for i in range(cost_mat.shape[0]):
        for j in range(cost_mat.shape[1]):
            cost_dict[(i, j)] = cost_mat[i][j]
    with open('cost_dict_size_' + str(size) + '.pkl', 'wb') as fp:
        pickle.dump(cost_dict, fp)

with open('cost_dict_size_' + str(DATA_SIZE) + '.pkl', 'rb') as fp:
    cost_dict = pickle.load(fp)

def compute1EMD(mtd:int, num1: int, num2: int):
    image1 = pd.read_csv(data_filenames[num1], header=None)
    image2 = pd.read_csv(data_filenames[num2], header=None)
    mu_vec, nu_vec = np.zeros(DATA_SIZE**2), np.zeros(DATA_SIZE**2)
    for i1 in range(DATA_SIZE):
        for i2 in range(DATA_SIZE):
            mu_vec[i1*DATA_SIZE+i2] = image1.iat[i1, i2]
    for j1 in range(DATA_SIZE):
        for j2 in range(DATA_SIZE):
            nu_vec[j1*DATA_SIZE+j2] = image2.iat[j1, j2]
    mu_vec_sum, nu_vec_sum = mu_vec.sum(), nu_vec.sum()
    mu, nu = dict(), dict()
    for i in range(DATA_SIZE**2):
        mu[i] = mu_vec[i] / mu_vec_sum  # normalize mu
    for j in range(DATA_SIZE**2):
        nu[j] = nu_vec[j] / nu_vec_sum  # normalize nu

    coord, val = gp.multidict(cost_dict)

    model_name = DATASET_DIR + '_' + str(DATA_SIZE) + '_' + str(num1) + '_' + str(num2) + '_' + str(mtd)
    model = gp.Model(model_name)
    model.Params.Method = mtd  # 0: primal simplex, 1: dual simplex, 2: barrier, 3: concurrent
    x = model.addVars(coord, vtype=GRB.CONTINUOUS, name="x")
    row_sum = model.addConstrs( (x.sum(i, '*') == mu[i] for i in range(DATA_SIZE**2)), name='row_sum')
    col_sum = model.addConstrs( (x.sum('*', j) == nu[j] for j in range(DATA_SIZE**2)), name='col_sum')
    model.setObjective(x.prod(val), GRB.MINIMIZE)
    # model.write(model_name + '.lp')
    model.optimize()

def main():
    for mtd in OPTIMIZE_METHODs:
        for i in range(0, 9):
            for j in range(i + 1, 10):
                compute1EMD(mtd, i, j)
                print('FINISH::' + DATASET_DIR + '::' + str(DATA_SIZE) + '::' + str(i) + '::' + str(j) + '::' + str(mtd))

if __name__ == '__main__':
    main()
