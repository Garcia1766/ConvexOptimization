import os, sys
import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB

DATA_DIR = 'random_data'  # 'random_data' or 'test_data'
OPTIMIZE_METHODs = [0, 1, 2]  # 0: primal simplex, 1: dual simplex, 2: barrier, 3: concurrent)
f = open('gurobi_' + DATA_DIR + '.log', 'w')
sys.stdout = f
sys.stderr = f
    
def computeEMD(mtd: int):
    C_df = pd.read_csv(os.path.join(DATA_DIR, 'C.csv'), header=None)
    COST_SIZE = len(C_df)
    mu_df = pd.read_csv(os.path.join(DATA_DIR, 'mu.csv'), header=None)
    nu_df = pd.read_csv(os.path.join(DATA_DIR, 'nu.csv'), header=None)
    cost_dict, mu, nu = dict(), dict(), dict()
    for i in range(COST_SIZE):
        for j in range(COST_SIZE):
            cost_dict[(i, j)] = C_df.iat[i, j]
    for i in range(COST_SIZE):
        mu[i] = mu_df.iat[0, i]
    for j in range(COST_SIZE):
        nu[j] = nu_df.iat[0, j]

    coord, val = gp.multidict(cost_dict)

    model_name = 'gurobi_' + DATA_DIR
    model = gp.Model(model_name)
    model.Params.Method = mtd  # 0: primal simplex, 1: dual simplex, 2: barrier, 3: concurrent
    x = model.addVars(coord, vtype=GRB.CONTINUOUS, name="x")
    row_sum = model.addConstrs( (x.sum(i, '*') == mu[i] for i in range(COST_SIZE)), name='row_sum')
    col_sum = model.addConstrs( (x.sum('*', j) == nu[j] for j in range(COST_SIZE)), name='col_sum')
    model.setObjective(x.prod(val), GRB.MINIMIZE)
    # model.write(model_name + '.lp')
    model.optimize()

def main():
    for mtd in OPTIMIZE_METHODs:
        computeEMD(mtd)
        print('FINISH::' + DATA_DIR + '::' + str(mtd))

if __name__ == '__main__':
    main()
