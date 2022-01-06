import numpy as np
import torch
import pandas as pd
import sys, os
import time
import pickle

DATASET_DIR = 'GRFrough'
DATA_SIZE = 32
DATA_NUMs = [i for i in range(1001, 1011)]

RAND_SEED = 2333
EPS = 1e-15
LEAST_ITER = 1
MAX_ITER = 1000000
HYPER_T = 5e-6  # sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_filenames = [os.path.join('Data', DATASET_DIR, 'data'+str(DATA_SIZE)+'_'+str(DATA_NUM)+'.csv') for DATA_NUM in DATA_NUMs]

def A_mul(X: torch.Tensor):
    return X.sum(dim=1), X.sum(dim=0)

def AT_mul(y1: np.ndarray, y2: np.ndarray):
    return y1.reshape(-1,1)+y2.reshape(1,-1)

def admm(
    C: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    n: int,
    eps: float,
    t: float,
    X0: torch.Tensor = None
) -> (np.ndarray, int, float):
    assert(C.shape == (n, n))
    assert(mu.shape == (n, ))
    assert(nu.shape == (n, ))

    if X0 is None:
        X = np.random.rand(n, n)
    else:
        X = X0

    S = np.random.rand(n, n)
    y = np.random.rand(2*n)
    X, S, y = torch.from_numpy(X), torch.from_numpy(S), torch.from_numpy(y)
    c1, c2 = A_mul(C)

    C = C.to(device)
    c1 = c1.to(device)
    c2 = c2.to(device)
    S = S.to(device)
    X = X.to(device)
    mu = mu.to(device)
    nu = nu.to(device)

    X_old = X
    err = 1
    itr = 0
    print('itr,err,ans')
    t0 = time.time()
    while itr < MAX_ITER:
        itr += 1
        # renew y
        x1, x2 = A_mul(X)
        s1, s2 = A_mul(S)
        y1  = (x1  - mu)/t + s1 - c1
        y2  = (x2  - nu)/t + s2 - c2
        # y = (A_mul(X) - b)/t + A_mul(S-C)
        y11 = 4*n*y1 - 3*y1.sum() + y2.sum()
        y22 = 4*n*y2 - 3*y2.sum() + y1.sum()
        ATy = AT_mul(y11, y22) / (-4*n*n)

        # renew s
        S = C - ATy - X/t
        S[S < 0] = 0

        # renew X
        X = X + t*(ATy + S - C)

        # err = np.linalg.norm((X - X_old).reshape(-1,))
        err = torch.norm((X - X_old).reshape(-1,))
        X_old = X
        if itr % 10 == 0:
            print(f'{itr},{err},{float((C*X).sum())}')
    print(f',,{time.time() - t0}')
    return X, itr, time.time() - t0

def read_data(num1: int, num2: int):
    cost_mat = np.load('cost_mat_size_' + str(DATA_SIZE) + '.npy', allow_pickle=True)
    image1 = pd.read_csv(data_filenames[num1], header=None)
    image2 = pd.read_csv(data_filenames[num2], header=None)
    mu_vec, nu_vec, mu, nu = np.zeros(DATA_SIZE**2), np.zeros(DATA_SIZE**2), np.zeros(DATA_SIZE**2), np.zeros(DATA_SIZE**2)
    for i1 in range(DATA_SIZE):
        for i2 in range(DATA_SIZE):
            mu_vec[i1*DATA_SIZE+i2] = image1.iat[i1, i2]
    for j1 in range(DATA_SIZE):
        for j2 in range(DATA_SIZE):
            nu_vec[j1*DATA_SIZE+j2] = image2.iat[j1, j2]
    mu_vec_sum, nu_vec_sum = mu_vec.sum(), nu_vec.sum()
    for i in range(DATA_SIZE**2):
        mu[i] = mu_vec[i] / mu_vec_sum
        nu[i] = nu_vec[i] / nu_vec_sum
    return cost_mat, mu, nu, DATA_SIZE**2

if __name__ == '__main__':
    f = open('admm_' + DATASET_DIR + '_' + str(RAND_SEED) + '_' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '.csv', 'w')
    sys.stdout = f
    sys.stderr = f

    np.random.seed(RAND_SEED)
    torch.random.manual_seed(RAND_SEED)
    C, mu, nu, n = read_data(int(sys.argv[2]), int(sys.argv[3]))
    X, itr, cpu_time = admm(torch.from_numpy(C), torch.from_numpy(mu), torch.from_numpy(nu), n, EPS, float(sys.argv[1]))
    C = torch.from_numpy(C).to(device)
    a = (C*X).sum()
    pass
