import numpy as np
import torch
import pandas as pd
import sys
import time

DATA_DIR = 'random_data'
RAND_SEED = 2333
EPS = 1e-15
LEAST_ITER = 1
MAX_ITER = 10000
HYPER_T = 0.1  # sys.argv[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(torch.cuda.is_available(), device)

def A_mul(X: torch.Tensor):
    return X.sum(dim=1), X.sum(dim=0)

def AT_mul(y1: torch.Tensor, y2: torch.Tensor):
    return y1.reshape(-1,1)+y2.reshape(1,-1)

def admm(
    C: torch.Tensor,
    mu: torch.Tensor,
    nu: torch.Tensor,
    n: int,
    eps: float,
    t: float,
    X0: torch.Tensor = None
) -> torch.Tensor:
    assert(C.shape == (n, n))
    assert(mu.shape == (n, ))
    assert(nu.shape == (n, ))

    if X0 is None:
        X = torch.rand((n, n), device=device)
        # X = np.random.rand(n, n)
    else:
        X = X0
    S = torch.rand((n, n), device=device)
    y = np.random.rand(2*n)
    y = torch.from_numpy(y)
    c1, c2 = A_mul(C)

    C = C.to(device)
    c1 = c1.to(device)
    c2 = c2.to(device)
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

        err = torch.norm((X - X_old).reshape(-1,))
        X_old = X
        if itr % 10 == 0:
            print(f'{itr},{err},{float((C*X).sum())}')
    print(f',,{time.time() - t0}')
    return X

def read_data(file_name=DATA_DIR):
    cost_mat = pd.read_csv(f'{file_name}/C.csv', header=None)
    mu_mat = pd.read_csv(f'{file_name}/mu.csv', header=None)
    nu_mat = pd.read_csv(f'{file_name}/nu.csv', header=None)

    m, n = cost_mat.shape[0:2]
    assert(m == n)

    return cost_mat.values, mu_mat.values.reshape(-1,), nu_mat.values.reshape(-1,), n

if __name__ == '__main__':
    f = open('admm_' + DATA_DIR + '_' + str(RAND_SEED) + '_' + str(sys.argv[1]) + '.csv', 'w')
    sys.stdout = f
    sys.stderr = f
    np.random.seed(RAND_SEED)
    torch.random.manual_seed(RAND_SEED)
    C, mu, nu, n = read_data()
    C, mu, nu = torch.from_numpy(C), torch.from_numpy(mu), torch.from_numpy(nu)

    # print(mu.shape, nu.shape)
    X = admm(C, mu, nu, n, EPS, float(sys.argv[1]))
    C = torch.from_numpy(C).to(device)
    a = (C*X).sum()
    print(f',,{float(a)}')
    pass
