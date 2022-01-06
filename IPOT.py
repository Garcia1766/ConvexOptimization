import numpy as np
import pandas as pd

def ipot(
    C: np.ndarray,
    mu: np.ndarray,
    nu: np.ndarray,
    n: int,
    eps: float = 1e-6,
) -> np.ndarray:
    assert(C.shape == (n, n))
    assert(mu.shape == (n, ))
    assert(nu.shape == (n, ))

    X = np.ones((n,n))
    G = np.exp(-C)
    a = np.ones(n)/n
    b = np.ones(n)/n
    L = 1

    err = 1
    X_old = X
    itr = 0
    while itr < 20000:
        itr += 1
        Q = G*X
        for _ in range(L):
            a = mu / np.matmul(Q,b)
            b = nu / np.matmul(Q.T,a)
        X = (Q*b).T*a

        err = np.linalg.norm(X - X_old)
        X_old = X
        if itr % 10 == 0:
            a = np.sum(C*X)
            print(f'itr: {itr}, err: {err}, ans: {a}')
    return X

def read_data(file_name='test_data'):
    cost_mat = pd.read_csv(f'{file_name}/C.csv', header=None)
    mu_mat = pd.read_csv(f'{file_name}/mu.csv', header=None)
    nu_mat = pd.read_csv(f'{file_name}/nu.csv', header=None)

    m, n = cost_mat.shape[0:2]
    assert(m == n)

    return cost_mat.values, mu_mat.values.reshape(-1,), nu_mat.values.reshape(-1,), n

if __name__ == '__main__':
    np.random.seed(233)
    C, mu, nu, n = read_data()
    print(mu.shape, nu.shape)
    X = ipot(C, mu, nu, n)
    pass