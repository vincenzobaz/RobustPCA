import numpy as np # Array manipulation


def l1_norm(mat):
    return np.sum(np.abs(mat))


def frob_norm(mat):
    return np.sqrt(np.sum(np.power(mat, 2)))


def shrinkage(tau, X):
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def thresholding(tau, X):
    u, s, vh = np.linalg.svd(X, full_matrices=False, compute_uv=True)
    s = shrinkage(tau, s)
    return np.multiply(u, s) @ vh


def robust_pca(M, max_iters, delta=10**(-7)):
    h, w = M.shape
    S = np.zeros(M.shape)
    Y = np.zeros(M.shape)
    L = None
    m_norm = frob_norm(M)
    mu = h * w / (4 * l1_norm(M))
    muinv = 1 / mu
    lambda_ = 1 / np.sqrt(max(h, w))

    converged = False
    n_iter = 0
    while n_iter < max_iters and not converged:
        muinvY = muinv * Y
        L = thresholding(muinv, M - S + muinvY)
        S = shrinkage(lambda_ * muinv, M - L + muinvY)
        Y = Y + mu * (M - L - S)

        converged = frob_norm(M - L - S) < delta * m_norm
        n_iter += 1

    print(f'Completed with {n_iter} SVDs')
    return L, S, n_iter

