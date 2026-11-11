"""Nearest neighbours and matrix factorisation: reference solution."""

import numpy as np


def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return the (n, m) matrix of squared distances between rows of A and rows of B."""
    d = (A**2).sum(axis=1)[:, None] - 2.0 * A @ B.T + (B**2).sum(axis=1)[None, :]
    return np.maximum(d, 0.0)


def _neighbours(Xtr: np.ndarray, Xte: np.ndarray, k: int) -> np.ndarray:
    d = pairwise_sq_dists(Xte, Xtr)
    return np.argsort(d, axis=1, kind="stable")[:, :k]


def knn_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int) -> np.ndarray:
    """Majority vote of the k nearest training rows; ties go to the smallest label."""
    idx = _neighbours(Xtr, Xte, k)
    n_classes = int(ytr.max()) + 1
    out = np.empty(len(Xte), dtype=int)
    for i, row in enumerate(idx):
        out[i] = np.argmax(np.bincount(ytr[row], minlength=n_classes))
    return out


def knn_regress(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int) -> np.ndarray:
    """Mean target of the k nearest training rows."""
    return ytr[_neighbours(Xtr, Xte, k)].mean(axis=1)


def loo_knn_error(X: np.ndarray, y: np.ndarray, k: int) -> float:
    """Leave-one-out misclassification rate of k-NN: each row is predicted from the others."""
    d = pairwise_sq_dists(X, X)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1, kind="stable")[:, :k]
    n_classes = int(y.max()) + 1
    wrong = 0
    for i, row in enumerate(idx):
        if np.argmax(np.bincount(y[row], minlength=n_classes)) != y[i]:
            wrong += 1
    return wrong / len(y)


def distance_ratio(d: int, n: int, rng: np.random.Generator) -> float:
    """Nearest over farthest Euclidean distance from one query to n uniform points."""
    X = rng.uniform(size=(n, d))
    q = rng.uniform(size=(1, d))
    dist = np.sqrt(pairwise_sq_dists(q, X))[0]
    return float(dist.min() / dist.max())


def masked_loss(R, M, U, V, lam) -> float:
    """Squared error on observed entries plus lam times the squared sizes of U and V."""
    resid = (R - U @ V.T) * M
    return float(np.sum(resid**2) + lam * (np.sum(U**2) + np.sum(V**2)))


def als_step(R, M, U, V, lam) -> np.ndarray:
    """Return U with every row replaced by its ridge solve against the observed items."""
    k = U.shape[1]
    new_U = np.empty_like(U)
    for i in range(R.shape[0]):
        obs = M[i] > 0
        Vo = V[obs]
        new_U[i] = np.linalg.solve(Vo.T @ Vo + lam * np.eye(k), Vo.T @ R[i, obs])
    return new_U


def fit_als(R, M, k, lam, iters, rng):
    """Alternate als_step on U and on V; return U, V, and the loss before and after each sweep."""
    U = 0.1 * rng.normal(size=(R.shape[0], k))
    V = 0.1 * rng.normal(size=(R.shape[1], k))
    history = [masked_loss(R, M, U, V, lam)]
    for _ in range(iters):
        U = als_step(R, M, U, V, lam)
        V = als_step(R.T, M.T, V, U, lam)
        history.append(masked_loss(R, M, U, V, lam))
    return U, V, history


def masked_rmse(R, M, P) -> float:
    """Root mean squared error over the entries where M is 1."""
    return float(np.sqrt(np.sum(M * (R - P) ** 2) / np.sum(M)))


def user_bias_baseline(R, M) -> np.ndarray:
    """Global mean plus each user's average offset from it, as an (n, m) table."""
    mu = np.sum(R * M) / np.sum(M)
    counts = M.sum(axis=1)
    offset = np.where(counts > 0, np.sum((R - mu) * M, axis=1) / np.maximum(counts, 1), 0.0)
    return np.broadcast_to(mu + offset[:, None], R.shape).copy()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xtr = np.array([[0, 0], [1, 0], [0, 2], [3, 3], [4, 3]], dtype=float)
    ytr = np.array([0, 0, 1, 1, 1])
    q = np.array([[1.0, 1.0]])
    print("squared distances:", pairwise_sq_dists(q, Xtr))
    print("k=3 vote:", knn_predict(Xtr, ytr, q, 3))
    print("distance ratio at d=2 and d=1000:",
          distance_ratio(2, 200, rng), distance_ratio(1000, 200, rng))
    U0, V0 = rng.normal(size=(60, 2)), rng.normal(size=(40, 2))
    R = U0 @ V0.T + 0.2 * rng.normal(size=(60, 40))
    obs = rng.uniform(size=R.shape) < 0.3
    M = obs.astype(float)
    U, V, hist = fit_als(R, M, 2, 1.0, 20, rng)
    print("loss first and last:", hist[0], hist[-1])
