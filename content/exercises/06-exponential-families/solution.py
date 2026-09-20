"""Reference solution for the exponential families and GLM exercise."""

import numpy as np

FAMILIES = ("gaussian", "bernoulli", "poisson")


def _softplus(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def log_partition(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return 0.5 * eta ** 2
    if family == "bernoulli":
        return _softplus(eta)
    if family == "poisson":
        return np.exp(eta)
    raise ValueError(f"unknown family {family!r}")


def mean_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return eta.copy()
    if family == "bernoulli":
        return _sigmoid(eta)
    if family == "poisson":
        return np.exp(eta)
    raise ValueError(f"unknown family {family!r}")


def variance_from_natural(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return np.ones_like(eta)
    if family == "bernoulli":
        p = _sigmoid(eta)
        return p * (1.0 - p)
    if family == "poisson":
        return np.exp(eta)
    raise ValueError(f"unknown family {family!r}")


def glm_gradient(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    eta = X @ w
    mu = mean_from_natural(family, eta)
    return X.T @ (mu - y)


def glm_fisher_information(family: str, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    eta = X @ w
    v = variance_from_natural(family, eta)
    return (X * v[:, None]).T @ X


def fit_fisher_scoring(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> tuple[np.ndarray, list[float]]:
    w = np.zeros(X.shape[1])
    history: list[float] = []
    for _ in range(max_iter):
        eta = X @ w
        history.append(float(np.sum(log_partition(family, eta) - y * eta)))
        g = glm_gradient(family, w, X, y)
        if np.linalg.norm(g) < tol:
            break
        info = glm_fisher_information(family, w, X)
        w = w - np.linalg.solve(info, g)
    return w, history


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 400, 3
    X = np.column_stack([np.ones(n), rng.normal(size=(n, d - 1))])
    w_true = np.array([0.3, 0.5, -0.4])
    y = rng.poisson(np.exp(X @ w_true)).astype(float)

    w_hat, history = fit_fisher_scoring("poisson", X, y, max_iter=30)
    print(f"fisher scoring: {len(history)} iterations")
    print(f"true coefficients:      {w_true}")
    print(f"recovered coefficients: {w_hat}")
    print(f"max absolute error:     {np.abs(w_hat - w_true).max():.3f}")
