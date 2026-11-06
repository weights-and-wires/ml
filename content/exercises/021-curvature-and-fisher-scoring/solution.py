"""Reference solution for the Fisher scoring exercise."""

import numpy as np

FAMILIES = ("gaussian", "bernoulli", "poisson")


# --- Written in the chapter "One dial, three distributions". Provided, already solved. ---


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


def log_partition(family: str, eta: np.ndarray) -> np.ndarray:
    eta = np.asarray(eta, dtype=float)
    if family == "gaussian":
        return 0.5 * eta ** 2
    if family == "bernoulli":
        return np.maximum(eta, 0.0) + np.log1p(np.exp(-np.abs(eta)))
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


# --- Written in the chapter "One loss and one gradient for every family". Provided, already solved. ---


def glm_loss(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ w
    return float(np.sum(log_partition(family, eta) - y * eta))


def glm_gradient(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu = mean_from_natural(family, X @ w)
    return X.T @ (mu - y)


# --- Write these in this chapter. ---


def glm_fisher_information(family: str, w: np.ndarray, X: np.ndarray) -> np.ndarray:
    var = variance_from_natural(family, X @ w)
    return (X * var[:, None]).T @ X


def newton_step(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    grad = glm_gradient(family, w, X, y)
    info = glm_fisher_information(family, w, X)
    return np.linalg.solve(info, grad)


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
        history.append(glm_loss(family, w, X, y))
        if np.linalg.norm(glm_gradient(family, w, X, y)) < tol:
            break
        w = w - newton_step(family, w, X, y)
    return w, history


if __name__ == "__main__":
    # The three hourly counts at hours 0, 1, 2: the running example.
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 1.0, 2.0])
    w0 = np.zeros(2)
    print("information at zero weights:\n", glm_fisher_information("poisson", w0, X), "\n(expected [[3, 3], [3, 5]])")
    print("first step delta:", newton_step("poisson", w0, X, y), "(expected [1/6, -0.5])")
    w, history = fit_fisher_scoring("poisson", X, y, max_iter=30)
    print("weights:", np.round(w, 5))
    print("losses: ", [round(v, 4) for v in history])
