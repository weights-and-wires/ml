"""Reference solution for the one-loss-one-gradient exercise."""

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


# --- Written in this chapter. ---


def glm_loss(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    eta = X @ w
    return float(np.sum(log_partition(family, eta) - y * eta))


def glm_gradient(family: str, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    mu = mean_from_natural(family, X @ w)
    return X.T @ (mu - y)


def gradient_descent_glm(
    family: str,
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.01,
    steps: int = 500,
) -> tuple[np.ndarray, list[float]]:
    w = np.zeros(X.shape[1])
    history = [glm_loss(family, w, X, y)]
    for _ in range(steps):
        w = w - lr * glm_gradient(family, w, X, y)
        history.append(glm_loss(family, w, X, y))
    return w, history


if __name__ == "__main__":
    # Three hourly counts at hours 0, 1, 2: the running example.
    X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([1.0, 1.0, 2.0])
    w0 = np.zeros(2)
    print("loss at zero weights:    ", glm_loss("poisson", w0, X, y), "(expected 3)")
    print("gradient at zero weights:", glm_gradient("poisson", w0, X, y), "(expected [-1, -2])")
    w, history = gradient_descent_glm("poisson", X, y, lr=0.05, steps=500)
    print("weights after 500 steps: ", np.round(w, 4))
    print("loss after 500 steps:    ", round(history[-1], 5))
