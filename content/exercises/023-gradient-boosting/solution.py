"""Reference solution for the gradient boosting exercise."""

import numpy as np


def _fit_stump(X: np.ndarray, y: np.ndarray) -> dict:
    """Fit a depth-1 regression tree: the single split that minimises squared error.

    Returns a dict with keys "feature", "threshold", "left_value", and
    "right_value". A point goes left when `X[:, feature] <= threshold`, and each
    side predicts the mean of its targets. "feature" is `None` when no feature
    admits a split, and then both values equal the mean of `y`.
    """
    n, d = X.shape
    best = None
    for j in range(d):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        ys = y[order]
        cumsum = np.cumsum(ys)
        cumsum2 = np.cumsum(ys ** 2)
        total, total2 = cumsum[-1], cumsum2[-1]

        counts = np.arange(1, n)
        left_sum, left_sum2 = cumsum[:-1], cumsum2[:-1]
        right_sum, right_sum2 = total - left_sum, total2 - left_sum2
        right_counts = n - counts
        sse = (left_sum2 - left_sum ** 2 / counts) + (right_sum2 - right_sum ** 2 / right_counts)
        sse = np.where(xs[:-1] != xs[1:], sse, np.inf)

        i = int(np.argmin(sse))
        if not np.isfinite(sse[i]):
            continue
        if best is None or sse[i] < best[0]:
            best = (
                float(sse[i]),
                j,
                float(0.5 * (xs[i] + xs[i + 1])),
                float(left_sum[i] / counts[i]),
                float(right_sum[i] / right_counts[i]),
            )

    if best is None:
        value = float(np.mean(y))
        return {"feature": None, "threshold": None, "left_value": value, "right_value": value}
    _, feature, threshold, left_value, right_value = best
    return {"feature": feature, "threshold": threshold, "left_value": left_value, "right_value": right_value}


def _predict_stump(stump: dict, X: np.ndarray) -> np.ndarray:
    """Predict with a stump returned by `_fit_stump`. Returns shape (n,)."""
    if stump["feature"] is None:
        return np.full(X.shape[0], stump["left_value"])
    goes_left = X[:, stump["feature"]] <= stump["threshold"]
    return np.where(goes_left, stump["left_value"], stump["right_value"])


def squared_error_neg_grad(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    return y - F


def absolute_error_neg_grad(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    return np.sign(y - F)


def gradient_boost(
    X: np.ndarray,
    y: np.ndarray,
    neg_grad,
    n_rounds: int,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, list[dict]]:
    F = np.zeros(X.shape[0])
    history = []
    for _ in range(n_rounds):
        target = neg_grad(y, F)
        stump = _fit_stump(X, target)
        F = F + learning_rate * _predict_stump(stump, X)
        history.append({"target": target, "prediction": F.copy()})
    return F, history


def newton_leaf_value(g: np.ndarray, h: np.ndarray, lam: float) -> float:
    return float(-np.sum(g) / (np.sum(h) + lam))


def split_gain(
    g: np.ndarray,
    h: np.ndarray,
    left: np.ndarray,
    lam: float,
    gamma: float,
) -> float:
    g = np.asarray(g, dtype=float)
    h = np.asarray(h, dtype=float)
    left = np.asarray(left, dtype=bool)

    def score(mask: np.ndarray) -> float:
        return float(np.sum(g[mask]) ** 2 / (np.sum(h[mask]) + lam))

    parent = float(np.sum(g) ** 2 / (np.sum(h) + lam))
    return 0.5 * (score(left) + score(~left) - parent) - float(gamma)


if __name__ == "__main__":
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 3.0, 8.0, 10.0])
    F, history = gradient_boost(X, y, squared_error_neg_grad, n_rounds=3, learning_rate=0.5)
    for m, step in enumerate(history, start=1):
        loss = 0.5 * np.mean((y - step["prediction"]) ** 2)
        print(f"round {m}: target {step['target']}, F {step['prediction']}, loss {loss:.4f}")

    g = -y
    h = np.ones_like(y)
    left = np.array([True, True, False, False])
    print("left Newton leaf, lam 0:", newton_leaf_value(g[left], h[left], 0.0))
    print("left Newton leaf, lam 1:", newton_leaf_value(g[left], h[left], 1.0))
    print("split gain, lam 0, gamma 0:", split_gain(g, h, left, 0.0, 0.0))

    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(200, 1))
    y = np.sin(2 * np.pi * X[:, 0]) + 0.3 * rng.normal(size=200)
    for name, grad in (("squared error", squared_error_neg_grad), ("absolute error", absolute_error_neg_grad)):
        F, history = gradient_boost(X, y, grad, n_rounds=50, learning_rate=0.3)
        print(f"{name}: mean squared miss after 1 round {np.mean((y - history[0]['prediction']) ** 2):.3f}, after 50 rounds {np.mean((y - F) ** 2):.3f}")
