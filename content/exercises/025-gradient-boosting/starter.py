"""Gradient boosting: fit what is still wrong.

Start from a model that predicts 0 everywhere. Each round, fit a stump to what
the current model still gets wrong, and add a fraction of it. What is wrong is
the negative gradient of the loss with respect to the predictions. You write
five functions, in this order:

1. `squared_error_neg_grad`: the negative gradient of 0.5 * (y - F) ** 2.
2. `absolute_error_neg_grad`: the negative gradient of |y - F|.
3. `gradient_boost`: the loop that adds one stump per round.
4. `newton_leaf_value`: the Newton leaf, minus the total slope over the total bend plus lam.
5. `split_gain`: the regularised gain of sending some rows left and the rest right.

`_fit_stump` and `_predict_stump` are already written.
"""

import numpy as np


# --- Provided helpers: a depth-1 regression tree. ---


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


# --- Write these in this chapter. ---


def squared_error_neg_grad(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Return the negative gradient of 0.5 * (y - F) ** 2 with respect to F.

    Args:
        y: Targets, shape (n,).
        F: Current predictions, shape (n,).

    Returns:
        An array of shape (n,), one entry per row.
    """
    raise NotImplementedError("squared_error_neg_grad")


def absolute_error_neg_grad(y: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Return the negative gradient of |y - F| with respect to F.

    The loss has a corner at y == F. Return 0 there.

    Args:
        y: Targets, shape (n,).
        F: Current predictions, shape (n,).

    Returns:
        An array of shape (n,), one entry per row.
    """
    raise NotImplementedError("absolute_error_neg_grad")


def gradient_boost(
    X: np.ndarray,
    y: np.ndarray,
    neg_grad,
    n_rounds: int,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, list[dict]]:
    """Boost stumps: each round fits the negative gradient and adds a fraction of it.

    Start from `F = np.zeros(n)`. In each round, compute
    `target = neg_grad(y, F)`, fit `_fit_stump(X, target)`, and update
    `F = F + learning_rate * _predict_stump(stump, X)`.

    Args:
        X: Features, shape (n, d).
        y: Targets, shape (n,).
        neg_grad: A function `neg_grad(y, F)` that returns the negative
            gradient of the loss with respect to F, shape (n,).
        n_rounds: Number of rounds.
        learning_rate: The fraction of each stump that is added.

    Returns:
        A tuple `(F, history)`. `F` has shape (n,). `history` has `n_rounds`
        dicts, where `history[m]["target"]` is the target that round m fit and
        `history[m]["prediction"]` is a copy of F after round m.
    """
    raise NotImplementedError("gradient_boost")


def newton_leaf_value(g: np.ndarray, h: np.ndarray, lam: float) -> float:
    """Return the Newton leaf value for one group of rows.

    `g` is the slope of the loss in the prediction, shape (n,), and it points
    uphill. For squared error that slope is `F - y`, not the residual.
    `h` is the bend, the second derivative, shape (n,). `lam` is the penalty
    on a large leaf, and it belongs in the denominator.

    The value is `-sum(g) / (sum(h) + lam)`.

    Args:
        g: Slopes, shape (n,).
        h: Bends, shape (n,).
        lam: Penalty on the leaf value. Use 0 for an unpenalised leaf.

    Returns:
        One float, the number added to every prediction in the leaf.
    """
    raise NotImplementedError("newton_leaf_value")


def split_gain(
    g: np.ndarray,
    h: np.ndarray,
    left: np.ndarray,
    lam: float,
    gamma: float,
) -> float:
    """Return the regularised gain of a split.

    `left` is a boolean mask of shape (n,): True sends the row left, False
    sends it right. Sum the slopes `g` and the bends `h` on each side. A
    group's score is the square of its total slope, divided by its total bend
    plus `lam`. The gain is half of (left score + right score - parent score),
    then minus `gamma`.

    Args:
        g: Slopes, shape (n,). Same convention as `newton_leaf_value`.
        h: Bends, shape (n,).
        left: Boolean mask, shape (n,).
        lam: Penalty in every denominator.
        gamma: Cost subtracted once for the extra leaf.

    Returns:
        One float. A positive gain is a split worth making.
    """
    raise NotImplementedError("split_gain")


if __name__ == "__main__":
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 3.0, 8.0, 10.0])
    F, history = gradient_boost(X, y, squared_error_neg_grad, n_rounds=3, learning_rate=0.5)
    for m, step in enumerate(history, start=1):
        loss = 0.5 * np.mean((y - step["prediction"]) ** 2)
        print(f"round {m}: target {step['target']}, F {step['prediction']}, loss {loss:.4f}")

    # Slopes of the squared loss at F = 0 are -y. Bends are 1.
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
