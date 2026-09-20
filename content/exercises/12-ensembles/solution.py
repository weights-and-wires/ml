"""Reference solution for the bagging and boosting exercise."""

import numpy as np


def _fit_stump(X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None) -> dict:
    n, d = X.shape
    w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    best = None

    for j in range(d):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        ys = y[order]
        ws = w[order]

        cw = np.cumsum(ws)
        cwy = np.cumsum(ws * ys)
        cwy2 = np.cumsum(ws * ys**2)
        total_w, total_wy, total_wy2 = cw[-1], cwy[-1], cwy2[-1]

        valid = xs[:-1] != xs[1:]
        left_w, left_wy, left_wy2 = cw[:-1], cwy[:-1], cwy2[:-1]
        right_w = total_w - left_w
        right_wy = total_wy - left_wy
        right_wy2 = total_wy2 - left_wy2
        valid &= (left_w > 0) & (right_w > 0)
        if not np.any(valid):
            continue

        left_sse = left_wy2 - left_wy**2 / np.where(left_w > 0, left_w, 1.0)
        right_sse = right_wy2 - right_wy**2 / np.where(right_w > 0, right_w, 1.0)
        total_sse = np.where(valid, left_sse + right_sse, np.inf)

        i = int(np.argmin(total_sse))
        if not np.isfinite(total_sse[i]):
            continue
        candidate = (
            float(total_sse[i]),
            j,
            0.5 * (xs[i] + xs[i + 1]),
            float(left_wy[i] / left_w[i]),
            float(right_wy[i] / right_w[i]),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        value = float(np.average(y, weights=w))
        return {"feature": None, "threshold": None, "left_value": value, "right_value": value}

    _, feature, threshold, left_value, right_value = best
    return {"feature": feature, "threshold": threshold, "left_value": left_value, "right_value": right_value}


def _predict_stump(stump: dict, X: np.ndarray) -> np.ndarray:
    if stump["feature"] is None:
        return np.full(X.shape[0], stump["left_value"])
    mask = X[:, stump["feature"]] <= stump["threshold"]
    return np.where(mask, stump["left_value"], stump["right_value"])


def bootstrap_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, n, size=n)


def bagging_variance(rho: float, sigma2: float, B: int) -> float:
    return rho * sigma2 + (1.0 - rho) * sigma2 / B


def fit_bagged_stumps(
    X: np.ndarray, y: np.ndarray, n_estimators: int, rng: np.random.Generator
) -> tuple[list[dict], list[np.ndarray]]:
    n = X.shape[0]
    stumps = []
    in_bag_indices = []
    for _ in range(n_estimators):
        idx = bootstrap_indices(n, rng)
        stumps.append(_fit_stump(X[idx], y[idx]))
        in_bag_indices.append(idx)
    return stumps, in_bag_indices


def out_of_bag_error(
    X: np.ndarray, y: np.ndarray, stumps: list[dict], in_bag_indices: list[np.ndarray]
) -> float:
    n = X.shape[0]
    total = np.zeros(n)
    count = np.zeros(n)
    for stump, idx in zip(stumps, in_bag_indices):
        in_bag = np.zeros(n, dtype=bool)
        in_bag[idx] = True
        oob = ~in_bag
        if not np.any(oob):
            continue
        total[oob] += _predict_stump(stump, X[oob])
        count[oob] += 1.0

    covered = count > 0
    if not np.any(covered):
        raise ValueError("no training row was ever out-of-bag; use more estimators")
    oob_pred = total[covered] / count[covered]
    return float(np.mean((oob_pred - y[covered]) ** 2))


def gradient_boost(
    X: np.ndarray,
    y: np.ndarray,
    loss_grad,
    n_rounds: int,
    learning_rate: float = 0.1,
) -> tuple[np.ndarray, list[dict]]:
    n = X.shape[0]
    F = np.zeros(n)
    history = []
    for _ in range(n_rounds):
        target = loss_grad(y, F)
        stump = _fit_stump(X, target)
        F = F + learning_rate * _predict_stump(stump, X)
        history.append({"target": target, "prediction": F.copy()})
    return F, history


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    X = rng.normal(size=(n, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.3 * rng.normal(size=n)

    print(f"bagging_variance(rho=0.3, sigma2=1.0, B=50) = {bagging_variance(0.3, 1.0, 50):.4f}")

    stumps, in_bag = fit_bagged_stumps(X, y, n_estimators=60, rng=rng)
    oob = out_of_bag_error(X, y, stumps, in_bag)
    print(f"out-of-bag MSE over {len(stumps)} bagged stumps: {oob:.4f}")

    F, history = gradient_boost(X, y, loss_grad=lambda yy, FF: yy - FF, n_rounds=20, learning_rate=0.3)
    losses = [0.5 * float(np.mean((y - h["prediction"]) ** 2)) for h in history]
    print(f"gradient boosting training loss: round 1 = {losses[0]:.4f}, round 20 = {losses[-1]:.4f}")
