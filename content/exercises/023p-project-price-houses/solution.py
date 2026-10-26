"""Reference solution: price houses honestly, then predict which ones sell fast."""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular

FEATURES = [
    "area_m2", "area_ft2", "rooms", "age", "age_gap_sq", "dist_km",
    "noise_1", "noise_2", "noise_3", "noise_4", "noise_5",
]


def make_data(seed: int = 0, n: int = 500):
    """Build the synthetic listings.

    Returns (X, price, sold, names): X is (n, 11), price is in thousands of
    pounds, sold is 0/1 (sold within 30 days). Planted truth: price depends on
    area_m2, age, age_gap_sq, and dist_km only. area_ft2 is a noisy copy of
    area_m2, rooms follows the area, and the five noise columns do nothing.
    """
    rng = np.random.default_rng(seed)
    area = rng.uniform(45.0, 220.0, n)
    area_ft2 = 10.7639 * area + rng.normal(0.0, 2.0, n)
    rooms = np.clip(np.round(area / 35.0 + rng.normal(0.0, 0.7, n)), 1, 9)
    age = rng.uniform(0.0, 80.0, n)
    dist = rng.uniform(0.5, 20.0, n)
    noise = rng.normal(0.0, 1.0, (n, 5))
    X = np.column_stack([area, area_ft2, rooms, age, (age - 40.0) ** 2, dist, noise])
    price = (
        60.0 + 2.2 * area - 1.0 * age + 0.04 * (age - 40.0) ** 2 - 6.0 * dist
        + rng.normal(0.0, 25.0, n)
    )
    z = lambda v: (v - v.mean()) / v.std()
    logit = 0.2 - 1.4 * z(dist) - 0.9 * z(age) - 0.7 * z(area)
    sold = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return X, price, sold, list(FEATURES)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def softplus(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    z = X @ w
    return float(np.sum(softplus(z) - y * z)) + 0.5 * ridge * float(w @ w)


def split_indices(n: int, seed: int, frac_val: float = 0.2, frac_test: float = 0.2):
    order = np.random.default_rng(seed).permutation(n)
    n_val = int(round(frac_val * n))
    n_test = int(round(frac_test * n))
    return order[n_val + n_test:], order[:n_val], order[n_val:n_val + n_test]


def fit_scaler(X_train: np.ndarray):
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    return mean, np.where(std > 0, std, 1.0)


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def fit_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(X, mode="reduced")
    return solve_triangular(r, q.T @ y, lower=False)


def ridge_path(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    c = u.T @ y
    factors = s[None, :] / (s[None, :] ** 2 + np.asarray(lambdas, float)[:, None])
    return (factors * c[None, :]) @ vt


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def choose_lambda(X_tr, y_tr, X_val, y_val, lambdas):
    W = ridge_path(X_tr, y_tr, lambdas)
    errors = np.array([rmse(X_val @ w, y_val) for w in W])
    best = int(np.argmin(errors))
    return float(lambdas[best]), errors, W[best]


def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def lambda_max(X: np.ndarray, y: np.ndarray) -> float:
    return float(np.max(np.abs(X.T @ y)))


def fista_lasso(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 500) -> np.ndarray:
    step = 1.0 / np.linalg.eigvalsh(X.T @ X).max()
    w = np.zeros(X.shape[1])
    z, t = w.copy(), 1.0
    for _ in range(iters):
        w_new = soft_threshold(z - step * (X.T @ (X @ z - y)), step * lam)
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
        z = w_new + ((t - 1.0) / t_new) * (w_new - w)
        w, t = w_new, t_new
    return w


def add_intercept(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def fit_logistic_newton(X, y, ridge: float = 1.0, max_iter: int = 25, tol: float = 1e-8):
    w = np.zeros(X.shape[1])
    for _ in range(max_iter):
        p = sigmoid(X @ w)
        g = X.T @ (p - y) + ridge * w
        if np.linalg.norm(g) < tol:
            break
        H = (X * (p * (1.0 - p))[:, None]).T @ X + ridge * np.eye(X.shape[1])
        w = w - np.linalg.solve(H, g)
    return w


def mean_log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


LAMBDAS = np.logspace(-2, 4, 61)
LASSO_FRACTION = 0.05


def run_pipeline(seed: int = 0) -> dict:
    X, price, sold, names = make_data(seed)
    tr, va, te = split_indices(len(X), seed)
    mean, std = fit_scaler(X[tr])
    Z = apply_scaler(X, mean, std)
    y_mean = price[tr].mean()
    yc = price - y_mean

    w_ols = fit_qr(Z[tr], yc[tr])
    lam, val_errors, w_ridge = choose_lambda(Z[tr], yc[tr], Z[va], yc[va], LAMBDAS)
    lam_l = LASSO_FRACTION * lambda_max(Z[tr], yc[tr])
    w_lasso = fista_lasso(Z[tr], yc[tr], lam_l)

    A = add_intercept(Z)
    w_log = fit_logistic_newton(A[tr], sold[tr])
    p_test = sigmoid(A[te] @ w_log)
    base = np.full(len(te), sold[tr].mean())
    return {
        "names": names,
        "rmse_ols": rmse(Z[te] @ w_ols, yc[te]),
        "rmse_ridge": rmse(Z[te] @ w_ridge, yc[te]),
        "rmse_lasso": rmse(Z[te] @ w_lasso, yc[te]),
        "lambda": lam,
        "w_ols": w_ols,
        "w_ridge": w_ridge,
        "w_lasso": w_lasso,
        "log_loss": mean_log_loss(p_test, sold[te]),
        "base_log_loss": mean_log_loss(base, sold[te]),
        "val_errors": val_errors,
        "cond": float(np.linalg.cond(Z[tr])),
    }


if __name__ == "__main__":
    r = run_pipeline(0)
    print(f"test RMSE (thousands): OLS {r['rmse_ols']:.2f}, ridge {r['rmse_ridge']:.2f}, lasso {r['rmse_lasso']:.2f}")
    print("chosen lambda:", round(r["lambda"], 3))
    print("lasso keeps:", [n for n, w in zip(r["names"], r["w_lasso"]) if w != 0.0])
    print(f"test log loss {r['log_loss']:.3f} vs base rate {r['base_log_loss']:.3f}")
