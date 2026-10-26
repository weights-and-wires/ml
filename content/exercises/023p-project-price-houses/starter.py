"""Project: price houses honestly.

An estate agency wants a price estimate for each listing and the chance that it
sells within 30 days. You build the pipeline stage by stage, in this order:

1. `split_indices`: a seeded train / validation / test split.
2. `fit_scaler` and `apply_scaler`: standardise with training statistics only.
3. `fit_qr`: least squares through QR, not the normal equations.
4. `ridge_path`: ridge weights for every lambda from one SVD.
5. `choose_lambda`: pick lambda on the validation rows.
6. `lambda_max` and `fista_lasso`: the lasso, to see which columns matter.
7. `add_intercept`, `fit_logistic_newton`, and `mean_log_loss`: the classifier.
8. `run_pipeline`: wire the stages together and report honest numbers.

`make_data`, `sigmoid`, `softplus`, `logistic_loss`, `soft_threshold`, `rmse`,
and the two constants are provided.

Shapes used everywhere: X is (n, d), one row per house; price and sold are (n,).
"""

from __future__ import annotations

import numpy as np

FEATURES = [
    "area_m2", "area_ft2", "rooms", "age", "age_gap_sq", "dist_km",
    "noise_1", "noise_2", "noise_3", "noise_4", "noise_5",
]

# Provided: the ridge grid and the lasso strength (a fraction of lambda_max).
LAMBDAS = np.logspace(-2, 4, 61)
LASSO_FRACTION = 0.05


def make_data(seed: int = 0, n: int = 500):
    """Build the synthetic listings (provided, seeded, instant).

    Returns (X, price, sold, names): X is (n, 11), price is in thousands of
    pounds, sold is 0 or 1 (sold within 30 days).

    Planted truth. Price depends on exactly four columns: area_m2,
    age, age_gap_sq (the squared distance of age from 40 years, a U-shaped
    effect), and dist_km. area_ft2 is area_m2 times 10.7639 plus a little noise,
    so the two columns are nearly copies. rooms follows the area and adds
    nothing. The five noise columns do nothing. Whether a house sells quickly
    depends on distance, age, and area through a logistic model.
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


# Provided: you wrote sigmoid in "From a score to a probability".
def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# Provided: you wrote softplus and the loss in "How wrong is a probability? The log loss".
def softplus(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray, ridge: float = 0.0) -> float:
    """Summed log loss plus (ridge / 2) * ||w||^2, for scores z = X @ w."""
    z = X @ w
    return float(np.sum(softplus(z) - y * z)) + 0.5 * ridge * float(w @ w)


# Provided: you wrote soft_threshold in "Lasso and soft thresholding".
def soft_threshold(x: np.ndarray, t: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Root mean squared error of predictions against targets."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def split_indices(n: int, seed: int, frac_val: float = 0.2, frac_test: float = 0.2):
    """Shuffle the row numbers 0..n-1 with a seeded generator and cut three groups.

    Args:
        n: Number of rows.
        seed: Seed for the shuffle. The same seed must give the same split.
        frac_val: Fraction of rows for validation (rounded to a whole count).
        frac_test: Fraction of rows for the test set (rounded the same way).

    Returns:
        (train_idx, val_idx, test_idx): three integer arrays that do not overlap
        and together hold every row number exactly once.

    Steps:
        1. order = np.random.default_rng(seed).permutation(n)
        2. Count the validation and test rows with round().
        3. Cut the order into validation, test, and the remaining training rows.
    """
    raise NotImplementedError


def fit_scaler(X_train: np.ndarray):
    """Learn column means and standard deviations from the training rows only.

    Args:
        X_train: Training matrix, shape (n_train, d).

    Returns:
        (mean, std), each shape (d,). std uses np.std (divide by n). A column
        with std 0 gets std 1, so applying the scaler never divides by zero.

    Steps:
        1. Take the mean and std over axis 0.
        2. Replace any zero std with 1.
    """
    raise NotImplementedError


def apply_scaler(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Return (X - mean) / std, shape (n, d), using the stored statistics.

    The mean and std come from fit_scaler on the training rows. Do not
    recompute them from X, or validation and test rows leak into the scaling.
    """
    raise NotImplementedError


def fit_qr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares weights, shape (d,), through a reduced QR factorisation.

    Must stay accurate when X has two nearly identical columns. Do not form
    X.T @ X, which squares the condition number.

    Steps:
        1. Q, R = np.linalg.qr(X, mode="reduced").
        2. Solve the triangular system R w = Q.T @ y (scipy.linalg.solve_triangular).
    """
    raise NotImplementedError


def ridge_path(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Ridge weights for every lambda from a single SVD of X.

    Args:
        X: Matrix, shape (n, d).
        y: Targets, shape (n,).
        lambdas: Fees, shape (m,), all at least 0.

    Returns:
        Weights, shape (m, d). Row k solves
        minimise ||X w - y||^2 + lambdas[k] * ||w||^2.

    Steps:
        1. U, s, Vt = np.linalg.svd(X, full_matrices=False), computed once.
        2. c = U.T @ y.
        3. factors = s / (s**2 + lam) for each lambda, shape (m, r), by broadcasting.
        4. Row k is (factors[k] * c) @ Vt.
    """
    raise NotImplementedError


def choose_lambda(X_tr, y_tr, X_val, y_val, lambdas):
    """Pick the ridge fee with the lowest validation RMSE.

    Returns:
        (best_lambda, val_errors, w_best): a float taken from lambdas, the
        validation RMSE for every lambda (shape (m,)), and the training-fit
        weights for the best lambda (shape (d,)).

    Steps:
        1. Get all weights with ridge_path on the training rows.
        2. Score each row of weights on the validation rows with rmse.
        3. Take the argmin. Never look at test rows here.
    """
    raise NotImplementedError


def lambda_max(X: np.ndarray, y: np.ndarray) -> float:
    """Smallest lasso fee at which every weight is exactly zero.

    The lasso objective is 0.5 * ||X w - y||^2 + lam * ||w||_1. At w = 0 the
    gradient is -X.T @ y, and zero stays optimal while lam is at least its
    largest absolute entry.
    """
    raise NotImplementedError


def fista_lasso(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 500) -> np.ndarray:
    """Minimise 0.5 * ||X w - y||^2 + lam * ||w||_1 with FISTA. Returns w, shape (d,).

    Start from zeros. Use step = 1 / L, where L is the largest eigenvalue of
    X.T @ X.

    Steps:
        1. Keep w, an extrapolated point z (start z = w), and t = 1.
        2. w_new = soft_threshold(z - step * X.T @ (X @ z - y), step * lam).
        3. t_new = (1 + sqrt(1 + 4 t^2)) / 2.
        4. z = w_new + ((t - 1) / t_new) * (w_new - w), then w, t = w_new, t_new.
        5. Return w after the loop, not z, so the zeros are exact.
    """
    raise NotImplementedError


def add_intercept(X: np.ndarray) -> np.ndarray:
    """Return X with a column of ones in front, shape (n, d + 1)."""
    raise NotImplementedError


def fit_logistic_newton(X, y, ridge: float = 1.0, max_iter: int = 25, tol: float = 1e-8):
    """Fit logistic regression with Newton's method. Returns w, shape (d,).

    Minimises logistic_loss(w, X, y, ridge), a sum over rows. Start at zeros.

    Steps:
        1. p = sigmoid(X @ w); g = X.T @ (p - y) + ridge * w.
        2. Stop when the norm of g is below tol.
        3. H = (X * (p * (1 - p))[:, None]).T @ X + ridge * np.eye(d).
        4. w = w - np.linalg.solve(H, g).
    """
    raise NotImplementedError


def mean_log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Average log loss of probabilities p against 0/1 labels y.

    Clip p into [eps, 1 - eps] first, so a probability of exactly 0 or 1 gives
    a large finite number, not infinity.
    """
    raise NotImplementedError


def run_pipeline(seed: int = 0) -> dict:
    """Run every stage on make_data(seed) and report honest numbers.

    Use split_indices(len(X), seed) for the split. Fit the scaler on the
    training rows, apply it everywhere, and centre price with the training mean
    (the standardised columns then need no intercept). Fit on training rows,
    choose on validation rows, and report on test rows.

    Returns a dict with these keys:
        "rmse_ols":      test RMSE of fit_qr weights.
        "rmse_ridge":    test RMSE of the ridge weights from choose_lambda over LAMBDAS.
        "rmse_lasso":    test RMSE of the lasso weights.
        "lambda":        the ridge lambda chosen on validation rows.
        "w_ols", "w_ridge", "w_lasso": weights on the standardised columns, shape (11,).
                         The lasso uses lam = LASSO_FRACTION * lambda_max on the training rows.
        "log_loss":      test mean_log_loss of the Newton logistic model for `sold`
                         (standardised columns plus an intercept, ridge 1.0).
        "base_log_loss": test mean_log_loss of always predicting the training sell rate.
    """
    raise NotImplementedError


if __name__ == "__main__":
    r = run_pipeline(0)
    print(f"test RMSE (thousands): OLS {r['rmse_ols']:.2f}, ridge {r['rmse_ridge']:.2f}, lasso {r['rmse_lasso']:.2f}")
    print("chosen lambda:", round(r["lambda"], 3))
    print("lasso keeps:", [n for n, w in zip(FEATURES, r["w_lasso"]) if w != 0.0])
    print(f"test log loss {r['log_loss']:.3f} vs base rate {r['base_log_loss']:.3f}")
