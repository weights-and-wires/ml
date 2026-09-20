"""Reference solution for the generalisation and model selection exercise."""

import numpy as np


def k_fold_indices(n: int, k: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = rng.permutation(n)
    return [np.sort(chunk) for chunk in np.array_split(order, k)]


def cross_val_score(fit, predict, X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    n = X.shape[0]
    rng = np.random.default_rng(0)
    folds = k_fold_indices(n, k, rng)
    all_idx = np.arange(n)
    scores = np.zeros(k)
    for j, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=True)
        model = fit(X[train_idx], y[train_idx])
        pred = predict(model, X[test_idx])
        scores[j] = float(np.mean((y[test_idx] - pred) ** 2))
    return scores


def loo_linear_smoother(X: np.ndarray, y: np.ndarray, lam: float) -> float:
    d = X.shape[1]
    gram_inv = np.linalg.inv(X.T @ X + lam * np.eye(d))
    hat = X @ gram_inv @ X.T
    fitted = hat @ y
    residual = y - fitted
    leverage = np.diag(hat)
    loo_residual = residual / (1.0 - leverage)
    return float(np.mean(loo_residual ** 2))


def learning_curve(
    fit,
    predict,
    X: np.ndarray,
    y: np.ndarray,
    train_sizes,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = int(round(val_fraction * n))
    val_idx, pool_idx = order[:n_val], order[n_val:]

    train_errors = np.zeros(len(train_sizes))
    val_errors = np.zeros(len(train_sizes))
    for i, size in enumerate(train_sizes):
        train_idx = pool_idx[:size]
        model = fit(X[train_idx], y[train_idx])
        train_pred = predict(model, X[train_idx])
        val_pred = predict(model, X[val_idx])
        train_errors[i] = float(np.mean((y[train_idx] - train_pred) ** 2))
        val_errors[i] = float(np.mean((y[val_idx] - val_pred) ** 2))
    return train_errors, val_errors


def empirical_rademacher(
    losses_matrix: np.ndarray, rng: np.random.Generator, num_draws: int = 2000
) -> float:
    num_hypotheses, n = losses_matrix.shape
    signs = rng.choice(np.array([-1.0, 1.0]), size=(num_draws, n))
    inner_products = losses_matrix @ signs.T
    suprema = inner_products.max(axis=0)
    return float(np.mean(suprema) / n)


if __name__ == "__main__":
    def ridge_fit(X, y, lam):
        d = X.shape[1]
        return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)

    def ridge_predict(w, X):
        return X @ w

    rng = np.random.default_rng(0)
    n, d = 60, 30
    Z = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, d))
    X = Z @ loadings + 0.5 * rng.normal(size=(n, d))
    w_true = 0.3 * rng.normal(size=d)
    y = X @ w_true + rng.normal(size=n)

    lambdas = np.logspace(-2, 3, 15)
    cv_curve = [
        cross_val_score(lambda Xt, yt, l=l: ridge_fit(Xt, yt, l), ridge_predict, X, y, k=5).mean()
        for l in lambdas
    ]
    best = lambdas[int(np.argmin(cv_curve))]
    print(f"cross-validated ridge penalty: {best:.3g}")

    loo_mse = [loo_linear_smoother(X, y, l) for l in lambdas]
    best_loo = lambdas[int(np.argmin(loo_mse))]
    print(f"leave-one-out ridge penalty:   {best_loo:.3g}")
