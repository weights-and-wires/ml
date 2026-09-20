"""Checks for the ridge and lasso exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


def correlated_design(n: int = 100, d: int = 40, seed: int = 0):
    """A design matrix with correlated columns, and a sparse true w."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 5))
    loadings = rng.normal(size=(5, d))
    X = latent @ loadings + 0.5 * rng.normal(size=(n, d))
    w_true = np.zeros(d)
    w_true[:5] = rng.normal(size=5) * 2.0
    y = X @ w_true + 0.1 * rng.normal(size=n)
    return X, y, w_true


@check("soft_threshold matches the elementwise formula on hand-picked values")
def test_soft_threshold_basic(sub):
    got = has(sub, "soft_threshold")
    x = np.array([3.0, -3.0, 0.5, -0.5, 0.0, 2.0])
    thresh = 1.0
    want = np.array([2.0, -2.0, 0.0, 0.0, 0.0, 1.0])
    close(got(x, thresh), want, atol=1e-12, label="soft_threshold(x, 1.0)")


@check("ridge_closed_form agrees with an independent augmented least squares solve")
def test_ridge_closed_form(sub):
    got = has(sub, "ridge_closed_form")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 6))
    y = rng.normal(size=30)
    lam = 4.0

    w = got(X, y, lam)
    shape(w, (6,), label="ridge_closed_form output")

    # Ridge is least squares on the design stacked with sqrt(lam) * I and the
    # targets padded with zeros: an independent route to the same answer.
    stacked_X = np.vstack([X, np.sqrt(lam) * np.eye(6)])
    stacked_y = np.concatenate([y, np.zeros(6)])
    reference = np.linalg.lstsq(stacked_X, stacked_y, rcond=None)[0]
    close(w, reference, atol=1e-8, label="against the augmented least squares solve")


@check("effective_degrees_of_freedom matches its defining sum, scalar and array")
def test_effective_degrees_of_freedom(sub):
    got = has(sub, "effective_degrees_of_freedom")
    s = np.array([10.0, 5.0, 1.0, 0.1])

    almost_zero = got(s, 0.0)
    if abs(almost_zero - 4.0) > 1e-8:
        fail(f"at lambda=0, degrees of freedom should equal the rank 4, got {almost_zero:.6g}")

    huge = got(s, 1e9)
    if huge > 1e-3:
        fail(f"at an enormous lambda, degrees of freedom should be near 0, got {huge:.6g}")

    lambdas = np.array([0.0, 1.0, 100.0])
    dof = got(s, lambdas)
    shape(dof, (3,), label="effective_degrees_of_freedom with an array of lambdas")
    want = np.array([np.sum(s**2 / (s**2 + l)) for l in lambdas])
    close(dof, want, atol=1e-10, label="against the defining sum")


@check("ridge_svd_path matches ridge_closed_form at every lambda")
def test_ridge_svd_path(sub):
    closed_form = has(sub, "ridge_closed_form")
    got = has(sub, "ridge_svd_path")
    X, y, _ = correlated_design(n=60, d=10, seed=3)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    lambdas = np.array([0.01, 0.5, 5.0, 50.0, 500.0])

    path = got(U, s, Vt, y, lambdas)
    shape(path, (5, 10), label="ridge_svd_path output")
    for i, lam in enumerate(lambdas):
        close(
            path[i],
            closed_form(X, y, lam),
            atol=1e-8,
            label=f"ridge_svd_path at lambda={lam:g} against ridge_closed_form",
        )


@check("soft_threshold is the exact proximal operator of the l1 norm")
def test_soft_threshold_is_prox(sub):
    got = has(sub, "soft_threshold")
    rng = np.random.default_rng(4)
    grid = np.linspace(-5.0, 5.0, 4001)

    for x in rng.uniform(-3.0, 3.0, size=8):
        for thresh in (0.3, 1.0, 2.5):
            candidate = float(got(np.array([x]), thresh)[0])
            objective = 0.5 * (grid - x) ** 2 + thresh * np.abs(grid)
            best_on_grid = grid[np.argmin(objective)]
            candidate_value = 0.5 * (candidate - x) ** 2 + thresh * abs(candidate)
            best_value = objective.min()
            if candidate_value > best_value + 1e-3:
                fail(
                    f"soft_threshold({x:.3f}, {thresh:.3f}) = {candidate:.4f} scores "
                    f"{candidate_value:.6f} on 0.5(z-x)^2 + thresh|z|, but a grid "
                    f"search near z={best_on_grid:.4f} scores {best_value:.6f}"
                )


@check("ISTA's objective never increases")
def test_ista_monotonic(sub):
    got = has(sub, "ista")
    X, y, _ = correlated_design(seed=5)
    L = np.linalg.eigvalsh(X.T @ X).max()
    _, history = got(X, y, lam=2.0, step=1.0 / L, iters=100)
    shape(history, (101,), label="ista history")
    increases = np.diff(history)
    if np.any(increases > 1e-8):
        fail(
            f"ISTA's objective increased by up to {increases.max():.3g} between "
            "consecutive steps; a proximal-gradient step with step <= 1/L must "
            "never increase the objective"
        )


@check("FISTA's objective falls faster than ISTA's on an ill-conditioned lasso")
def test_fista_faster_than_ista(sub):
    ista_fn = has(sub, "ista")
    fista_fn = has(sub, "fista")
    X, y, _ = correlated_design(seed=0)
    L = np.linalg.eigvalsh(X.T @ X).max()
    step = 1.0 / L
    iters = 80

    _, ista_history = ista_fn(X, y, lam=3.0, step=step, iters=iters)
    _, fista_history = fista_fn(X, y, lam=3.0, step=step, iters=iters)

    if fista_history[-1] > 0.7 * ista_history[-1]:
        fail(
            f"after {iters} steps ISTA's objective is {ista_history[-1]:.4f} and "
            f"FISTA's is {fista_history[-1]:.4f}; FISTA should be substantially "
            "lower on this ill-conditioned problem"
        )


@check("The lasso solution satisfies the KKT stationarity condition")
def test_lasso_kkt(sub):
    got = has(sub, "fista")
    X, y, _ = correlated_design(n=80, d=15, seed=1)
    L = np.linalg.eigvalsh(X.T @ X).max()
    lam = 1.5
    w, _ = got(X, y, lam, step=1.0 / L, iters=800)

    grad_smooth = X.T @ (X @ w - y)
    nonzero = np.abs(w) > 1e-6
    if nonzero.any():
        stationarity = grad_smooth[nonzero] + lam * np.sign(w[nonzero])
        worst = float(np.abs(stationarity).max())
        if worst > 1e-2:
            fail(
                "at a nonzero coordinate the lasso's KKT condition requires "
                f"X_i^T(Xw - y) + lambda * sign(w_i) = 0, but the largest "
                f"violation is {worst:.4g}"
            )
    zero = ~nonzero
    if zero.any():
        slack = float(np.abs(grad_smooth[zero]).max())
        if slack > lam + 1e-2:
            fail(
                f"at a zero coordinate the KKT condition requires "
                f"|X_i^T(Xw - y)| <= lambda = {lam}, but the largest value is {slack:.4g}"
            )
