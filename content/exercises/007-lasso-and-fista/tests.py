"""Checks for the lasso exercise: soft thresholding, ISTA, FISTA, and the KKT test."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, shape


def correlated_design(n: int = 100, d: int = 40, seed: int = 0):
    """A design with correlated columns and a sparse true weight vector."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 5))
    X = latent @ rng.normal(size=(5, d)) + 0.5 * rng.normal(size=(n, d))
    w_true = np.zeros(d)
    w_true[:5] = rng.normal(size=5) * 2.0
    y = X @ w_true + 0.1 * rng.normal(size=n)
    return X, y


def step_for(X):
    return 1.0 / np.linalg.eigvalsh(X.T @ X).max()


COLUMN = np.array([[1.0], [2.0]])
LABELS = np.array([1.0, 2.0])


@check("soft_threshold maps 3, -3, 0.5, -0.5, 0, 2 to 2, -2, 0, 0, 0, 1 at threshold 1, keeps the shape, is odd, and returns exact zeros")
def test_soft_threshold(sub):
    x = np.array([3.0, -3.0, 0.5, -0.5, 0.0, 2.0])
    close(sub.soft_threshold(x, 1.0), np.array([2.0, -2.0, 0.0, 0.0, 0.0, 1.0]), atol=1e-12)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(4, 5)) * 2
    out = sub.soft_threshold(x, 0.7)
    shape(out, (4, 5), label="soft_threshold output")
    close(sub.soft_threshold(-x, 0.7), -out, atol=1e-12, label="S(-x) against -S(x)")

    x = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
    out = sub.soft_threshold(x, 1.0)
    equal(np.abs(out), np.zeros(5), label="soft_threshold at threshold 1 for entries inside [-1, 1]")


@check("soft_threshold is the exact minimiser of 0.5 (z - x)^2 + t |z|")
def test_soft_threshold_is_prox(sub):
    rng = np.random.default_rng(4)
    grid = np.linspace(-5.0, 5.0, 4001)
    for x in rng.uniform(-3.0, 3.0, size=8):
        for t in (0.3, 1.0, 2.5):
            z = float(sub.soft_threshold(np.array([x]), t)[0])
            objective = 0.5 * (grid - x) ** 2 + t * np.abs(grid)
            value = 0.5 * (z - x) ** 2 + t * abs(z)
            if value > objective.min() + 1e-6:
                fail(
                    f"soft_threshold({x:.3f}, {t}) = {z:.4f} scores {value:.6f}, but a grid search "
                    f"finds {objective.min():.6f} near z = {grid[np.argmin(objective)]:.4f}"
                )


@check("lasso_objective scores the hand example (4 at (3, 1), 3.25 at (1.5, 0.5), 3 at (2, 0)) with no half on the fee")
def test_lasso_objective(sub):
    X = np.eye(2)
    y = np.array([3.0, 1.0])
    for w, want in (([3.0, 1.0], 4.0), ([1.5, 0.5], 3.25), ([2.0, 0.0], 3.0)):
        almost(sub.lasso_objective(X, y, np.array(w), 1.0), want, 1e-12, label=f"score at {tuple(w)}")

    rng = np.random.default_rng(1)
    X = rng.normal(size=(20, 4))
    y = rng.normal(size=20)
    w = rng.normal(size=4)
    base = sub.lasso_objective(X, y, w, 0.0)
    almost(base, 0.5 * np.sum((X @ w - y) ** 2), 1e-10, label="score at lam = 0")
    for lam in (0.5, 3.0):
        almost(sub.lasso_objective(X, y, w, lam) - base, lam * np.sum(np.abs(w)), 1e-9, label=f"fee at lam = {lam}")


@check("lasso_identity solves the chapter's example, (2, 0) and not ridge's (1.5, 0.5), and no nearby weights beat it")
def test_lasso_identity(sub):
    got = sub.lasso_identity(np.array([3.0, 1.0]), 1.0)
    close(got, np.array([2.0, 0.0]), atol=1e-12, label="lasso_identity(y = (3, 1), lam = 1)")
    zeros = sub.lasso_identity(np.array([3.0, -1.0, 0.2]), 3.0)
    equal(zeros, np.zeros(3), label="a fee at least as large as every |y| sets every weight to 0")

    rng = np.random.default_rng(5)
    y = rng.normal(size=6) * 2
    lam = 0.8
    w = sub.lasso_identity(y, lam)
    shape(w, (6,), label="lasso_identity output")
    X = np.eye(6)

    def score(v):
        return 0.5 * np.sum((X @ v - y) ** 2) + lam * np.sum(np.abs(v))

    best = score(w)
    for _ in range(300):
        other = w + 0.1 * rng.normal(size=6)
        if score(other) < best - 1e-12:
            fail(f"weights {np.round(other, 3)} score {score(other):.6f}, below lasso_identity's {best:.6f}")
    for i in range(6):
        for delta in (-0.01, 0.01):
            other = w.copy()
            other[i] += delta
            if score(other) < best - 1e-12:
                fail(f"moving weight {i} by {delta} lowers the score, so it is not the exact minimiser")


@check("ista on the one-column example steps from 0 to 0.8 and the score from 2.5 to 0.9, and it leaves w0 untouched")
def test_ista_hand(sub):
    w, history = sub.ista(COLUMN, LABELS, lam=1.0, step=0.2, iters=1)
    close(w, np.array([0.8]), atol=1e-12, label="weight after one step")
    close(history, np.array([2.5, 0.9]), atol=1e-12, label="history after one step")

    X, y = correlated_design(n=30, d=8, seed=2)
    w0 = np.full(8, 0.5)
    w0_copy = w0.copy()
    w, history = sub.ista(X, y, lam=1.0, step=step_for(X), iters=12, w0=w0)
    shape(w, (8,), label="ista weights")
    shape(history, (13,), label="ista history")
    almost(history[0], 0.5 * np.sum((X @ w0 - y) ** 2) + 1.0 * np.sum(np.abs(w0)), 1e-9, label="history[0]")
    equal(w0, w0_copy, label="the caller's w0 after the call")


@check("ista never raises the score at step 1/L, and soft thresholding lands exactly on 0")
def test_ista_monotone_and_zeros(sub):
    X, y = correlated_design(seed=5)
    w, history = sub.ista(X, y, lam=20.0, step=step_for(X), iters=100)
    rises = np.diff(history)
    if np.any(rises > 1e-8):
        fail(f"the score rose by up to {rises.max():.3g} in one step; at step 1/L it cannot")
    if not np.any(w == 0.0):
        fail("no weight is exactly 0 after 100 steps at lam = 20; soft thresholding should write exact zeros")
    huge, _ = sub.ista(X, y, lam=1e6, step=step_for(X), iters=5)
    equal(huge, np.zeros(40), label="weights when the fee is enormous")


@check("kkt_violation is 0 at the answer 0.8 and 4 at the start 0, and reports the largest breach over a mix of weights")
def test_kkt(sub):
    almost(sub.kkt_violation(COLUMN, LABELS, np.array([0.8]), 1.0), 0.0, 1e-9, label="breach at w = 0.8")
    almost(sub.kkt_violation(COLUMN, LABELS, np.array([0.0]), 1.0), 4.0, 1e-9, label="breach at w = 0")

    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 5))
    y = rng.normal(size=20)
    w = np.array([0.0, 0.7, 0.0, -0.4, 1.1])
    lam = 2.0
    g = X.T @ (X @ w - y)
    want = max(
        abs(g[1] + lam), abs(g[3] - lam), abs(g[4] + lam),
        max(abs(g[0]) - lam, 0.0), max(abs(g[2]) - lam, 0.0),
    )
    almost(sub.kkt_violation(X, y, w, lam), want, 1e-9, label="largest breach")


@check("fista matches ista for the first step, and with a step of 0.1 reaches 0.7282 at step 3")
def test_fista_hand(sub):
    w1, h1 = sub.fista(COLUMN, LABELS, lam=1.0, step=0.2, iters=1)
    close(w1, np.array([0.8]), atol=1e-12, label="fista weight after one step")
    close(h1, np.array([2.5, 0.9]), atol=1e-12, label="fista history after one step")
    w3, h3 = sub.fista(COLUMN, LABELS, lam=1.0, step=0.1, iters=3)
    shape(h3, (4,), label="fista history")
    close(w3, np.array([0.728175]), atol=1e-5, label="fista weight after three steps at step 0.1")
    w3_ista, _ = sub.ista(COLUMN, LABELS, lam=1.0, step=0.1, iters=3)
    close(w3_ista, np.array([0.7]), atol=1e-12, label="ista weight after three steps at step 0.1")


@check("after 80 steps on correlated columns FISTA's score is under 0.7 times ISTA's")
def test_fista_faster(sub):
    X, y = correlated_design(seed=0)
    step = step_for(X)
    _, slow = sub.ista(X, y, lam=3.0, step=step, iters=80)
    _, fast = sub.fista(X, y, lam=3.0, step=step, iters=80)
    if fast[-1] > 0.7 * slow[-1]:
        fail(f"after 80 steps ISTA scores {slow[-1]:.4f} and FISTA {fast[-1]:.4f}; FISTA should be well under 0.7 times ISTA")


@check("800 FISTA steps end where kkt_violation is tiny, and an enormous fee gives all zeros")
def test_fista_kkt(sub):
    X, y = correlated_design(n=80, d=15, seed=1)
    w, _ = sub.fista(X, y, lam=1.5, step=step_for(X), iters=800)
    breach = sub.kkt_violation(X, y, w, 1.5)
    if breach > 1e-2:
        fail(f"the largest KKT breach after 800 steps is {breach:.4g}, expected under 0.01")
    huge, _ = sub.fista(X, y, lam=1e6, step=step_for(X), iters=5)
    equal(huge, np.zeros(15), label="fista weights when the fee is enormous")
