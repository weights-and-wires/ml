"""Checks for the ISTA, FISTA, and KKT exercise."""

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


@check("ista on the one-column example steps from 0 to 0.8 and the score from 2.5 to 0.9")
def test_ista_hand(sub):
    w, history = sub.ista(COLUMN, LABELS, lam=1.0, step=0.2, iters=1)
    close(w, np.array([0.8]), atol=1e-12, label="weight after one step")
    close(history, np.array([2.5, 0.9]), atol=1e-12, label="history after one step")


@check("ista returns a history of length iters + 1, uses w0, and leaves w0 untouched")
def test_ista_shapes(sub):
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


@check("kkt_violation is 0 at the answer 0.8 and 4 at the start 0 on the one-column example")
def test_kkt_hand(sub):
    almost(sub.kkt_violation(COLUMN, LABELS, np.array([0.8]), 1.0), 0.0, 1e-9, label="breach at w = 0.8")
    almost(sub.kkt_violation(COLUMN, LABELS, np.array([0.0]), 1.0), 4.0, 1e-9, label="breach at w = 0")


@check("kkt_violation reports the largest breach over a mix of zero and nonzero weights")
def test_kkt_formula(sub):
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
