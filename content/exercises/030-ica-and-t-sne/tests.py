"""Checks for the ICA and t-SNE exercise."""

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite_difference, has, shape

FOUR = np.sqrt(2.0) * np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


def _sources(seed=0, n=2000, k=3):
    rng = np.random.default_rng(seed)
    cols = [
        rng.uniform(-1, 1, n) * np.sqrt(3.0),
        rng.laplace(size=n) / np.sqrt(2.0),
        np.sign(rng.normal(size=n)),
    ][:k]
    return np.column_stack(cols)


def _mixed(sub, k, seed=0):
    S = _sources(seed, k=k)
    A = np.random.default_rng(seed + 10).normal(size=(k, k))
    return S, sub.whiten(S @ A.T)


def _best_match(Z, S):
    """Absolute correlation of each recovered column with its best true source, and the matched index."""
    C = np.abs(np.corrcoef(Z.T, S.T)[: Z.shape[1], Z.shape[1] :])
    return C.max(axis=1), C.argmax(axis=1)


def _sq_dists(X):
    sq = np.sum(X**2, axis=1)
    D = sq[:, None] + sq[None, :] - 2 * X @ X.T
    D = np.maximum(D, 0.0)
    np.fill_diagonal(D, 0.0)
    return D


def _blobs(seed=0, per=30, d=10, k=3):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(k, d)) * 4.0
    X = np.vstack([c + rng.normal(size=(per, d)) for c in centres])
    return X, np.repeat(np.arange(k), per)


def _kl(P, Y):
    """Independent KL(P || Q) with Student-t Q, used for the finite-difference check."""
    d2 = _sq_dists(Y)
    inv = 1.0 / (1.0 + d2)
    np.fill_diagonal(inv, 0.0)
    Q = inv / inv.sum()
    mask = P > 0
    return float(np.sum(P[mask] * np.log(P[mask] / Q[mask])))


@check("contrast_terms returns g and g' for logcosh and kurtosis")
def test_contrast(sub):
    fn = has(sub, "contrast_terms")
    g, gp = fn(np.array([0.0, 1.0]), "logcosh")
    close(g, np.array([0.0, np.tanh(1.0)]), atol=1e-12, label="tanh values")
    close(gp, np.array([1.0, 1.0 - np.tanh(1.0) ** 2]), atol=1e-12, label="1 - tanh^2 values")
    g, gp = fn(np.array([1.0, 2.0, -1.0]), "kurtosis")
    close(g, np.array([1.0, 8.0, -1.0]), atol=1e-12, label="u^3 values")
    close(gp, np.array([3.0, 12.0, 3.0]), atol=1e-12, label="3 u^2 values")


@check("fastica_one_unit takes the hand-worked step: (0.6, 0.8) becomes (-0.7050, -0.7092) with the kurtosis contrast")
def test_one_step(sub):
    fn = has(sub, "fastica_one_unit")
    w = fn(FOUR, np.array([0.6, 0.8]), "kurtosis", 1)
    shape(w, (2,), label="w")
    # The update gives (-1.368, -1.376) before scaling to length 1. A negative sign is correct here.
    close(w, np.array([-1.368, -1.376]) / np.linalg.norm([-1.368, -1.376]), atol=1e-6, label="w after one update")
    almost(float(np.linalg.norm(w)), 1.0, tol=1e-9, label="length of w")


@check("fastica_one_unit finds one planted source with either contrast")
def test_one_unit_converges(sub):
    fn = has(sub, "fastica_one_unit")
    S, Xw = _mixed(sub, 2)
    for g in ("logcosh", "kurtosis"):
        w = fn(Xw, np.array([1.0, 0.3]), g, 100)
        best, _ = _best_match((Xw @ w)[:, None], S)
        if best[0] < 0.99:
            fail(f"contrast {g!r}: expected |correlation| of at least 0.99 with one source, got {best[0]:.3f}")


@check("fastica recovers three planted sources up to sign and order, with orthonormal rows")
def test_fastica(sub):
    fn = has(sub, "fastica")
    S, Xw = _mixed(sub, 3)
    W = fn(Xw, 3, np.random.default_rng(1))
    shape(W, (3, 3), label="W")
    close(W @ W.T, np.eye(3), atol=1e-6, label="W W^T")
    # A wrong implementation without deflation sends every row to the same source.
    # It fails here: two rows match the same column and one source is never found.
    best, which = _best_match(Xw @ W.T, S)
    if best.min() < 0.98:
        fail(f"expected each recovered column to match a source with |correlation| >= 0.98, got {np.round(best, 3)}")
    equal(sorted(which.tolist()), [0, 1, 2], label="sources matched, one per row")


@check("conditional_p gives rows that sum to 1 with a zero diagonal: distances (1, 4) at beta 1 give (0.9526, 0.0474)")
def test_conditional(sub):
    fn = has(sub, "conditional_p")
    D = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    P = fn(D, np.ones(3))
    shape(P, (3, 3), label="P")
    close(np.diag(P), np.zeros(3), atol=1e-12, label="diagonal")
    close(P.sum(axis=1), np.ones(3), atol=1e-12, label="row sums")
    e1, e4 = np.exp(-1.0), np.exp(-4.0)
    close(P[0], np.array([0.0, e1, e4]) / (e1 + e4), atol=1e-10, label="row 0")
    big = fn(np.array([[0.0, 1e4], [1e4, 0.0]]), np.array([1.0, 1.0]))
    close(big, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12, label="rows that would underflow")


@check("perplexity_bandwidths hits the target perplexity in bits, including the boundary perplexity n - 1")
def test_bandwidths(sub):
    fn = has(sub, "perplexity_bandwidths")
    cond = has(sub, "conditional_p")
    X, _ = _blobs(0)
    D = _sq_dists(X)
    n = len(D)
    # The convention is bits: the row entropy is -sum p log2 p and the target is log2(perplexity).
    # A wrong implementation that measures entropy in nats against log2(perplexity) lands on the
    # wrong perplexity for every row. It fails here.
    for perp in (5.0, 15.0, 40.0):
        P = cond(D, fn(D, perp))
        H = -np.sum(np.where(P > 0, P * np.log2(np.where(P > 0, P, 1.0)), 0.0), axis=1)
        close(2.0**H, np.full(n, perp), rtol=2e-3, label=f"perplexity of every row, target {perp}")
    Dsmall = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 9.0], [4.0, 9.0, 0.0]])
    P = cond(Dsmall, fn(Dsmall, 2.0))
    close(P[P > 0], np.full(6, 0.5), atol=2e-3, label="perplexity n - 1 makes every row uniform")
    b_lo, b_hi = fn(D, 5.0), fn(D, 40.0)
    if not np.all(b_hi < b_lo):
        fail("a larger perplexity needs a wider Gaussian, so every beta should shrink as the perplexity grows")


@check("joint_p is symmetric, has a zero diagonal, and sums to 1")
def test_joint(sub):
    fn = has(sub, "joint_p")
    X, _ = _blobs(1)
    P = fn(_sq_dists(X), 15.0)
    shape(P, (90, 90), label="P")
    close(P, P.T, atol=1e-12, label="P versus its transpose")
    close(np.diag(P), np.zeros(90), atol=1e-12, label="diagonal")
    almost(float(P.sum()), 1.0, tol=1e-9, label="sum of P")
    if P.min() < 0:
        fail(f"P has a negative entry: {P.min()}")


@check("student_q matches the hand numbers for three points on a line, and kl_divergence is 0 for equal inputs")
def test_q_and_kl(sub):
    q_fn = has(sub, "student_q")
    kl = has(sub, "kl_divergence")
    Y = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    Q, inv = q_fn(Y)
    close(inv, np.array([[0.0, 0.5, 0.2], [0.5, 0.0, 0.5], [0.2, 0.5, 0.0]]), atol=1e-12, label="1 / (1 + d^2)")
    want = np.array([[0.0, 0.5, 0.2], [0.5, 0.0, 0.5], [0.2, 0.5, 0.0]]) / 2.4
    close(Q, want, atol=1e-12, label="Q")
    almost(float(Q.sum()), 1.0, tol=1e-12, label="sum of Q")
    almost(float(kl(Q, Q)), 0.0, tol=1e-12, label="KL of Q against itself")
    P = np.array([[0.0, 0.3, 0.2], [0.3, 0.0, 0.1], [0.2, 0.1, 0.0]]) / 1.2
    almost(float(kl(P, Q)), _kl(P, Y), tol=1e-10, label="KL(P || Q)")


@check("tsne_grad matches finite differences of the KL with a Student-t Q")
def test_grad(sub):
    fn = has(sub, "tsne_grad")
    rng = np.random.default_rng(0)
    X, _ = _blobs(2, per=8, d=5)
    D = _sq_dists(X)
    A = np.exp(-D / 20.0)
    np.fill_diagonal(A, 0.0)
    P = A / A.sum()
    Y = rng.normal(size=(len(P), 2))
    got = fn(P, Y)
    shape(got, Y.shape, label="gradient")
    # A wrong implementation that uses a Gaussian Q (plain SNE) drops the (1 + d^2)^-1 factor
    # from the gradient, so it is off by a large margin from the finite differences. It fails here.
    want = finite_difference(lambda Yv: _kl(P, Yv), Y.copy())
    close(got, want, rtol=1e-4, atol=1e-7, label="tsne_grad")


@check("tsne_embed lowers the KL and keeps the planted clusters apart: 5-nearest-neighbour agreement above 0.95")
def test_embed(sub):
    P_fn = has(sub, "joint_p")
    embed = has(sub, "tsne_embed")
    X, labels = _blobs(0)
    P = P_fn(_sq_dists(X), 15.0)
    Y0 = 1e-2 * np.random.default_rng(1).normal(size=(len(X), 2))
    Y = embed(P, Y0)
    shape(Y, (90, 2), label="Y")
    if not np.all(np.isfinite(Y)):
        fail("the map has non-finite entries; check the step size and the gradient sign")
    kl_start, kl_end = _kl(P, Y0), _kl(P, Y)
    if not kl_end < 0.5 * kl_start:
        fail(f"expected the KL to fall below half its start, got {kl_start:.3f} then {kl_end:.3f}")
    d2 = _sq_dists(Y)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :5]
    agree = float((labels[nn] == labels[:, None]).mean())
    if agree < 0.95:
        fail(f"expected at least 0.95 of the 5 nearest map neighbours to share a label, got {agree:.3f}")
