"""Checks for the speculative decoding exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has

P = np.array([0.7, 0.2, 0.1])
Q = np.array([0.1, 0.1, 0.8])


@check("expected_accept_rate is the overlap of the two distributions")
def test_accept_rate(sub):
    got = has(sub, "expected_accept_rate")
    almost(got(P, Q), 0.3, tol=1e-12, label="rate for p=[.7,.2,.1], q=[.1,.1,.8]")
    almost(got(P, P), 1.0, tol=1e-12, label="rate when the draft equals the target")
    almost(got(np.array([1.0, 0.0]), np.array([0.0, 1.0])), 0.0, tol=1e-12, label="rate for disjoint distributions")


@check("expected_accept_rate is symmetric and equals one minus the leftover target mass")
def test_accept_rate_identity(sub):
    got = has(sub, "expected_accept_rate")
    rng = np.random.default_rng(0)
    for _ in range(10):
        p = rng.dirichlet(np.ones(6))
        q = rng.dirichlet(np.ones(6))
        almost(got(p, q), got(q, p), tol=1e-12, label="symmetry")
        almost(got(p, q), 1.0 - np.maximum(p - q, 0).sum(), tol=1e-12, label="1 minus the residual mass")


@check("residual_distribution is max(p - q, 0) normalised")
def test_residual(sub):
    got = has(sub, "residual_distribution")
    close(got(P, Q), [6 / 7, 1 / 7, 0.0], atol=1e-12, label="residual for the worked example")
    out = got(np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.2, 0.6]))
    close(out, [0.3 / 0.4, 0.1 / 0.4, 0.0], atol=1e-12, label="residual for a second pair")
    almost(float(out.sum()), 1.0, tol=1e-12, label="sums to 1")


@check("residual_distribution falls back to p when p equals q")
def test_residual_equal(sub):
    got = has(sub, "residual_distribution")
    out = got(P, P)
    if not np.all(np.isfinite(out)):
        fail(f"residual_distribution(p, p) returned {out}; it must not divide by zero")
    close(out, P, atol=1e-12, label="residual of identical distributions")


@check("speculative_accept always keeps a token the target likes at least as much as the draft")
def test_accept_always(sub):
    got = has(sub, "speculative_accept")
    rng = np.random.default_rng(1)
    for token in (0, 1):
        for _ in range(200):
            out = got(P, Q, token, rng)
            if out != token:
                fail(f"draft token {token} has p >= q, so it must always be accepted, but got {out}")
    same = [got(P, P, 1, rng) for _ in range(200)]
    if any(t != 1 for t in same):
        fail("with q equal to p every draft token must be accepted")


@check("After a rejection the replacement comes from the residual, not from p")
def test_replacement(sub):
    got = has(sub, "speculative_accept")
    rng = np.random.default_rng(2)
    n = 20_000
    # Draft token 2: accepted with probability 0.125, else replaced from [6/7, 1/7, 0].
    # Output: token 0 with 0.875 * 6/7 = 0.75, token 1 with 0.125, token 2 with 0.125.
    draws = np.array([got(P, Q, 2, rng) for _ in range(n)])
    freq = np.bincount(draws, minlength=3) / n
    close(freq, [0.75, 0.125, 0.125], atol=0.015, label="outputs given draft token 2")


@check("With drafts drawn from q, the output follows p even for a badly wrong draft")
def test_output_distribution(sub):
    got = has(sub, "speculative_accept")
    rng = np.random.default_rng(7)
    n = 20_000
    drafts = rng.choice(3, size=n, p=Q)
    out = np.array([got(P, Q, int(d), rng) for d in drafts])
    close(np.bincount(out, minlength=3) / n, P, atol=0.02, label="empirical output vs target p")
    q2 = np.array([0.25, 0.25, 0.5])
    drafts = rng.choice(3, size=n, p=q2)
    out = np.array([got(P, q2, int(d), rng) for d in drafts])
    close(np.bincount(out, minlength=3) / n, P, atol=0.02, label="empirical output vs p, milder draft")
