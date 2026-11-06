"""Checks for the KL leash and best-policy exercise."""

import warnings

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

REF = np.array([0.5, 0.3, 0.2])
REWARDS = np.array([1.0, 0.0, 0.0])


def _softmax_rows(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@check("kl_divergence gives 0.0851 for [0.7, 0.2, 0.1] against the reference, 0 for identical, and depends on the order")
def test_kl_hand(sub):
    kl = has(sub, "kl_divergence")
    p = np.array([0.7, 0.2, 0.1])
    almost(float(kl(p, REF)), 0.085123, tol=1e-5, label="KL(p || ref)")
    almost(float(kl(REF, p)), 0.092033, tol=1e-5, label="KL(ref || p)")
    almost(float(kl(REF, REF)), 0.0, tol=1e-12, label="KL of a distribution with itself")


@check("kl_divergence counts a term with p = 0 as 0, with no nan and no warning")
def test_kl_zero_terms(sub):
    kl = has(sub, "kl_divergence")
    p = np.array([1.0, 0.0, 0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            value = float(kl(p, REF))
        except FloatingPointError as exc:
            fail(f"kl_divergence warned on a zero-probability term: {exc}")
    almost(value, np.log(2.0), tol=1e-10, label="KL of [1, 0, 0] against the reference")
    almost(float(kl(np.array([0.5, 0.5]), np.array([0.99, 0.01]))), 0.5 * np.log(0.5 / 0.99) + 0.5 * np.log(50.0), tol=1e-10, label="mass where the reference has little")


@check("kl_divergence works on a batch and sums only the last axis")
def test_kl_batch(sub):
    kl = has(sub, "kl_divergence")
    rng = np.random.default_rng(0)
    p = _softmax_rows(rng.normal(size=(4, 5)))
    q = _softmax_rows(rng.normal(size=(4, 5)))
    got = kl(p, q)
    shape(got, (4,), label="KL on a (4, 5) batch")
    want = np.array([np.sum(p[i] * np.log(p[i] / q[i])) for i in range(4)])
    close(got, want, atol=1e-12, label="row-wise KL")
    if np.any(got < 0):
        fail(f"KL must be non-negative, got {got}")


@check("optimal_policy gives [0.7311, 0.1614, 0.1076] on the three-answer example")
def test_optimal_hand(sub):
    best = has(sub, "optimal_policy")
    close(best(np.log(REF), REWARDS, 1.0), [0.731059, 0.161365, 0.107577], atol=1e-5, label="best policy at beta = 1")


@check("optimal_policy normalises each row of a batch and matches the closed form")
def test_optimal_batch(sub):
    best = has(sub, "optimal_policy")
    rng = np.random.default_rng(1)
    ref = _softmax_rows(rng.normal(size=(4, 6)))
    r = rng.normal(size=(4, 6))
    beta = 0.7
    pi = best(np.log(ref), r, beta)
    shape(pi, (4, 6), label="best policy")
    close(pi.sum(axis=-1), np.ones(4), atol=1e-12, label="row sums")
    w = ref * np.exp(r / beta)
    close(pi, w / w.sum(axis=-1, keepdims=True), atol=1e-12, label="closed form")


@check("optimal_policy is the reference at a huge beta and the top-reward answer at a tiny beta")
def test_optimal_limits(sub):
    best = has(sub, "optimal_policy")
    close(best(np.log(REF), REWARDS, 1e6), REF, atol=1e-5, label="beta = 1e6")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            small = best(np.log(REF), REWARDS, 1e-3)
        except FloatingPointError as exc:
            fail(f"optimal_policy overflowed at beta = 0.001: {exc}")
    close(small, [1.0, 0.0, 0.0], atol=1e-9, label="beta = 0.001")


@check("penalised_objective scores the reference 0.5, the best policy 0.6201, and all-in 0.3069")
def test_objective_hand(sub):
    obj = has(sub, "penalised_objective")
    best = has(sub, "optimal_policy")
    almost(float(obj(REF, REF, REWARDS, 1.0)), 0.5, tol=1e-10, label="reference")
    almost(float(obj(best(np.log(REF), REWARDS, 1.0), REF, REWARDS, 1.0)), 0.620115, tol=1e-5, label="best policy")
    almost(float(obj(np.array([1.0, 0.0, 0.0]), REF, REWARDS, 1.0)), 0.306853, tol=1e-5, label="all on the top reward")


@check("No random policy beats optimal_policy, and score = beta log Z - beta KL(policy || best)")
def test_nothing_beats_best(sub):
    obj = has(sub, "penalised_objective")
    best = has(sub, "optimal_policy")
    kl = has(sub, "kl_divergence")
    rng = np.random.default_rng(2)
    ref = _softmax_rows(rng.normal(size=5))
    r = rng.normal(size=5) * 2
    beta = 0.8
    pi_star = best(np.log(ref), r, beta)
    top = float(obj(pi_star, ref, r, beta))
    log_z = np.log(np.sum(ref * np.exp(r / beta)))
    almost(top, beta * log_z, tol=1e-10, label="score of the best policy against beta * log Z")
    for _ in range(300):
        p = rng.dirichlet(np.ones(5))
        score = float(obj(p, ref, r, beta))
        if score > top + 1e-12:
            fail(f"a random policy scored {score:.6f}, above the best policy's {top:.6f}")
        almost(score, beta * log_z - beta * float(kl(p, pi_star)), tol=1e-9, label="score decomposition")
