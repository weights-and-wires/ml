"""Checks for the reward-model and KL-leash exercise."""

import warnings

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape

REF = np.array([0.5, 0.3, 0.2])
REWARDS = np.array([1.0, 0.0, 0.0])


def _softmax_rows(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


@check("preference_probability gives 0.5, 0.8808 and 0.2689 at gaps 0, 2 and -1, stays finite at +-800, and swapping gives 1 - p")
def test_probability(sub):
    got = has(sub, "preference_probability")
    out = got(np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    close(out, [0.5, 0.880797, 0.268941], atol=1e-5, label="chance the chosen answer wins")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            big = got(np.array([400.0, -400.0]), np.array([-400.0, 400.0]))
        except FloatingPointError as exc:
            fail(f"preference_probability overflowed at a gap of 800: {exc}")
    if not np.all(np.isfinite(big)):
        fail(f"non-finite values at a gap of +-800: {big}")
    close(big, [1.0, 0.0], atol=1e-12, label="chance at gaps +800 and -800")
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=7), rng.normal(size=7)
    close(got(a, b) + got(b, a), np.ones(7), atol=1e-12, label="p(a beats b) + p(b beats a)")
    shape(got(a, b), (7,), label="output")


@check("bradley_terry_loss: a tie costs log 2, the two-pair example costs 0.8133, and a constant shift changes nothing")
def test_loss_known(sub):
    loss = has(sub, "bradley_terry_loss")
    tied = np.zeros(5)
    almost(loss(tied, tied), np.log(2.0), tol=1e-9, label="tied rewards")
    almost(loss(np.array([2.0, 0.0]), np.array([1.0, 1.0])), 0.813262, tol=1e-5, label="two-pair example")
    rng = np.random.default_rng(1)
    c, r = rng.normal(size=8), rng.normal(size=8)
    almost(loss(c + 3.7, r + 3.7), loss(c, r), tol=1e-10, label="loss after adding 3.7 to every reward")


@check("bradley_terry_loss is near 0 for a wide right gap, near the gap for a wide wrong gap, and finite at -800")
def test_loss_extremes(sub):
    loss = has(sub, "bradley_terry_loss")
    hi, lo = np.full(4, 10.0), np.zeros(4)
    if loss(hi, lo) > 1e-4:
        fail(f"a gap of +10 should cost almost nothing, got {loss(hi, lo):.3g}")
    almost(loss(lo, hi), 10.0, tol=1e-3, label="a confidently wrong gap of -10")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            value = loss(np.array([-400.0]), np.array([400.0]))
        except FloatingPointError as exc:
            fail(f"bradley_terry_loss overflowed at a gap of -800: {exc}")
    almost(value, 800.0, tol=1e-6, label="loss at a gap of -800")


@check("bradley_terry_grad gives [-0.1345, -0.3655] on the two-pair example and matches finite differences")
def test_grad(sub):
    loss = has(sub, "bradley_terry_loss")
    grad = has(sub, "bradley_terry_grad")
    gc, gr = grad(np.array([2.0, 0.0]), np.array([1.0, 1.0]))
    close(gc, [-0.134471, -0.365529], atol=1e-5, label="grad_chosen")
    close(gr, [0.134471, 0.365529], atol=1e-5, label="grad_rejected")

    rng = np.random.default_rng(2)
    n = 6
    c, r = rng.normal(size=n) * 2, rng.normal(size=n) * 2
    gc, gr = grad(c, r)
    shape(gc, (n,), label="grad_chosen")
    shape(gr, (n,), label="grad_rejected")
    fd = finite_difference(lambda x: loss(x[:n], x[n:]), np.concatenate([c, r]))
    close(np.concatenate([gc, gr]), fd, rtol=1e-4, atol=1e-7, label="gradient against finite differences")


@check("kl_divergence gives 0.0851 for [0.7, 0.2, 0.1] against the reference, 0 for identical, and depends on the order")
def test_kl_hand(sub):
    kl = has(sub, "kl_divergence")
    p = np.array([0.7, 0.2, 0.1])
    almost(float(kl(p, REF)), 0.085123, tol=1e-5, label="KL(p || ref)")
    almost(float(kl(REF, p)), 0.092033, tol=1e-5, label="KL(ref || p)")
    almost(float(kl(REF, REF)), 0.0, tol=1e-12, label="KL of a distribution with itself")


@check("kl_divergence counts a term with p = 0 as 0 with no warning, and sums only the last axis of a batch")
def test_kl_zero_and_batch(sub):
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

    rng = np.random.default_rng(0)
    p = _softmax_rows(rng.normal(size=(4, 5)))
    q = _softmax_rows(rng.normal(size=(4, 5)))
    got = kl(p, q)
    shape(got, (4,), label="KL on a (4, 5) batch")
    want = np.array([np.sum(p[i] * np.log(p[i] / q[i])) for i in range(4)])
    close(got, want, atol=1e-12, label="row-wise KL")
    if np.any(got < 0):
        fail(f"KL must be non-negative, got {got}")


@check("optimal_policy gives [0.7311, 0.1614, 0.1076] on the three-answer example and normalises each row of a batch")
def test_optimal(sub):
    best = has(sub, "optimal_policy")
    close(best(np.log(REF), REWARDS, 1.0), [0.731059, 0.161365, 0.107577], atol=1e-5, label="best policy at beta = 1")

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
