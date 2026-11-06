"""Checks for the reward-from-preferences exercise."""

import warnings

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


@check("preference_probability gives 0.5, 0.8808 and 0.2689 at gaps 0, 2 and -1")
def test_probability_hand(sub):
    got = has(sub, "preference_probability")
    out = got(np.array([0.0, 2.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    close(out, [0.5, 0.880797, 0.268941], atol=1e-5, label="chance the chosen answer wins")


@check("preference_probability is finite at gaps of +-800 and swapping the answers gives 1 - p")
def test_probability_stable(sub):
    got = has(sub, "preference_probability")
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


@check("bradley_terry_loss: a tie costs log 2, the two-pair example costs 0.8133")
def test_loss_known(sub):
    loss = has(sub, "bradley_terry_loss")
    tied = np.zeros(5)
    almost(loss(tied, tied), np.log(2.0), tol=1e-9, label="tied rewards")
    almost(loss(np.array([2.0, 0.0]), np.array([1.0, 1.0])), 0.813262, tol=1e-5, label="two-pair example")


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


@check("bradley_terry_loss does not change when one constant is added to every reward")
def test_loss_shift(sub):
    loss = has(sub, "bradley_terry_loss")
    rng = np.random.default_rng(1)
    c, r = rng.normal(size=8), rng.normal(size=8)
    almost(loss(c + 3.7, r + 3.7), loss(c, r), tol=1e-10, label="loss after adding 3.7 to every reward")


@check("bradley_terry_grad matches finite differences and has the shapes (n,) and (n,)")
def test_grad_fd(sub):
    loss = has(sub, "bradley_terry_loss")
    grad = has(sub, "bradley_terry_grad")
    rng = np.random.default_rng(2)
    n = 6
    c, r = rng.normal(size=n) * 2, rng.normal(size=n) * 2
    gc, gr = grad(c, r)
    shape(gc, (n,), label="grad_chosen")
    shape(gr, (n,), label="grad_rejected")
    fd = finite_difference(lambda x: loss(x[:n], x[n:]), np.concatenate([c, r]))
    close(np.concatenate([gc, gr]), fd, rtol=1e-4, atol=1e-7, label="gradient against finite differences")


@check("bradley_terry_grad gives [-0.1345, -0.3655] on the two-pair example, so the wrong pair is pushed harder")
def test_grad_hand(sub):
    grad = has(sub, "bradley_terry_grad")
    gc, gr = grad(np.array([2.0, 0.0]), np.array([1.0, 1.0]))
    close(gc, [-0.134471, -0.365529], atol=1e-5, label="grad_chosen")
    close(gr, [0.134471, 0.365529], atol=1e-5, label="grad_rejected")


@check("A few gradient steps lower the loss and fix a pair that started out ranked wrong")
def test_training(sub):
    loss = has(sub, "bradley_terry_loss")
    grad = has(sub, "bradley_terry_grad")
    c, r = np.array([2.0, 0.0]), np.array([1.0, 1.0])
    before = loss(c, r)
    for _ in range(5):
        gc, gr = grad(c, r)
        c, r = c - 1.0 * gc, r - 1.0 * gr
    after = loss(c, r)
    if after >= before:
        fail(f"loss should fall under gradient descent: {before:.4f} before, {after:.4f} after")
    if not c[1] > r[1] - 0.0:
        fail(f"pair 2 should be ranked right after 5 steps, got chosen {c[1]:.3f} and rejected {r[1]:.3f}")
