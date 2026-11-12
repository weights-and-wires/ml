"""Checks for the interpretability exercise.

Plausible wrong implementations these checks catch:
- exact_shapley with weight 1/n for every subset instead of
  |S|! (n - |S| - 1)! / n!. It fails "exact_shapley matches the hand values"
  because feature 0 of the worked model comes out 4, not 3.
- integrated_gradients that returns the average path gradient and forgets to
  multiply by (x - baseline). It fails "integrated_gradients gives [3, 3, 2]"
  and the completeness checks.
- sae_step that skips the unit-norm rescaling. It fails "sae_step lowers the
  loss ... and keeps every decoder row at unit length".
"""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


# The worked model of the chapter: f(x) = x0 * x1 + 2 * x2.
def _f(v):
    return float(v[0] * v[1] + 2.0 * v[2])


def _grad_f(v):
    return np.array([v[1], v[0], 2.0])


X_W = np.array([2.0, 3.0, 1.0])


def _smooth_model(seed=0, d=5):
    """A nonlinear model with an exact gradient: tanh(a . x) + 0.5 * sum(x^2)."""
    a = np.random.default_rng(seed).normal(size=d)

    def f(v):
        return float(np.tanh(a @ v) + 0.5 * np.sum(v * v))

    def g(v):
        return (1.0 - np.tanh(a @ v) ** 2) * a + v

    return f, g


@check("grad_times_input gives [6, 6, 2] on the worked model at x = [2, 3, 1]")
def test_gti(sub):
    fn = has(sub, "grad_times_input")
    got = fn(_grad_f, X_W)
    shape(got, (3,), label="grad_times_input")
    close(got, np.array([6.0, 6.0, 2.0]), atol=1e-12, label="grad_times_input")


@check("integrated_gradients gives [3, 3, 2] on the worked model, and its sum matches f(x) - f(baseline) for a nonzero baseline")
def test_ig_worked(sub):
    ig = has(sub, "integrated_gradients")
    got = ig(_f, _grad_f, X_W, np.zeros(3), 4)
    close(got, np.array([3.0, 3.0, 2.0]), atol=1e-9, label="IG with a zero baseline")
    base = np.array([1.0, -1.0, 0.5])
    got = ig(_f, _grad_f, X_W, base, 10)
    almost(float(got.sum()), _f(X_W) - _f(base), 1e-9, label="sum of IG vs f(x) - f(baseline)")


@check("integrated_gradients completeness error shrinks as steps grow on a nonlinear model")
def test_ig_converges(sub):
    ig = has(sub, "integrated_gradients")
    f, g = _smooth_model()
    rng = np.random.default_rng(1)
    x, base = rng.normal(size=5), rng.normal(size=5)
    want = f(x) - f(base)
    errs = [abs(float(ig(f, g, x, base, s).sum()) - want) for s in (2, 8, 32, 128)]
    if not (errs[0] > errs[1] > errs[2] > errs[3]):
        fail(f"completeness errors for steps 2, 8, 32, 128 should fall each time, got {errs}")
    if errs[3] > 1e-4:
        fail(f"with 128 steps the sum should be within 1e-4 of {want:.6f}, got error {errs[3]:.2e}")


@check("exact_shapley matches the hand values [3, 3, 2], sums to f(x) - f(baseline), and equals w * (x - b) for a linear model")
def test_shapley(sub):
    sh = has(sub, "exact_shapley")
    got = sh(_f, X_W, np.zeros(3))
    close(got, np.array([3.0, 3.0, 2.0]), atol=1e-9, label="Shapley values of the worked model")
    rng = np.random.default_rng(2)
    n = 6
    x, base = rng.normal(size=n), rng.normal(size=n)
    f, _ = _smooth_model(3, n)
    phi = sh(f, x, base)
    almost(float(phi.sum()), f(x) - f(base), 1e-9, label="efficiency: sum of Shapley values")
    w = rng.normal(size=n)
    phi = sh(lambda v: float(w @ v + 0.7), x, base)
    close(phi, w * (x - base), atol=1e-9, label="Shapley values of a linear model")


@check("probe_loss_grad matches finite differences for w and b, and stays finite for huge scores")
def test_probe_grad(sub):
    fn = has(sub, "probe_loss_grad")
    rng = np.random.default_rng(0)
    H = rng.normal(size=(12, 4))
    y = (rng.uniform(size=12) < 0.5).astype(float)
    w, b, l2 = rng.normal(size=4), 0.3, 0.1
    loss, gw, gb = fn(w, b, H, y, l2)
    want_w = finite_difference(lambda v: fn(v, b, H, y, l2)[0], w)
    close(gw, want_w, rtol=1e-5, atol=1e-7, label="grad_w vs finite differences")
    want_b = finite_difference(lambda v: fn(w, float(v[0]), H, y, l2)[0], np.array([b]))[0]
    almost(float(gb), float(want_b), 1e-6, label="grad_b vs finite differences")
    big, _, _ = fn(np.full(4, 1e4), 0.0, H, y, 0.0)
    if not np.isfinite(big):
        fail(f"loss with huge scores should be finite, got {big}")


@check("a probe on planted activations reaches 0.9 held-out accuracy, while a random network and shuffled labels stay near chance")
def test_probe_control(sub):
    train = has(sub, "train_probe")
    acc = has(sub, "probe_accuracy")
    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(size=(n, 6))
    y = (X[:, 0] * X[:, 1] > 0).astype(float)
    Wp, Wr = rng.normal(size=(6, 7)), rng.normal(size=(6, 8))
    H_rand = np.maximum(X @ Wr, 0.0)
    planted_unit = np.tanh(2 * X[:, 0]) * np.tanh(2 * X[:, 1])
    H_plant = np.concatenate([np.maximum(X @ Wp, 0.0), planted_unit[:, None]], axis=1)
    tr, te = slice(0, 500), slice(500, n)

    def held_out(H, labels):
        w, b = train(H[tr], labels[tr])
        return acc(w, b, H[te], labels[te])

    a_plant = held_out(H_plant, y)
    a_rand = held_out(H_rand, y)
    a_shuf = held_out(H_plant, rng.permutation(y))
    if a_plant < 0.9:
        fail(f"planted activations: expected held-out accuracy of at least 0.9, got {a_plant:.3f}")
    if a_rand > 0.75:
        fail(f"random network: expected at most 0.75, got {a_rand:.3f}")
    if a_shuf > 0.65:
        fail(f"shuffled labels: expected at most 0.65, got {a_shuf:.3f}")


@check("match_score is 1 for permuted, sign-flipped, rescaled copies and 0 for orthogonal directions")
def test_match_score(sub):
    ms = has(sub, "match_score")
    rng = np.random.default_rng(0)
    D = rng.normal(size=(5, 4))
    almost(ms(-3.0 * D[::-1], D), 1.0, 1e-9, label="permuted, flipped, scaled copy")
    almost(ms(np.array([[0.0, 1.0]]), np.array([[1.0, 0.0]])), 0.0, 1e-9, label="orthogonal pair")
    got = ms(np.array([[1.0, 0.0], [1.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]]))
    almost(got, (1.0 + 0.5**0.5) / 2, 1e-9, label="one exact and one 45 degree match")


def _sae_params(seed=0, d=3, k=4):
    rng = np.random.default_rng(seed)
    Wd = rng.normal(size=(k, d))
    Wd /= np.linalg.norm(Wd, axis=1, keepdims=True)
    return {"We": rng.normal(size=(d, k)) * 0.5, "be": rng.normal(size=k) * 0.1, "Wd": Wd}


@check("sae_loss_grads matches finite differences for We, be, and Wd")
def test_sae_grads(sub):
    fn = has(sub, "sae_loss_grads")
    rng = np.random.default_rng(1)
    X = rng.normal(size=(9, 3))
    P = _sae_params()
    l1 = 0.2
    loss, grads = fn(P, X, l1)
    for key in ("We", "be", "Wd"):
        shape(grads[key], P[key].shape, label=f"grad {key}")

        def as_loss(v, key=key):
            Q = dict(P)
            Q[key] = v.reshape(P[key].shape)
            return fn(Q, X, l1)[0]

        want = finite_difference(as_loss, P[key].ravel()).reshape(P[key].shape)
        close(grads[key], want, rtol=1e-4, atol=1e-6, label=f"grad {key} vs finite differences")


@check("sae_step lowers the loss on a small step and keeps every decoder row at unit length")
def test_sae_step(sub):
    step = has(sub, "sae_step")
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 3))
    P = _sae_params(3)
    P0 = {k: v.copy() for k, v in P.items()}
    new, loss0 = step(P, X, 0.1, 0.01)
    for k in P0:
        close(P[k], P0[k], atol=0.0, label=f"input {k} must not change in place")
    close(np.linalg.norm(new["Wd"], axis=1), np.ones(4), atol=1e-9, label="decoder row norms")
    _, loss1 = step(new, X, 0.1, 0.01)
    if not loss1 < loss0:
        fail(f"expected the loss to fall after one step, got {loss0:.6f} then {loss1:.6f}")


@check("train_sae recovers planted sparse features: mean best cosine above 0.9, far above a random dictionary")
def test_train_sae(sub):
    train = has(sub, "train_sae")
    ms = has(sub, "match_score")
    rng = np.random.default_rng(0)
    m, d, n = 16, 8, 3000
    D = rng.normal(size=(m, d))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    X = ((rng.uniform(size=(n, m)) < 0.04) * 1.0) @ D
    P = train(X, m, l1=0.3, lr=0.2, steps=4000, seed=100)
    close(np.linalg.norm(P["Wd"], axis=1), np.ones(m), atol=1e-9, label="decoder row norms")
    score = ms(P["Wd"], D)
    base = ms(rng.normal(size=(m, d)), D)
    if score < 0.9:
        fail(f"expected a mean best cosine of at least 0.9, got {score:.3f} (a random dictionary scores {base:.3f})")
    if score < base + 0.15:
        fail(f"expected the trained score to beat a random dictionary ({base:.3f}) by 0.15, got {score:.3f}")
