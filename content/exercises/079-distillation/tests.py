"""Checks for the distillation exercise.

The wrong implementations these checks are written to catch:

* Distillation loss without the T^2 factor. The "soft term at T = 3" and
  "gradient size stays level as T grows" checks catch it, because the loss is
  9 times too small and the gradient norm shrinks like 1/T.
* KL with its arguments swapped, KL(student || teacher), inside distill_loss.
  The "soft term at T = 3" check and the finite-difference check catch it,
  because the loss value changes and the gradient no longer matches.
"""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape

Z_T = np.array([[6.0, 3.0, 0.0]])
Z_S = np.array([[4.0, 3.0, 1.0]])


@check("soft_targets: [6, 3, 0] gives [0.9503, 0.0473, 0.0024] at T = 1 and [0.6652, 0.2447, 0.0900] at T = 3")
def test_soft_targets(sub):
    fn = has(sub, "soft_targets")
    close(fn(np.array([6.0, 3.0, 0.0]), 1.0), np.array([0.95033, 0.04731, 0.00236]), atol=1e-4, label="T = 1")
    close(fn(np.array([6.0, 3.0, 0.0]), 3.0), np.array([0.66524, 0.24473, 0.09003]), atol=1e-4, label="T = 3")
    rows = fn(np.array([[1000.0, 999.0, 0.0], [0.0, 0.0, 0.0]]), 2.0)
    shape(rows, (2, 3), label="batch shape")
    if not np.all(np.isfinite(rows)):
        fail(f"soft_targets overflowed on logits near 1000: {rows}")
    close(rows.sum(axis=1), np.ones(2), atol=1e-12, label="rows sum to 1")


@check("A larger temperature flattens the distribution: entropy rises and the largest probability falls")
def test_temperature_flattens(sub):
    fn = has(sub, "soft_targets")
    z = np.array([5.0, 2.0, 0.0, -1.0])
    prev_entropy, prev_max = -1.0, 2.0
    for T in (0.5, 1.0, 2.0, 5.0, 50.0):
        p = fn(z, T)
        entropy = float(-np.sum(p * np.log(p)))
        if not (entropy > prev_entropy and p.max() < prev_max):
            fail(f"T = {T}: entropy {entropy:.4f} (previous {prev_entropy:.4f}), max {p.max():.4f} (previous {prev_max:.4f})")
        prev_entropy, prev_max = entropy, float(p.max())
    close(fn(z, 1e6), np.full(4, 0.25), atol=1e-4, label="T = 1e6 is uniform")


@check("kl_divergence: known value, zero for equal inputs, and not symmetric")
def test_kl(sub):
    kl = has(sub, "kl_divergence")
    p = np.array([0.5, 0.5, 0.0])
    q = np.array([0.25, 0.25, 0.5])
    almost(float(kl(p, q)), np.log(2.0), 1e-9, label="KL([.5,.5,0] || [.25,.25,.5])")
    almost(float(kl(p, p)), 0.0, 1e-12, label="KL(p || p)")
    a = np.array([0.7, 0.2, 0.1])
    b = np.array([0.4, 0.4, 0.2])
    want_ab = float(np.sum(a * np.log(a / b)))
    want_ba = float(np.sum(b * np.log(b / a)))
    almost(float(kl(a, b)), want_ab, 1e-9, label="KL(a || b)")
    almost(float(kl(b, a)), want_ba, 1e-9, label="KL(b || a)")
    out = kl(np.array([a, b]), np.array([b, a]))
    shape(out, (2,), label="one divergence per row")


@check("distill_loss: the soft term at T = 3 is 0.6633 and the mixed loss is 0.5062")
def test_distill_loss_value(sub):
    loss = has(sub, "distill_loss")
    y = np.array([0])
    # Expected pieces: T^2 * KL(teacher || student) = 0.6633, cross-entropy of the label = 0.3490.
    almost(loss(Z_S, Z_T, y, 3.0, 1.0), 0.66331, 2e-4, label="alpha = 1, T = 3 (soft term only)")
    almost(loss(Z_S, Z_T, y, 3.0, 0.0), 0.34901, 2e-4, label="alpha = 0 (hard term only)")
    almost(loss(Z_S, Z_T, y, 3.0, 0.5), 0.50616, 2e-4, label="alpha = 0.5")
    almost(loss(Z_T, Z_T, np.array([0]), 3.0, 1.0), 0.0, 1e-12, label="student equal to teacher")
    zs = np.array([[4.0, 3.0, 1.0], [0.0, 2.0, 1.0]])
    zt = np.array([[6.0, 3.0, 0.0], [1.0, 1.0, 1.0]])
    one = [loss(zs[i:i + 1], zt[i:i + 1], np.array([0]), 2.0, 0.3) for i in range(2)]
    almost(loss(zs, zt, np.array([0, 0]), 2.0, 0.3), float(np.mean(one)), 1e-10, label="mean over rows")


@check("distill_grad matches finite differences, for several temperatures and mixes")
def test_distill_grad(sub):
    loss = has(sub, "distill_loss")
    grad = has(sub, "distill_grad")
    rng = np.random.default_rng(0)
    zs = rng.normal(size=(4, 5)) * 2
    zt = rng.normal(size=(4, 5)) * 3
    y = np.array([0, 3, 1, 4])
    for T, alpha in ((1.0, 1.0), (3.0, 0.7), (5.0, 0.0), (2.0, 0.5)):
        want = finite_difference(lambda a: loss(a, zt, y, T, alpha), zs)
        got = grad(zs, zt, y, T, alpha)
        shape(got, zs.shape, label=f"gradient shape at T={T}")
        close(got, want, rtol=1e-5, atol=1e-7, label=f"gradient at T={T}, alpha={alpha}")


@check("The T^2 factor keeps the soft gradient the same size as T grows")
def test_t_squared(sub):
    grad = has(sub, "distill_grad")
    y = np.array([0])
    norms = {T: float(np.linalg.norm(grad(Z_S, Z_T, y, T, 1.0))) for T in (4.0, 8.0, 16.0)}
    ratio = norms[16.0] / norms[4.0]
    if not 0.9 < ratio < 1.1:
        fail(f"gradient norms at T = 4, 8, 16 were {norms}; expected the T = 16 to T = 4 ratio near 1, got {ratio:.3f}")
    # Expected near 0.714 at T = 4 for this teacher and student, one row (n = 1).
    almost(norms[4.0], 0.7139, 2e-3, label="gradient norm at T = 4")


@check("At a very large T the soft gradient equals the centred logit difference over K")
def test_logit_limit(sub):
    match = has(sub, "logit_match_grad")
    grad = has(sub, "distill_grad")
    rng = np.random.default_rng(1)
    zs = rng.normal(size=(3, 4)) * 2
    zt = rng.normal(size=(3, 4)) * 3
    y = np.zeros(3, dtype=int)
    got = grad(zs, zt, y, 1000.0, 1.0) * 3  # distill_grad divides by n = 3
    want = match(zs, zt)
    shape(want, zs.shape, label="logit_match_grad shape")
    close(got, want, rtol=0, atol=2e-3, label="T = 1000 gradient against (z_s_c - z_t_c) / K")
    almost(float(want.sum()), 0.0, 1e-12, label="the limit gradient sums to zero over classes")
    shifted = match(zs + 7.0, zt - 2.0)
    close(shifted, want, atol=1e-12, label="adding a constant to a row of logits changes nothing")


@check("feature_distill_loss and its gradient in P match the formula and finite differences")
def test_feature(sub):
    loss = has(sub, "feature_distill_loss")
    grad = has(sub, "feature_distill_grad")
    hs = np.array([[1.0, 0.0], [0.0, 2.0]])
    ht = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    # h_s @ P = [[1, 0, 0], [0, 2, 2]]; residual = [[0, -1, 0], [0, 2, -1]]; sum of squares = 6; loss = 6 / 4.
    almost(loss(hs, ht, P), 1.5, 1e-12, label="worked example")
    rng = np.random.default_rng(2)
    h_s = rng.normal(size=(6, 4))
    h_t = rng.normal(size=(6, 7))
    P0 = rng.normal(size=(4, 7))
    want = finite_difference(lambda p: loss(h_s, h_t, p), P0)
    got = grad(h_s, h_t, P0)
    shape(got, (4, 7), label="gradient shape")
    close(got, want, rtol=1e-5, atol=1e-7, label="gradient in P")


def _problem(seed, n=30, d=10, K=3):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(d, K))
    X = rng.normal(size=(n, d))
    X_test = rng.normal(size=(2000, d))
    p = np.exp(X @ W)
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(K, p=row) for row in p])
    z_t = X @ W + 0.3 * rng.normal(size=(n, K))
    truth = (X_test @ W).argmax(axis=1)
    return X, z_t, y, X_test, truth


@check("With 30 examples, a student trained on soft targets beats one trained on hard labels")
def test_student(sub):
    fit = has(sub, "fit_student")
    soft, hard = [], []
    for seed in range(5):
        X, z_t, y, X_test, truth = _problem(seed)
        W_soft = fit(X, z_t, y, 3.0, 1.0, 400, 0.1)
        W_hard = fit(X, z_t, y, 3.0, 0.0, 400, 0.1)
        shape(W_soft, (10, 3), label="W shape")
        soft.append(float(np.mean((X_test @ W_soft).argmax(axis=1) == truth)))
        hard.append(float(np.mean((X_test @ W_hard).argmax(axis=1) == truth)))
    if not (np.mean(soft) > np.mean(hard) + 0.1 and np.mean(soft) > 0.9):
        fail(
            f"expected soft-target agreement above 0.9 and at least 0.1 above hard-label agreement; "
            f"got soft {np.round(soft, 3)} (mean {np.mean(soft):.3f}) and hard {np.round(hard, 3)} (mean {np.mean(hard):.3f})"
        )
