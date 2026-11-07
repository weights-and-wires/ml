"""Checks for the dropout and label smoothing exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


@check("dropout_mask holds only 0s and 1s, keeps about a 1 - p share, and repeats under the same seed")
def test_dropout_mask_holds_only_0s_and_1s_keeps(sub):
    dropout_mask = has(sub, "dropout_mask")
    p = 0.3
    mask = dropout_mask((400, 500), p, np.random.default_rng(0))
    shape(mask, (400, 500), label="mask shape")
    if not np.all((mask == 0.0) | (mask == 1.0)):
        fail("the mask should hold only 0.0 and 1.0")
    almost(float(mask.mean()), 1.0 - p, tol=0.01, label="share of units kept")
    again = dropout_mask((400, 500), p, np.random.default_rng(0))
    close(again, mask, atol=0.0, label="mask with the same seed")
    other = dropout_mask((400, 500), p, np.random.default_rng(1))
    if np.array_equal(other, mask):
        fail("different seeds should give different masks")


@check("dropout_forward returns x unchanged at test time and x * mask / (1 - p) in training")
def test_dropout_forward_returns_x_unchanged_at_t(sub):
    dropout_forward = has(sub, "dropout_forward")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4))
    out, mask = dropout_forward(x, p=0.5, rng=rng, training=False)
    close(out, x, atol=1e-12, label="output at test time")
    close(mask, np.ones_like(x), atol=1e-12, label="mask at test time")
    dropout_forward = has(sub, "dropout_forward")
    dropout_mask = has(sub, "dropout_mask")
    x = np.array([[2.0, 4.0, 6.0, 8.0], [1.0, -3.0, 5.0, 7.0]])
    # p = 0.5 cannot tell 1 - p from p, so also try p = 0.25.
    for p in (0.5, 0.25):
        out, mask = dropout_forward(x, p, np.random.default_rng(5), True)
        shape(out, x.shape, label="output shape")
        close(mask, dropout_mask(x.shape, p, np.random.default_rng(5)), atol=0.0, label=f"mask from the same seed at p={p}")
        close(out, x * mask / (1.0 - p), atol=1e-12, label=f"output vs x * mask / (1 - p) at p={p}")
        if np.any(out[mask == 0.0] != 0.0):
            fail("dropped units should output exactly 0")


@check("the average output over many draws matches the input, so the layer sum averages 20")
def test_the_average_output_over_many_draws_match(sub):
    dropout_forward = has(sub, "dropout_forward")
    rng = np.random.default_rng(1)
    x = np.array([2.0, 4.0, 6.0, 8.0])
    trials = 4000
    total = np.zeros(4)
    for _ in range(trials):
        out, _ = dropout_forward(x, 0.5, rng, True)
        total += out
    mean = total / trials
    close(mean, x, atol=0.6, label="mean output per unit, near [2, 4, 6, 8]")
    almost(float(mean.sum()), 20.0, tol=1.0, label="mean layer sum")


@check("dropout_backward turns dout = [1, 1, 1, 1] into [2, 0, 2, 2] and matches a finite difference with the mask held fixed")
def test_dropout_backward_turns_dout_1_1_1_1_into(sub):
    dropout_backward = has(sub, "dropout_backward")
    dx = dropout_backward(np.ones(4), np.array([1.0, 0.0, 1.0, 1.0]), 0.5)
    close(dx, np.array([2.0, 0.0, 2.0, 2.0]), atol=1e-12, label="dx for the worked example")
    dx = dropout_backward(np.ones(4), np.array([1.0, 0.0, 1.0, 1.0]), 0.25)
    close(dx, np.array([4 / 3, 0.0, 4 / 3, 4 / 3]), atol=1e-12, label="dx at p = 0.25 (divide by 1 - p, not p)")
    dropout_forward = has(sub, "dropout_forward")
    dropout_backward = has(sub, "dropout_backward")
    p = 0.4
    x = np.linspace(-2.0, 2.0, 12).reshape(3, 4)
    dout = np.linspace(1.0, -1.0, 12).reshape(3, 4)

    _, mask = dropout_forward(x, p, np.random.default_rng(7), True)
    dx = dropout_backward(dout, mask, p)

    def f(x_):
        out_, _ = dropout_forward(x_, p, np.random.default_rng(7), True)  # same seed, same mask
        return float(np.sum(dout * out_))

    close(dx, finite_difference(f, x), atol=1e-6, rtol=1e-4, label="dx vs finite difference")


@check("smooth_targets gives [0.9333, 0.0333, 0.0333] for class 0 with K = 3 and eps = 0.1")
def test_smooth_targets_gives_0_9333_0_0333_0_033(sub):
    smooth_targets = has(sub, "smooth_targets")
    out = smooth_targets(np.array([0, 2]), 3, 0.1)
    shape(out, (2, 3), label="smooth_targets output")
    want = np.array([[0.9 + 0.1 / 3, 0.1 / 3, 0.1 / 3], [0.1 / 3, 0.1 / 3, 0.9 + 0.1 / 3]])
    close(out, want, atol=1e-12, label="smoothed targets")
    close(out.sum(axis=1), np.ones(2), atol=1e-12, label="row sums")
    close(smooth_targets(np.array([1, 0]), 2, 0.0), np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12, label="eps = 0 gives one-hot")


@check("label_smoothing_loss is 0.3729 on the worked row, and is the mean over rows")
def test_label_smoothing_loss_is_0_3729_on_the_wo(sub):
    loss = has(sub, "label_smoothing_loss")
    logits = np.array([[2.0, 0.0, 0.0]])
    targets = np.array([0])
    got = loss(logits, targets, 0.1)
    if not isinstance(got, float):
        fail(f"label_smoothing_loss should return a float, got {type(got).__name__}")
    almost(got, 0.372878, tol=1e-5, label="loss with eps = 0.1")
    almost(loss(logits, targets, 0.0), 0.239545, tol=1e-5, label="loss with eps = 0")
    loss = has(sub, "label_smoothing_loss")
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 5))
    targets = np.array([0, 3, 1, 4])
    singles = [loss(logits[i:i + 1], targets[i:i + 1], 0.2) for i in range(4)]
    almost(loss(logits, targets, 0.2), float(np.mean(singles)), tol=1e-10, label="batch loss vs mean of single-row losses")


@check("label_smoothing_loss stays finite at scores of 800, where it equals 2 * (eps / K) * 800")
def test_label_smoothing_loss_stays_finite_at_sco(sub):
    loss = has(sub, "label_smoothing_loss")
    logits = np.array([[800.0, 0.0, 0.0]])
    got = loss(logits, np.array([0]), 0.1)
    if not np.isfinite(got):
        fail(f"loss is not finite at scores of 800: {got}")
    almost(got, 2 * (0.1 / 3) * 800, tol=1e-6, label="loss at a gap of 800")
    almost(loss(logits, np.array([0]), 0.0), 0.0, tol=1e-9, label="loss at a gap of 800 with eps = 0")


@check("label_smoothing_grad matches a finite difference, and vanishes where p equals the smoothed target")
def test_label_smoothing_grad_matches_a_finite_di(sub):
    loss = has(sub, "label_smoothing_loss")
    grad = has(sub, "label_smoothing_grad")
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(5, 4))
    targets = np.array([0, 1, 2, 3, 1])
    eps = 0.15
    got = grad(logits, targets, eps)
    shape(got, logits.shape, label="label_smoothing_grad output")
    close(got, finite_difference(lambda z: loss(z, targets, eps), logits), atol=1e-6, rtol=1e-4, label="grad vs finite difference")
    grad = has(sub, "label_smoothing_grad")
    got = grad(np.array([[2.0, 0.0, 0.0]]), np.array([0]), 0.1)
    close(got, np.array([[-0.146347, 0.073174, 0.073174]]), atol=1e-5, label="gradient on the worked row")
    close(got.sum(axis=1), np.zeros(1), atol=1e-12, label="gradient row sum")
    eps, k = 0.2, 4
    optimal = np.log(np.array([[1 - eps + eps / k, eps / k, eps / k, eps / k]]))
    at_optimum = grad(optimal, np.array([0]), eps)
    if float(np.abs(at_optimum).max()) > 1e-9:
        fail(f"the gradient should vanish where softmax equals the smoothed target, but its largest entry is {float(np.abs(at_optimum).max()):.3g}")


@check("the loss over score gaps is lowest near ln 28 = 3.33, and the one-hot loss never turns up")
def test_the_loss_over_score_gaps_is_lowest_near(sub):
    loss = has(sub, "label_smoothing_loss")
    gaps = np.linspace(0.0, 8.0, 801)
    targets = np.array([0])
    smoothed = np.array([loss(np.array([[g, 0.0, 0.0]]), targets, 0.1) for g in gaps])
    best = float(gaps[int(np.argmin(smoothed))])
    almost(best, float(np.log(28.0)), tol=0.02, label="gap with the lowest smoothed loss")
    plain = np.array([loss(np.array([[g, 0.0, 0.0]]), targets, 0.0) for g in gaps])
    if np.any(np.diff(plain) >= 0):
        fail("with eps = 0 the loss should keep falling as the gap grows")
