"""Checks for the gradient boosting exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has, shape

X_HAND = np.array([[1.0], [2.0], [3.0], [4.0]])
Y_HAND = np.array([1.0, 3.0, 8.0, 10.0])


def squared_loss(y, F):
    return 0.5 * float(np.mean((y - F) ** 2))


def finite_difference_neg_grad(loss, y, F, h=1e-6):
    """Return -d loss / d F_i for each row, by central differences."""
    out = np.zeros_like(F)
    for i in range(len(F)):
        up, down = F.copy(), F.copy()
        up[i] += h
        down[i] -= h
        out[i] = -(loss(y[i], up[i]) - loss(y[i], down[i])) / (2 * h)
    return out


@check("squared_error_neg_grad matches a finite-difference slope of 0.5 * (y - F) ** 2")
def test_squared_grad(sub):
    grad = has(sub, "squared_error_neg_grad")
    rng = np.random.default_rng(0)
    y = rng.normal(size=12)
    F = rng.normal(size=12)
    got = grad(y, F)
    shape(got, (12,), label="squared_error_neg_grad output")
    want = finite_difference_neg_grad(lambda yi, Fi: 0.5 * (yi - Fi) ** 2, y, F)
    close(got, want, atol=1e-6, label="negative gradient of the squared loss")
    close(grad(Y_HAND, np.zeros(4)), Y_HAND, atol=1e-12, label="at F = 0 the negative gradient is y")


@check("absolute_error_neg_grad matches a finite-difference slope of |y - F|")
def test_absolute_grad(sub):
    grad = has(sub, "absolute_error_neg_grad")
    rng = np.random.default_rng(1)
    y = rng.normal(size=12)
    F = y + rng.choice([-1.0, 1.0], size=12) * rng.uniform(0.2, 2.0, size=12)
    got = grad(y, F)
    shape(got, (12,), label="absolute_error_neg_grad output")
    want = finite_difference_neg_grad(lambda yi, Fi: abs(yi - Fi), y, F)
    close(got, want, atol=1e-6, label="negative gradient of the absolute loss")
    close(grad(np.array([2.0]), np.array([2.0])), np.array([0.0]), atol=1e-12, label="at the corner y == F")


@check("gradient_boost starts from zero, so round 1 fits y itself")
def test_first_round(sub):
    boost = has(sub, "gradient_boost")
    sq = has(sub, "squared_error_neg_grad")
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 3))
    y = 2.0 * X[:, 0] - X[:, 1] + 0.2 * rng.normal(size=80)
    F, history = boost(X, y, sq, n_rounds=4, learning_rate=0.5)
    if len(history) != 4:
        fail(f"expected a history of length 4, got {len(history)}")
    shape(F, (80,), label="final F")
    close(history[0]["target"], y, atol=1e-10, label="round-1 target under squared error")
    close(F, history[-1]["prediction"], atol=1e-12, label="final F equals the last history prediction")


@check("gradient_boost reproduces the hand example round by round")
def test_hand_example(sub):
    boost = has(sub, "gradient_boost")
    sq = has(sub, "squared_error_neg_grad")
    F, history = boost(X_HAND, Y_HAND, sq, n_rounds=2, learning_rate=0.5)
    close(history[0]["prediction"], np.array([1.0, 1.0, 4.5, 4.5]), atol=1e-12, label="F after round 1")
    close(history[1]["target"], np.array([0.0, 2.0, 3.5, 5.5]), atol=1e-12, label="round-2 target is the residual")
    close(history[1]["prediction"], np.array([1.5, 1.5, 6.75, 6.75]), atol=1e-12, label="F after round 2")
    almost(squared_loss(Y_HAND, history[0]["prediction"]), 5.8125, tol=1e-12, label="loss after round 1")
    almost(squared_loss(Y_HAND, F), 1.828125, tol=1e-12, label="loss after round 2")


@check("each round lowers the training loss under squared error")
def test_monotone(sub):
    boost = has(sub, "gradient_boost")
    sq = has(sub, "squared_error_neg_grad")
    rng = np.random.default_rng(6)
    X = rng.normal(size=(150, 4))
    y = np.sin(X[:, 0]) + X[:, 1] - 0.5 * X[:, 2] + 0.2 * rng.normal(size=150)
    _, history = boost(X, y, sq, n_rounds=25, learning_rate=0.4)
    losses = [squared_loss(y, step["prediction"]) for step in history]
    for m in range(len(losses) - 1):
        if losses[m + 1] > losses[m] + 1e-12:
            fail(f"the loss rose from {losses[m]:.6f} at round {m + 1} to {losses[m + 1]:.6f} at round {m + 2}")
    if losses[-1] > 0.5 * losses[0]:
        fail(f"25 rounds should at least halve the loss, but it went from {losses[0]:.4f} to {losses[-1]:.4f}")


@check("each round's loss drop equals eta * (1 - eta / 2) * mean(h ** 2)")
def test_shrinkage_identity(sub):
    boost = has(sub, "gradient_boost")
    sq = has(sub, "squared_error_neg_grad")
    rng = np.random.default_rng(7)
    X = rng.normal(size=(120, 3))
    y = 1.5 * X[:, 0] - X[:, 2] + 0.15 * rng.normal(size=120)
    for eta in (0.1, 0.3, 0.6, 1.0, 1.5):
        _, history = boost(X, y, sq, n_rounds=5, learning_rate=eta)
        before = np.zeros(120)
        for m, step in enumerate(history, start=1):
            h = (step["prediction"] - before) / eta
            want = squared_loss(y, before) - eta * (1 - eta / 2) * float(np.mean(h ** 2))
            almost(
                squared_loss(y, step["prediction"]),
                want,
                tol=1e-9,
                label=f"loss after round {m} at learning_rate={eta}",
            )
            before = step["prediction"]


@check("each history entry is its own copy of F")
def test_history_copies(sub):
    boost = has(sub, "gradient_boost")
    sq = has(sub, "squared_error_neg_grad")
    rng = np.random.default_rng(8)
    X = rng.normal(size=(60, 2))
    y = X[:, 0] + 0.1 * rng.normal(size=60)
    F, history = boost(X, y, sq, n_rounds=5, learning_rate=0.5)
    if np.allclose(history[0]["prediction"], history[-1]["prediction"]):
        fail("history[0] and history[-1] hold the same predictions, so each entry must be a copy: use F.copy()")
    keep = history[-1]["prediction"].copy()
    F[:] = 0.0
    close(history[-1]["prediction"], keep, atol=0.0, label="history after the returned F is overwritten")


@check("the neg_grad hook decides what each round fits")
def test_hook(sub):
    boost = has(sub, "gradient_boost")
    absolute = has(sub, "absolute_error_neg_grad")
    rng = np.random.default_rng(9)
    X = rng.normal(size=(50, 2))
    y = X[:, 0] + 0.3 * rng.normal(size=50)

    F, history = boost(X, y, lambda yy, FF: np.ones_like(yy), n_rounds=3, learning_rate=0.2)
    close(F, np.full(50, 0.6), atol=1e-12, label="constant target 1 for 3 rounds at learning_rate 0.2")

    F, history = boost(X, y, absolute, n_rounds=40, learning_rate=0.3)
    close(history[0]["target"], np.sign(y), atol=1e-12, label="round-1 target under absolute error")
    before = float(np.mean(np.abs(y)))
    after = float(np.mean(np.abs(y - F)))
    if after > 0.8 * before:
        fail(f"40 rounds of absolute-error boosting should cut the mean |y - F| from {before:.4f} well below, got {after:.4f}")


def _mean_log_loss(y, F):
    """Mean of log(1 + exp(F)) - y F, for labels y in {0, 1}."""
    return float(np.mean(np.log1p(np.exp(F)) - y * F))


# Dropping lam from the denominator still matches the squared-loss leaf at
# lam = 0, and it returns 2 instead of 4/3 on the left leaf when lam is 1.
@check("newton_leaf_value matches the gradient step on squared loss, and lambda stays in the denominator")
def test_newton_leaf_squared(sub):
    leaf = has(sub, "newton_leaf_value")
    g = -Y_HAND  # slope of 0.5 * (y - F) ** 2 at F = 0
    h = np.ones(4)
    left = np.array([True, True, False, False])
    got0 = leaf(g[left], h[left], 0.0)
    almost(got0, 2.0, tol=1e-12, label="left Newton leaf at lam=0")
    almost(
        got0,
        float((-g[left]).mean()),
        tol=1e-12,
        label="at lam=0 the Newton leaf equals the mean of the downhill steps",
    )
    almost(leaf(g[left], h[left], 1.0), 4.0 / 3.0, tol=1e-12, label="left Newton leaf at lam=1")
    almost(leaf(g, h, 0.0), 5.5, tol=1e-12, label="one leaf over all four rows at lam=0")

    rng = np.random.default_rng(3)
    g = rng.normal(size=7)
    h = np.ones(7)
    almost(
        leaf(g, h, 0.0),
        float(-g.mean()),
        tol=1e-12,
        label="squared loss, lam=0, Newton step equals the gradient step",
    )
    want = float(-g.sum() / (h.sum() + 0.8))
    almost(leaf(g, h, 0.8), want, tol=1e-12, label="squared loss, lam=0.8")


# Using the gradient step -mean(g) as the Newton leaf leaves the two losses
# equal on this logistic example, so the comparison fails. Forgetting the 1/2
# in split_gain returns 49 instead of 24.5.
@check("one Newton step on logistic loss lowers the loss more than a gradient step, and split_gain uses the regularised formula")
def test_logistic_newton_and_gain(sub):
    leaf = has(sub, "newton_leaf_value")
    gain = has(sub, "split_gain")
    y = np.array([1.0, 1.0, 0.0])
    F = np.zeros(3)
    p = 1.0 / (1.0 + np.exp(-F))
    g = p - y
    h = p * (1.0 - p)
    w_newton = float(leaf(g, h, 0.0))
    w_grad = float(-g.mean())
    almost(w_newton, 2.0 / 3.0, tol=1e-12, label="Newton leaf on labels [1, 1, 0] at F = 0")
    loss_start = _mean_log_loss(y, F)
    loss_grad = _mean_log_loss(y, F + w_grad)
    loss_newton = _mean_log_loss(y, F + w_newton)
    if not (loss_newton < loss_grad < loss_start):
        fail(
            f"expected Newton loss < gradient loss < start loss; "
            f"got Newton loss {loss_newton:.6f} from leaf {w_newton:.6f}, "
            f"gradient loss {loss_grad:.6f} from leaf {w_grad:.6f}, "
            f"start loss {loss_start:.6f}"
        )

    g4 = np.array([-1.0, -3.0, -8.0, -10.0])
    h4 = np.ones(4)
    left = np.array([True, True, False, False])
    almost(gain(g4, h4, left, 0.0, 0.0), 24.5, tol=1e-12, label="gain at lam=0, gamma=0")
    almost(gain(g4, h4, left, 1.0, 0.0), 124.0 / 15.0, tol=1e-9, label="gain at lam=1, gamma=0")
    almost(
        gain(g4, h4, left, 1.0, 0.1),
        124.0 / 15.0 - 0.1,
        tol=1e-9,
        label="gain at lam=1, gamma=0.1",
    )
