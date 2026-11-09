"""Checks for the debugging and tuning exercise.

Two plausible wrong implementations are named here so a passing suite has
actually rejected them:

- ``check_initial_loss`` that treats a loss of ``10 * log(C)`` as close enough,
  for example ``return loss <= 10 * expected_initial_loss(model.n_classes)``.
  The fixed-loss model in the second check has that loss and must be flagged.
- ``pick_lr`` that returns the smallest rate, ``float(np.min(lrs))``, instead
  of the rate at the bottom of the bowl. On the hand-built bowl that rate is
  ``1e-4`` and the bottom is ``0.1``.

A log-uniform draw that instead calls ``rng.uniform(low, high)`` on the raw
scale misses the seeded samples in the search check.
"""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has


def _bowl_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    y = (X[:, 0] > 0).astype(int)
    return X, y


def _reference_losses(model_fn, X, y, lrs, steps):
    losses = np.empty(len(lrs), dtype=float)
    for index, lr in enumerate(lrs):
        model = model_fn()
        finished = True
        for _ in range(int(steps)):
            loss, grads = model.loss_and_grads(X, y)
            if not np.isfinite(loss):
                finished = False
                break
            model.sgd_step(grads, float(lr))
        if not finished:
            losses[index] = np.inf
            continue
        final, _grads = model.loss_and_grads(X, y)
        losses[index] = final if np.isfinite(final) else np.inf
    return losses


def _log_uniform_samples(space, n, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        config = {}
        for name in sorted(space):
            low, high = space[name]
            config[name] = float(10 ** rng.uniform(np.log10(low), np.log10(high)))
        samples.append(config)
    return samples


class _FixedLoss:
    """A stand-in whose mean loss is whatever the check is given."""

    def __init__(self, n_classes, loss_value):
        self.n_classes = n_classes
        self._loss = float(loss_value)

    def loss_and_grads(self, X, y):
        return self._loss, {}


@check("expected_initial_loss is log(C): 2 classes give 0.693147, 4 give 1.386294, 10 give 2.302585")
def test_expected_initial_loss(sub):
    fn = has(sub, "expected_initial_loss")
    almost(fn(2), np.log(2), tol=1e-12, label="log 2")
    almost(fn(4), np.log(4), tol=1e-12, label="log 4")
    almost(fn(10), np.log(10), tol=1e-12, label="log 10")


@check("check_initial_loss accepts a small init and flags a loss of 10 * log(C)")
def test_check_initial_loss_flags_ten_times(sub):
    # Wrong implementation: return loss <= 10 * log(C). That accepts the
    # fixed model below, whose loss is log(C) times ten.
    check_initial_loss = has(sub, "check_initial_loss")
    X, y = _bowl_data()
    healthy = sub.MLP(2, 8, 2, np.random.default_rng(1), scale=0.1)
    healthy_loss, _grads = healthy.loss_and_grads(X, y)
    want = float(np.log(2))
    got = check_initial_loss(healthy, X, y)
    if bool(got) is not True:
        fail(
            f"expected True for loss {healthy_loss:.6g} against log(2)={want:.6g}, got {got}"
        )
    ten_times = 10 * float(np.log(10))
    flagged = check_initial_loss(_FixedLoss(10, ten_times), X, y)
    if bool(flagged) is not False:
        fail(
            f"expected False for loss {ten_times:.6g}, which is 10 * log(10) = {np.log(10):.6g} times ten, got {flagged}"
        )


@check("check_initial_loss flags a real network whose weights were drawn at scale 2")
def test_check_initial_loss_flags_a_wide_init(sub):
    check_initial_loss = has(sub, "check_initial_loss")
    X, y = _bowl_data()
    model = sub.MLP(2, 8, 2, np.random.default_rng(1), scale=2.0)
    loss, _grads = model.loss_and_grads(X, y)
    want = float(np.log(model.n_classes))
    if abs(loss - want) <= 0.5:
        fail(
            f"fixture loss {loss:.6g} is within 0.5 of log(C)={want:.6g}, so the wide init is not a bug detector"
        )
    got = check_initial_loss(model, X, y)
    if bool(got) is not False:
        fail(f"expected False for loss {loss:.6g} against log(C)={want:.6g}, got {got}")


@check("pick_lr returns the bottom of a hand-built bowl, not the smallest rate")
def test_pick_lr_bottom_of_hand_bowl(sub):
    # Wrong implementation: return float(np.min(lrs)). On this bowl that is
    # 1e-4, and the bottom is 0.1, which is a thousand times larger.
    pick_lr = has(sub, "pick_lr")
    lrs = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    losses = np.array([2.2, 1.4, 0.5, 0.08, 0.9, 4.0])
    best = 0.1
    got = float(pick_lr(lrs, losses))
    almost(got, best, tol=1e-12, label="learning rate at the bottom")
    spiked = np.array([0.4, np.inf, 0.2])
    spiked_rates = np.array([0.01, 0.1, 1.0])
    got_finite = float(pick_lr(spiked_rates, spiked))
    if abs(got_finite - 1.0) > 1e-12:
        fail(f"expected 1 when the middle loss is inf, got {got_finite:.6g}")


@check("lr_range_test matches a fresh full-batch run at each rate, for 40 steps and for 8")
def test_lr_range_test_matches_reference(sub):
    lr_range_test = has(sub, "lr_range_test")
    X, y = _bowl_data()
    lrs = np.logspace(-3, 1, 9)

    def model_fn():
        return sub.MLP(2, 8, 2, np.random.default_rng(1), scale=0.1)

    for steps in (40, 8):
        got = np.asarray(lr_range_test(model_fn, X, y, lrs, steps=steps), dtype=float)
        want = _reference_losses(model_fn, X, y, lrs, steps)
        if got.shape != want.shape:
            fail(f"expected shape {want.shape} for {steps} steps, got {got.shape}")
        finite = np.isfinite(want)
        if not np.array_equal(np.isfinite(got), finite):
            fail(
                f"non-finite pattern for {steps} steps differs.\n  got:  {got}\n  want: {want}"
            )
        if np.any(finite):
            close(got[finite], want[finite], atol=1e-8, rtol=1e-6, label=f"losses after {steps} steps")


@check("pick_lr on the seeded bowl is within a factor of ten of the best rate")
def test_pick_lr_on_seeded_bowl(sub):
    lr_range_test = has(sub, "lr_range_test")
    pick_lr = has(sub, "pick_lr")
    X, y = _bowl_data()
    lrs = np.logspace(-3, 1, 9)

    def model_fn():
        return sub.MLP(2, 8, 2, np.random.default_rng(1), scale=0.1)

    losses = np.asarray(lr_range_test(model_fn, X, y, lrs, steps=40), dtype=float)
    best = float(lrs[int(np.argmin(np.where(np.isfinite(losses), losses, np.inf)))])
    got = float(pick_lr(lrs, losses))
    if not (best / 10 <= got <= best * 10):
        fail(f"expected a learning rate within a factor of ten of {best:.6g}, got {got:.6g}")
    smallest = float(lrs[0])
    if abs(got - smallest) <= 1e-15:
        fail(f"expected the bottom near {best:.6g}, got the smallest rate {got:.6g}")


@check("random_search draws the seeded log-uniform samples and keeps the lowest score")
def test_random_search_seeded_draw(sub):
    random_search = has(sub, "random_search")
    space = {"decay": (1e-4, 1e-1), "lr": (1e-3, 1.0)}

    def objective(config):
        return (np.log10(config["lr"]) + 1.0) ** 2 + (np.log10(config["decay"]) + 2.0) ** 2

    found = random_search(objective, space, 4, 0)
    samples = found["samples"]
    scores = found["scores"]
    want = _log_uniform_samples(space, 4, 0)
    if len(samples) != 4:
        fail(f"expected 4 samples, got {len(samples)}")
    for index, (got, expected) in enumerate(zip(samples, want)):
        for name in ("decay", "lr"):
            almost(
                got[name],
                expected[name],
                tol=1e-12,
                label=f"sample {index} {name}",
            )
    want_scores = [objective(config) for config in want]
    close(np.asarray(scores, dtype=float), np.asarray(want_scores), atol=1e-10, rtol=1e-8, label="scores")
    best_index = int(np.argmin(want_scores))
    for name in ("decay", "lr"):
        almost(
            found["best"][name],
            want[best_index][name],
            tol=1e-12,
            label=f"best {name}",
        )


@check("update ratios match a finite-difference gradient at learning rate 0.01")
def test_update_ratio_matches_finite_difference(sub):
    update_ratio_stats = has(sub, "update_ratio_stats")
    X, y = _bowl_data()
    model = sub.MLP(2, 8, 2, np.random.default_rng(1), scale=0.1)
    lr = 0.01
    before = {name: getattr(model, name).copy() for name in ("W1", "W2")}
    got = update_ratio_stats(model, X, y, lr)
    for name in ("W1", "W2"):
        if not np.allclose(getattr(model, name), before[name]):
            fail(f"{name} changed; update_ratio_stats should only read the gradient")

        def loss_of(weight, name=name):
            saved = getattr(model, name)
            setattr(model, name, weight)
            try:
                value, _grads = model.loss_and_grads(X, y)
                return value
            finally:
                setattr(model, name, saved)

        gradient = finite_difference(loss_of, before[name])
        want = float(np.linalg.norm(lr * gradient) / np.linalg.norm(before[name]))
        almost(got[name], want, tol=1e-4, label=f"{name} update ratio")


@check("the update ratio is near 1e-3 at learning rate 0.01 and far from it at learning rate 1")
def test_update_ratio_band(sub):
    # ||g|| / ||W|| on this seed is 0.331 for W1 and 0.133 for W2, so learning
    # rate 0.01 produces ratios 0.00331 and 0.00133. The band [1e-4, 1e-2] is
    # a factor of ten around 1e-3 and still contains both. Learning rate 1
    # produces 0.331 and 0.133, which sit outside that band.
    update_ratio_stats = has(sub, "update_ratio_stats")
    X, y = _bowl_data()
    low, high = 1e-4, 1e-2
    good = update_ratio_stats(sub.MLP(2, 8, 2, np.random.default_rng(1), scale=0.1), X, y, 0.01)
    for name in ("W1", "W2"):
        value = float(good[name])
        if not (low <= value <= high):
            fail(
                f"expected {name} at learning rate 0.01 to lie in [{low:.0e}, {high:.0e}] around 1e-3, got {value:.6g}"
            )
    bad = update_ratio_stats(sub.MLP(2, 8, 2, np.random.default_rng(1), scale=0.1), X, y, 1.0)
    for name in ("W1", "W2"):
        value = float(bad[name])
        if low <= value <= high:
            fail(
                f"expected {name} at learning rate 1 to fall outside [{low:.0e}, {high:.0e}], got {value:.6g}"
            )
