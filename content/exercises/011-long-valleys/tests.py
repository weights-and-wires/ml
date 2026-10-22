"""Checks for the long-valleys exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def random_spd(d: int, mu: float, L: float, seed: int) -> np.ndarray:
    """A symmetric matrix, tilted away from the coordinate axes, with extreme eigenvalues mu and L."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigs = np.concatenate([[mu], rng.uniform(mu, L, size=d - 2), [L]])
    return (q * eigs) @ q.T


def walk(sub, curvature: float, start, iters: int, step: float | None = None):
    """Gradient descent on diag(1, curvature) with bottom (1, 1), at step 1/curvature by default."""
    gd = has(sub, "gradient_descent")
    A = np.diag([1.0, curvature])
    w_star = np.array([1.0, 1.0])
    b = A @ w_star
    path = gd(lambda w: A @ w - b, np.array(start, dtype=float), step or 1.0 / curvature, iters)
    return path, w_star


@check("condition_number is L / mu, also for a tilted matrix")
def test_condition_number(sub):
    got = has(sub, "condition_number")
    close(np.array([got(np.diag([1.0, 10.0]))]), np.array([10.0]), atol=1e-9, label="kappa of diag(1, 10)")
    close(np.array([got(np.eye(3) * 2.0)]), np.array([1.0]), atol=1e-9, label="kappa of 2 I (a round bowl)")
    close(np.array([got(random_spd(4, 2.0, 200.0, seed=0))]), np.array([100.0]), atol=1e-6, label="kappa of a tilted matrix")


@check("contraction_ratios divides each error by the one before it")
def test_ratios_hand_path(sub):
    got = has(sub, "contraction_ratios")
    path = np.array([[2.0, 0.0], [1.0, 0.0], [0.5, 0.0]])
    out = np.asarray(got(path, np.zeros(2)))
    shape(out, (2,), label="contraction_ratios")
    close(out, np.array([0.5, 0.5]), atol=1e-12, label="ratios of a halving path")
    path = np.array([[3.0, 4.0], [0.0, 2.5], [0.0, 0.5]])
    close(np.asarray(got(path, np.zeros(2))), np.array([0.5, 0.2]), atol=1e-12, label="ratios of lengths, not coordinates")


@check("From (0, 0) on the kappa = 10 bowl the ratios are 0.9 / sqrt(2), then exactly 0.9")
def test_ratios_on_walk(sub):
    got = has(sub, "contraction_ratios")
    path, w_star = walk(sub, 10.0, (0.0, 0.0), 6)
    out = np.asarray(got(path, w_star))
    want = np.array([0.9 / np.sqrt(2.0)] + [0.9] * 5)
    close(out, want, atol=1e-9, label="ratios along the walk")


@check("Along the flat axis the ratio is exactly 1 - mu / L for any kappa")
def test_ratio_is_one_minus_inverse_kappa(sub):
    got = has(sub, "contraction_ratios")
    for curvature in (4.0, 50.0):
        path, w_star = walk(sub, curvature, (0.0, 1.0), 30)
        out = np.asarray(got(path, w_star))
        close(out, np.full(30, 1.0 - 1.0 / curvature), atol=1e-10, label=f"ratios at kappa = {curvature:g}")


@check("steps_to_tolerance returns the first row under the target, or -1")
def test_steps_hand_path(sub):
    got = has(sub, "steps_to_tolerance")
    path = np.array([[4.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.5, 0.0], [0.25, 0.0]])
    equal(got(path, np.zeros(2), 0.2), 3, label="steps to reach 0.2 of the start (error below 0.8)")
    equal(got(path, np.zeros(2), 0.05), -1, label="steps to reach 0.05 of the start (never)")


@check("A round bowl is solved in one step, and kappa = 10 and 100 need 66 and 688 steps")
def test_step_counts(sub):
    got = has(sub, "steps_to_tolerance")
    gd = has(sub, "gradient_descent")
    A = 3.0 * np.eye(3)
    w_star = np.array([0.5, -1.0, 2.0])
    b = A @ w_star
    path = gd(lambda w: A @ w - b, np.array([2.0, 2.0, -1.0]), 1.0 / 3.0, 5)
    equal(got(path, w_star, 1e-6), 1, label="steps for kappa = 1")

    p10, w_star = walk(sub, 10.0, (0.0, 1.0), 200)
    p100, _ = walk(sub, 100.0, (0.0, 1.0), 1500)
    s10 = got(p10, w_star, 1e-3)
    s100 = got(p100, w_star, 1e-3)
    equal(s10, 66, label="steps for kappa = 10")
    equal(s100, 688, label="steps for kappa = 100")
    ratio = s100 / s10
    if not 9.0 <= ratio <= 11.5:
        fail(f"kappa grew by 10 but the step count grew by {ratio:.2f}; expected about 10")
