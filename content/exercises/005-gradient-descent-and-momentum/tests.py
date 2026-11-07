"""Checks for the gradient descent and momentum exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape

A10 = np.diag([1.0, 10.0])
B10 = np.array([1.0, 10.0])
grad10 = lambda w: A10 @ w - B10


def random_spd(d: int, mu: float, L: float, seed: int) -> np.ndarray:
    """A symmetric matrix, tilted away from the coordinate axes, with extreme eigenvalues mu and L."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigs = np.concatenate([[mu], rng.uniform(mu, L, size=d - 2), [L]])
    return (q * eigs) @ q.T


def steps_to_millionth(path: np.ndarray, w_star: np.ndarray) -> int:
    errors = np.linalg.norm(path - w_star, axis=1)
    hit = np.nonzero(errors < 1e-6 * errors[0])[0]
    if hit.size == 0:
        fail("the walk did not reach one millionth of the starting miss within the step budget")
    return int(hit[0])


@check("quadratic_loss_and_grad gives heights 0, -5.5, -5 and gradients (-1, -10), (0, -10), and matches a finite difference")
def test_loss_and_grad(sub):
    got = has(sub, "quadratic_loss_and_grad")
    for point, want in [((0.0, 0.0), 0.0), ((1.0, 1.0), -5.5), ((0.0, 1.0), -5.0)]:
        loss, _ = got(A10, B10, np.array(point))
        if not isinstance(loss, float):
            fail(f"the loss must be a Python float, got {type(loss).__name__}")
        if abs(loss - want) > 1e-9:
            fail(f"loss at {point} is {loss}, expected {want}")
    _, g0 = got(A10, B10, np.array([0.0, 0.0]))
    _, g1 = got(A10, B10, np.array([1.0, 0.0]))
    close(g0, np.array([-1.0, -10.0]), atol=1e-9, label="gradient at (0, 0)")
    close(g1, np.array([0.0, -10.0]), atol=1e-9, label="gradient at (1, 0)")

    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4))
    Asym = M @ M.T + np.eye(4)
    b = rng.normal(size=4)
    w = rng.normal(size=4)
    _, grad = got(Asym, b, w)
    want = finite_difference(lambda x: 0.5 * x @ Asym @ x - b @ x, w)
    close(grad, want, atol=1e-5, label="analytic vs finite-difference gradient")


@check("gradient_descent keeps w0 intact, follows the hand path (0.1, 1.0), (0.19, 1.0), (0.271, 1.0), and with step 0.1 lowers the loss at every step down to near (1, 1)")
def test_hand_path(sub):
    got = has(sub, "gradient_descent")
    w0 = np.array([0.0, 0.0])
    path = got(grad10, w0, 0.1, 3)
    shape(path, (4, 2), label="gradient_descent path")
    close(w0, np.array([0.0, 0.0]), label="the input w0, which gradient_descent must not change")
    want = np.array([[0.0, 0.0], [0.1, 1.0], [0.19, 1.0], [0.271, 1.0]])
    close(path, want, atol=1e-9, label="path")

    path = got(grad10, np.array([0.0, 0.0]), 0.1, 100)
    losses = np.array([0.5 * p @ A10 @ p - B10 @ p for p in path])
    rises = np.nonzero(np.diff(losses) >= 0)[0]
    if rises.size:
        k = int(rises[0])
        fail(f"the loss did not fall at step {k + 1}: {losses[k]:.6f} then {losses[k + 1]:.6f}")
    close(path[-1], np.array([1.0, 1.0]), atol=1e-3, label="last point of the path")
    equal(bool(losses[-1] < -5.49), True, label="loss close to the minimum -5.5")


@check("eigenvalue_range returns (mu, L) and condition_number returns L / mu, also for a tilted matrix")
def test_curvatures(sub):
    rng_range = has(sub, "eigenvalue_range")
    kappa = has(sub, "condition_number")
    mu, L = rng_range(np.diag([1.0, 10.0]))
    close(np.array([mu, L]), np.array([1.0, 10.0]), atol=1e-9, label="(mu, L) of diag(1, 10)")
    mu, L = rng_range(random_spd(4, 2.0, 50.0, seed=0))
    close(np.array([mu, L]), np.array([2.0, 50.0]), atol=1e-8, label="(mu, L) of a tilted matrix")
    close(np.array([kappa(np.diag([1.0, 10.0]))]), np.array([10.0]), atol=1e-9, label="kappa of diag(1, 10)")
    close(np.array([kappa(np.eye(3) * 2.0)]), np.array([1.0]), atol=1e-9, label="kappa of 2 I (a round bowl)")
    close(np.array([kappa(random_spd(4, 2.0, 200.0, seed=0))]), np.array([100.0]), atol=1e-6, label="kappa of a tilted matrix")


@check("axis_factors gives [0.7, 2.0] at step 0.3, [0.9, 0.0] at 0.1, [0.82, 0.8] at 0.18, and predicts a real step along tilted axes")
def test_axis_factors(sub):
    got = has(sub, "axis_factors")
    gd = has(sub, "gradient_descent")
    for step, want in [(0.3, [0.7, 2.0]), (0.1, [0.9, 0.0]), (0.18, [0.82, 0.8])]:
        out = np.asarray(got(A10, step))
        shape(out, (2,), label=f"axis_factors at step {step}")
        close(out, np.array(want), atol=1e-9, label=f"axis_factors at step {step}")

    A = random_spd(3, 1.0, 10.0, seed=1)
    w_star = np.array([0.5, -1.0, 2.0])
    b = A @ w_star
    grad = lambda w: A @ w - b
    step = 0.15
    _, vectors = np.linalg.eigh(A)
    factors = np.asarray(got(A, step))
    for k in range(3):
        w0 = w_star + vectors[:, k]
        w1 = gd(grad, w0, step, 1)[1]
        ratio = np.linalg.norm(w1 - w_star) / np.linalg.norm(w0 - w_star)
        if abs(ratio - factors[k]) > 1e-8:
            fail(f"along axis {k} one step shrank the miss by {ratio:.6g}, but axis_factors says {factors[k]:.6g}")


@check("A step of 1.9 / L converges and a step of 2.1 / L blows up")
def test_two_over_l(sub):
    rng_range = has(sub, "eigenvalue_range")
    gd = has(sub, "gradient_descent")
    A = random_spd(3, 1.0, 10.0, seed=2)
    w_star = np.array([1.0, -1.0, 0.5])
    b = A @ w_star
    grad = lambda w: A @ w - b
    _, L = rng_range(A)
    w0 = np.zeros(3)
    start = np.linalg.norm(w0 - w_star)

    safe = np.linalg.norm(gd(grad, w0, 1.9 / L, 80)[-1] - w_star)
    if not safe < 0.01 * start:
        fail(f"with step 1.9 / L the miss went from {start:.3g} to {safe:.3g} over 80 steps, expected it to shrink below {0.01 * start:.3g}")
    wild = np.linalg.norm(gd(grad, w0, 2.1 / L, 80)[-1] - w_star)
    if not wild > 100 * start:
        fail(f"with step 2.1 / L the miss went from {start:.3g} to {wild:.3g} over 80 steps, expected it to grow past {100 * start:.3g}")


def walk(sub, curvature: float, start, iters: int):
    """Gradient descent on diag(1, curvature) with bottom (1, 1), at step 1/curvature."""
    gd = has(sub, "gradient_descent")
    A = np.diag([1.0, curvature])
    w_star = np.array([1.0, 1.0])
    b = A @ w_star
    path = gd(lambda w: A @ w - b, np.array(start, dtype=float), 1.0 / curvature, iters)
    return path, w_star


@check("steps_to_tolerance returns the first row under the target or -1, and a round bowl takes 1 step, kappa = 10 takes 66, kappa = 100 takes 688")
def test_step_counts(sub):
    got = has(sub, "steps_to_tolerance")
    gd = has(sub, "gradient_descent")
    path = np.array([[4.0, 0.0], [2.0, 0.0], [1.0, 0.0], [0.5, 0.0], [0.25, 0.0]])
    equal(got(path, np.zeros(2), 0.2), 3, label="steps to reach 0.2 of the start (error below 0.8)")
    equal(got(path, np.zeros(2), 0.05), -1, label="steps to reach 0.05 of the start (never)")

    A = 3.0 * np.eye(3)
    w_star = np.array([0.5, -1.0, 2.0])
    b = A @ w_star
    round_path = gd(lambda w: A @ w - b, np.array([2.0, 2.0, -1.0]), 1.0 / 3.0, 5)
    equal(got(round_path, w_star, 1e-6), 1, label="steps for kappa = 1")

    p10, w_star = walk(sub, 10.0, (0.0, 1.0), 200)
    p100, _ = walk(sub, 100.0, (0.0, 1.0), 1500)
    s10 = got(p10, w_star, 1e-3)
    s100 = got(p100, w_star, 1e-3)
    equal(s10, 66, label="steps for kappa = 10")
    equal(s100, 688, label="steps for kappa = 100")
    ratio = s100 / s10
    if not 9.0 <= ratio <= 11.5:
        fail(f"kappa grew by 10 but the step count grew by {ratio:.2f}; expected about 10")


@check("heavy_ball starts at w0, follows the hand path (0.24, 1.5), (0.386, 1.25), and equals plain descent at momentum 0")
def test_heavy_ball(sub):
    got = has(sub, "heavy_ball")
    gd = has(sub, "gradient_descent")
    w0 = np.array([0.0, 0.0])
    path = got(grad10, w0, 0.1, 0.5, 3)
    shape(path, (4, 2), label="heavy_ball path")
    close(w0, np.zeros(2), label="the input w0, which heavy_ball must not change")
    want = np.array([[0.0, 0.0], [0.1, 1.0], [0.24, 1.5], [0.386, 1.25]])
    close(path, want, atol=1e-9, label="hand path")

    start = np.array([-1.0, 2.0])
    close(got(grad10, start, 0.07, 0.0, 12), gd(grad10, start, 0.07, 12), atol=1e-12, label="heavy_ball with momentum 0")

    A = random_spd(3, 1.0, 25.0, seed=2)
    w_star = np.array([0.5, -0.5, 0.2])
    b = A @ w_star
    grad = lambda w: A @ w - b
    w0 = np.array([2.0, 1.0, -1.0])
    step, momentum, iters = 0.02, 0.4, 15
    trajectory = got(grad, w0, step, momentum, iters)
    expected = np.empty_like(trajectory)
    expected[0] = w0
    w_prev, w = w0.copy(), w0.copy()
    for k in range(iters):
        w_next = w - step * grad(w) + momentum * (w - w_prev)
        w_prev, w = w, w_next
        expected[k + 1] = w
    close(trajectory, expected, atol=1e-10, label="heavy-ball iterates on a tilted bowl")


@check("nesterov lands at (0.235, 1.0) on step 2, equals plain descent at momentum 0, and follows its own recursion")
def test_nesterov(sub):
    got = has(sub, "nesterov")
    gd = has(sub, "gradient_descent")
    path = got(grad10, np.array([0.0, 0.0]), 0.1, 0.5, 2)
    shape(path, (3, 2), label="nesterov path")
    close(path[1], np.array([0.1, 1.0]), atol=1e-12, label="first step, which has no past to look ahead along")
    close(path[2], np.array([0.235, 1.0]), atol=1e-9, label="second step, with the gradient taken at the look-ahead point")

    start = np.array([-1.0, 2.0])
    close(got(grad10, start, 0.07, 0.0, 12), gd(grad10, start, 0.07, 12), atol=1e-12, label="nesterov with momentum 0")

    A = random_spd(3, 1.0, 25.0, seed=3)
    w_star = np.array([-0.3, 0.1, 0.4])
    b = A @ w_star
    grad = lambda w: A @ w - b
    w0 = np.array([1.5, -2.0, 0.5])
    step, momentum, iters = 1.0 / 25.0, 0.6, 15
    trajectory = got(grad, w0, step, momentum, iters)
    expected = np.empty_like(trajectory)
    expected[0] = w0
    w_prev, w = w0.copy(), w0.copy()
    for k in range(iters):
        y = w + momentum * (w - w_prev)
        w_next = y - step * grad(y)
        w_prev, w = w, w_next
        expected[k + 1] = w
    close(trajectory, expected, atol=1e-10, label="nesterov iterates on a tilted bowl")


@check("best_heavy_ball_params gives (0.231, 0.270) for mu = 1, L = 10, and every axis shrinks by about sqrt(momentum) per step")
def test_best_params(sub):
    got = has(sub, "best_heavy_ball_params")
    hb = has(sub, "heavy_ball")
    step, momentum = got(1.0, 10.0)
    close(np.array([step, momentum]), np.array([0.230886, 0.269874]), atol=1e-5, label="(step, momentum) for mu = 1, L = 10")
    step, momentum = got(1.0, 100.0)
    close(np.array([step, momentum]), np.array([4.0 / 121.0, 81.0 / 121.0]), atol=1e-9, label="(step, momentum) for mu = 1, L = 100")

    mu, L = 1.0, 100.0
    A = np.diag([mu, L])
    rate = np.sqrt(momentum)
    iters = 80
    for k, name in enumerate(["flat", "steep"]):
        e = np.zeros(2)
        e[k] = 1.0
        path = hb(lambda w: A @ w, e, step, momentum, iters)
        miss = np.linalg.norm(path[-1])
        # A repeated root gives a factor of about k * rate**k, so allow a polynomial factor.
        if not miss < 3.0 * iters * rate**iters:
            fail(f"on the {name} axis the miss after {iters} steps is {miss:.3g}, expected below {3.0 * iters * rate**iters:.3g} (shrink sqrt(momentum) = {rate:.3f} per step)")


@check("Heavy ball and Nesterov need far fewer steps than plain descent when kappa = 100")
def test_speedups(sub):
    gd = has(sub, "gradient_descent")
    hb = has(sub, "heavy_ball")
    nag = has(sub, "nesterov")
    best = has(sub, "best_heavy_ball_params")
    mu, L = 1.0, 100.0
    A = np.diag([mu, L])
    w_star = np.array([0.3, -0.2])
    b = A @ w_star
    grad = lambda w: A @ w - b
    w0 = w_star + np.array([1.0, 1.0])

    gd_steps = steps_to_millionth(gd(grad, w0, 1.0 / L, 3000), w_star)
    step, momentum = best(mu, L)
    hb_steps = steps_to_millionth(hb(grad, w0, step, momentum, 1000), w_star)
    speedup = gd_steps / hb_steps
    root_kappa = np.sqrt(L / mu)
    if not (0.4 * root_kappa <= speedup <= 3.0 * root_kappa):
        fail(f"plain descent took {gd_steps} steps and heavy ball took {hb_steps}, a speedup of {speedup:.1f}; expected the scale of sqrt(kappa) = {root_kappa:.0f}")

    nag_momentum = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    nag_steps = steps_to_millionth(nag(grad, w0, 1.0 / L, nag_momentum, 1000), w_star)
    if gd_steps < 5 * nag_steps:
        fail(f"plain descent took {gd_steps} steps and nesterov took {nag_steps}; expected nesterov to need at least 5 times fewer")
