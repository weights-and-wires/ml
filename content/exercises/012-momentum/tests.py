"""Checks for the momentum exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape

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


@check("heavy_ball returns iters + 1 rows, starts at w0, and its first step is a plain step")
def test_heavy_ball_first_step(sub):
    got = has(sub, "heavy_ball")
    w0 = np.array([0.0, 0.0])
    path = got(grad10, w0, 0.1, 0.5, 4)
    shape(path, (5, 2), label="heavy_ball path")
    close(path[0], np.array([0.0, 0.0]), label="row 0")
    close(path[1], np.array([0.1, 1.0]), atol=1e-12, label="first step, which has no past to remember")
    close(w0, np.zeros(2), label="the input w0, which heavy_ball must not change")


@check("heavy_ball follows the hand path (0.24, 1.5), (0.386, 1.25) and its own recursion exactly")
def test_heavy_ball_path(sub):
    got = has(sub, "heavy_ball")
    path = got(grad10, np.array([0.0, 0.0]), 0.1, 0.5, 3)
    want = np.array([[0.0, 0.0], [0.1, 1.0], [0.24, 1.5], [0.386, 1.25]])
    close(path, want, atol=1e-9, label="hand path")

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


@check("nesterov lands at (0.235, 1.0) on step 2 and follows its own recursion exactly")
def test_nesterov_path(sub):
    got = has(sub, "nesterov")
    path = got(grad10, np.array([0.0, 0.0]), 0.1, 0.5, 2)
    shape(path, (3, 2), label="nesterov path")
    close(path[1], np.array([0.1, 1.0]), atol=1e-12, label="first step, which has no past to look ahead along")
    close(path[2], np.array([0.235, 1.0]), atol=1e-9, label="second step, with the gradient taken at the look-ahead point")

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


@check("With momentum 0, heavy_ball and nesterov both equal plain gradient descent")
def test_zero_momentum(sub):
    hb = has(sub, "heavy_ball")
    nag = has(sub, "nesterov")
    gd = has(sub, "gradient_descent")
    w0 = np.array([-1.0, 2.0])
    plain = gd(grad10, w0, 0.07, 12)
    close(hb(grad10, w0, 0.07, 0.0, 12), plain, atol=1e-12, label="heavy_ball with momentum 0")
    close(nag(grad10, w0, 0.07, 0.0, 12), plain, atol=1e-12, label="nesterov with momentum 0")


@check("best_heavy_ball_params gives (0.231, 0.270) for mu = 1, L = 10 and (4/121, 81/121) for L = 100")
def test_best_params(sub):
    got = has(sub, "best_heavy_ball_params")
    step, momentum = got(1.0, 10.0)
    close(np.array([step, momentum]), np.array([0.230886, 0.269874]), atol=1e-5, label="(step, momentum) for mu = 1, L = 10")
    step, momentum = got(1.0, 100.0)
    close(np.array([step, momentum]), np.array([4.0 / 121.0, 81.0 / 121.0]), atol=1e-9, label="(step, momentum) for mu = 1, L = 100")


@check("With the best settings every axis shrinks by about sqrt(momentum) per step, flat or steep")
def test_best_params_equal_rate(sub):
    hb = has(sub, "heavy_ball")
    best = has(sub, "best_heavy_ball_params")
    mu, L = 1.0, 100.0
    A = np.diag([mu, L])
    step, momentum = best(mu, L)
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


@check("Heavy ball with the best settings needs far fewer steps than plain descent when kappa = 100")
def test_heavy_ball_speedup(sub):
    gd = has(sub, "gradient_descent")
    hb = has(sub, "heavy_ball")
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


@check("Nesterov with step 1 / L and momentum (sqrt(L) - sqrt(mu)) / (sqrt(L) + sqrt(mu)) also beats plain descent")
def test_nesterov_speedup(sub):
    gd = has(sub, "gradient_descent")
    nag = has(sub, "nesterov")
    mu, L = 1.0, 100.0
    A = np.diag([mu, L])
    w_star = np.array([0.3, -0.2])
    b = A @ w_star
    grad = lambda w: A @ w - b
    w0 = w_star + np.array([1.0, 1.0])

    gd_steps = steps_to_millionth(gd(grad, w0, 1.0 / L, 3000), w_star)
    momentum = (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))
    nag_steps = steps_to_millionth(nag(grad, w0, 1.0 / L, momentum, 1000), w_star)
    if gd_steps < 5 * nag_steps:
        fail(f"plain descent took {gd_steps} steps and nesterov took {nag_steps}; expected nesterov to need at least 5 times fewer")
