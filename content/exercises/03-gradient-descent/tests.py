"""Checks for the gradient descent exercise."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


def random_spd(d: int, mu: float, L: float, seed: int) -> np.ndarray:
    """A symmetric matrix with prescribed extreme eigenvalues mu and L."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigs = np.concatenate([[mu], rng.uniform(mu, L, size=d - 2), [L]])
    return (q * eigs) @ q.T


@check("quadratic_loss_and_grad's gradient matches a finite difference")
def test_gradient_matches_finite_difference(sub):
    got = has(sub, "quadratic_loss_and_grad")
    A = random_spd(4, 1.0, 10.0, seed=0)
    b = np.array([1.0, -2.0, 0.5, 0.3])
    w = np.array([0.2, -0.1, 0.4, 0.0])

    loss, grad = got(A, b, w)
    want_loss = float(0.5 * w @ A @ w - b @ w)
    if abs(loss - want_loss) > 1e-8:
        fail(f"loss is {loss:.6g}, expected {want_loss:.6g}")

    numeric_grad = finite_difference(lambda x: 0.5 * x @ A @ x - b @ x, w)
    close(grad, numeric_grad, atol=1e-5, label="analytic vs finite-difference gradient")


@check("exact_line_search_step returns the closed-form optimal step")
def test_exact_line_search_step(sub):
    got = has(sub, "exact_line_search_step")
    A = random_spd(3, 1.0, 20.0, seed=1)
    b = np.array([0.4, -0.2, 0.1])
    w = np.array([2.0, -1.0, 0.5])
    g = A @ w - b  # the gradient at w, which is what exact line search steps along

    alpha = got(A, g)
    want = float((g @ g) / (g @ A @ g))
    if abs(alpha - want) > 1e-10 * max(1.0, abs(want)):
        fail(f"got alpha={alpha:.8g}, expected the closed form {want:.8g}")

    # It should genuinely minimise f(w - alpha g) along this direction: a
    # perturbation in either direction must not do better.
    f = lambda a: 0.5 * (w - a * g) @ A @ (w - a * g) - b @ (w - a * g)
    base = f(alpha)
    for delta in (-1e-3, 1e-3):
        if f(alpha + delta) < base - 1e-9:
            fail(
                f"alpha={alpha:.6g} is not a local minimiser of the line search: "
                f"f(alpha + {delta:g}) = {f(alpha + delta):.8g} < f(alpha) = {base:.8g}"
            )


@check("gradient_descent diverges when the step exceeds 2 / L")
def test_divergence_above_two_over_l(sub):
    got = has(sub, "gradient_descent")
    mu, L = 1.0, 100.0
    A = np.diag([mu, L])
    w_star = np.array([0.3, -0.2])
    b = A @ w_star
    grad = lambda w: A @ w - b
    w0 = w_star + np.array([1.0, 1.0])

    trajectory = got(grad, w0, 2.1 / L, 60)
    shape(trajectory, (61, 2), label="gradient_descent trajectory")
    errors = np.linalg.norm(trajectory - w_star, axis=1)
    if not (errors[-1] > 50 * errors[0]):
        fail(
            "a step size of 2.1 / L should make the iterates diverge, but the "
            f"error went from {errors[0]:.3g} to {errors[-1]:.3g} over 60 steps"
        )


@check("Fixed-step gradient descent contracts by exactly 1 - mu / L")
def test_contraction_factor(sub):
    got = has(sub, "gradient_descent")
    mu, L = 2.0, 200.0
    A = np.diag([mu, L])
    w_star = np.array([0.1, 0.4])
    b = A @ w_star
    grad = lambda w: A @ w - b
    # Displace w0 from w* only along the slow (minimum-eigenvalue) direction,
    # so gradient descent with a fixed step decays geometrically with no
    # mixing from the fast direction, and the rate 1 - mu / L is exact.
    w0 = w_star + np.array([1.0, 0.0])

    trajectory = got(grad, w0, 1.0 / L, 40)
    errors = np.linalg.norm(trajectory - w_star, axis=1)
    ratios = errors[1:] / errors[:-1]
    close(
        ratios,
        np.full(40, 1.0 - mu / L),
        rtol=1e-8,
        atol=1e-10,
        label="per-step contraction ratio against 1 - mu / L",
    )


@check("heavy_ball follows its own recursion exactly")
def test_heavy_ball_recursion(sub):
    got = has(sub, "heavy_ball")
    A = random_spd(3, 1.0, 25.0, seed=2)
    w_star = np.array([0.5, -0.5, 0.2])
    b = A @ w_star
    grad = lambda w: A @ w - b
    w0 = np.array([2.0, 1.0, -1.0])
    step, momentum, iters = 0.02, 0.4, 15

    trajectory = got(grad, w0, step, momentum, iters)
    shape(trajectory, (iters + 1, 3), label="heavy_ball trajectory")

    expected = np.empty_like(trajectory)
    expected[0] = w0
    w_prev, w = w0.copy(), w0.copy()
    for k in range(iters):
        w_next = w - step * grad(w) + momentum * (w - w_prev)
        w_prev, w = w, w_next
        expected[k + 1] = w
    close(trajectory, expected, atol=1e-10, label="heavy-ball iterates")


@check("nesterov follows its own recursion exactly")
def test_nesterov_recursion(sub):
    got = has(sub, "nesterov")
    A = random_spd(3, 1.0, 25.0, seed=3)
    w_star = np.array([-0.3, 0.1, 0.4])
    b = A @ w_star
    grad = lambda w: A @ w - b
    w0 = np.array([1.5, -2.0, 0.5])
    step, momentum, iters = 1.0 / 25.0, 0.6, 15

    trajectory = got(grad, w0, step, momentum, iters)
    shape(trajectory, (iters + 1, 3), label="nesterov trajectory")

    expected = np.empty_like(trajectory)
    expected[0] = w0
    w_prev, w = w0.copy(), w0.copy()
    for k in range(iters):
        y = w + momentum * (w - w_prev)
        w_next = y - step * grad(y)
        w_prev, w = w, w_next
        expected[k + 1] = w
    close(trajectory, expected, atol=1e-10, label="nesterov iterates")


@check("Heavy-ball momentum reaches a target accuracy in roughly sqrt(kappa) fewer iterations")
def test_momentum_speedup(sub):
    gd = has(sub, "gradient_descent")
    hb = has(sub, "heavy_ball")

    mu, L = 1.0, 100.0
    kappa = L / mu
    A = np.diag([mu, L])
    w_star = np.array([0.3, -0.2])
    b = A @ w_star
    grad = lambda w: A @ w - b
    w0 = w_star + np.array([1.0, 1.0])
    target = 1e-6 * np.linalg.norm(w0 - w_star)

    def iterations_to_target(trajectory: np.ndarray) -> int:
        errors = np.linalg.norm(trajectory - w_star, axis=1)
        hit = np.nonzero(errors < target)[0]
        if hit.size == 0:
            fail("the method did not reach the target accuracy within the iteration budget")
        return int(hit[0])

    gd_trajectory = gd(grad, w0, 1.0 / L, 20000)
    gd_iters = iterations_to_target(gd_trajectory)

    hb_step = (2.0 / (np.sqrt(L) + np.sqrt(mu))) ** 2
    hb_momentum = ((np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu))) ** 2
    hb_trajectory = hb(grad, w0, hb_step, hb_momentum, 2000)
    hb_iters = iterations_to_target(hb_trajectory)

    speedup = gd_iters / hb_iters
    expected = np.sqrt(kappa)
    if not (0.4 * expected <= speedup <= 3.0 * expected):
        fail(
            f"gradient descent took {gd_iters} iterations and heavy ball took "
            f"{hb_iters}, a speedup of {speedup:.2f}x; expected roughly "
            f"sqrt(kappa) = {expected:.2f}x"
        )
