"""Checks for the step size and long valleys exercise."""

import numpy as np

from mlcheck import check, close, equal, fail, has, shape


def random_spd(d: int, mu: float, L: float, seed: int) -> np.ndarray:
    """A symmetric matrix, tilted away from the coordinate axes, with extreme eigenvalues mu and L."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eigs = np.concatenate([[mu], rng.uniform(mu, L, size=d - 2), [L]])
    return (q * eigs) @ q.T


@check("eigenvalue_range returns (mu, L), also for a tilted matrix")
def test_eigenvalue_range(sub):
    got = has(sub, "eigenvalue_range")
    mu, L = got(np.diag([1.0, 10.0]))
    close(np.array([mu, L]), np.array([1.0, 10.0]), atol=1e-9, label="(mu, L) of diag(1, 10)")
    mu, L = got(random_spd(4, 2.0, 50.0, seed=0))
    close(np.array([mu, L]), np.array([2.0, 50.0]), atol=1e-8, label="(mu, L) of a tilted matrix")


@check("axis_factors gives [0.7, 2.0] at step 0.3, [0.9, 0.0] at 0.1, and [0.82, 0.8] at 0.18")
def test_axis_factors_hand_values(sub):
    got = has(sub, "axis_factors")
    A = np.diag([1.0, 10.0])
    for step, want in [(0.3, [0.7, 2.0]), (0.1, [0.9, 0.0]), (0.18, [0.82, 0.8])]:
        out = np.asarray(got(A, step))
        shape(out, (2,), label=f"axis_factors at step {step}")
        close(out, np.array(want), atol=1e-9, label=f"axis_factors at step {step}")


@check("axis_factors predicts the real shrink along each tilted axis")
def test_factors_match_a_real_step(sub):
    got = has(sub, "axis_factors")
    gd = has(sub, "gradient_descent")
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


@check("exact_line_search_step is the closed form, equals 101 / 1001 at the running start, and the next gradient is perpendicular")
def test_exact_line_search_step(sub):
    got = has(sub, "exact_line_search_step")
    A = random_spd(3, 1.0, 20.0, seed=3)
    b = np.array([0.4, -0.2, 0.1])
    w = np.array([2.0, -1.0, 0.5])
    g = A @ w - b

    alpha = got(A, g)
    want = float((g @ g) / (g @ A @ g))
    if abs(alpha - want) > 1e-10 * max(1.0, abs(want)):
        fail(f"got alpha = {alpha:.8g}, expected the closed form {want:.8g}")

    f = lambda a: 0.5 * (w - a * g) @ A @ (w - a * g) - b @ (w - a * g)
    base = f(alpha)
    for delta in (-1e-3, 1e-3):
        if f(alpha + delta) < base - 1e-9:
            fail(f"alpha = {alpha:.6g} is not the lowest point: f(alpha + {delta:g}) = {f(alpha + delta):.8g} < {base:.8g}")

    A = np.diag([1.0, 10.0])
    b = np.array([1.0, 10.0])
    w = np.zeros(2)
    g = A @ w - b
    alpha = got(A, g)
    if abs(alpha - 101.0 / 1001.0) > 1e-10:
        fail(f"got alpha = {alpha:.8g}, expected 101 / 1001 = {101.0 / 1001.0:.8g}")
    g_next = A @ (w - alpha * g) - b
    dot = float(g_next @ g)
    if abs(dot) > 1e-9:
        fail(f"after the best step the new gradient should be perpendicular to the old one, but their dot product is {dot:.3g}")


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


@check("On the walk the ratios are 0.9 / sqrt(2) then 0.9, and along the flat axis exactly 1 - mu / L")
def test_ratios_on_walk(sub):
    got = has(sub, "contraction_ratios")
    path, w_star = walk(sub, 10.0, (0.0, 0.0), 6)
    out = np.asarray(got(path, w_star))
    want = np.array([0.9 / np.sqrt(2.0)] + [0.9] * 5)
    close(out, want, atol=1e-9, label="ratios along the walk from (0, 0)")
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
