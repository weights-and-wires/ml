"""Checks for the how-big-a-step exercise."""

import numpy as np

from mlcheck import check, close, fail, has, shape


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


@check("exact_line_search_step is the closed form and lowers the height most along its line")
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


@check("At the running bowl's start the best step is 101 / 1001, and the next gradient is perpendicular")
def test_line_search_hand_value_and_perpendicular(sub):
    got = has(sub, "exact_line_search_step")
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
