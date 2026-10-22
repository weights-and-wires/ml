"""Checks for the compute-optimal-split exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, has

E_, A_, B_, ALPHA, BETA = 1.69, 400.0, 1100.0, 0.34, 0.34


def loss(n, d):
    return E_ + A_ / n**ALPHA + B_ / d**BETA


@check("chinchilla_optimal spends exactly the budget and matches the chapter's example")
def test_budget(sub):
    co = has(sub, "chinchilla_optimal")
    for compute in (1e18, 1e20, 1e22, 1e24):
        n, d = co(compute)
        if n <= 0 or d <= 0:
            fail(f"chinchilla_optimal({compute:.0e}) returned a non-positive value")
        close(6.0 * n * d, compute, rtol=1e-9, label=f"6 N* D* at compute={compute:.0e}")
    n, d = co(1e20)
    close(n, 9.2225e8, rtol=1e-4, label="N* at 1e20")
    close(d, 1.8072e10, rtol=1e-4, label="D* at 1e20")


@check("N* is the lowest point along the constant-compute curve, and the two marginal gains match there")
def test_is_minimum(sub):
    co = has(sub, "chinchilla_optimal")
    for compute in (1e19, 1e21):
        n, d = co(compute)
        best = loss(n, d)
        for factor in (0.8, 0.95, 1.05, 1.25):
            other = loss(factor * n, compute / (6.0 * factor * n))
            if other <= best:
                fail(f"at compute={compute:.0e}, scaling N* by {factor} gave loss {other:.6f}, not above {best:.6f}")
        close(ALPHA * A_ / n**ALPHA, BETA * B_ / d**BETA, rtol=1e-9, label="alpha A / N**alpha versus beta B / D**beta")


@check("With equal exponents, D*/N* is about 19.6 at every budget and N* grows as the square root of compute")
def test_ratio(sub):
    co = has(sub, "chinchilla_optimal")
    for compute in (1e18, 1e21, 1e24):
        n, d = co(compute)
        almost(d / n, 19.5954, tol=1e-3, label=f"D*/N* at compute={compute:.0e}")
    n1, _ = co(1e20)
    n2, _ = co(1e22)
    almost(n2 / n1, 10.0, tol=1e-9, label="N* ratio for a hundredfold budget")


@check("grid_optimal returns a split that spends the budget")
def test_grid_budget(sub):
    go = has(sub, "grid_optimal")
    n, d = go(1e20)
    close(6.0 * n * d, 1e20, rtol=1e-9, label="6 N D from the grid")


@check("grid_optimal agrees with the closed form to within 1 percent")
def test_grid_agrees(sub):
    go, co = has(sub, "grid_optimal"), has(sub, "chinchilla_optimal")
    for compute in (1e18, 1e20, 1e23):
        ng, dg = go(compute)
        n, d = co(compute)
        if abs(ng / n - 1) > 0.01 or abs(dg / d - 1) > 0.01:
            fail(f"at compute={compute:.0e}, grid gave N={ng:.4e}, D={dg:.4e}; closed form gave N={n:.4e}, D={d:.4e}")


@check("lifetime_flops is 6 N D_train + 2 N served, and survives numpy int64 inputs")
def test_lifetime(sub):
    lf = has(sub, "lifetime_flops")
    almost(lf(1e9, 2e10, 0) / 1.2e20, 1.0, tol=1e-12, label="training only")
    almost(lf(1e9, 2e10, 1e12) / 2.12e21, 1.0, tol=1e-12, label="training plus serving 1e12 tokens")
    got = lf(np.int64(1_000_000_000), np.int64(20_000_000_000), np.int64(1_000_000_000_000))
    if not np.isfinite(got) or abs(got / 2.12e21 - 1) > 1e-12:
        fail(f"lifetime_flops with numpy int64 inputs returned {got!r}, expected 2.12e21. Convert to float first.")
    almost(lf(1e9, 2e10, 2e12) - lf(1e9, 2e10, 1e12), 2e21, tol=1e9, label="each served token costs 2 N")


@check("A half-size model trained to the same loss is cheaper overall once enough tokens are served")
def test_break_even(sub):
    co, lf = has(sub, "chinchilla_optimal"), has(sub, "lifetime_flops")
    n_big, d_big = co(1.2e20)
    n_small = n_big / 2
    d_small = (B_ / (loss(n_big, d_big) - E_ - A_ / n_small**ALPHA)) ** (1.0 / BETA)
    if lf(n_big, d_big, 0) >= lf(n_small, d_small, 0):
        fail("with nothing served, the balanced model should be the cheaper one overall")
    served = (6 * n_small * d_small - 6 * n_big * d_big) / (2 * (n_big - n_small))
    almost(served / 2.8553e10, 1.0, tol=1e-3, label="break-even served tokens, relative to 2.8553e10")
    close(lf(n_big, d_big, served), lf(n_small, d_small, served), rtol=1e-9, label="equal lifetime flops at break-even")
    if lf(n_small, d_small, 1e12) >= lf(n_big, d_big, 1e12):
        fail("after serving 1e12 tokens the half-size model should be the cheaper one")
