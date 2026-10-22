"""Checks for the backward-walking engine."""

import numpy as np

from mlcheck import check, close, fail, has


@check("central_difference recovers the slope of 3a at a = 3 and of a*a at a = 3")
def test_central_difference(sub):
    cd = has(sub, "central_difference")
    close(cd(lambda a: 3.0 * a, 3.0), 3.0, atol=1e-6, label="slope of 3a")
    close(cd(lambda a: a * a, 3.0), 6.0, atol=1e-6, label="slope of a*a")
    close(cd(lambda a: 3.0 * a, 3.0, h=0.001), 3.0, atol=1e-6, label="slope of 3a with h = 0.001")


@check("central_difference is two-sided: its error on a**3 at 2 is far below a one-sided nudge")
def test_two_sided(sub):
    cd = has(sub, "central_difference")
    got = cd(lambda a: a**3, 2.0, h=1e-3)
    one_sided = ((2.0 + 1e-3) ** 3 - 2.0**3) / 1e-3
    err, err_one = abs(got - 12.0), abs(one_sided - 12.0)
    if err > 1e-5:
        fail(f"error {err:.2e} at h = 0.001; a two-sided nudge should be within 1e-5 of 12")
    if err >= err_one:
        fail("your estimate is no better than a one-sided nudge")


@check("__mul__ sends each input the incoming slope times the other input")
def test_mul(sub):
    Tensor = has(sub, "Tensor")
    a, b = Tensor(3.0), Tensor(2.0)
    p = a * b
    close(p.value, 6.0, label="value of a*b")
    p.backward()
    close(a.grad, 2.0, label="slope in a")
    close(b.grad, 3.0, label="slope in b")
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=4), rng.normal(size=4)
    X, Y = Tensor(x), Tensor(y)
    (X * Y).backward()
    close(X.grad, y, atol=1e-12, label="slope in X when the output slope is ones")
    close(Y.grad, x, atol=1e-12, label="slope in Y when the output slope is ones")


@check("c = a*b + a at a = 3, b = 2 gives slope 3 in a and 3 in b")
def test_running_example(sub):
    Tensor = has(sub, "Tensor")
    cd = has(sub, "central_difference")
    a, b = Tensor(3.0), Tensor(2.0)
    c = a * b + a
    c.backward()
    close(c.value, 9.0, label="value of c")
    close(a.grad, 3.0, label="slope in a")
    close(b.grad, 3.0, label="slope in b")
    fd_a = cd(lambda av: float((Tensor(av) * Tensor(2.0) + Tensor(av)).value), 3.0)
    fd_b = cd(lambda bv: float((Tensor(3.0) * Tensor(bv) + Tensor(3.0)).value), 2.0)
    close(a.grad, fd_a, atol=1e-6, label="slope in a against central_difference")
    close(b.grad, fd_b, atol=1e-6, label="slope in b against central_difference")


@check("A value used twice collects both contributions: a*a at 3 gives 6")
def test_accumulation(sub):
    Tensor = has(sub, "Tensor")
    a = Tensor(3.0)
    (a * a).backward()
    close(a.grad, 6.0, label="slope of a*a at 3 (3 + 3, not 3)")


@check("The diamond b = a*a, c = a + a, d = b + c gives slope 6 at a = 2")
def test_diamond(sub):
    Tensor = has(sub, "Tensor")
    a = Tensor(2.0)
    b = a * a
    c = a + a
    d = b + c
    d.backward()
    close(d.value, 8.0, label="value of d")
    close(a.grad, 6.0, label="slope in a through both routes")


@check("A node is finished before it forwards its slope: the diamond hung under x = 1, a = 2x")
def test_order(sub):
    Tensor = has(sub, "Tensor")
    x = Tensor(1.0)
    a = x * 2.0
    b = a * a
    c = a + a
    d = b + c
    d.backward()
    close(a.grad, 6.0, label="slope in a")
    close(x.grad, 12.0, label="slope in x (8 means a forwarded its slope before c arrived)")


@check("Mixed graphs match central_difference at several points")
def test_random_graph(sub):
    Tensor = has(sub, "Tensor")
    cd = has(sub, "central_difference")

    def build(av, bv):
        a, b = Tensor(av), Tensor(bv)
        p = a * b
        q = p + a
        r = q * q + b * a
        return a, b, r

    for av, bv in [(0.7, -1.3), (2.0, 0.5), (-1.1, 1.9)]:
        a, b, r = build(av, bv)
        r.backward()
        fd_a = cd(lambda t: float(build(t, bv)[2].value), av)
        fd_b = cd(lambda t: float(build(av, t)[2].value), bv)
        close(a.grad, fd_a, atol=1e-5, label=f"slope in a at ({av}, {bv})")
        close(b.grad, fd_b, atol=1e-5, label=f"slope in b at ({av}, {bv})")
