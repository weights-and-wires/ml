"""Checks for the backward-walking engine with array rules."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


@check("central_difference is two-sided: it recovers the slope of 3a and a*a, and beats a one-sided nudge on a**3")
def test_central_difference(sub):
    cd = has(sub, "central_difference")
    close(cd(lambda a: 3.0 * a, 3.0), 3.0, atol=1e-6, label="slope of 3a")
    close(cd(lambda a: a * a, 3.0), 6.0, atol=1e-6, label="slope of a*a")
    close(cd(lambda a: 3.0 * a, 3.0, h=0.001), 3.0, atol=1e-6, label="slope of 3a with h = 0.001")
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


@check("c = a*b + a gives slope 3 in a and 3 in b, and a value used twice collects both slopes: a*a at 3 gives 6")
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
    a = Tensor(3.0)
    (a * a).backward()
    close(a.grad, 6.0, label="slope of a*a at 3 (3 + 3, not 3)")


@check("The diamond gives slope 6 at a = 2, and hung under x = 1 a node is finished before it forwards (slope 12)")
def test_diamond_and_order(sub):
    Tensor = has(sub, "Tensor")
    a = Tensor(2.0)
    b = a * a
    c = a + a
    d = b + c
    d.backward()
    close(d.value, 8.0, label="value of d")
    close(a.grad, 6.0, label="slope in a through both routes")
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


@check("unbroadcast sums a (2, 2) slope down to a (2,) bias slope, and (4, 3) to (4, 1) and ()")
def test_unbroadcast(sub):
    unbroadcast = has(sub, "unbroadcast")
    g = np.array([[1.0, 2.0], [3.0, 4.0]])
    got = unbroadcast(g, (2,))
    shape(got, (2,), label="bias slope")
    close(got, np.array([4.0, 6.0]), label="column sums of [[1, 2], [3, 4]]")
    rng = np.random.default_rng(0)
    big = rng.normal(size=(4, 3))
    got = unbroadcast(big, (4, 1))
    shape(got, (4, 1), label="slope for a (4, 1) value")
    close(got, big.sum(axis=1, keepdims=True), label="row sums")
    same = unbroadcast(big, (4, 3))
    close(same, big, label="a shape that was not broadcast comes back unchanged")
    scalar = unbroadcast(big, ())
    equal(np.shape(scalar), (), label="shape of a scalar slope")
    close(scalar, big.sum(), label="slope for a scalar value")


@check("A bias added to every row, and a row-shaped multiplier, match finite differences (add and mul use unbroadcast)")
def test_broadcast(sub):
    Tensor = has(sub, "Tensor")
    rng = np.random.default_rng(1)
    A = rng.normal(size=(5, 3))
    bias = rng.normal(size=3)
    weights = rng.normal(size=(5, 3))

    def total(bv):
        return float(((Tensor(A) + Tensor(bv)) * Tensor(weights)).sum().value)

    fd = finite_difference(total, bias)
    a, b = Tensor(A), Tensor(bias)
    ((a + b) * Tensor(weights)).sum().backward()
    shape(b.grad, (3,), label="bias slope")
    close(b.grad, fd, atol=1e-5, label="bias slope against finite differences")
    close(a.grad, weights, atol=1e-10, label="slope in the matrix")

    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 3))
    row = rng.normal(size=(1, 3))
    weights = rng.normal(size=(4, 3))

    def loss(av, rv):
        return float(((Tensor(av) * Tensor(rv)) * Tensor(weights)).sum().value)

    a, r = Tensor(A), Tensor(row)
    ((a * r) * Tensor(weights)).sum().backward()
    shape(r.grad, (1, 3), label="slope of the row")
    close(r.grad, finite_difference(lambda v: loss(A, v), row), atol=1e-5, label="row slope")
    close(a.grad, finite_difference(lambda v: loss(v, row), A), atol=1e-5, label="matrix slope")


@check("__matmul__ reproduces the 2x3 times 3x2 worked example, and matches finite differences on a non-square product")
def test_matmul(sub):
    Tensor = has(sub, "Tensor")
    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = Tensor([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    Z = A @ B
    close(Z.value, np.array([[5.0, 11.0], [14.0, 23.0]]), label="value of A @ B")
    Zbar = np.array([[1.0, 2.0], [3.0, 4.0]])
    (Z * Tensor(Zbar)).sum().backward()
    shape(A.grad, (2, 3), label="slope in A")
    shape(B.grad, (3, 2), label="slope in B")
    close(A.grad, np.array([[1.0, 4.0, 6.0], [3.0, 10.0, 12.0]]), label="slope in A (Zbar times B transposed)")
    close(B.grad, np.array([[13.0, 18.0], [17.0, 24.0], [21.0, 30.0]]), label="slope in B (A transposed times Zbar)")

    rng = np.random.default_rng(3)
    A = rng.normal(size=(4, 5))
    B = rng.normal(size=(5, 3))
    W = rng.normal(size=(4, 3))

    def loss(av, bv):
        return float(((Tensor(av) @ Tensor(bv)) * Tensor(W)).sum().value)

    a, b = Tensor(A), Tensor(B)
    ((a @ b) * Tensor(W)).sum().backward()
    close(a.grad, finite_difference(lambda v: loss(v, B), A), atol=1e-5, label="slope in A")
    close(b.grad, finite_difference(lambda v: loss(A, v), B), atol=1e-5, label="slope in B")


@check("relu blocks the slope where the input is negative and passes it where positive")
def test_relu_gate(sub):
    Tensor = has(sub, "Tensor")
    x = Tensor([-1.0, 2.0, 0.5, -3.0])
    y = x.relu()
    close(y.value, np.array([0.0, 2.0, 0.5, 0.0]), label="relu value")
    (y * Tensor([5.0, 7.0, 1.0, 2.0])).sum().backward()
    close(x.grad, np.array([0.0, 7.0, 1.0, 0.0]), label="slope after the gates")


@check("A whole layer, relu(X @ W + b), matches finite differences, and tensors used twice collect both slopes")
def test_layer(sub):
    Tensor = has(sub, "Tensor")
    rng = np.random.default_rng(4)
    X = rng.normal(size=(6, 3))
    W = rng.normal(size=(3, 4))
    b = rng.normal(size=4) * 0.3
    C = rng.normal(size=(6, 4))

    def loss(Wv, bv):
        return float((((Tensor(X) @ Tensor(Wv)) + Tensor(bv)).relu() * Tensor(C)).sum().value)

    Wt, bt = Tensor(W), Tensor(b)
    ((((Tensor(X) @ Wt) + bt).relu()) * Tensor(C)).sum().backward()
    shape(Wt.grad, (3, 4), label="W slope")
    shape(bt.grad, (4,), label="b slope")
    close(Wt.grad, finite_difference(lambda v: loss(v, b), W), atol=1e-4, label="W slope against finite differences")
    close(bt.grad, finite_difference(lambda v: loss(W, v), b), atol=1e-4, label="b slope against finite differences")

    x = Tensor([-1.0, 2.0, 0.5])
    (x.relu() + x).sum().backward()
    close(x.grad, np.array([1.0, 2.0, 2.0]), label="slope in x for relu(x) + x (relu passes 1 where x > 0, the add passes 1 everywhere)")
    A = rng.normal(size=(3, 3))
    W2 = rng.normal(size=(3, 3))
    a = Tensor(A)
    ((a @ a) * Tensor(W2)).sum().backward()
    fd = finite_difference(lambda v: float(((Tensor(v) @ Tensor(v)) * Tensor(W2)).sum().value), A)
    close(a.grad, fd, atol=1e-5, label="slope of A @ A (A is both factors, so both rules must use +=)")
