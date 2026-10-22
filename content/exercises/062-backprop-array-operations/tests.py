"""Checks for the array rules."""

import numpy as np

from mlcheck import check, close, equal, fail, finite_difference, has, shape


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


@check("A bias added to every row gets one slope per column, matching finite differences")
def test_broadcast_add(sub):
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


@check("Multiplying by a row-shaped value matches finite differences for both operands")
def test_broadcast_mul(sub):
    Tensor = has(sub, "Tensor")
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


@check("__matmul__ reproduces the 2x3 times 3x2 worked example, with gradients of the right shapes")
def test_matmul_example(sub):
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


@check("__matmul__ gradients match finite differences on a non-square product")
def test_matmul_fd(sub):
    Tensor = has(sub, "Tensor")
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


@check("A whole layer, relu(X @ W + b), matches finite differences for W and b")
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


@check("A tensor used by relu and by an add, or on both sides of a square product, collects both slopes")
def test_accumulation(sub):
    Tensor = has(sub, "Tensor")
    x = Tensor([-1.0, 2.0, 0.5])
    (x.relu() + x).sum().backward()
    close(x.grad, np.array([1.0, 2.0, 2.0]), label="slope in x for relu(x) + x (relu passes 1 where x > 0, the add passes 1 everywhere)")
    rng = np.random.default_rng(5)
    A = rng.normal(size=(3, 3))
    W = rng.normal(size=(3, 3))
    a = Tensor(A)
    ((a @ a) * Tensor(W)).sum().backward()
    fd = finite_difference(lambda v: float(((Tensor(v) @ Tensor(v)) * Tensor(W)).sum().value), A)
    close(a.grad, fd, atol=1e-5, label="slope of A @ A (A is both factors, so both rules must use +=)")
