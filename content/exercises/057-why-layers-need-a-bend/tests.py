"""Checks for the exercise on bends between layers."""

import numpy as np

from mlcheck import check, close, fail, finite_difference, has, shape


@check("relu keeps positives, zeroes negatives, and keeps the shape")
def test_relu(sub):
    relu = has(sub, "relu")
    out = relu(np.array([2.0, -3.0, -2.0]))
    close(out, np.array([2.0, 0.0, 0.0]), atol=1e-12, label="relu on [2, -3, -2]")
    m = np.array([[1.5, -0.5], [0.0, -7.0]])
    out = relu(m)
    shape(out, (2, 2), label="relu on a matrix")
    close(out, np.array([[1.5, 0.0], [0.0, 0.0]]), atol=1e-12, label="relu on a matrix")


@check("relu_grad is 1 where the input is positive and 0 where it is negative")
def test_relu_grad_values(sub):
    relu_grad = has(sub, "relu_grad")
    out = relu_grad(np.array([2.0, -3.0, -2.0]))
    if out.dtype.kind != "f":
        fail(f"relu_grad should return floats, got dtype {out.dtype}")
    close(out, np.array([1.0, 0.0, 0.0]), atol=1e-12, label="relu_grad on [2, -3, -2]")


@check("relu_grad matches finite differences away from the corner")
def test_relu_grad_fd(sub):
    relu, relu_grad = has(sub, "relu"), has(sub, "relu_grad")
    rng = np.random.default_rng(0)
    x = rng.normal(size=12) * 3.0
    x = np.where(np.abs(x) < 0.05, 0.3, x)
    fd = finite_difference(lambda v: float(np.sum(relu(v))), x)
    close(relu_grad(x), fd, atol=1e-5, label="relu_grad against finite differences")


@check("compose_affine reproduces the hand example: W = [3, 2], b = 7")
def test_compose_hand(sub):
    compose_affine = has(sub, "compose_affine")
    W, b = compose_affine(
        np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([1.0, 0.0]),
        np.array([[3.0, 1.0]]), np.array([4.0]),
    )
    shape(W, (1, 2), label="merged W")
    close(W, np.array([[3.0, 2.0]]), atol=1e-12, label="merged W")
    close(b, np.array([7.0]), atol=1e-12, label="merged b")


@check("The merged layer gives the same answers as the two layers in a row")
def test_compose_random(sub):
    compose_affine = has(sub, "compose_affine")
    rng = np.random.default_rng(1)
    W1, b1 = rng.normal(size=(4, 6)), rng.normal(size=4)
    W2, b2 = rng.normal(size=(3, 4)), rng.normal(size=3)
    W, b = compose_affine(W1, b1, W2, b2)
    shape(W, (3, 6), label="merged W")
    shape(b, (3,), label="merged b")
    X = rng.normal(size=(10, 6))
    two_steps = (X @ W1.T + b1) @ W2.T + b2
    one_step = X @ W.T + b
    close(one_step, two_steps, atol=1e-10, label="merged layer against two layers")


@check("A narrow middle layer limits the rank of the merged weights")
def test_compose_rank(sub):
    compose_affine = has(sub, "compose_affine")
    rng = np.random.default_rng(2)
    W1, b1 = rng.normal(size=(2, 6)), rng.normal(size=2)
    W2, b2 = rng.normal(size=(5, 2)), rng.normal(size=5)
    W, _ = compose_affine(W1, b1, W2, b2)
    rank = np.linalg.matrix_rank(W)
    if rank > 2:
        fail(f"merged weights have rank {rank}, but a middle layer of 2 units allows at most 2")
    if rank < 2:
        fail(f"merged weights have rank {rank}; random layers with a middle width of 2 should reach 2")


@check("Two ReLU units turn XOR into a problem for one straight line")
def test_xor(sub):
    relu = has(sub, "relu")
    points = np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    on = np.array([False, False, True, True])
    W_hidden = np.array([[1.0, -1.0], [-1.0, 1.0]])
    hidden = relu(points @ W_hidden.T)
    score = hidden.sum(axis=1) - 0.5
    if not np.all((score > 0) == on):
        fail(f"h1 + h2 - 0.5 should be positive only for the lamp-on points, got {score}")


@check("A ReLU between two layers changes the answer, so they no longer merge")
def test_relu_breaks_merge(sub):
    relu = has(sub, "relu")
    compose_affine = has(sub, "compose_affine")
    W1, b1 = np.array([[1.0, 0.0], [0.0, 2.0]]), np.array([1.0, 0.0])
    W2, b2 = np.array([[3.0, 1.0]]), np.array([4.0])
    x = np.array([2.0, -3.0])
    W, b = compose_affine(W1, b1, W2, b2)
    merged = W @ x + b
    with_bend = W2 @ relu(W1 @ x + b1) + b2
    close(with_bend, np.array([13.0]), atol=1e-12, label="two layers with a ReLU between them")
    if np.allclose(merged, with_bend):
        fail("the merged layer should disagree with the network that has a ReLU in the middle")
