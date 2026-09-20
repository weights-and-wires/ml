"""A small reverse-mode autodiff engine, and an MLP loss built from it.

Every op below builds a new Tensor whose `_children` point at the tensors it
was computed from, and whose `_backward_fn` propagates `out.grad` into each
child's `.grad` using that op's vector-Jacobian product. `backward()` then
runs a topological sort over `_children` and calls every `_backward_fn` in
reverse order, so a tensor's gradient is fully accumulated before it
propagates further.

`_unbroadcast` is provided: it undoes whatever shape broadcasting numpy
performed in the forward pass, so a bias vector added to a batch of
activations gets a gradient of the right shape. You do not need to change it.
"""

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum grad back down to shape, undoing whatever numpy broadcast."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """A node in the computational graph: a value, a gradient, and a backward closure."""

    def __init__(self, value, children=()):
        self.value = np.asarray(value, dtype=float)
        self.grad = np.zeros_like(self.value)
        self._children = children
        self._backward_fn = lambda: None

    def __repr__(self) -> str:
        return f"Tensor({self.value!r})"

    @staticmethod
    def _coerce(other):
        return other if isinstance(other, Tensor) else Tensor(other)

    def __add__(self, other):
        """Elementwise addition, with numpy broadcasting.

        Build `out` from `self.value + other.value`, and give it a
        `_backward_fn` that adds `out.grad`, unbroadcast to each operand's
        own shape, into that operand's `.grad`.
        """
        raise NotImplementedError("Tensor.__add__")

    def __mul__(self, other):
        """Elementwise multiplication, with numpy broadcasting.

        The gradient with respect to one operand is `out.grad` times the
        *other* operand's value (the product rule), unbroadcast to that
        operand's own shape.
        """
        raise NotImplementedError("Tensor.__mul__")

    def __matmul__(self, other):
        """Matrix multiplication of two 2-dimensional tensors.

        For `out = self @ other`, the vector-Jacobian products are
        `self.grad += out.grad @ other.value.T` and
        `other.grad += self.value.T @ out.grad`.
        """
        raise NotImplementedError("Tensor.__matmul__")

    def relu(self):
        """Elementwise ReLU. The gradient is out.grad where self.value > 0, else 0."""
        raise NotImplementedError("Tensor.relu")

    def sum(self):
        """Sum every entry to a scalar Tensor.

        The gradient with respect to self is out.grad broadcast back to
        self's shape, since every entry contributed additively to the sum.
        """
        raise NotImplementedError("Tensor.sum")

    def backward(self):
        """Run backpropagation from this tensor.

        Build a topological order of every tensor reachable through
        `_children` (a node is appended only after all of its children have
        been), set this tensor's gradient to ones of its own shape, then call
        every node's `_backward_fn` in reverse topological order.
        """
        raise NotImplementedError("Tensor.backward")


def softmax_cross_entropy(logits: "Tensor", labels: np.ndarray) -> "Tensor":
    """Mean cross-entropy loss of softmax(logits) against integer labels.

    Args:
        logits: A Tensor of shape (n, c), the unnormalised class scores.
        labels: An integer array of shape (n,), values in range(c). This is
            plain data, not a Tensor: the loss is not differentiated with
            respect to the labels.

    Returns:
        A scalar Tensor equal to -mean_i log(softmax(logits_i)[labels_i]),
        whose `_backward_fn` sets logits.grad using the closed-form gradient
        (softmax(logits) - one_hot(labels)) / n, derived in the lesson by
        combining softmax's Jacobian with cross-entropy's gradient.

    Compute the forward value with a numerically stable log-sum-exp, and
    build a graph node with `logits` as its only child.
    """
    raise NotImplementedError("softmax_cross_entropy")


def mlp_loss_and_grad(params: list[dict], X: np.ndarray, y: np.ndarray):
    """Compute the loss and every parameter gradient of a small ReLU MLP.

    Args:
        params: A list of layers, each a dict {"W": ndarray(in, out), "b":
            ndarray(out,)}. Every layer but the last is followed by relu; the
            last layer's output (the logits) feeds softmax_cross_entropy
            directly, with no activation.
        X: Inputs, shape (n, d), where d is the first layer's input size.
        y: Integer class labels, shape (n,).

    Returns:
        A tuple (loss, grads):
            loss: The scalar loss value, as a float.
            grads: A list matching params, each entry {"W": ndarray, "b":
                ndarray} of the same shapes as the corresponding parameter,
                holding dloss/dW and dloss/db.

    Wrap every parameter in a Tensor, run the forward pass using the Tensor
    ops above, call backward() once on the loss, and read the gradients back
    off the parameter Tensors.
    """
    raise NotImplementedError("mlp_loss_and_grad")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d, hidden, classes = 8, 4, 5, 3
    X = rng.normal(size=(n, d))
    y = rng.integers(classes, size=n)
    params = [
        {"W": rng.normal(size=(d, hidden)) * 0.5, "b": np.zeros(hidden)},
        {"W": rng.normal(size=(hidden, classes)) * 0.5, "b": np.zeros(classes)},
    ]
    loss, grads = mlp_loss_and_grad(params, X, y)
    print(f"loss: {loss:.4f}")
    for i, g in enumerate(grads):
        print(f"layer {i}: ||dW|| = {np.linalg.norm(g['W']):.4f}, ||db|| = {np.linalg.norm(g['b']):.4f}")
