"""Slopes through array operations.

The engine from the previous chapter is here with its add and multiply rules
already upgraded to call unbroadcast. Fill in unbroadcast, Tensor.__matmul__,
and Tensor.relu. Tensor.sum and Tensor.backward are provided.
"""

import numpy as np


def central_difference(f, x, h=1e-5):
    """Two-sided slope estimate. Written in 'Walking slopes backwards through a calculation'."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def unbroadcast(grad, shape):
    """Sum a slope back down to the shape its value had before broadcasting.

    Args:
        grad: An array, the slope that arrived at the broadcast result.
        shape: A tuple, the shape of the value that was broadcast.

    Returns:
        An array of exactly `shape`. Axes that numpy added on the left, or
        stretched from size 1, are summed away, because one value used many
        times collects the slope from every copy. For example, grad of shape
        (4, 3) and shape (3,) returns the column sums, shape (3,).
    """
    raise NotImplementedError("unbroadcast")


class Tensor:
    """A node in the graph: a value, a gradient, and the inputs it came from."""

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

    # Written in 'Walking slopes backwards through a calculation'; upgraded here to unbroadcast.
    def __add__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value + other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.value.shape)
            other.grad += unbroadcast(out.grad, other.value.shape)

        out._backward_fn = _backward
        return out

    # Written in 'Walking slopes backwards through a calculation'; upgraded here to unbroadcast.
    def __mul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value * other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad * other.value, self.value.shape)
            other.grad += unbroadcast(out.grad * self.value, other.value.shape)

        out._backward_fn = _backward
        return out

    def __matmul__(self, other):
        """Matrix product of two 2-dimensional tensors, shapes (n, k) and (k, m).

        Build `out` from `self.value @ other.value`, with children (self,
        other). Its `_backward_fn` adds a slope of shape (n, k) into
        `self.grad` and a slope of shape (k, m) into `other.grad`. Each
        gradient must have the shape of its own value.
        """
        raise NotImplementedError("Tensor.__matmul__")

    def relu(self):
        """Elementwise max(x, 0).

        The slope passes through where the input was positive and stops
        where it was zero or negative.
        """
        raise NotImplementedError("Tensor.relu")

    # Provided: the checks use sum as the final step that turns an array into one number.
    def sum(self):
        out = Tensor(self.value.sum(), children=(self,))

        def _backward():
            self.grad += out.grad * np.ones_like(self.value)

        out._backward_fn = _backward
        return out

    # Written in 'Walking slopes backwards through a calculation'.
    def backward(self):
        order = []
        seen = set()

        def visit(node):
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in node._children:
                visit(child)
            order.append(node)

        visit(self)
        self.grad = np.ones_like(self.value)
        for node in reversed(order):
            node._backward_fn()


if __name__ == "__main__":
    A = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = Tensor([[1.0, 0.0], [2.0, 1.0], [0.0, 3.0]])
    bias = Tensor([10.0, -30.0])
    Z = A @ B + bias
    Zbar = np.array([[1.0, 2.0], [3.0, 4.0]])
    loss = (Z.relu() * Tensor(Zbar)).sum()
    loss.backward()
    print("Z =", Z.value.tolist())
    print("slope in A:", A.grad.tolist())
    print("slope in B:", B.grad.tolist())
    print("slope in bias:", bias.grad.tolist())
