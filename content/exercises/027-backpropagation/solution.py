"""Reference solution: an engine that walks slopes backwards through arrays."""

import numpy as np


def central_difference(f, x, h=1e-5):
    """Two-sided slope estimate."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


def unbroadcast(grad, shape):
    """Sum grad back down to shape, undoing whatever numpy broadcast."""
    grad = np.asarray(grad, dtype=float)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


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

    def __add__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value + other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.value.shape)
            other.grad += unbroadcast(out.grad, other.value.shape)

        out._backward_fn = _backward
        return out

    def __mul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value * other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad * other.value, self.value.shape)
            other.grad += unbroadcast(out.grad * self.value, other.value.shape)

        out._backward_fn = _backward
        return out

    def __matmul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value @ other.value, children=(self, other))

        def _backward():
            self.grad += out.grad @ other.value.T
            other.grad += self.value.T @ out.grad

        out._backward_fn = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(self.value, 0.0), children=(self,))

        def _backward():
            self.grad += out.grad * (self.value > 0.0)

        out._backward_fn = _backward
        return out

    # Provided: the checks use sum as the final step that turns an array into one number.
    def sum(self):
        out = Tensor(self.value.sum(), children=(self,))

        def _backward():
            self.grad += out.grad * np.ones_like(self.value)

        out._backward_fn = _backward
        return out

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
    a, b = Tensor(3.0), Tensor(2.0)
    c = a * b + a
    c.backward()
    print(f"c = {float(c.value)}")
    print(f"slope of c in a = {float(a.grad)} (expect 3)")
    print(f"slope of c in b = {float(b.grad)} (expect 3)")

    def c_of_a(av):
        return float((Tensor(av) * Tensor(2.0) + Tensor(av)).value)

    print(f"two-sided check at a = 3: {central_difference(c_of_a, 3.0, h=0.001):.4f}")

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
