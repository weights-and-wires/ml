"""Reference solution: a tiny engine that walks slopes backwards."""

import numpy as np


def central_difference(f, x, h=1e-5):
    """Estimate the slope of a scalar function f at x with a two-sided nudge."""
    return (f(x + h) - f(x - h)) / (2.0 * h)


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
        # Provided: study this rule, then write __mul__ in the same style.
        other = Tensor._coerce(other)
        out = Tensor(self.value + other.value, children=(self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward_fn = _backward
        return out

    def __mul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value * other.value, children=(self, other))

        def _backward():
            self.grad += out.grad * other.value
            other.grad += out.grad * self.value

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
