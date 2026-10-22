"""A tiny engine that walks slopes backwards.

Each Tensor remembers its value, its gradient, and its children: the tensors
it was computed from. Fill in central_difference, Tensor.__mul__, and
Tensor.backward. Tensor.__add__ is provided as a model for the multiply rule.
"""

import numpy as np


def central_difference(f, x, h=1e-5):
    """Estimate the slope of the scalar function f at the number x.

    Args:
        f: A function from a float to a float.
        x: The point to measure the slope at.
        h: How far to nudge x in each direction.

    Returns:
        A float: (f(x + h) - f(x - h)) / (2 h).
    """
    raise NotImplementedError("central_difference")


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
        """Elementwise product of two tensors of the same shape (or a scalar).

        Build `out` from `self.value * other.value`, with children (self,
        other). Give it a `_backward_fn` that adds `out.grad * other.value`
        into `self.grad`, and `out.grad * self.value` into `other.grad`.
        Use `+=`, never `=`.
        """
        raise NotImplementedError("Tensor.__mul__")

    def backward(self):
        """Walk slopes back from this tensor to every input.

        Build a list of every tensor reachable through `_children`, in an
        order where each tensor comes after all of its children (a
        depth-first walk with a visited set does it). Set this tensor's grad
        to ones of its own shape. Then call `_backward_fn` on each tensor
        in the reverse of that list.
        """
        raise NotImplementedError("Tensor.backward")


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
