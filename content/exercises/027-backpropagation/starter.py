"""A tiny engine that walks slopes backwards, through numbers and through arrays.

Each Tensor remembers its value, its gradient, and its children: the tensors
it was computed from. Fill in central_difference, Tensor.__mul__,
Tensor.backward, unbroadcast, Tensor.__matmul__, and Tensor.relu, in the order
the lesson builds them. Tensor.__add__ is provided as a model for the other
rules. Once you have written unbroadcast, edit __add__ and __mul__ so that
every contribution goes through it.
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

    def __add__(self, other):
        # Provided: study this rule, then write __mul__ in the same style.
        # It must also call unbroadcast once you have written that function.
        other = Tensor._coerce(other)
        out = Tensor(self.value + other.value, children=(self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward_fn = _backward
        return out

    def __mul__(self, other):
        """Elementwise product of two tensors, with numpy broadcasting.

        Build `out` from `self.value * other.value`, with children (self,
        other). Give it a `_backward_fn` that adds `out.grad * other.value`
        into `self.grad`, and `out.grad * self.value` into `other.grad`, each
        sent back to the shape of its own value. Use `+=`, never `=`.
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
