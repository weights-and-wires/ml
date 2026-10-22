"""Reference solution: softmax and cross-entropy with a readable gradient."""

import numpy as np


# Written in 'The backward rule for each array operation'.
def unbroadcast(grad, shape):
    grad = np.asarray(grad, dtype=float)
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    """The engine from the previous two chapters, provided complete."""

    def __init__(self, value, children=()):
        self.value = np.asarray(value, dtype=float)
        self.grad = np.zeros_like(self.value)
        self._children = children
        self._backward_fn = lambda: None

    @staticmethod
    def _coerce(other):
        return other if isinstance(other, Tensor) else Tensor(other)

    # Written in 'Walking slopes backwards through a calculation'.
    def __add__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value + other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad, self.value.shape)
            other.grad += unbroadcast(out.grad, other.value.shape)

        out._backward_fn = _backward
        return out

    # Written in 'Walking slopes backwards through a calculation'.
    def __mul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value * other.value, children=(self, other))

        def _backward():
            self.grad += unbroadcast(out.grad * other.value, self.value.shape)
            other.grad += unbroadcast(out.grad * self.value, other.value.shape)

        out._backward_fn = _backward
        return out

    # Written in 'The backward rule for each array operation'.
    def __matmul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value @ other.value, children=(self, other))

        def _backward():
            self.grad += out.grad @ other.value.T
            other.grad += self.value.T @ out.grad

        out._backward_fn = _backward
        return out

    # Written in 'The backward rule for each array operation'.
    def relu(self):
        out = Tensor(np.maximum(self.value, 0.0), children=(self,))

        def _backward():
            self.grad += out.grad * (self.value > 0.0)

        out._backward_fn = _backward
        return out

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


def softmax(z):
    """Row-wise softmax that does not overflow."""
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z):
    """Row-wise log of softmax, computed without ever forming a tiny probability."""
    z = np.asarray(z, dtype=float)
    shifted = z - z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def softmax_cross_entropy(logits, labels):
    """Mean cross-entropy of softmax(logits) against integer labels, as a scalar Tensor."""
    z = logits.value
    n = z.shape[0]
    log_probs = log_softmax(z)
    loss = -log_probs[np.arange(n), labels].mean()
    out = Tensor(loss, children=(logits,))

    def _backward():
        g = np.exp(log_probs)
        g[np.arange(n), labels] -= 1.0
        logits.grad += out.grad * g / n

    out._backward_fn = _backward
    return out


if __name__ == "__main__":
    z = np.array([[2.0, 1.0, 0.0]])
    print("softmax:", np.round(softmax(z), 6).tolist())
    logits = Tensor(z)
    loss = softmax_cross_entropy(logits, np.array([0]))
    loss.backward()
    print(f"loss for class 0: {float(loss.value):.6f}")
    print("slope on the logits:", np.round(logits.grad, 6).tolist())
