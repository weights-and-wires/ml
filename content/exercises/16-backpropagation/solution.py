"""Reference solution for the backpropagation exercise."""

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
        other = Tensor._coerce(other)
        out = Tensor(self.value + other.value, children=(self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad, self.value.shape)
            other.grad += _unbroadcast(out.grad, other.value.shape)

        out._backward_fn = _backward
        return out

    def __mul__(self, other):
        other = Tensor._coerce(other)
        out = Tensor(self.value * other.value, children=(self, other))

        def _backward():
            self.grad += _unbroadcast(out.grad * other.value, self.value.shape)
            other.grad += _unbroadcast(out.grad * self.value, other.value.shape)

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

    def sum(self):
        out = Tensor(self.value.sum(), children=(self,))

        def _backward():
            self.grad += out.grad * np.ones_like(self.value)

        out._backward_fn = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._children:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = np.ones_like(self.value)
        for node in reversed(topo):
            node._backward_fn()


def softmax_cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean cross-entropy loss of softmax(logits) against integer labels."""
    z = logits.value
    n = z.shape[0]
    m = z.max(axis=1, keepdims=True)
    exp = np.exp(z - m)
    row_sums = exp.sum(axis=1, keepdims=True)
    probs = exp / row_sums
    log_probs = (z - m) - np.log(row_sums)
    loss_value = -log_probs[np.arange(n), labels].mean()

    out = Tensor(loss_value, children=(logits,))

    def _backward():
        grad_logits = probs.copy()
        grad_logits[np.arange(n), labels] -= 1.0
        grad_logits /= n
        logits.grad += out.grad * grad_logits

    out._backward_fn = _backward
    return out


def mlp_loss_and_grad(params: list[dict], X: np.ndarray, y: np.ndarray):
    """Forward and backward pass through a ReLU MLP with a softmax-cross-entropy head.

    params: list of {"W": (in, out), "b": (out,)}. Every layer but the last is
    followed by relu; the last layer's output feeds softmax_cross_entropy.
    """
    tensor_params = [
        {"W": Tensor(layer["W"]), "b": Tensor(layer["b"])} for layer in params
    ]

    h = Tensor(X)
    for i, layer in enumerate(tensor_params):
        z = h @ layer["W"] + layer["b"]
        h = z.relu() if i < len(tensor_params) - 1 else z

    loss = softmax_cross_entropy(h, y)
    loss.backward()

    grads = [
        {"W": layer["W"].grad.copy(), "b": layer["b"].grad.copy()}
        for layer in tensor_params
    ]
    return float(loss.value), grads


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
