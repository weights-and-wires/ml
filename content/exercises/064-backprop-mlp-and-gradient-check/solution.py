"""Reference solution: gradients for a whole network, and the check that proves them."""

import numpy as np


# Written in 'Walking slopes backwards through a calculation'.
def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2.0 * h)


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
    """The engine from the previous chapters, provided complete."""

    def __init__(self, value, children=()):
        self.value = np.asarray(value, dtype=float)
        self.grad = np.zeros_like(self.value)
        self._children = children
        self._backward_fn = lambda: None

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


# Written in 'Softmax and cross-entropy: a loss with a gradient you can read'.
def softmax_cross_entropy(logits, labels):
    z = logits.value
    n = z.shape[0]
    shifted = z - z.max(axis=1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    out = Tensor(-log_probs[np.arange(n), labels].mean(), children=(logits,))

    def _backward():
        g = np.exp(log_probs)
        g[np.arange(n), labels] -= 1.0
        logits.grad += out.grad * g / n

    out._backward_fn = _backward
    return out


def mlp_loss_and_grad(params, X, y):
    tensors = [{"W": Tensor(layer["W"]), "b": Tensor(layer["b"])} for layer in params]
    h = Tensor(X)
    for i, layer in enumerate(tensors):
        z = h @ layer["W"] + layer["b"]
        h = z.relu() if i < len(tensors) - 1 else z
    loss = softmax_cross_entropy(h, y)
    loss.backward()
    grads = [{"W": t["W"].grad.copy(), "b": t["b"].grad.copy()} for t in tensors]
    return float(loss.value), grads


def numeric_grad(loss_fn, array, h=1e-5):
    x = np.array(array, dtype=float)
    grad = np.zeros_like(x)
    for index in np.ndindex(*x.shape):
        original = x[index]
        x[index] = original + h
        high = loss_fn(x)
        x[index] = original - h
        low = loss_fn(x)
        x[index] = original
        grad[index] = (high - low) / (2.0 * h)
    return grad


def gradient_check(params, X, y):
    _, grads = mlp_loss_and_grad(params, X, y)
    worst = 0.0
    for i, layer in enumerate(params):
        for key in ("W", "b"):

            def loss_fn(value, i=i, key=key):
                trial = [{"W": p["W"], "b": p["b"]} for p in params]
                trial[i][key] = value
                return mlp_loss_and_grad(trial, X, y)[0]

            numeric = numeric_grad(loss_fn, layer[key])
            analytic = grads[i][key]
            gap = np.linalg.norm(analytic - numeric)
            scale = np.linalg.norm(analytic) + np.linalg.norm(numeric)
            worst = max(worst, float(gap / scale) if scale > 0 else 0.0)
    return worst


# Provided: one step downhill. The next chapters build better steps.
def sgd_step(params, grads, lr):
    return [
        {"W": p["W"] - lr * g["W"], "b": p["b"] - lr * g["b"]}
        for p, g in zip(params, grads)
    ]


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
    print(f"gradient check, worst relative error: {gradient_check(params, X, y):.2e}")
    for step in range(5):
        params = sgd_step(params, grads, lr=0.5)
        loss, grads = mlp_loss_and_grad(params, X, y)
        print(f"after step {step + 1}: loss {loss:.4f}")
