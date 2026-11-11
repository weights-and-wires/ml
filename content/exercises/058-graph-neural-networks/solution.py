"""Graph neural networks. Reference solution."""

from __future__ import annotations

import numpy as np


def normalised_adjacency(A: np.ndarray) -> np.ndarray:
    """Return D^-1/2 (A + I) D^-1/2, where D holds the row sums of A + I.

    Args:
        A: Array of shape (n, n), a symmetric 0/1 adjacency matrix with a zero diagonal.

    Returns:
        Array of shape (n, n), symmetric.
    """
    A_loop = A + np.eye(A.shape[0])
    inv_sqrt = 1.0 / np.sqrt(A_loop.sum(axis=1))
    return A_loop * inv_sqrt[:, None] * inv_sqrt[None, :]


def gcn_layer(A_hat: np.ndarray, H: np.ndarray, W: np.ndarray, relu: bool = True) -> np.ndarray:
    """One graph convolution: A_hat @ H @ W, then ReLU when relu is True.

    Args:
        A_hat: Array of shape (n, n), from normalised_adjacency.
        H: Node features, shape (n, d_in).
        W: Weights, shape (d_in, d_out).
        relu: Apply max(0, x) to the result when True.

    Returns:
        Array of shape (n, d_out).
    """
    out = A_hat @ H @ W
    return np.maximum(out, 0.0) if relu else out


def gcn_layer_backward(
    A_hat: np.ndarray, H: np.ndarray, W: np.ndarray, d_out: np.ndarray, relu: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of a scalar loss through gcn_layer.

    Args:
        A_hat: Array of shape (n, n), symmetric.
        H: Node features, shape (n, d_in).
        W: Weights, shape (d_in, d_out).
        d_out: Gradient of the loss with respect to the layer output, shape (n, d_out).
        relu: The same flag you passed to gcn_layer.

    Returns:
        (dH, dW) with the shapes of H and W.
    """
    pre = A_hat @ H @ W
    d_pre = d_out * (pre > 0) if relu else d_out
    dW = (A_hat @ H).T @ d_pre
    dH = A_hat.T @ d_pre @ W.T
    return dH, dW


def message_pass(A: np.ndarray, H: np.ndarray, agg: str) -> np.ndarray:
    """Aggregate each node's neighbour features, without the node itself.

    Args:
        A: Array of shape (n, n), a 0/1 adjacency matrix.
        H: Node features, shape (n, d).
        agg: "sum", "mean", or "max".

    Returns:
        Array of shape (n, d). A node with no neighbours gets zeros.
    """
    out = np.zeros_like(H, dtype=float)
    for i in range(A.shape[0]):
        nbrs = np.flatnonzero(A[i])
        if len(nbrs) == 0:
            continue
        rows = H[nbrs]
        if agg == "sum":
            out[i] = rows.sum(axis=0)
        elif agg == "mean":
            out[i] = rows.mean(axis=0)
        elif agg == "max":
            out[i] = rows.max(axis=0)
        else:
            raise ValueError(f"unknown aggregation {agg!r}")
    return out


def readout(H: np.ndarray, kind: str = "mean") -> np.ndarray:
    """Collapse node features of shape (n, d) into one graph vector of shape (d,)."""
    if kind == "sum":
        return H.sum(axis=0)
    if kind == "mean":
        return H.mean(axis=0)
    if kind == "max":
        return H.max(axis=0)
    raise ValueError(f"unknown readout {kind!r}")


def oversmoothing_curve(A: np.ndarray, H: np.ndarray, layers: int) -> np.ndarray:
    """Spread of node features after 0, 1, ..., layers weightless GCN steps.

    Each step replaces H with A_hat @ H. Before measuring, divide row i by
    sqrt(deg_i + 1), where deg_i + 1 is the row sum of A + I. The spread is the
    mean over feature columns of (max over nodes - min over nodes).

    Args:
        A: Array of shape (n, n), a 0/1 adjacency matrix.
        H: Array of shape (n, d), the starting features.
        layers: Number of steps.

    Returns:
        Array of shape (layers + 1,). Entry k is the spread after k steps.
    """
    A_hat = normalised_adjacency(A)
    scale = 1.0 / np.sqrt(A.sum(axis=1) + 1.0)
    out = []
    X = np.asarray(H, dtype=float)
    for _ in range(layers + 1):
        S = X * scale[:, None]
        out.append(float(np.mean(S.max(axis=0) - S.min(axis=0))))
        X = A_hat @ X
    return np.array(out)


def two_community_graph(seed: int = 3, n: int = 60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A seeded stochastic block model with weak node features.

    Returns:
        (A, X, y): adjacency (n, n), features (n, 2), and labels (n,) in {0, 1}.
    """
    rng = np.random.default_rng(seed)
    y = np.repeat([0, 1], n // 2)
    prob = np.where(y[:, None] == y[None, :], 0.25, 0.02)
    upper = np.triu(rng.uniform(size=(n, n)) < prob, 1)
    A = (upper | upper.T).astype(float)
    signal = y * 0.6 - 0.3
    X = np.stack([signal, -signal], axis=1) + rng.normal(size=(n, 2))
    return A, X, y


def train_gcn(
    A: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    hidden: int = 8,
    steps: int = 200,
    lr: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """Train a two-layer GCN on the labelled nodes and predict every node.

    The model is logits = A_hat @ relu(A_hat @ X @ W1) @ W2, with softmax
    cross-entropy averaged over train_idx and plain gradient descent.

    Returns:
        Integer class predictions, shape (n,).
    """
    rng = np.random.default_rng(seed)
    A_hat = normalised_adjacency(A)
    n_classes = int(y.max()) + 1
    W1 = 0.3 * rng.normal(size=(X.shape[1], hidden))
    W2 = 0.3 * rng.normal(size=(hidden, n_classes))
    onehot = np.eye(n_classes)[y[train_idx]]
    for _ in range(steps):
        H1 = gcn_layer(A_hat, X, W1)
        logits = gcn_layer(A_hat, H1, W2, relu=False)
        z = logits[train_idx]
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        d_logits = np.zeros_like(logits)
        d_logits[train_idx] = (p - onehot) / len(train_idx)
        dH1, dW2 = gcn_layer_backward(A_hat, H1, W2, d_logits, relu=False)
        _, dW1 = gcn_layer_backward(A_hat, X, W1, dH1)
        W1 -= lr * dW1
        W2 -= lr * dW2
    logits = gcn_layer(A_hat, gcn_layer(A_hat, X, W1), W2, relu=False)
    return logits.argmax(axis=1)


if __name__ == "__main__":
    A = np.array([[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=float)
    A_hat = normalised_adjacency(A)
    print("A_hat:\n", A_hat.round(4))
    H = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    W = np.array([[1.0], [-1.0]])
    print("gcn_layer:", gcn_layer(A_hat, H, W).ravel().round(4))
    print("sum / mean / max of neighbours:")
    for agg in ("sum", "mean", "max"):
        print(" ", agg, message_pass(A, H[:, :1], agg).ravel())
    print("spread by depth:", oversmoothing_curve(A, H, 4).round(4))
    A2, X2, y2 = two_community_graph()
    idx = np.array([0, 1, 2, 30, 31, 32])
    pred = train_gcn(A2, X2, y2, idx)
    rest = np.setdiff1d(np.arange(len(y2)), idx)
    print("accuracy on unlabelled nodes:", (pred[rest] == y2[rest]).mean())
