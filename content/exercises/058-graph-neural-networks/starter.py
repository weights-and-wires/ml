"""Graph neural networks. Fill in every function that raises NotImplementedError."""

from __future__ import annotations

import numpy as np


def normalised_adjacency(A: np.ndarray) -> np.ndarray:
    """Return D^-1/2 (A + I) D^-1/2, where D holds the row sums of A + I.

    Args:
        A: Array of shape (n, n), a symmetric 0/1 adjacency matrix with a zero diagonal.

    Returns:
        Array of shape (n, n), symmetric.
    """
    raise NotImplementedError("write normalised_adjacency")


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
    raise NotImplementedError("write gcn_layer")


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
    raise NotImplementedError("write gcn_layer_backward")


def message_pass(A: np.ndarray, H: np.ndarray, agg: str) -> np.ndarray:
    """Aggregate each node's neighbour features, without the node itself.

    Args:
        A: Array of shape (n, n), a 0/1 adjacency matrix.
        H: Node features, shape (n, d).
        agg: "sum", "mean", or "max".

    Returns:
        Array of shape (n, d). A node with no neighbours gets zeros.
    """
    raise NotImplementedError("write message_pass")


def readout(H: np.ndarray, kind: str = "mean") -> np.ndarray:
    """Collapse node features of shape (n, d) into one graph vector of shape (d,)."""
    raise NotImplementedError("write readout")


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
    raise NotImplementedError("write oversmoothing_curve")


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
    raise NotImplementedError("write train_gcn")


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
