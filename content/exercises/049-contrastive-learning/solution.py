"""Reference solution for the contrastive-learning exercise."""

from __future__ import annotations

from typing import Callable

import numpy as np


def l2_normalize(z: np.ndarray) -> np.ndarray:
    """Scale every row of z to length 1.

    Args:
        z: Array of shape (N, d). No row is all zeros.

    Returns:
        Array of shape (N, d) whose rows have length 1.
    """
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def _row_logsumexp(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=1, keepdims=True)))[:, 0]


def info_nce(z1: np.ndarray, z2: np.ndarray, tau: float) -> float:
    """InfoNCE: the mean cross-entropy of picking the positive in each row.

    Args:
        z1: Array of shape (N, d), view 1. Row i is the query for example i.
        z2: Array of shape (N, d), view 2. Row i is the positive for query i;
            the other N - 1 rows are its negatives.
        tau: Temperature, a positive float.

    Returns:
        The mean over i of -log softmax(n1[i] @ n2.T / tau)[i], as a float,
        where n1 and n2 are the unit-length rows of z1 and z2.
    """
    logits = l2_normalize(z1) @ l2_normalize(z2).T / tau
    return float(np.mean(_row_logsumexp(logits) - np.diag(logits)))


def _back_through_normalize(z: np.ndarray, n: np.ndarray, dn: np.ndarray) -> np.ndarray:
    return (dn - n * np.sum(n * dn, axis=1, keepdims=True)) / np.linalg.norm(z, axis=1, keepdims=True)


def info_nce_grad(z1: np.ndarray, z2: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of info_nce with respect to the raw embeddings.

    Args:
        z1: Array of shape (N, d).
        z2: Array of shape (N, d).
        tau: Temperature.

    Returns:
        A pair (dz1, dz2), each of shape (N, d).
    """
    n1, n2 = l2_normalize(z1), l2_normalize(z2)
    logits = n1 @ n2.T / tau
    p = np.exp(logits - _row_logsumexp(logits)[:, None])
    g = (p - np.eye(len(z1))) / (len(z1) * tau)
    return _back_through_normalize(z1, n1, g @ n2), _back_through_normalize(z2, n2, g.T @ n1)


def clip_loss(img: np.ndarray, txt: np.ndarray, tau: float) -> float:
    """CLIP's symmetric loss: image-to-text and text-to-image, averaged.

    Args:
        img: Array of shape (N, d), image embeddings.
        txt: Array of shape (N, d), caption embeddings. Row i matches img[i].
        tau: Temperature.

    Returns:
        Half the sum of the row-wise and column-wise cross-entropies, as a float.
    """
    return 0.5 * (info_nce(img, txt, tau) + info_nce(txt, img, tau))


def alignment(z1: np.ndarray, z2: np.ndarray) -> float:
    """Mean squared distance between the unit rows of two views.

    Args:
        z1: Array of shape (N, d).
        z2: Array of shape (N, d). Row i is a positive pair with z1[i].

    Returns:
        A float. It is 0 when every pair lands on the same point.
    """
    return float(np.mean(np.sum((l2_normalize(z1) - l2_normalize(z2)) ** 2, axis=1)))


def uniformity(z: np.ndarray, t: float = 2.0) -> float:
    """Log of the mean Gaussian kernel between distinct unit rows.

    Args:
        z: Array of shape (N, d), N >= 2.
        t: Kernel sharpness.

    Returns:
        log mean over pairs i < j of exp(-t * squared distance), as a float.
        It is 0 when every point is the same and more negative when points spread out.
    """
    n = l2_normalize(z)
    d2 = np.sum((n[:, None, :] - n[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(len(n), k=1)
    return float(np.log(np.mean(np.exp(-t * d2[iu]))))


def train_encoder(
    make_views: Callable[[np.random.Generator], tuple[np.ndarray, np.ndarray]],
    d_in: int,
    d_out: int,
    tau: float,
    steps: int,
    lr: float,
    seed: int,
) -> np.ndarray:
    """Train a linear encoder z = x @ W by gradient descent on InfoNCE.

    Args:
        make_views: Called as make_views(rng). Returns two arrays of shape (N, d_in),
            two augmented views of the same N examples.
        d_in: Input width.
        d_out: Embedding width.
        tau: Temperature.
        steps: Number of gradient steps. Every step draws fresh views.
        lr: Learning rate.
        seed: Seed for np.random.default_rng.

    Returns:
        The weight matrix W of shape (d_in, d_out).
    """
    rng = np.random.default_rng(seed)
    w = 0.1 * rng.normal(size=(d_in, d_out))
    for _ in range(steps):
        v1, v2 = make_views(rng)
        dz1, dz2 = info_nce_grad(v1 @ w, v2 @ w, tau)
        w = w - lr * (v1.T @ dz1 + v2.T @ dz2)
    return w


def knn_accuracy(
    train_z: np.ndarray, train_y: np.ndarray, test_z: np.ndarray, test_y: np.ndarray, k: int = 5
) -> float:
    """Accuracy of a k-nearest-neighbour vote by cosine similarity.

    Args:
        train_z: Array of shape (M, d).
        train_y: Integer labels of shape (M,).
        test_z: Array of shape (T, d).
        test_y: Integer labels of shape (T,).
        k: Number of neighbours that vote.

    Returns:
        The fraction of test rows whose majority label matches. Ties go to the
        smaller label. Equal similarities go to the earlier training row.
    """
    sim = l2_normalize(test_z) @ l2_normalize(train_z).T
    idx = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    classes = int(max(train_y.max(), test_y.max())) + 1
    pred = np.array([np.bincount(train_y[row], minlength=classes).argmax() for row in idx])
    return float(np.mean(pred == test_y))


if __name__ == "__main__":
    z1 = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    z2 = np.array([[0.8, 0.6], [0.6, 0.8], [0.0, 1.0]])
    print("info_nce(z1, z2) =", info_nce(z1, z2, 0.5))
    print("info_nce(z2, z1) =", info_nce(z2, z1, 0.5))
    print("clip_loss        =", clip_loss(z1, z2, 0.5))
    print("log 3            =", np.log(3))
