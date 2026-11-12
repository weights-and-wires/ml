"""Exercise: pool a decoder's hidden states, train with InfoNCE, score retrieval.

The helpers _row_softmax and _order are given. Fill in every other function."""

from __future__ import annotations

import numpy as np


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale every row of x to length 1.

    Args:
        x: Array of shape (n, d).
        eps: Floor for the row length, so a zero row stays zero instead of dividing by zero.

    Returns:
        Array of shape (n, d) whose rows have length 1 (or 0 for a zero row).
    """
    raise NotImplementedError("write l2_normalize")


def last_token_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the hidden state of the last real token of each sequence.

    Args:
        hidden: Array of shape (batch, time, d), the decoder's last-layer states.
        mask: Array of shape (batch, time), 1 for a real token and 0 for padding.
            Padding can sit on the right, on the left, or both.

    Returns:
        Array of shape (batch, d).
    """
    raise NotImplementedError("write last_token_pool")


def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the average hidden state over the real tokens of each sequence.

    Args:
        hidden: Array of shape (batch, time, d).
        mask: Array of shape (batch, time), 1 for a real token and 0 for padding.

    Returns:
        Array of shape (batch, d).
    """
    raise NotImplementedError("write mean_pool")


def embed(hidden: np.ndarray, mask: np.ndarray, pooling: str = "last") -> np.ndarray:
    """Pool the hidden states into one vector per sequence and scale it to length 1.

    Args:
        hidden: Array of shape (batch, time, d).
        mask: Array of shape (batch, time), 1 for a real token and 0 for padding.
        pooling: "last" for last-token pooling, "mean" for mean pooling.

    Returns:
        Array of shape (batch, d) with unit-length rows.
    """
    raise NotImplementedError("write embed")


def _row_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def info_nce_loss(q: np.ndarray, docs: np.ndarray, tau: float) -> float:
    """InfoNCE for query-passage pairs with in-batch and hard negatives.

    Args:
        q: Array of shape (B, d), one query vector per row.
        docs: Array of shape (M, d) with M >= B. Row i (i < B) is the positive
            passage of query i. Rows B and beyond are hard negatives. Every
            row of docs is a negative for every query except its owner.
        tau: Temperature.

    Returns:
        The mean over queries of -log softmax(q @ docs.T / tau)[i, i].
    """
    raise NotImplementedError("write info_nce_loss")


def info_nce_grad(q: np.ndarray, docs: np.ndarray, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of info_nce_loss with respect to q and docs.

    Args:
        q: Array of shape (B, d).
        docs: Array of shape (M, d) with M >= B.
        tau: Temperature.

    Returns:
        A pair (dq, ddocs) with the shapes of q and docs.
    """
    raise NotImplementedError("write info_nce_grad")


def _order(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


def recall_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """Fraction of the relevant passages that appear in the top k.

    Args:
        scores: Array of shape (n,), higher is better.
        relevance: Integer array of shape (n,); a passage is relevant when its grade is above 0.
        k: Cutoff.

    Returns:
        A number in [0, 1]. Returns 0.0 when nothing is relevant.
    """
    raise NotImplementedError("write recall_at_k")


def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """Normalised discounted cumulative gain at k, with gain equal to the grade.

    Args:
        scores: Array of shape (n,), higher is better.
        relevance: Non-negative grades of shape (n,).
        k: Cutoff.

    Returns:
        DCG of the ranking by scores divided by DCG of the best possible ranking.
        Returns 0.0 when every grade is 0.
    """
    raise NotImplementedError("write ndcg_at_k")


def search(q_emb: np.ndarray, doc_emb: np.ndarray, k: int) -> np.ndarray:
    """Rank passages for each query by cosine similarity.

    Args:
        q_emb: Array of shape (Q, d), unit-length rows.
        doc_emb: Array of shape (N, d), unit-length rows.
        k: Number of passages to return.

    Returns:
        Integer array of shape (Q, k): the indices of the k best passages per query, best first.
    """
    raise NotImplementedError("write search")


if __name__ == "__main__":
    hidden = np.array([[[1.0, 0.0], [0.0, 2.0], [2.0, 2.0], [9.0, 9.0]]])
    mask = np.array([[1, 1, 1, 0]])
    print("last-token pool:", last_token_pool(hidden, mask))
    print("mean pool:", mean_pool(hidden, mask))
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    docs = np.array([[0.8, 0.6], [0.6, 0.8], [0.6, -0.8], [0.8, -0.6]])
    print("InfoNCE at tau 0.1:", info_nce_loss(q, docs, 0.1))
    scores = np.array([0.95, 0.90, 0.80, 0.60, 0.30, -0.20])
    grades = np.array([0, 3, 0, 2, 1, 0])
    print("recall@3:", recall_at_k(scores, grades, 3), "nDCG@3:", ndcg_at_k(scores, grades, 3))
