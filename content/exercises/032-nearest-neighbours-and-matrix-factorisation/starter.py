"""Nearest neighbours and matrix factorisation.

Fill in the ten functions below, in this order. The first five predict from
similar rows (k-NN) and measure why that gets hard in high dimension. The last
five fit a ratings table with a low-rank model by alternating least squares
(ALS), using only the entries you observed.
"""

import numpy as np


def pairwise_sq_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return squared distances between every row of A and every row of B.

    Args:
        A: Points, shape (n, d).
        B: Points, shape (m, d).

    Returns:
        Array of shape (n, m) whose entry (i, j) is ||A[i] - B[j]||^2. Use
        ||a||^2 - 2 a.b + ||b||^2 with one matrix product, and clip at 0
        because rounding can leave tiny negative values.
    """
    raise NotImplementedError("pairwise_sq_dists")


def knn_predict(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int) -> np.ndarray:
    """Classify each test row by a vote among its k nearest training rows.

    Args:
        Xtr: Training points, shape (n, d).
        ytr: Integer labels in range(c), shape (n,).
        Xte: Query points, shape (q, d).
        k: Number of neighbours.

    Returns:
        Integer labels, shape (q,). Rank neighbours with a stable sort, so a
        tie in distance goes to the lower training index. When two labels tie
        in the vote, return the smaller label.
    """
    raise NotImplementedError("knn_predict")


def knn_regress(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, k: int) -> np.ndarray:
    """Predict each test row as the mean target of its k nearest training rows.

    Args:
        Xtr: Training points, shape (n, d).
        ytr: Real targets, shape (n,).
        Xte: Query points, shape (q, d).
        k: Number of neighbours.

    Returns:
        Predictions, shape (q,). Use the same neighbour order as knn_predict.
    """
    raise NotImplementedError("knn_regress")


def loo_knn_error(X: np.ndarray, y: np.ndarray, k: int) -> float:
    """Return the leave-one-out misclassification rate of k-NN.

    Args:
        X: Points, shape (n, d).
        y: Integer labels, shape (n,).
        k: Number of neighbours.

    Returns:
        The fraction of rows whose label differs from the vote of their k
        nearest OTHER rows. A row must never count as its own neighbour. Ties
        follow knn_predict.
    """
    raise NotImplementedError("loo_knn_error")


def distance_ratio(d: int, n: int, rng: np.random.Generator) -> float:
    """Return nearest over farthest distance from one query to n random points.

    Args:
        d: Dimension.
        n: Number of points.
        rng: Seeded generator. Draw the points first, rng.uniform(size=(n, d)),
            then the query, rng.uniform(size=(1, d)).

    Returns:
        min distance divided by max distance, using Euclidean (not squared)
        distances, as a Python float.
    """
    raise NotImplementedError("distance_ratio")


def masked_loss(R, M, U, V, lam) -> float:
    """Return the regularised squared error on the observed entries.

    Args:
        R: Ratings, shape (n, m). Unobserved entries hold any number.
        M: Mask, shape (n, m), 1.0 where observed and 0.0 elsewhere.
        U: User factors, shape (n, k).
        V: Item factors, shape (m, k).
        lam: Ridge penalty.

    Returns:
        sum of M * (R - U V^T)^2 plus lam * (sum of U^2 + sum of V^2), as a
        Python float.
    """
    raise NotImplementedError("masked_loss")


def als_step(R, M, U, V, lam) -> np.ndarray:
    """Return U with each row replaced by its best fit, holding V fixed.

    Args:
        R: Ratings, shape (n, m).
        M: Mask, shape (n, m).
        U: Current user factors, shape (n, k). Do not modify it.
        V: Item factors, shape (m, k).
        lam: Ridge penalty.

    Returns:
        New U, shape (n, k). Row i solves a small ridge regression of the
        ratings the user gave on the rows of V for those items. A user with no
        ratings gets a row of zeros.
    """
    raise NotImplementedError("als_step")


def fit_als(R, M, k, lam, iters, rng):
    """Fit R ~ U V^T on the observed entries by alternating least squares.

    Args:
        R: Ratings, shape (n, m).
        M: Mask, shape (n, m).
        k: Rank of the factorisation.
        lam: Ridge penalty.
        iters: Number of sweeps. A sweep updates U, then V (call als_step on
            the transposed table for V).
        rng: Seeded generator. Start U = 0.1 * rng.normal(size=(n, k)), then
            V = 0.1 * rng.normal(size=(m, k)).

    Returns:
        (U, V, history). history holds masked_loss at the start and after every
        sweep, so it has iters + 1 entries.
    """
    raise NotImplementedError("fit_als")


def masked_rmse(R, M, P) -> float:
    """Return the root mean squared error of P against R over entries where M is 1."""
    raise NotImplementedError("masked_rmse")


def user_bias_baseline(R, M) -> np.ndarray:
    """Return the user-bias baseline as an (n, m) table.

    Args:
        R: Ratings, shape (n, m).
        M: Mask, shape (n, m).

    Returns:
        Entry (i, j) is the mean of all observed ratings, plus user i's mean
        offset from that global mean over their observed ratings (0 for a user
        with no ratings).
    """
    raise NotImplementedError("user_bias_baseline")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    Xtr = np.array([[0, 0], [1, 0], [0, 2], [3, 3], [4, 3]], dtype=float)
    ytr = np.array([0, 0, 1, 1, 1])
    q = np.array([[1.0, 1.0]])
    print("squared distances:", pairwise_sq_dists(q, Xtr))
    print("k=3 vote:", knn_predict(Xtr, ytr, q, 3))
    print("distance ratio at d=2 and d=1000:",
          distance_ratio(2, 200, rng), distance_ratio(1000, 200, rng))
    U0, V0 = rng.normal(size=(60, 2)), rng.normal(size=(40, 2))
    R = U0 @ V0.T + 0.2 * rng.normal(size=(60, 40))
    obs = rng.uniform(size=R.shape) < 0.3
    M = obs.astype(float)
    U, V, hist = fit_als(R, M, 2, 1.0, 20, rng)
    print("loss first and last:", hist[0], hist[-1])
