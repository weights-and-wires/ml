"""Reference solution for the learning-theory exercise."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np


# ---------------------------------------------------------------- Hoeffding


def hoeffding_bound(n: int, eps: float) -> float:
    """Upper bound on P(|average - mean| > eps) for n i.i.d. values in [0, 1].

    Args:
        n: Number of examples.
        eps: Size of the deviation.

    Returns:
        2 exp(-2 n eps^2), capped at 1 because a probability cannot exceed 1.
    """
    return float(min(1.0, 2.0 * math.exp(-2.0 * n * eps**2)))


def deviation_frequency(n: int, eps: float, trials: int, seed: int, p: float = 0.5) -> float:
    """Fraction of trials where an average of n coin flips misses p by more than eps.

    Args:
        n: Flips per trial.
        eps: Size of the deviation.
        trials: Number of repeated trials.
        seed: Seed for np.random.default_rng.
        p: Chance of heads.

    Returns:
        The fraction of trials with |average - p| > eps.
    """
    rng = np.random.default_rng(seed)
    averages = rng.binomial(n, p, size=trials) / n
    return float(np.mean(np.abs(averages - p) > eps))


def union_bound_sample_size(k: int, eps: float, delta: float) -> int:
    """Smallest n for which k models all land within eps with chance 1 - delta.

    Args:
        k: Number of models in the class.
        eps: Allowed gap between training average and true average.
        delta: Allowed failure chance.

    Returns:
        ceil(log(2k / delta) / (2 eps^2)).
    """
    return int(math.ceil(math.log(2.0 * k / delta) / (2.0 * eps**2)))


def union_bound_eps(k: int, n: int, delta: float) -> float:
    """Gap that holds for all k models at once with chance 1 - delta.

    Args:
        k: Number of models in the class.
        n: Number of examples.
        delta: Allowed failure chance.

    Returns:
        sqrt(log(2k / delta) / (2n)).
    """
    return float(math.sqrt(math.log(2.0 * k / delta) / (2.0 * n)))


# ----------------------------------------------------------------- shattering


def threshold_class(grid: np.ndarray) -> list:
    """Hypotheses h(P) = 1 when the first coordinate is at least t, for each t in grid."""
    return [lambda P, t=t: (np.asarray(P)[:, 0] >= t).astype(int) for t in grid]


def interval_class(grid: np.ndarray) -> list:
    """Hypotheses h(P) = 1 when a <= first coordinate <= b, for each a <= b in grid."""
    return [
        lambda P, a=a, b=b: ((np.asarray(P)[:, 0] >= a) & (np.asarray(P)[:, 0] <= b)).astype(int)
        for i, a in enumerate(grid)
        for b in grid[i:]
    ]


def halfplane_class(n_angles: int, offsets: np.ndarray) -> list:
    """Hypotheses h(P) = 1 when P . (cos a, sin a) >= c, for each angle a and offset c."""
    out = []
    for a in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        direction = np.array([np.cos(a), np.sin(a)])
        for c in offsets:
            out.append(lambda P, d=direction, c=c: (np.asarray(P) @ d >= c).astype(int))
    return out


def num_dichotomies(points: np.ndarray, hypotheses: list) -> int:
    """Count the distinct label patterns the hypotheses produce on the points.

    Args:
        points: Array of shape (m, d).
        hypotheses: Functions that map points to an array of m labels in {0, 1}.

    Returns:
        The number of different label tuples.
    """
    return len({tuple(int(v) for v in h(points)) for h in hypotheses})


def shatters(points: np.ndarray, hypotheses: list) -> bool:
    """True when the hypotheses produce all 2^m label patterns on the m points."""
    m = len(points)
    return num_dichotomies(points, hypotheses) == 2**m


def largest_shattered(candidates: np.ndarray, hypotheses: list) -> int:
    """Size of the largest subset of the candidate points that the class shatters.

    Args:
        candidates: Array of shape (c, d).
        hypotheses: Functions that map points to labels in {0, 1}.

    Returns:
        The largest m such that some m of the candidates are shattered, or 0.
    """
    best = 0
    for m in range(1, len(candidates) + 1):
        found = any(
            shatters(candidates[list(idx)], hypotheses)
            for idx in combinations(range(len(candidates)), m)
        )
        if not found:
            break
        best = m
    return best


# ------------------------------------------------------------------ perceptron


def perceptron(X: np.ndarray, y: np.ndarray, max_epochs: int = 1000) -> tuple[np.ndarray, int]:
    """Train a perceptron without a bias, cycling through rows in order.

    Args:
        X: Array of shape (n, d).
        y: Labels in {-1, +1}, shape (n,).
        max_epochs: Maximum passes over the rows.

    Returns:
        (w, mistakes): the weights, shape (d,), and the total number of
        updates. A row is a mistake when y_i * (w . x_i) <= 0.
    """
    w = np.zeros(X.shape[1])
    mistakes = 0
    for _ in range(max_epochs):
        clean = True
        for x_i, y_i in zip(X, y):
            if y_i * (w @ x_i) <= 0:
                w = w + y_i * x_i
                mistakes += 1
                clean = False
        if clean:
            break
    return w, mistakes


def perceptron_mistake_bound(X: np.ndarray, y: np.ndarray, w_star: np.ndarray) -> float:
    """(R / margin)^2 for data that w_star separates.

    Args:
        X: Array of shape (n, d).
        y: Labels in {-1, +1}.
        w_star: A separating direction, shape (d,).

    Returns:
        (R / gamma)^2, where R is the largest row norm and gamma is the
        smallest y_i (w_star . x_i) / ||w_star||. Returns inf when w_star
        does not separate the data.
    """
    R = np.max(np.linalg.norm(X, axis=1))
    gamma = np.min(y * (X @ w_star)) / np.linalg.norm(w_star)
    if gamma <= 0:
        return float("inf")
    return float((R / gamma) ** 2)


# --------------------------------------------------------------- double descent


def draw_problem(n: int, p: int, rng: np.random.Generator, d: int = 3, noise: float = 0.3):
    """Draw one random-features regression problem. Given in full.

    The target is f(x) = sin(x . beta) with x and beta standard normal in d
    dimensions. The training labels carry noise with standard deviation `noise`.
    The p random ReLU features have weights W of shape (d, p) and offsets b.

    Returns:
        (X, y, X_test, f_test, W, b) with X of shape (n, d), y of shape (n,),
        X_test of shape (400, d), f_test of shape (400,), W of shape (d, p),
        and b of shape (p,).
    """
    beta = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    X_test = rng.normal(size=(400, d))
    W = rng.normal(size=(d, p))
    b = rng.normal(size=p)
    y = np.sin(X @ beta) + noise * rng.normal(size=n)
    return X, y, X_test, np.sin(X_test @ beta), W, b


def relu_features(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return max(0, X W + b), shape (n, p)."""
    return np.maximum(X @ W + b, 0.0)


def min_norm_fit(F: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Weights of the smallest norm that minimise ||F w - y||^2.

    Args:
        F: Feature matrix, shape (n, p). p can be smaller or larger than n.
        y: Targets, shape (n,).

    Returns:
        w of shape (p,): the pseudo-inverse solution pinv(F) y.
    """
    return np.linalg.pinv(F) @ y


def min_norm_test_error(n: int, p: int, trials: int = 40, seed: int = 0, noise: float = 0.3) -> float:
    """Median test error of min-norm least squares on p random ReLU features.

    Each trial calls draw_problem with one shared np.random.default_rng(seed),
    fits min_norm_fit, and measures mean((F_test w - f_test)^2) + noise^2,
    which is the error against a fresh noisy label.

    Returns:
        The median over trials.
    """
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(trials):
        X, y, X_test, f_test, W, b = draw_problem(n, p, rng, noise=noise)
        w = min_norm_fit(relu_features(X, W, b), y)
        pred = relu_features(X_test, W, b) @ w
        errors.append(np.mean((pred - f_test) ** 2) + noise**2)
    return float(np.median(errors))


if __name__ == "__main__":
    print("Hoeffding, n=100, eps=0.1:", hoeffding_bound(100, 0.1))
    print("Monte Carlo:", deviation_frequency(100, 0.1, 20000, 0))
    print("Sample size for k=1000, eps=0.05, delta=0.05:", union_bound_sample_size(1000, 0.05, 0.05))
    grid = np.arange(0.5, 6.0, 1.0)
    pts = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    print("VC of thresholds:", largest_shattered(pts, threshold_class(grid)))
    print("VC of intervals:", largest_shattered(pts, interval_class(grid)))
    X = np.array([[2.0, 1.0], [-1.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])
    y = np.array([1.0, -1.0, 1.0, -1.0])
    w, mistakes = perceptron(X, y)
    print("perceptron:", w, mistakes, "bound", perceptron_mistake_bound(X, y, np.array([1.0, 0.0])))
    for p in (15, 30, 300):
        print("p =", p, "test error", min_norm_test_error(30, p, trials=20))
