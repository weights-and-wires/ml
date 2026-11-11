"""Reference solution for the approximation-theory exercise."""

from __future__ import annotations

import math

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def tent(x: np.ndarray) -> np.ndarray:
    """Fold [0, 1] onto [0, 1] with a two-unit ReLU layer."""
    x = np.asarray(x, dtype=float)
    return 2.0 * _relu(x) - 4.0 * _relu(x - 0.5)


def sawtooth(x: np.ndarray, k: int) -> np.ndarray:
    """Apply the tent map k times."""
    out = np.asarray(x, dtype=float)
    for _ in range(k):
        out = tent(out)
    return out


def count_pieces(f, xs: np.ndarray, tol: float = 1e-6) -> int:
    """Count the straight pieces of f on a sorted grid."""
    xs = np.asarray(xs, dtype=float)
    slopes = np.diff(f(xs)) / np.diff(xs)
    return 1 + int(np.sum(np.abs(np.diff(slopes)) > tol))


def deep_units_used(k: int) -> int:
    """Hidden units in k folded tent layers."""
    return 2 * k


def shallow_units_needed(k: int) -> int:
    """Fewest hidden units in one layer that can make 2**k pieces."""
    return 2**k - 1


def interp_error(f, n_knots: int, a: float = 0.0, b: float = 1.0) -> float:
    """Largest gap between f and its chord interpolant on [a, b]."""
    knots = np.linspace(a, b, n_knots)
    xs = np.linspace(a, b, 20001)
    return float(np.max(np.abs(f(xs) - np.interp(xs, knots, f(knots)))))


def loglog_slope(ns: np.ndarray, errs: np.ndarray) -> float:
    """Slope of the best line through (log n, log error)."""
    return float(np.polyfit(np.log(np.asarray(ns, dtype=float)), np.log(np.asarray(errs, dtype=float)), 1)[0])


def pieces_for_error(eps: float, curvature: float, length: float = 1.0) -> int:
    """Segments needed so that h^2 M / 8 <= eps over a length."""
    return int(math.ceil(length * math.sqrt(curvature / (8.0 * eps)) - 1e-9))


def grid_points_needed(eps: float, d: int, curvature: float = 1.0) -> int:
    """Cells of a d-dimensional grid on the unit cube."""
    return pieces_for_error(eps, curvature) ** d


if __name__ == "__main__":
    xs = np.linspace(0, 1, 2**10 + 1)
    for k in range(1, 6):
        print(k, count_pieces(lambda x: sawtooth(x, k), xs), deep_units_used(k), shallow_units_needed(k))
    ns = np.array([3, 5, 9, 17, 33, 65])
    errs = np.array([interp_error(np.sin, n, 0.0, math.pi) for n in ns])
    print("log-log slope:", loglog_slope(ns - 1, errs))
    print("cells at eps=0.01, d=10:", grid_points_needed(0.01, 10, curvature=8.0))
