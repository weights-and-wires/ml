"""Starter for the approximation-theory exercise. Fill in each function."""

from __future__ import annotations

import math

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def tent(x: np.ndarray) -> np.ndarray:
    """Fold [0, 1] onto [0, 1] with a two-unit ReLU layer (the tent map).

    Args:
        x: Array of numbers in [0, 1].

    Returns:
        The folded values, same shape as x, with peak 1 at x = 0.5.
    """
    raise NotImplementedError("write tent")


def sawtooth(x: np.ndarray, k: int) -> np.ndarray:
    """Apply the tent map k times.

    Args:
        x: Array of numbers in [0, 1].
        k: Number of folds (0 returns x).

    Returns:
        Array of the same shape with 2**k straight pieces.
    """
    raise NotImplementedError("write sawtooth")


def count_pieces(f, xs: np.ndarray, tol: float = 1e-6) -> int:
    """Count the straight pieces of f on a sorted grid.

    Args:
        f: Vectorised function of one variable.
        xs: Sorted 1-D grid that contains the bends.
        tol: A slope change larger than tol starts a new piece.

    Returns:
        The number of pieces, as an int (slope changes plus one).
    """
    raise NotImplementedError("write count_pieces")


def deep_units_used(k: int) -> int:
    """Hidden units in k folded tent layers.

    Args:
        k: Number of folds.

    Returns:
        The unit count, as an int.
    """
    raise NotImplementedError("write deep_units_used")


def shallow_units_needed(k: int) -> int:
    """Fewest hidden units in one layer that can make 2**k pieces.

    Args:
        k: Number of folds of the matching sawtooth.

    Returns:
        The unit count, as an int.
    """
    raise NotImplementedError("write shallow_units_needed")


def interp_error(f, n_knots: int, a: float = 0.0, b: float = 1.0) -> float:
    """Largest gap between f and its chord interpolant on [a, b].

    Args:
        f: Vectorised function of one variable.
        n_knots: Number of evenly spaced knots, including both ends.
        a: Left end.
        b: Right end.

    Returns:
        The largest absolute gap on a fine grid, as a float.
    """
    raise NotImplementedError("write interp_error")


def loglog_slope(ns: np.ndarray, errs: np.ndarray) -> float:
    """Slope of the best line through (log n, log error).

    Args:
        ns: Positive counts, shape (m,).
        errs: Positive errors, shape (m,).

    Returns:
        The fitted slope, as a float.
    """
    raise NotImplementedError("write loglog_slope")


def pieces_for_error(eps: float, curvature: float, length: float = 1.0) -> int:
    """Segments needed so that h^2 M / 8 <= eps over a length.

    Args:
        eps: Target largest error.
        curvature: Bound M on the second derivative.
        length: Length of the interval.

    Returns:
        The smallest whole number of equal segments whose error bound is
        at most eps, as an int.
    """
    raise NotImplementedError("write pieces_for_error")


def grid_points_needed(eps: float, d: int, curvature: float = 1.0) -> int:
    """Cells of a d-dimensional grid on the unit cube.

    Args:
        eps: Target largest error.
        d: Number of input dimensions.
        curvature: Bound M on the second derivative.

    Returns:
        The cell count, as a Python int.
    """
    raise NotImplementedError("write grid_points_needed")


if __name__ == "__main__":
    xs = np.linspace(0, 1, 2**10 + 1)
    for k in range(1, 6):
        print(k, count_pieces(lambda x: sawtooth(x, k), xs), deep_units_used(k), shallow_units_needed(k))
    ns = np.array([3, 5, 9, 17, 33, 65])
    errs = np.array([interp_error(np.sin, n, 0.0, math.pi) for n in ns])
    print("log-log slope:", loglog_slope(ns - 1, errs))
    print("cells at eps=0.01, d=10:", grid_points_needed(0.01, 10, curvature=8.0))
