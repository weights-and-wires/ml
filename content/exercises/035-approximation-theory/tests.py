"""Checks for the approximation-theory exercise.

Plausible wrong implementations these checks catch:
- A tent scaled to peak at 2 (4 relu(x) - 8 relu(x - 0.5), or 4 min(x, 1 - x)).
  It passes a one-fold check by eye but fails the range check in
  test_tent_range and the 2**k piece counts in test_sawtooth_pieces.
- An error slope fitted on unlogged values. It returns a number near zero and
  fails test_loglog_slope_exact.
"""

import math

import numpy as np

from mlcheck import almost, check, close, equal, fail, has


def _grid(m=12):
    return np.linspace(0.0, 1.0, 2**m + 1)


@check("tent: values at 0, 0.25, 0.5, 0.75, 1 are 0, 0.5, 1, 0.5, 0")
def test_tent_values(sub):
    tent = has(sub, "tent")
    got = np.asarray(tent(np.array([0.0, 0.25, 0.5, 0.75, 1.0])), dtype=float)
    close(got, np.array([0.0, 0.5, 1.0, 0.5, 0.0]), atol=1e-12, label="tent on five points")


@check("tent: the output never leaves [0, 1] on a fine grid")
def test_tent_range(sub):
    tent = has(sub, "tent")
    out = np.asarray(tent(_grid(10)), dtype=float)
    if out.max() > 1.0 + 1e-12 or out.min() < -1e-12:
        fail(f"tent must map [0, 1] onto [0, 1]; got min {out.min():.4f} and max {out.max():.4f}")
    almost(out.max(), 1.0, 1e-12, label="peak of the tent")


@check("sawtooth: T(T(x)) matches the lesson's nine heights, and k = 0 returns x")
def test_sawtooth_values(sub):
    sawtooth = has(sub, "sawtooth")
    xs = np.linspace(0.0, 1.0, 9)
    close(
        np.asarray(sawtooth(xs, 2), dtype=float),
        np.array([0, 0.5, 1, 0.5, 0, 0.5, 1, 0.5, 0.0]),
        atol=1e-12,
        label="T(T(x)) on 9 points",
    )
    close(np.asarray(sawtooth(xs, 0), dtype=float), xs, atol=1e-12, label="zero folds")
    out = np.asarray(sawtooth(_grid(10), 6), dtype=float)
    if out.max() > 1.0 + 1e-9 or out.min() < -1e-9:
        fail(f"six folds left [0, 1]: min {out.min():.4f}, max {out.max():.4f}")


@check("count_pieces: a line has 1 piece, a tent has 2, a hand-built zigzag has 4")
def test_count_pieces(sub):
    count_pieces = has(sub, "count_pieces")
    xs = np.linspace(0.0, 1.0, 1025)
    equal(int(count_pieces(lambda x: 3.0 * x - 1.0, xs)), 1, label="pieces of a line")
    equal(int(count_pieces(lambda x: 2.0 * np.minimum(x, 1.0 - x), xs)), 2, label="pieces of a tent")
    knots = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    heights = np.array([0.0, 1.0, 0.5, 0.5, 2.0])
    got = int(count_pieces(lambda x: np.interp(x, knots, heights), xs))
    equal(got, 4, label="pieces of a four-segment zigzag")


@check("the sawtooth has exactly 2**k pieces for k = 1 to 8")
def test_sawtooth_pieces(sub):
    sawtooth, count_pieces = has(sub, "sawtooth"), has(sub, "count_pieces")
    xs = _grid(12)
    got = [int(count_pieces(lambda x, k=k: sawtooth(x, k), xs)) for k in range(1, 9)]
    want = [2**k for k in range(1, 9)]
    if got != want:
        fail(f"pieces for k = 1..8: expected {want}, got {got}")


@check("unit counts: deep 2k against shallow 2**k - 1, and depth wins from k = 4")
def test_unit_counts(sub):
    deep, shallow = has(sub, "deep_units_used"), has(sub, "shallow_units_needed")
    for k, d_want, s_want in [(1, 2, 1), (3, 6, 7), (4, 8, 15), (10, 20, 1023), (20, 40, 1048575)]:
        equal(int(deep(k)), d_want, label=f"deep_units_used({k})")
        equal(int(shallow(k)), s_want, label=f"shallow_units_needed({k})")
    if not all(deep(k) < shallow(k) for k in range(4, 21)):
        fail("expected the deep count to be smaller than the shallow count for every k from 4 to 20")


@check("interp_error: x^2 with 3 knots misses by 0.0625, and sin follows the h^2/8 bound")
def test_interp_error(sub):
    interp_error = has(sub, "interp_error")
    almost(float(interp_error(lambda x: x**2, 3)), 0.0625, 1e-6, label="largest error of x^2, 3 knots")
    almost(float(interp_error(lambda x: 2.0 * x + 1.0, 4)), 0.0, 1e-12, label="a line has no error")
    for n_knots in (5, 9, 17):
        h = math.pi / (n_knots - 1)
        bound = h * h / 8.0
        got = float(interp_error(np.sin, n_knots, 0.0, math.pi))
        if not (0.85 * bound <= got <= bound + 1e-9):
            fail(f"sin with {n_knots} knots: expected an error between {0.85 * bound:.6f} and {bound:.6f}, got {got:.6f}")


@check("loglog_slope: exactly -2 on c / n^2, and close to -2 for sin interpolation errors")
def test_loglog_slope(sub):
    loglog_slope, interp_error = has(sub, "loglog_slope"), has(sub, "interp_error")
    ns = np.array([2.0, 4.0, 8.0, 16.0, 32.0, 64.0])
    almost(float(loglog_slope(ns, 5.0 / ns**2)), -2.0, 1e-9, label="slope of 5 / n^2")
    almost(float(loglog_slope(ns, 3.0 * ns**-1.0)), -1.0, 1e-9, label="slope of 3 / n")
    segs = np.array([8, 16, 32, 64, 128])
    errs = np.array([float(interp_error(np.sin, int(n) + 1, 0.0, math.pi)) for n in segs])
    almost(float(loglog_slope(segs, errs)), -2.0, 0.05, label="slope of the measured sin errors")


@check("pieces_for_error and grid_points_needed: 10 per axis at eps = 0.01, 10^d cells, and the eps^(-d/2) rule")
def test_budget(sub):
    pieces, cells = has(sub, "pieces_for_error"), has(sub, "grid_points_needed")
    equal(int(pieces(0.01, 8.0)), 10, label="pieces at eps 0.01, M 8")
    equal(int(pieces(0.0025, 8.0)), 20, label="a quarter of the error needs twice the pieces")
    equal(int(pieces(1e-3, 1.0, math.pi)), 36, label="sin on [0, pi] at eps 0.001")
    equal(int(pieces(1e-4, 1.0, math.pi)), 112, label="sin on [0, pi] at eps 0.0001")
    equal(int(cells(0.01, 3, 8.0)), 1000, label="cells in d = 3")
    got = cells(0.01, 10, 8.0)
    equal(int(got), 10**10, label="cells in d = 10")
    if not isinstance(got, int):
        fail(f"grid_points_needed must return a Python int, got {type(got).__name__}")
