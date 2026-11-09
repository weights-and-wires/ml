#!/usr/bin/env python3
"""Numbers for the likelihood, MAP, and Bayesian regression figures.

The lesson SVG uses these posteriors. Figure 1 plots size from -0.4 to 6.4
on the left-hand band and the right-hand variance curve. Figure 2 plots the
one-standard-deviation ellipses of the prior, posterior(1), and posterior(2).

The chapter's two houses are a studio, size 0, sold at the baseline price 0,
and a house of size 2 sold at price 2. The model is an intercept plus a slope.
Noise variance and prior variance are both 1. A second observation of the same
two houses is the "more data" posterior.

Prints the hand-check values used in the lesson, then writes nothing else.
The SVG polylines are printed after the numbers so the lesson can quote them.
"""

from __future__ import annotations

import numpy as np

SIGMA2 = 1.0
TAU2 = 1.0


def design(copies: int = 1) -> tuple[np.ndarray, np.ndarray]:
    row_x = np.array([[1.0, 0.0], [1.0, 2.0]])
    row_y = np.array([0.0, 2.0])
    return np.vstack([row_x] * copies), np.concatenate([row_y] * copies)


def posterior(copies: int) -> tuple[np.ndarray, np.ndarray]:
    x_mat, y = design(copies)
    lam = SIGMA2 / TAU2
    gram = x_mat.T @ x_mat + lam * np.eye(2)
    mean = np.linalg.solve(gram, x_mat.T @ y)
    cov = SIGMA2 * np.linalg.solve(gram, np.eye(2))
    return mean, cov


def features(size: float) -> np.ndarray:
    return np.array([1.0, size])


def pred_var(size: float, cov: np.ndarray) -> float:
    x = features(size)
    return float(SIGMA2 + x @ cov @ x)


def neg_log_joint(w: np.ndarray) -> float:
    x_mat, y = design(1)
    resid = x_mat @ w - y
    nll = 0.5 * len(y) * np.log(2 * np.pi * SIGMA2) + 0.5 * np.dot(resid, resid) / SIGMA2
    prior = 0.5 * len(w) * np.log(2 * np.pi * TAU2) + 0.5 * np.dot(w, w) / TAU2
    return float(nll + prior)


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def panel_map(x0: float, x1: float, y0: float, y1: float, left: float, right: float, top: float, bottom: float):
    def px(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - y0) / (y1 - y0) * (bottom - top)

    return px, py


def ellipse(mean: np.ndarray, cov: np.ndarray, px, py, steps: int = 80) -> str:
    vals, vecs = np.linalg.eigh(cov)
    angles = np.linspace(0, 2 * np.pi, steps, endpoint=True)
    circle = np.stack([np.cos(angles), np.sin(angles)])
    pts = mean[:, None] + vecs @ (np.sqrt(vals)[:, None] * circle)
    return polyline([(px(x), py(y)) for x, y in pts.T])


def main() -> None:
    mean1, cov1 = posterior(1)
    mean2, cov2 = posterior(2)
    sizes = [0.0, 2.0 / 3.0, 2.0, 6.0]
    print("posterior 1 mean", mean1, "cov", cov1, "trace", np.trace(cov1))
    print("posterior 2 mean", mean2, "cov", cov2, "trace", np.trace(cov2))
    for size in sizes:
        x = features(size)
        quad = float(x @ cov1 @ x)
        print(
            f"size {size:.4f} mean {float(np.dot(x, mean1)):.6f} "
            f"quad {quad:.6f} var {pred_var(size, cov1):.6f} std {np.sqrt(pred_var(size, cov1)):.6f}"
        )
    for label, w in (
        ("mle", np.array([0.0, 1.0])),
        ("map", mean1),
        ("zero", np.array([0.0, 0.0])),
    ):
        print(label, "joint", neg_log_joint(w))

    # Minimum of the predictive variance along size.
    grid = np.linspace(-1.0, 8.0, 9001)
    variances = np.array([pred_var(s, cov1) for s in grid])
    print("variance minimum at", grid[int(np.argmin(variances))], variances.min())

    # Figure 1. Left: price band. Right: predictive variance.
    px, py = panel_map(-1.0, 7.0, -2.0, 8.0, 46, 292, 28, 214)
    qx, qy = panel_map(-1.0, 7.0, 0.0, 10.0, 348, 600, 28, 214)
    curve_x = np.linspace(-1.0, 7.0, 97)
    upper = []
    lower = []
    mean_line = []
    var_line = []
    for size in curve_x:
        x = features(float(size))
        height = float(np.dot(x, mean1))
        spread = float(np.sqrt(pred_var(float(size), cov1)))
        upper.append((px(size), py(height + spread)))
        lower.append((px(size), py(height - spread)))
        mean_line.append((px(size), py(height)))
        var_line.append((qx(size), qy(pred_var(float(size), cov1))))
    least_squares = [(px(s), py(s)) for s in curve_x]
    band = polyline(upper + lower[::-1])

    def ticks_x(mapper, y_at: float) -> str:
        parts = []
        for tick in (0, 2, 6):
            parts.append(
                f'<line x1="{mapper(tick):.1f}" y1="{y_at:.1f}" x2="{mapper(tick):.1f}" '
                f'y2="{y_at + 4:.1f}" class="stroke-muted"/>'
                f'<text x="{mapper(tick):.1f}" y="{y_at + 16:.1f}" text-anchor="middle" '
                f'class="fig-small">{tick}</text>'
            )
        return "".join(parts)

    print("FIGURE1_BAND", band)
    print("FIGURE1_MEAN", polyline(mean_line))
    print("FIGURE1_LS", polyline(least_squares))
    print("FIGURE1_VAR", polyline(var_line))
    print("house0", px(0), py(0), "house2", px(2), py(2))
    print("var marks", [(qx(s), qy(pred_var(s, cov1))) for s in (0, 2 / 3, 2, 6)])
    print("mean marks", [(px(s), py(float(np.dot(features(s), mean1)))) for s in (0, 2, 6)])

    # Figure 2. One-standard-deviation ellipses in weight space.
    rx, ry = panel_map(-1.2, 1.5, -0.4, 1.6, 70, 330, 36, 236)
    prior = ellipse(np.zeros(2), np.eye(2), rx, ry)
    post1 = ellipse(mean1, cov1, rx, ry)
    post2 = ellipse(mean2, cov2, rx, ry)
    print("FIGURE2_PRIOR", prior)
    print("FIGURE2_POST1", post1)
    print("FIGURE2_POST2", post2)
    print("points", "origin", rx(0), ry(0), "ls", rx(0), ry(1), "map", rx(mean1[0]), ry(mean1[1]), "map2", rx(mean2[0]), ry(mean2[1]))
    print("eig1", np.linalg.eigh(cov1)[0], "eig2", np.linalg.eigh(cov2)[0])


if __name__ == "__main__":
    main()
