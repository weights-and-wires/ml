#!/usr/bin/env python3
"""Numbers for the autoencoder and VAE figures.

Figure 1 plots the worked matrix [[3, 1], [1, 3]], the line through its mean
along the top principal component, and the line along the mean of the rows.
Figure 2 plots the ELBO of the one-dimensional linear-Gaussian model at x = 2
and posterior mean 1, as the posterior variance moves. The bound touches
log p(x) at variance 1/2 and sits strictly below it at variance 2.

Prints the hand-check values, then the SVG coordinates the lesson quotes.
"""

from __future__ import annotations

import math

import numpy as np


def gaussian_kl(mu: float, logvar: float) -> float:
    return 0.5 * (mu**2 + math.exp(logvar) - 1.0 - logvar)


def expected_loglik(x: float, mu: float, logvar: float) -> float:
    var = math.exp(logvar)
    return -0.5 * math.log(2 * math.pi) - 0.5 * ((x - mu) ** 2 + var)


def log_marginal(x: float) -> float:
    return -0.5 * math.log(4 * math.pi) - 0.5 * x**2 / 2.0


def elbo(x: float, mu: float, logvar: float) -> float:
    return expected_loglik(x, mu, logvar) - gaussian_kl(mu, logvar)


def panel(x0, x1, y0, y1, left, right, top, bottom):
    def px(x: float) -> float:
        return left + (x - x0) / (x1 - x0) * (right - left)

    def py(y: float) -> float:
        return bottom - (y - y0) / (y1 - y0) * (bottom - top)

    return px, py


def polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)


def main() -> None:
    worked = np.array([[3.0, 1.0], [1.0, 3.0]])
    centred = worked - worked.mean(axis=0)
    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    print("centred")
    print(centred)
    print("singular values", singular)
    print("top right singular vector", vt[0])
    print("unit principal component", np.array([1.0, -1.0]) / math.sqrt(2))

    x = 2.0
    true_lv = math.log(0.5)
    wide_lv = math.log(2.0)
    print("log p(2)", log_marginal(x))
    print("true recon, kl, elbo", expected_loglik(x, 1.0, true_lv), gaussian_kl(1.0, true_lv), elbo(x, 1.0, true_lv))
    print("wide recon, kl, elbo", expected_loglik(x, 1.0, wide_lv), gaussian_kl(1.0, wide_lv), elbo(x, 1.0, wide_lv))
    print("KL at 0", gaussian_kl(0.0, 0.0) + gaussian_kl(0.0, 0.0))
    print("KL at mu=(1, 0)", gaussian_kl(1.0, 0.0) + gaussian_kl(0.0, 0.0))
    print(
        "KL at logvar=(1, -1)",
        gaussian_kl(0.0, 1.0) + gaussian_kl(0.0, -1.0),
    )

    px, py = panel(-0.5, 4.5, -0.5, 4.5, 36, 320, 18, 300)
    # Line through the mean along (1, -1), and the line along the row mean.
    pc = polyline([(px(2 + t), py(2 - t)) for t in (-2.3, 2.3)])
    mean_line = polyline([(px(t), py(t)) for t in (-0.3, 4.3)])
    print("figure 1 principal line", pc)
    print("figure 1 mean line", mean_line)
    for name, point in (("p31", (3, 1)), ("p13", (1, 3)), ("mean", (2, 2)), ("origin", (0, 0))):
        print(name, f"{px(point[0]):.1f},{py(point[1]):.1f}")

    qx, qy = panel(0.2, 2.8, -3.85, -2.05, 78, 560, 36, 300)
    variances = np.linspace(0.22, 2.55, 80)
    curve = []
    for var in variances:
        curve.append((qx(float(var)), qy(elbo(x, 1.0, math.log(float(var))))))
    print("figure 2 curve", polyline(curve))
    print("log p line", f"{qx(0.2):.1f},{qy(log_marginal(x)):.1f} {qx(2.8):.1f},{qy(log_marginal(x)):.1f}")
    for var, label in ((0.5, "true"), (2.0, "wide")):
        height = elbo(x, 1.0, math.log(var))
        print(label, f"{qx(var):.1f},{qy(height):.1f}", "elbo", height)


if __name__ == "__main__":
    main()
