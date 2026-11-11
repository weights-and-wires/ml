#!/usr/bin/env python3
"""Numbers and SVG for the Gaussian processes lesson figures.

Uses the six noisy points from the kernel ridge lesson, so the posterior mean
drawn here is the orange curve of that lesson's Figure 1 (gamma = 1, which is
length scale 1 / sqrt(2), and noise variance 0.1).

Prints the hand-check values quoted in the lesson, then the numbers behind each
figure. Pass `--out DIR` to also write figure1.svg, figure2.svg, and figure3.svg
into DIR, ready to paste into a `figure` block.

    python scripts/figures/022-gaussian-processes.py --out /tmp/gp-figs
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

X6 = np.array([0.51, 2.37, 3.21, 3.82, 5.93, 6.13])
Y6 = np.array([0.36, 0.73, -0.39, -0.58, -0.30, 0.17])
NOISE = 0.1
LS_019 = 1.0 / np.sqrt(2.0)  # gamma = 1 in the kernel ridge lesson


def rbf(a: np.ndarray, b: np.ndarray, length_scale: float) -> np.ndarray:
    sq = (a[:, None] - b[None, :]) ** 2
    return np.exp(-sq / (2.0 * length_scale**2))


def chol_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.solve(L.T, np.linalg.solve(L, b))


def lml_terms(x: np.ndarray, y: np.ndarray, ls: float, s2: float) -> tuple[float, float, float]:
    n = len(y)
    L = np.linalg.cholesky(rbf(x, x, ls) + s2 * np.eye(n))
    fit = -0.5 * float(y @ chol_solve(L, y))
    simple = -float(np.log(np.diag(L)).sum())
    const = -0.5 * n * np.log(2.0 * np.pi)
    return fit, simple, const


def hand_numbers() -> None:
    k = np.exp(-0.5)
    K = np.array([[1.0, k], [k, 1.0]])
    y = np.array([1.0, 2.0])
    print("== two-point hand example (distance 1, length scale 1)")
    print("k =", k)
    print("L =", np.linalg.cholesky(K).tolist())
    z = np.array([0.5, -1.0])
    print("L @ [0.5, -1.0] =", (np.linalg.cholesky(K) @ z).tolist())
    A = K + np.eye(2)
    w = np.linalg.solve(A, np.array([1.0, k]))
    print("(K + I)^-1 k_0 =", w.tolist(), " k_0 . that =", float(np.array([1.0, k]) @ w))
    print("latent variance at a training point =", 1.0 - float(np.array([1.0, k]) @ w))
    ks = np.exp(-0.125) * np.ones(2)
    print("mean at 0.5 =", float(ks @ np.linalg.solve(A, y)))
    print("latent variance at 0.5 =", 1.0 - float(ks @ np.linalg.solve(A, ks)))
    fit, simple, const = lml_terms(np.array([0.0, 1.0]), y, 1.0, 1.0)
    print("y^T A^-1 y =", float(y @ np.linalg.solve(A, y)))
    print("log det A =", float(np.linalg.slogdet(A)[1]))
    print("LML terms: fit", fit, "simplicity", simple, "constant", const, "total", fit + simple + const)
    print()
    x = np.linspace(0.0, 6.0, 50)
    Kg = rbf(x, x, LS_019)
    print("== 50-point grid on [0, 6], length scale", LS_019)
    print("smallest eigenvalue =", float(np.linalg.eigvalsh(Kg).min()))
    for jit in (0.0, 1e-10, 1e-8):
        try:
            np.linalg.cholesky(Kg + jit * np.eye(50))
            print("jitter", jit, "-> Cholesky works")
        except np.linalg.LinAlgError:
            print("jitter", jit, "-> Cholesky fails")
    print()


class Panel:
    def __init__(self, x0, x1, y0, y1, left, right, top, bottom):
        self.x0, self.x1, self.y0, self.y1 = x0, x1, y0, y1
        self.left, self.right, self.top, self.bottom = left, right, top, bottom

    def px(self, x):
        return self.left + (x - self.x0) / (self.x1 - self.x0) * (self.right - self.left)

    def py(self, y):
        return self.bottom - (y - self.y0) / (self.y1 - self.y0) * (self.bottom - self.top)

    def axes(self, xticks, yticks, xlabel="", zero_line=True) -> str:
        out = [
            f'<line x1="{self.left}" y1="{self.bottom}" x2="{self.right}" y2="{self.bottom}" class="stroke-muted"/>',
            f'<line x1="{self.left}" y1="{self.top}" x2="{self.left}" y2="{self.bottom}" class="stroke-muted"/>',
        ]
        if zero_line and self.y0 < 0.0 < self.y1:
            out.append(
                f'<line x1="{self.left}" y1="{self.py(0):.1f}" x2="{self.right}" y2="{self.py(0):.1f}" class="stroke-grid"/>'
            )
        for t in xticks:
            out.append(
                f'<text x="{self.px(t):.1f}" y="{self.bottom + 15}" text-anchor="middle" class="fig-small">{t:g}</text>'
            )
        for t in yticks:
            out.append(
                f'<text x="{self.left - 6}" y="{self.py(t) + 4:.1f}" text-anchor="end" class="fig-small">{t:g}</text>'
            )
        if xlabel:
            out.append(
                f'<text x="{(self.left + self.right) / 2:.1f}" y="{self.bottom + 31}" text-anchor="middle" class="fig-small">{xlabel}</text>'
            )
        return "".join(out)

    def path(self, xs, ys, cls, width=2.0, extra="") -> str:
        d = " ".join(
            ("M" if i == 0 else "L") + f"{self.px(x):.1f},{self.py(y):.1f}" for i, (x, y) in enumerate(zip(xs, ys))
        )
        return f'<path d="{d}" fill="none" class="{cls}" stroke-width="{width}"{extra}/>'

    def band(self, xs, lo, hi, cls) -> str:
        pts = [f"{self.px(x):.1f},{self.py(h):.1f}" for x, h in zip(xs, hi)]
        pts += [f"{self.px(x):.1f},{self.py(l):.1f}" for x, l in zip(xs[::-1], lo[::-1])]
        return f'<polygon points="{" ".join(pts)}" class="{cls}"/>'


def figure1(out: Path | None) -> None:
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 6.0, 80)
    draws = rng.normal(size=(80, 3))  # same standard normals for both panels
    svg = ['<svg viewBox="0 0 600 220" role="img" aria-label="Three random functions drawn from a Gaussian process prior with a short length scale on the left and a long one on the right.">']
    print("== figure 1: three prior draws, same standard normals, two length scales")
    for name, ls, left, right in (("short", 0.4, 46, 285), ("long", 1.5, 336, 575)):
        K = rbf(x, x, ls) + 1e-8 * np.eye(80)
        f = np.linalg.cholesky(K) @ draws
        p = Panel(0, 6, -3, 3, left, right, 24, 176)
        svg.append(p.axes([0, 2, 4, 6], [-2, 0, 2], "x"))
        for j, cls in enumerate(("stroke-a", "stroke-b", "stroke-c")):
            svg.append(p.path(x, f[:, j], cls, 2.0))
        svg.append(f'<text x="{(left + right) / 2:.0f}" y="14" text-anchor="middle" class="fig-bold">length scale {ls:g}</text>')
        print(name, "ls", ls, "draw ranges", [(round(float(f[:, j].min()), 2), round(float(f[:, j].max()), 2)) for j in range(3)])
        print(name, "draw standard deviations", [round(float(f[:, j].std()), 2) for j in range(3)])
        # Count sign changes as a rough wiggle count.
        print(name, "zero crossings per draw", [int((np.diff(np.sign(f[:, j])) != 0).sum()) for j in range(3)])
    svg.append("</svg>")
    if out:
        (out / "figure1.svg").write_text("\n".join(svg) + "\n")
    print()


def posterior(xs: np.ndarray, ls: float, s2: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(X6)
    L = np.linalg.cholesky(rbf(X6, X6, ls) + s2 * np.eye(n))
    Ks = rbf(xs, X6, ls)
    mean = Ks @ chol_solve(L, Y6)
    v = np.linalg.solve(L, Ks.T)
    var = 1.0 - (v**2).sum(axis=0)
    return mean, var


def figure2(out: Path | None) -> None:
    xs = np.linspace(-1.0, 8.0, 181)
    mean, var = posterior(xs, LS_019, NOISE)
    sd = np.sqrt(np.maximum(var, 0.0))
    print("== figure 2: posterior on six points, length scale", round(float(LS_019), 4), "noise", NOISE)
    for q in (0.51, 2.37, 4.5, 6.13, 8.0):
        m, v = posterior(np.array([q]), LS_019, NOISE)
        print(f"x = {q}: mean {float(m[0]):.3f}, latent variance {float(v[0]):.3f}, sd {float(np.sqrt(v[0])):.3f}")
    gap = np.linspace(4.0, 5.5, 301)
    _, vg = posterior(gap, LS_019, NOISE)
    print("largest variance between 4 and 5.5:", round(float(vg.max()), 3), "at x =", round(float(gap[int(vg.argmax())]), 2))
    print("minimum variance over the grid:", round(float(var.min()), 3), "at x =", round(float(xs[int(var.argmin())]), 2))
    print("noise level", NOISE)
    p = Panel(-1, 8, -2.6, 2.6, 50, 570, 20, 220)
    svg = ['<svg viewBox="0 0 600 260" role="img" aria-label="Six training points, the Gaussian process posterior mean, and a band of two standard deviations that pinches near the data and widens away from it.">']
    svg.append(p.axes([0, 2, 4, 6, 8], [-2, -1, 0, 1, 2], "x"))
    svg.append(p.band(xs, mean - 2 * sd, mean + 2 * sd, "fill-soft-b"))
    svg.append(p.path(xs, mean, "stroke-b", 2.2))
    for x, y in zip(X6, Y6):
        svg.append(f'<circle cx="{p.px(x):.1f}" cy="{p.py(y):.1f}" r="4.5" class="fill-a"/>')
    svg.append("</svg>")
    if out:
        (out / "figure2.svg").write_text("\n".join(svg) + "\n")
    print()


def figure3(out: Path | None) -> None:
    ls_grid = np.geomspace(0.1, 4.0, 120)
    fits, simples, totals = [], [], []
    for ls in ls_grid:
        fit, simple, const = lml_terms(X6, Y6, ls, NOISE)
        fits.append(fit)
        simples.append(simple)
        totals.append(fit + simple + const)
    best = int(np.argmax(totals))
    print("== figure 3: six points, noise", NOISE)
    for ls in (0.1, 0.2, 0.4, float(LS_019), 1.0, 1.5, 2.0, 4.0):
        fit, simple, const = lml_terms(X6, Y6, ls, NOISE)
        print(f"ls {ls:.4f}: fit {fit:.3f}, simplicity {simple:.3f}, LML {fit + simple + const:.3f}")
    print("best length scale on the grid", round(float(ls_grid[best]), 3), "LML", round(float(totals[best]), 3))
    lx = np.log10(ls_grid)
    left = Panel(np.log10(0.1), np.log10(4.0), -5, 3.6, 50, 270, 24, 190)
    right = Panel(np.log10(0.1), np.log10(4.0), -7.0, -5.2, 345, 565, 24, 190)
    svg = ['<svg viewBox="0 0 600 236" role="img" aria-label="Left: the data-fit term falls and the simplicity term rises as the length scale grows. Right: their sum, the log marginal likelihood, peaks near a length scale of 1.5.">']
    for pnl, ylab in ((left, [-4, -2, 0, 2]), (right, [-7, -6.5, -6, -5.5])):
        svg.append(pnl.axes([], ylab, zero_line=pnl is left))
        for v in (0.1, 0.3, 1.0, 3.0):
            svg.append(
                f'<text x="{pnl.px(np.log10(v)):.1f}" y="{pnl.bottom + 15}" text-anchor="middle" class="fig-small">{v:g}</text>'
            )
        svg.append(
            f'<text x="{(pnl.left + pnl.right) / 2:.1f}" y="{pnl.bottom + 31}" text-anchor="middle" class="fig-small">length scale</text>'
        )
    svg.append(left.path(lx, fits, "stroke-a", 2.2))
    svg.append(left.path(lx, simples, "stroke-c", 2.2))
    svg.append(f'<text x="{left.px(np.log10(0.1)) + 6:.0f}" y="{left.py(2.4):.0f}" class="fig-small">simplicity</text>')
    svg.append(f'<text x="{left.px(np.log10(0.8)):.0f}" y="{left.py(-4.2):.0f}" class="fig-small">data fit</text>')
    svg.append(right.path(lx, totals, "stroke-b", 2.2))
    svg.append(f'<circle cx="{right.px(lx[best]):.1f}" cy="{right.py(totals[best]):.1f}" r="4.5" class="fill-b"/>')
    svg.append(
        f'<text x="{right.px(lx[best]):.1f}" y="{right.py(totals[best]) - 10:.1f}" text-anchor="middle" class="fig-small">best</text>'
    )
    svg.append("</svg>")
    if out:
        (out / "figure3.svg").write_text("\n".join(svg) + "\n")
    print()


def cost_numbers() -> None:
    print("== cost")
    for n in (1_000, 10_000, 100_000):
        print(f"n = {n}: K holds {8 * n * n / 1e9:.2f} GB in float64, Cholesky needs about {n**3 / 3:.2e} flops")


def main() -> None:
    out = None
    if "--out" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
    hand_numbers()
    figure1(out)
    figure2(out)
    figure3(out)
    cost_numbers()


if __name__ == "__main__":
    main()
