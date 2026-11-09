#!/usr/bin/env python3
"""Numbers and SVG coordinates for the diffusion lesson.

Figure 1 draws the two-blob cloud and the same points after the closed-form
forward step at t = 3 of the worked schedule beta = [0.1, 0.2, 0.4].
Figure 2 draws ancestral samples from the exact denoiser of that mixture.

The script also prints the worked arithmetic, the Monte Carlo checks, the
seed-0 sample moments, and the short MLP run quoted in the lesson.
"""

from __future__ import annotations

import numpy as np


HAND_BETAS = np.array([0.1, 0.2, 0.4])
X0 = np.array([1.5, 0.0])
EPS = np.array([1.0, -1.0])
WEIGHT = np.array([0.4, 0.1])
NOISE = np.array([0.5, -0.25])
MEANS = np.array([[-1.5, 0.0], [1.5, 0.0]])
SAMPLE_BETAS = np.linspace(1e-4, 0.1, 80)


def alpha_bar(betas: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 - np.asarray(betas, dtype=float))


def q_sample(x0, t, eps, betas):
    a = float(alpha_bar(betas)[int(t) - 1])
    return np.sqrt(a) * np.asarray(x0, dtype=float) + np.sqrt(1.0 - a) * np.asarray(eps, dtype=float)


def mixture_eps(x, t, betas, means):
    a = float(alpha_bar(betas)[int(t) - 1])
    centers = np.sqrt(a) * means
    diff = x[:, None, :] - centers[None, :, :]
    logp = -0.5 * np.sum(diff * diff, axis=-1)
    logp -= logp.max(axis=1, keepdims=True)
    resp = np.exp(logp)
    resp /= resp.sum(axis=1, keepdims=True)
    post = (1.0 - a) * (resp @ means) + np.sqrt(a) * x
    return (x - np.sqrt(a) * post) / np.sqrt(1.0 - a)


def ddpm_step(x, eps_hat, t, betas, noise):
    abar = alpha_bar(betas)
    t = int(t)
    abar_t = float(abar[t - 1])
    alpha_t = 1.0 - float(betas[t - 1])
    beta_t = float(betas[t - 1])
    mean = (x - (beta_t / np.sqrt(1.0 - abar_t)) * eps_hat) / np.sqrt(alpha_t)
    if t == 1:
        return mean
    beta_tilde = (1.0 - float(abar[t - 2])) / (1.0 - abar_t) * beta_t
    return mean + np.sqrt(beta_tilde) * noise


def ddim_step(x, eps_hat, t, betas):
    abar = alpha_bar(betas)
    t = int(t)
    abar_t = float(abar[t - 1])
    abar_prev = 1.0 if t == 1 else float(abar[t - 2])
    x0_hat = (x - np.sqrt(1.0 - abar_t) * eps_hat) / np.sqrt(abar_t)
    return np.sqrt(abar_prev) * x0_hat + np.sqrt(1.0 - abar_prev) * eps_hat


def ddpm_sample(denoiser, n, betas, rng):
    x = rng.normal(size=(int(n), 2))
    for t in range(len(betas), 0, -1):
        eps_hat = denoiser(x, t)
        noise = np.zeros_like(x) if t == 1 else rng.normal(size=x.shape)
        x = ddpm_step(x, eps_hat, t, betas, noise)
    return x


def moments(x):
    mean = x.mean(axis=0)
    centered = x - mean
    cov = (centered.T @ centered) / len(x)
    return mean, cov


def panel(x0, x1, y0, y1, left, right, top, bottom):
    def px(x):
        return left + (x - x0) / (x1 - x0) * (right - left)

    def py(y):
        return bottom - (y - y0) / (y1 - y0) * (bottom - top)

    return px, py


def circles(points, px, py, cls, limit=5.2):
    parts = []
    for x, y in np.asarray(points, dtype=float):
        if abs(x) > limit or abs(y) > limit:
            continue
        parts.append(f'  <circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4" class="{cls}"/>')
    return parts


def train_mlp(steps=1500):
    """One hidden layer, width 48, Adam, seed 0. Returns the loss pair and samples."""
    betas = np.linspace(1e-4, 0.2, 40)
    abar = alpha_bar(betas)
    rng = np.random.default_rng(0)
    width = 48
    w1 = rng.normal(scale=0.4, size=(width, 4))
    b1 = np.zeros(width)
    w2 = rng.normal(scale=0.2, size=(2, width))
    b2 = np.zeros(2)
    params = [w1, b1, w2, b2]
    moment = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    lr, beta1, beta2 = 0.03, 0.9, 0.999
    batch = 256
    losses = []
    for step in range(1, steps + 1):
        which = rng.integers(0, 2, size=batch)
        clean = MEANS[which] + rng.normal(size=(batch, 2))
        t = int(rng.integers(1, len(betas) + 1))
        eps = rng.normal(size=(batch, 2))
        a = float(abar[t - 1])
        xt = np.sqrt(a) * clean + np.sqrt(1.0 - a) * eps
        feat = np.concatenate(
            [xt, np.full((batch, 1), np.sqrt(a)), np.full((batch, 1), np.sqrt(1.0 - a))],
            axis=1,
        )
        pre = feat @ w1.T + b1
        hidden = np.maximum(0.0, pre)
        pred = hidden @ w2.T + b2
        resid = pred - eps
        losses.append(float(np.mean(resid**2)))
        dpred = (2.0 / resid.size) * resid
        dhidden = dpred @ w2
        dpre = dhidden * (pre > 0)
        grads = [dpre.T @ feat, dpre.sum(axis=0), dpred.T @ hidden, dpred.sum(axis=0)]
        for i, (param, grad) in enumerate(zip(params, grads)):
            moment[i] = beta1 * moment[i] + (1.0 - beta1) * grad
            second[i] = beta2 * second[i] + (1.0 - beta2) * (grad * grad)
            param -= lr * (moment[i] / (1.0 - beta1**step)) / (
                np.sqrt(second[i] / (1.0 - beta2**step)) + 1e-8
            )

    def denoiser(x, t):
        a = float(abar[int(t) - 1])
        feat = np.concatenate(
            [x, np.full((len(x), 1), np.sqrt(a)), np.full((len(x), 1), np.sqrt(1.0 - a))],
            axis=1,
        )
        hidden = np.maximum(0.0, feat @ w1.T + b1)
        return hidden @ w2.T + b2

    samples = ddpm_sample(denoiser, 2000, betas, np.random.default_rng(1))
    return losses[0], float(np.mean(losses[-20:])), moments(samples)


def main() -> None:
    abar = alpha_bar(HAND_BETAS)
    print("hand abar", abar)
    xt = q_sample(X0, 2, EPS, HAND_BETAS)
    a = float(abar[1])
    eps_hat = np.sqrt(1.0 - a) * xt
    print("xt", xt)
    print("eps_hat", eps_hat)
    print("worked loss", float(np.mean((EPS - eps_hat) ** 2)))
    print("swapped loss", float(np.mean((EPS - np.sqrt(a) * xt) ** 2)))
    print("conditional score", -EPS / np.sqrt(1.0 - a))
    resid = EPS - WEIGHT * xt
    print("linear loss", float(np.mean(resid**2)))
    print("linear grad", -resid * xt)
    print("ddim", ddim_step(xt, EPS, 2, HAND_BETAS))
    print("ddpm mean", ddpm_step(xt, EPS, 2, HAND_BETAS, np.zeros(2)))
    print("ddpm step", ddpm_step(xt, EPS, 2, HAND_BETAS, NOISE))
    print("q_sample t=1", q_sample(X0, 1, EPS, HAND_BETAS))
    long = alpha_bar(np.linspace(1e-4, 0.02, 1000))
    print("abar_T_1000", float(long[-1]))
    print("sample abar_T", float(alpha_bar(SAMPLE_BETAS)[-1]))

    rng = np.random.default_rng(0)
    draw = rng.normal(size=(4000, 2))
    mc = q_sample(X0, 2, draw, HAND_BETAS)
    print("mc mean", mc.mean(0), "mc var", mc.var(0))
    loss_rng = np.random.default_rng(0)
    x0s = loss_rng.normal(size=(4000, 2))
    epss = loss_rng.normal(size=(4000, 2))
    xts = q_sample(x0s, 2, epss, HAND_BETAS)
    print("mc optimal loss", float(np.mean((epss - np.sqrt(1.0 - a) * xts) ** 2)))

    denoiser = lambda x, t: mixture_eps(x, t, SAMPLE_BETAS, MEANS)
    samples = ddpm_sample(denoiser, 2500, SAMPLE_BETAS, np.random.default_rng(0))
    mean, cov = moments(samples)
    print("seed0 mean", mean)
    print("seed0 cov", cov)
    cxx, cyy = [], []
    for seed in range(8):
        drawn = ddpm_sample(denoiser, 2500, SAMPLE_BETAS, np.random.default_rng(seed))
        _, cov_s = moments(drawn)
        cxx.append(float(cov_s[0, 0]))
        cyy.append(float(cov_s[1, 1]))
    print("seeds 0-7 cxx", min(cxx), max(cxx))
    print("seeds 0-7 cyy", min(cyy), max(cyy))

    first, last, (mlp_mean, mlp_cov) = train_mlp()
    print("mlp first", first, "mlp last20", last)
    print("mlp mean", mlp_mean)
    print("mlp cov", mlp_cov)

    # Figure 1. Same 36 points per blob, clean and after t = 3.
    cloud_rng = np.random.default_rng(0)
    west = MEANS[0] + cloud_rng.normal(size=(36, 2))
    east = MEANS[1] + cloud_rng.normal(size=(36, 2))
    clean = np.vstack([west, east])
    cloud_eps = cloud_rng.normal(size=clean.shape)
    noised = q_sample(clean, 3, cloud_eps, HAND_BETAS)
    px, py = panel(-5.0, 5.0, -4.2, 4.2, 48, 292, 46, 286)
    qx, qy = panel(-5.0, 5.0, -4.2, 4.2, 356, 600, 46, 286)
    print("FIGURE1")
    print("\n".join(circles(west, px, py, "fill-a")))
    print("\n".join(circles(east, px, py, "fill-d")))
    print("\n".join(circles(noised[:36], qx, qy, "fill-a")))
    print("\n".join(circles(noised[36:], qx, qy, "fill-d")))
    print("ENDFIGURE1")

    # Figure 2. Data cloud and 80 ancestral samples.
    show = ddpm_sample(denoiser, 80, SAMPLE_BETAS, np.random.default_rng(2))
    data_rng = np.random.default_rng(3)
    data = np.vstack(
        [
            MEANS[0] + data_rng.normal(size=(30, 2)),
            MEANS[1] + data_rng.normal(size=(30, 2)),
        ]
    )
    rx, ry = panel(-5.0, 5.0, -3.6, 3.6, 78, 560, 42, 292)
    print("FIGURE2")
    print("\n".join(circles(data[:30], rx, ry, "fill-soft-a")))
    print("\n".join(circles(data[30:], rx, ry, "fill-soft-d")))
    print("\n".join(circles(show, rx, ry, "fill-b")))
    for name, point in (("west", MEANS[0]), ("east", MEANS[1])):
        print(f"  mean {name} {rx(point[0]):.1f},{ry(point[1]):.1f}")
    print("ENDFIGURE2")


if __name__ == "__main__":
    main()
