"""Diffusion models: add noise, then walk it back."""

from __future__ import annotations

import numpy as np


def alpha_bar(betas: np.ndarray) -> np.ndarray:
    """Cumulative product of 1 - beta, one entry per time.

    Time is 1-indexed. Entry t - 1 holds the product through step t.

    Args:
        betas: Noise schedule, shape (T,). Each entry is in (0, 1).

    Returns:
        Array of shape (T,).
    """
    return np.cumprod(1.0 - np.asarray(betas, dtype=float))


def q_sample(x0: np.ndarray, t: int, eps: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Draw the closed-form noisy point at time t.

    Args:
        x0: Clean points. Shape (d,) or (n, d).
        t: Time, an integer from 1 through len(betas).
        eps: Standard normal noise, the same shape as x0.
        betas: Noise schedule, shape (T,).

    Returns:
        Noisy points, the same shape as x0.
    """
    signal = float(alpha_bar(betas)[int(t) - 1])
    clean = np.asarray(x0, dtype=float)
    noise = np.asarray(eps, dtype=float)
    return np.sqrt(signal) * clean + np.sqrt(1.0 - signal) * noise


def ddpm_loss(model, x0: np.ndarray, t: int, eps: np.ndarray, betas: np.ndarray) -> float:
    """Mean squared error of a noise prediction at one time.

    The model is called as model(x_t, t) and returns a prediction with the
    same shape as x_t.

    Args:
        model: Callable of (x_t, t) returning a noise prediction.
        x0: Clean points. Shape (d,) or (n, d).
        t: Time, an integer from 1 through len(betas).
        eps: Standard normal noise, the same shape as x0.
        betas: Noise schedule, shape (T,).

    Returns:
        The mean of the squared coordinates, as a Python float.
    """
    noisy = q_sample(x0, t, eps, betas)
    prediction = np.asarray(model(noisy, t), dtype=float)
    residual = np.asarray(eps, dtype=float) - prediction
    return float(np.mean(residual * residual))


def linear_eps_grad(
    weight: np.ndarray, x0: np.ndarray, t: int, eps: np.ndarray, betas: np.ndarray
) -> np.ndarray:
    """Gradient of ddpm_loss for the denoiser eps_hat = weight * x_t.

    weight broadcasts over the rows of x_t.

    Args:
        weight: Per-coordinate weights, shape (d,).
        x0: Clean points. Shape (d,) or (n, d).
        t: Time, an integer from 1 through len(betas).
        eps: Standard normal noise, the same shape as x0.
        betas: Noise schedule, shape (T,).

    Returns:
        Gradient with respect to weight, shape (d,).
    """
    weights = np.asarray(weight, dtype=float)
    noisy = q_sample(x0, t, eps, betas)
    residual = np.asarray(eps, dtype=float) - noisy * weights
    product = residual * noisy
    extra = tuple(range(product.ndim - weights.ndim))
    if extra:
        product = product.sum(axis=extra)
    return -2.0 / residual.size * product


def ddpm_step(
    x: np.ndarray,
    eps_hat: np.ndarray,
    t: int,
    betas: np.ndarray,
    noise: np.ndarray,
) -> np.ndarray:
    """One ancestral reverse step.

    At t = 1 the noise argument is ignored.

    Args:
        x: Noisy points at time t. Shape (d,) or (n, d).
        eps_hat: Noise prediction, the same shape as x.
        t: Time, an integer from 1 through len(betas).
        betas: Noise schedule, shape (T,).
        noise: Standard normal noise, the same shape as x.

    Returns:
        Points at time t - 1, the same shape as x.
    """
    schedule = np.asarray(betas, dtype=float)
    cumulative = alpha_bar(schedule)
    step = int(t)
    signal = float(cumulative[step - 1])
    alpha = 1.0 - float(schedule[step - 1])
    beta = float(schedule[step - 1])
    point = np.asarray(x, dtype=float)
    prediction = np.asarray(eps_hat, dtype=float)
    mean = (point - (beta / np.sqrt(1.0 - signal)) * prediction) / np.sqrt(alpha)
    if step == 1:
        return mean
    previous = float(cumulative[step - 2])
    beta_tilde = (1.0 - previous) / (1.0 - signal) * beta
    return mean + np.sqrt(beta_tilde) * np.asarray(noise, dtype=float)


def ddpm_sample(denoiser, n: int, betas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ancestral samples in 2 dimensions.

    Draw the starting cloud from rng, then step from t = T down to t = 1.
    Draw a fresh normal from rng only when t > 1. The denoiser is called as
    denoiser(x, t) and returns a noise prediction of the same shape as x.

    Args:
        denoiser: Callable of (x, t) returning a noise prediction.
        n: Number of rows.
        betas: Noise schedule, shape (T,).
        rng: NumPy random generator.

    Returns:
        Array of shape (n, 2).
    """
    schedule = np.asarray(betas, dtype=float)
    point = rng.normal(size=(int(n), 2))
    for step in range(len(schedule), 0, -1):
        prediction = denoiser(point, step)
        if step == 1:
            noise = np.zeros_like(point)
        else:
            noise = rng.normal(size=point.shape)
        point = ddpm_step(point, prediction, step, schedule, noise)
    return point


def ddim_step(x: np.ndarray, eps_hat: np.ndarray, t: int, betas: np.ndarray) -> np.ndarray:
    """One deterministic reverse step, with no extra normal.

    At t = 1 the previous cumulative product is 1.

    Args:
        x: Noisy points at time t. Shape (d,) or (n, d).
        eps_hat: Noise prediction, the same shape as x.
        t: Time, an integer from 1 through len(betas).
        betas: Noise schedule, shape (T,).

    Returns:
        Points at time t - 1, the same shape as x.
    """
    cumulative = alpha_bar(betas)
    step = int(t)
    signal = float(cumulative[step - 1])
    previous = 1.0 if step == 1 else float(cumulative[step - 2])
    point = np.asarray(x, dtype=float)
    prediction = np.asarray(eps_hat, dtype=float)
    clean = (point - np.sqrt(1.0 - signal) * prediction) / np.sqrt(signal)
    return np.sqrt(previous) * clean + np.sqrt(1.0 - previous) * prediction


def classifier_free_guidance(
    cond: np.ndarray, uncond: np.ndarray, scale: float
) -> np.ndarray:
    """Extrapolate a conditional noise prediction away from the unconditional one.

    Scale 0 returns the unconditional prediction. Scale 1 returns the
    conditional prediction.

    Args:
        cond: Conditional noise prediction, any shape.
        uncond: Unconditional noise prediction, the same shape as cond.
        scale: Guidance scale.

    Returns:
        Guided prediction, the same shape as cond.
    """
    conditional = np.asarray(cond, dtype=float)
    unconditional = np.asarray(uncond, dtype=float)
    return unconditional + float(scale) * (conditional - unconditional)


if __name__ == "__main__":
    schedule = alpha_bar(np.array([0.1, 0.2, 0.4]))
    print("cumulative product:", schedule)
