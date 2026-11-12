"""Generative adversarial networks and how to evaluate generators.

stable_sigmoid, softplus, and Adam are provided. The other functions are yours.
"""

from __future__ import annotations

import numpy as np


def stable_sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function that stays finite at large magnitudes."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


def softplus(z: np.ndarray) -> np.ndarray:
    """log(1 + exp(z)), computed without overflow."""
    return np.logaddexp(0.0, np.asarray(z, dtype=float))


class Adam:
    """Adam with beta1 = 0.5, which keeps the toy GAN steady."""

    def __init__(self, size: int, lr: float, beta1: float = 0.5, beta2: float = 0.9):
        self.m = np.zeros(size)
        self.v = np.zeros(size)
        self.t = 0
        self.lr, self.beta1, self.beta2 = lr, beta1, beta2

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Return the parameters after one Adam step."""
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad * grad
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)


def optimal_discriminator(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The discriminator that maximises the GAN objective for fixed p and q.

    Args:
        p: Real density values, any shape, all at least 0.
        q: Generated density values, the same shape as p.

    Returns:
        p / (p + q) elementwise. Where p + q is 0 there is no evidence
        either way, so return 0.5.
    """
    raise NotImplementedError("write optimal_discriminator")


def js_divergence(p: np.ndarray, q: np.ndarray, dx: float) -> float:
    """Jensen-Shannon divergence of two densities on a grid, in nats.

    Args:
        p: Density values on a grid, shape (n,).
        q: Density values on the same grid, shape (n,).
        dx: Grid spacing.

    Returns:
        0.5 KL(p || m) + 0.5 KL(q || m) with m = (p + q) / 2, where the
        integral is a sum times dx and a term with density 0 contributes 0.
    """
    raise NotImplementedError("write js_divergence")


def minimax_value(p: np.ndarray, q: np.ndarray, d: np.ndarray, dx: float) -> float:
    """The GAN objective V(D, G) = E_p[log D] + E_q[log(1 - D)] on a grid.

    Args:
        p: Real density values on a grid, shape (n,).
        q: Generated density values on the same grid, shape (n,).
        d: Discriminator outputs on the same grid, shape (n,). Clip them
            into [1e-12, 1 - 1e-12] before taking logs.
        dx: Grid spacing.

    Returns:
        sum(p log d + q log(1 - d)) * dx.
    """
    raise NotImplementedError("write minimax_value")


def gan_losses(d_real_logit: np.ndarray, d_fake_logit: np.ndarray) -> tuple[float, float, float]:
    """The three losses of a GAN, from discriminator logits.

    Args:
        d_real_logit: Discriminator logits on real samples, shape (n,).
        d_fake_logit: Discriminator logits on generated samples, shape (m,).

    Returns:
        A tuple (d_loss, g_saturating, g_nonsaturating) of means:
        d_loss = mean(-log D(real)) + mean(-log(1 - D(fake))),
        g_saturating = mean(log(1 - D(fake))),
        g_nonsaturating = mean(-log D(fake)),
        where D = sigmoid(logit). Stay finite at logits of +-800.
    """
    raise NotImplementedError("write gan_losses")


def generator_loss_grads(d_fake_logit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of the two generator losses with respect to the fake logits.

    Args:
        d_fake_logit: Discriminator logits on generated samples, shape (m,).

    Returns:
        A pair of arrays of shape (m,). The first is the gradient of
        g_saturating = mean(log(1 - D)), and the second is the gradient of
        g_nonsaturating = mean(-log D), each with respect to the logits.
    """
    raise NotImplementedError("write generator_loss_grads")


def sqrtm_psd(a: np.ndarray) -> np.ndarray:
    """Symmetric square root of a positive semi-definite matrix.

    Args:
        a: Symmetric matrix, shape (d, d). Eigenvalues a hair below 0 from
            rounding count as 0.

    Returns:
        The symmetric matrix r with r @ r = a, shape (d, d), built from an
        eigendecomposition.
    """
    raise NotImplementedError("write sqrtm_psd")


def frechet_distance(mu1: np.ndarray, s1: np.ndarray, mu2: np.ndarray, s2: np.ndarray) -> float:
    """Squared Frechet distance between N(mu1, s1) and N(mu2, s2).

    Args:
        mu1: Mean, shape (d,).
        s1: Covariance, shape (d, d).
        mu2: Mean, shape (d,).
        s2: Covariance, shape (d, d).

    Returns:
        |mu1 - mu2|^2 + trace(s1 + s2 - 2 (s1^(1/2) s2 s1^(1/2))^(1/2)),
        as a Python float.
    """
    raise NotImplementedError("write frechet_distance")


def train_gan_1d(target_mean: float, target_std: float, steps: int, seed: int) -> tuple[float, float]:
    """Train a one-dimensional GAN and return the generator's mean and std.

    The generator is g(z) = a z + b with z ~ N(0, 1), starting at a = 1 and
    b = 0. The discriminator is D(x) = sigmoid(w1 x + w2 x^2 + c), starting
    at zero. Each iteration takes one discriminator step and one generator
    step on batches of 256, with the non-saturating generator loss. Both
    use Adam (beta1 = 0.5, beta2 = 0.9); the learning rate is 0.1 for the
    discriminator and 0.02 for the generator.

    Args:
        target_mean: Mean of the real data.
        target_std: Standard deviation of the real data.
        steps: Number of iterations.
        seed: Seed for numpy's default_rng.

    Returns:
        (mean, std) of the generator, that is b and |a|, each averaged over
        the last 50 iterations.
    """
    raise NotImplementedError("write train_gan_1d")


if __name__ == "__main__":
    x = np.linspace(-6.0, 8.0, 2801)
    dx = x[1] - x[0]
    p = np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    q = np.exp(-0.5 * (x - 2.0) ** 2) / np.sqrt(2 * np.pi)
    d_star = optimal_discriminator(p, q)
    print("D*(0) =", d_star[np.argmin(np.abs(x))])
    print("JS =", js_divergence(p, q, dx), " V(D*) =", minimax_value(p, q, d_star, dx))
    print("losses at logits [2], [-6]:", gan_losses(np.array([2.0]), np.array([-6.0])))
    print("FID of N(0, I) vs N((1, 0), I):", frechet_distance(np.zeros(2), np.eye(2), np.array([1.0, 0.0]), np.eye(2)))
    print("1-D GAN (mean, std):", train_gan_1d(3.0, 1.5, 300, 0))
