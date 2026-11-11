"""Generative adversarial networks and how to evaluate generators."""

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
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    total = p + q
    return np.where(total > 0, p / np.where(total > 0, total, 1.0), 0.5)


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
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    safe_m = np.where(m > 0, m, 1.0)
    kl_p = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0) / safe_m), 0.0)
    kl_q = np.where(q > 0, q * np.log(np.where(q > 0, q, 1.0) / safe_m), 0.0)
    return float(0.5 * (kl_p.sum() + kl_q.sum()) * dx)


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
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    d = np.clip(np.asarray(d, dtype=float), 1e-12, 1.0 - 1e-12)
    return float(np.sum(p * np.log(d) + q * np.log(1.0 - d)) * dx)


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
    a = np.asarray(d_real_logit, dtype=float)
    b = np.asarray(d_fake_logit, dtype=float)
    # -log sigmoid(z) = softplus(-z), and log(1 - sigmoid(z)) = -softplus(z).
    d_loss = float(np.mean(softplus(-a)) + np.mean(softplus(b)))
    g_sat = float(np.mean(-softplus(b)))
    g_nonsat = float(np.mean(softplus(-b)))
    return d_loss, g_sat, g_nonsat


def generator_loss_grads(d_fake_logit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradients of the two generator losses with respect to the fake logits.

    Args:
        d_fake_logit: Discriminator logits on generated samples, shape (m,).

    Returns:
        A pair of arrays of shape (m,). The first is the gradient of
        g_saturating = mean(log(1 - D)), and the second is the gradient of
        g_nonsaturating = mean(-log D), each with respect to the logits.
    """
    b = np.asarray(d_fake_logit, dtype=float)
    m = b.size
    s = stable_sigmoid(b)
    return -s / m, (s - 1.0) / m


def sqrtm_psd(a: np.ndarray) -> np.ndarray:
    """Symmetric square root of a positive semi-definite matrix.

    Args:
        a: Symmetric matrix, shape (d, d). Eigenvalues a hair below 0 from
            rounding count as 0.

    Returns:
        The symmetric matrix r with r @ r = a, shape (d, d), built from an
        eigendecomposition.
    """
    a = np.asarray(a, dtype=float)
    vals, vecs = np.linalg.eigh(0.5 * (a + a.T))
    roots = np.sqrt(np.clip(vals, 0.0, None))
    return (vecs * roots) @ vecs.T


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
    mu1, mu2 = np.asarray(mu1, dtype=float), np.asarray(mu2, dtype=float)
    s1, s2 = np.asarray(s1, dtype=float), np.asarray(s2, dtype=float)
    root1 = sqrtm_psd(s1)
    cross = sqrtm_psd(root1 @ s2 @ root1)
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(cross))


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
    rng = np.random.default_rng(seed)
    batch = 256
    gen = np.array([1.0, 0.0])
    disc = np.zeros(3)
    opt_g, opt_d = Adam(2, 0.02), Adam(3, 0.1)

    def feats(x: np.ndarray) -> np.ndarray:
        return np.stack([x, x * x, np.ones_like(x)], axis=1)

    history = []
    for _ in range(steps):
        real = rng.normal(target_mean, target_std, batch)
        fake = gen[0] * rng.normal(size=batch) + gen[1]
        f_real, f_fake = feats(real), feats(fake)
        # d_loss = mean softplus(-logit_real) + mean softplus(logit_fake).
        grad_d = (-(1.0 - stable_sigmoid(f_real @ disc))[:, None] * f_real).mean(axis=0)
        grad_d += (stable_sigmoid(f_fake @ disc)[:, None] * f_fake).mean(axis=0)
        disc = opt_d.step(disc, grad_d)

        z = rng.normal(size=batch)
        fake = gen[0] * z + gen[1]
        logit = feats(fake) @ disc
        d_logit = (stable_sigmoid(logit) - 1.0) / batch  # non-saturating loss
        d_fake = d_logit * (disc[0] + 2.0 * disc[1] * fake)
        gen = opt_g.step(gen, np.array([np.sum(d_fake * z), np.sum(d_fake)]))
        history.append(gen.copy())
    last = np.array(history[-50:])
    return float(last[:, 1].mean()), float(np.abs(last[:, 0]).mean())


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
