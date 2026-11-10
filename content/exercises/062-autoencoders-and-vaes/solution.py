"""Reference solution: linear autoencoder, Gaussian KL, and a tiny VAE."""

from __future__ import annotations

import numpy as np

DECODER_NOISE = 0.05
LEARNING_RATE = 0.08


def gaussian_kl(mu: np.ndarray, logvar: np.ndarray) -> float:
    """KL of N(mu, diag(exp(logvar))) against the standard normal.

    The sum runs over every element. A matrix of codes returns the sum of
    the per-code KL values, not the mean.

    Args:
        mu: Posterior means, any shape.
        logvar: Log-variances, the same shape as mu.

    Returns:
        0.5 * sum(mu^2 + exp(logvar) - 1 - logvar), as a Python float.
    """
    mu_a = np.asarray(mu, dtype=float)
    logvar_a = np.asarray(logvar, dtype=float)
    if mu_a.shape != logvar_a.shape:
        raise ValueError(f"mu shape {mu_a.shape} does not match logvar shape {logvar_a.shape}")
    return float(0.5 * np.sum(mu_a**2 + np.exp(logvar_a) - 1.0 - logvar_a))


def gaussian_kl_grad(mu: np.ndarray, logvar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of gaussian_kl with respect to mu and logvar.

    Args:
        mu: Posterior means, any shape.
        logvar: Log-variances, the same shape as mu.

    Returns:
        A pair of arrays of the same shape. The mu gradient is mu. The
        logvar gradient is 0.5 * (exp(logvar) - 1).
    """
    mu_a = np.asarray(mu, dtype=float)
    logvar_a = np.asarray(logvar, dtype=float)
    if mu_a.shape != logvar_a.shape:
        raise ValueError(f"mu shape {mu_a.shape} does not match logvar shape {logvar_a.shape}")
    return mu_a.copy(), 0.5 * (np.exp(logvar_a) - 1.0)


def reparameterise(mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Draw a diagonal-Gaussian sample with the noise held fixed.

    Args:
        mu: Means, any shape.
        logvar: Log-variances, the same shape as mu.
        eps: Standard-normal noise, the same shape as mu.

    Returns:
        mu + exp(0.5 * logvar) * eps, the same shape as mu.
    """
    mu_a = np.asarray(mu, dtype=float)
    logvar_a = np.asarray(logvar, dtype=float)
    eps_a = np.asarray(eps, dtype=float)
    return mu_a + np.exp(0.5 * logvar_a) * eps_a


def reparameterise_grad(
    mu: np.ndarray, logvar: np.ndarray, eps: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise derivatives of reparameterise at a fixed eps.

    The map has no cross terms, so these are also the gradient of the sum
    of the sample.

    Args:
        mu: Means, any shape.
        logvar: Log-variances, the same shape as mu.
        eps: The noise held fixed, the same shape as mu.

    Returns:
        A pair of arrays. The mu derivative is 1 everywhere. The logvar
        derivative is 0.5 * exp(0.5 * logvar) * eps.
    """
    mu_a = np.asarray(mu, dtype=float)
    logvar_a = np.asarray(logvar, dtype=float)
    eps_a = np.asarray(eps, dtype=float)
    ones = np.ones_like(mu_a)
    d_logvar = 0.5 * np.exp(0.5 * logvar_a) * eps_a
    return ones, d_logvar


def elbo(x: np.ndarray, mu_q: np.ndarray, logvar_q: np.ndarray, reconstruction: np.ndarray) -> float:
    """Evidence lower bound, reconstruction term minus the KL.

    Args:
        x: The observation the reconstruction term belongs to. It must be
            finite. A scalar reconstruction is the expected log likelihood of
            this observation. An array must have the same shape as x, and
            this function sums it.
        mu_q: Posterior means.
        logvar_q: Posterior log-variances, the same shape as mu_q.
        reconstruction: Expected log p(x | z) under q, a float or an array
            the same shape as x.

    Returns:
        The reconstruction term minus gaussian_kl(mu_q, logvar_q).
    """
    observed = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(observed)):
        raise ValueError("x must be finite")
    recon = np.asarray(reconstruction, dtype=float)
    if recon.shape != () and recon.shape != observed.shape:
        raise ValueError(
            f"reconstruction shape {recon.shape} does not match x shape {observed.shape}"
        )
    return float(np.sum(recon) - gaussian_kl(mu_q, logvar_q))


def linear_autoencoder_fit(X: np.ndarray, rank: int) -> np.ndarray:
    """Decoder whose columns span the top principal components of X.

    The data are centred first, matching principal components. The columns
    are the leading right singular vectors of the centred matrix.

    Args:
        X: Data matrix, shape (n, d), one row per point.
        rank: How many components to keep. 1 <= rank <= d.

    Returns:
        Decoder, shape (d, rank). Column space matches the top principal
        subspace. Each column is a unit vector. The sign of a column follows
        the SVD.
    """
    data = np.asarray(X, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"X must be a matrix, got ndim {data.ndim}")
    centred = data - data.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:rank].T.copy()


def _loss_grads(X: np.ndarray, params: list[np.ndarray], beta: float, sigma2: float):
    """Mean negative ELBO and its gradients for a linear decoder."""
    weight_mu, bias_mu, weight_lv, bias_lv, weight_dec, bias_dec = params
    n, width = X.shape
    mu = X @ weight_mu + bias_mu
    logvar = X @ weight_lv + bias_lv
    var = np.exp(logvar)
    decoded = mu @ weight_dec + bias_dec
    # Variance that the posterior pushes through the linear decoder.
    spread = var @ (weight_dec**2)
    resid = X - decoded
    recon = 0.5 * np.sum((resid**2 + spread) / sigma2, axis=1) + 0.5 * width * np.log(
        2 * np.pi * sigma2
    )
    kl = 0.5 * np.sum(mu**2 + var - 1.0 - logvar, axis=1)
    value_recon = float(np.mean(recon))
    value_kl = float(np.mean(kl))
    inv_n = 1.0 / n
    grad_decoded = -(resid / sigma2) * inv_n
    grad_spread = np.full_like(spread, (0.5 / sigma2) * inv_n)
    grad_weight_dec = mu.T @ grad_decoded + 2.0 * weight_dec * (var.T @ grad_spread)
    grad_bias_dec = grad_decoded.sum(axis=0)
    grad_mu = grad_decoded @ weight_dec.T + beta * mu * inv_n
    # spread = var @ (W ** 2), so d(spread) / d(var) is W squared, transposed.
    grad_var = grad_spread @ (weight_dec**2).T + beta * 0.5 * inv_n
    grad_logvar = -beta * 0.5 * inv_n + grad_var * var
    grads = [
        X.T @ grad_mu,
        grad_mu.sum(axis=0),
        X.T @ grad_logvar,
        grad_logvar.sum(axis=0),
        grad_weight_dec,
        grad_bias_dec,
    ]
    return value_recon, value_kl, grads


def train_vae(
    X: np.ndarray, steps: int, seed: int, beta: float = 1.0
) -> tuple[np.ndarray, float, float]:
    """Train a linear-Gaussian VAE and draw decoder samples.

    The encoder and the decoder are affine maps with latent size 2. Decoder
    noise variance is 0.05. Adam uses learning rate 0.08. Parameter draws,
    in order, are weight_mu, bias_mu, weight_logvar, bias_logvar, weight_dec,
    bias_dec, from default_rng(seed). Samples use default_rng(seed + 1).

    Args:
        X: Data matrix, shape (n, 2).
        steps: Gradient steps. A few dozen is the whole run.
        seed: Seed for the initial weights and for the samples.
        beta: Weight on the KL. 1 is the ELBO. A large value pushes q
            toward the prior.

    Returns:
        samples: Array of shape (400, 2), drawn from the prior through the
            decoder, plus the decoder noise.
        reconstruction: Mean expected negative log likelihood, a cost.
        kl: Mean KL of q against the standard normal.
    """
    data = np.asarray(X, dtype=float)
    n, width = data.shape
    latent = 2
    rng = np.random.default_rng(seed)
    weight_mu = rng.normal(scale=0.1, size=(width, latent))
    bias_mu = np.zeros(latent)
    weight_lv = rng.normal(scale=0.05, size=(width, latent))
    bias_lv = np.full(latent, -0.5)
    weight_dec = rng.normal(scale=0.1, size=(latent, width))
    bias_dec = np.zeros(width)
    params = [weight_mu, bias_mu, weight_lv, bias_lv, weight_dec, bias_dec]
    moments = [(np.zeros_like(param), np.zeros_like(param)) for param in params]
    beta1, beta2, adam_eps = 0.9, 0.999, 1e-8
    recon, kl = 0.0, 0.0
    for step in range(1, steps + 1):
        recon, kl, grads = _loss_grads(data, params, beta, DECODER_NOISE)
        for param, grad, moment in zip(params, grads, moments):
            first, second = moment
            first += (1 - beta1) * (grad - first)
            second += (1 - beta2) * (grad * grad - second)
            first_hat = first / (1 - beta1**step)
            second_hat = second / (1 - beta2**step)
            param -= LEARNING_RATE * first_hat / (np.sqrt(second_hat) + adam_eps)
    sample_rng = np.random.default_rng(seed + 1)
    codes = sample_rng.normal(size=(400, latent))
    noise = sample_rng.normal(scale=np.sqrt(DECODER_NOISE), size=(400, width))
    samples = codes @ weight_dec + bias_dec + noise
    return samples, recon, kl


if __name__ == "__main__":
    origin = gaussian_kl(np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    print("KL at mu = 0, logvar = 0:", origin)
    sample = reparameterise(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    print("sample at eps = 1:", float(sample))
    matrix = np.array([[3.0, 1.0], [1.0, 3.0]])
    print("decoder", linear_autoencoder_fit(matrix, 1).ravel())
