"""Checks for the autoencoder and VAE exercise.

Four plausible wrong implementations are named below. Each check that would
accept one of them fails it with the expected value and the value it returns:

- kl_without_logvar keeps 0.5 * sum(mu**2) and drops the logvar terms. It
  fails at logvar = [1, -1].
- sample_without_mu returns exp(0.5 * logvar) * eps and leaves mu out. It
  fails the worked sample at mu = 2.
- reconstruction_only returns the reconstruction term and drops the KL. It
  fails the wider-q case.
- mean_direction uses the mean of the rows instead of the top singular
  vector. It fails the worked matrix.
"""

import numpy as np

from mlcheck import almost, check, close, fail, finite, finite_difference, has, shape


def kl_without_logvar(mu, logvar):
    """Wrong: the exp(logvar) - 1 - logvar terms are missing."""
    del logvar
    return float(0.5 * np.sum(np.asarray(mu, dtype=float) ** 2))


def sample_without_mu(mu, logvar, eps):
    """Wrong: the sample is only the scaled noise."""
    del mu
    return np.exp(0.5 * np.asarray(logvar, dtype=float)) * np.asarray(eps, dtype=float)


def reconstruction_only(x, mu_q, logvar_q, reconstruction):
    """Wrong: the KL is not subtracted."""
    del x, mu_q, logvar_q
    return float(np.sum(np.asarray(reconstruction, dtype=float)))


def mean_direction(X, rank):
    """Wrong: the decoder follows the mean of the rows."""
    data = np.asarray(X, dtype=float)
    direction = data.mean(axis=0)
    direction = direction / np.linalg.norm(direction)
    decoder = np.zeros((data.shape[1], rank))
    decoder[:, 0] = direction
    return decoder


def expected_loglik(x, mu, logvar):
    var = np.exp(logvar)
    return -0.5 * np.log(2 * np.pi) - 0.5 * ((x - mu) ** 2 + var)


def log_marginal(x):
    return -0.5 * np.log(4 * np.pi) - 0.5 * x**2 / 2.0


def projector(basis):
    columns = np.asarray(basis, dtype=float)
    if columns.ndim == 1:
        columns = columns[:, None]
    left, values, _ = np.linalg.svd(columns, full_matrices=False)
    kept = values > 1e-10
    if not np.any(kept):
        return np.zeros((columns.shape[0], columns.shape[0]))
    return left[:, kept] @ left[:, kept].T


def principal_basis(X, rank):
    centred = np.asarray(X, dtype=float)
    centred = centred - centred.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[:rank].T


def cloud():
    rng = np.random.default_rng(0)
    mean = np.array([1.5, -0.8])
    scale = np.array([1.0, 0.55])
    return rng.normal(loc=mean, scale=scale, size=(40, 2))


@check("gaussian_kl is 0 at the origin and 0.5 at mu = [1, 0], logvar = [0, 0]")
def test_gaussian_kl_worked(sub):
    fn = has(sub, "gaussian_kl")
    almost(
        fn(np.array([0.0, 0.0]), np.array([0.0, 0.0])),
        0.0,
        1e-12,
        label="KL at mu = [0, 0], logvar = [0, 0]",
    )
    almost(
        fn(np.array([1.0, 0.0]), np.array([0.0, 0.0])),
        0.5,
        1e-12,
        label="KL at mu = [1, 0], logvar = [0, 0]",
    )


@check("gaussian_kl keeps the logvar terms, and its gradient matches a finite difference")
def test_gaussian_kl_grad(sub):
    fn = has(sub, "gaussian_kl")
    grad = has(sub, "gaussian_kl_grad")
    mu = np.array([0.0, 0.0])
    logvar = np.array([1.0, -1.0])
    want = 0.5 * (np.exp(1.0) + np.exp(-1.0) - 2.0)
    almost(fn(mu, logvar), want, 1e-12, label="KL at mu = [0, 0], logvar = [1, -1]")
    missing = kl_without_logvar(mu, logvar)
    if abs(missing - want) <= 1e-8:
        fail(
            f"kl_without_logvar returned {missing:.6g}, expected a gap from the "
            f"full KL {want:.6g} at logvar = [1, -1]"
        )
    point_mu = np.array([0.4, -0.7, 0.2])
    point_lv = np.array([0.3, -1.1, 0.8])
    got_mu, got_lv = grad(point_mu, point_lv)
    shape(got_mu, (3,), label="gradient with respect to mu")
    shape(got_lv, (3,), label="gradient with respect to logvar")
    want_mu = finite_difference(lambda v: fn(v, point_lv), point_mu.copy())
    want_lv = finite_difference(lambda v: fn(point_mu, v), point_lv.copy())
    close(got_mu, want_mu, rtol=1e-5, atol=1e-6, label="finite-difference gradient of the KL in mu")
    close(got_lv, want_lv, rtol=1e-5, atol=1e-6, label="finite-difference gradient of the KL in logvar")


@check("reparameterise adds mu, and at eps = 1, mu = 0, logvar = 0 the sample is 1")
def test_reparameterise(sub):
    fn = has(sub, "reparameterise")
    close(
        fn(np.array([0.0]), np.array([0.0]), np.array([1.0])),
        np.array([1.0]),
        atol=1e-12,
        label="sample at mu = 0, logvar = 0, eps = 1",
    )
    close(
        fn(np.array([2.0]), np.array([0.0]), np.array([1.0])),
        np.array([3.0]),
        atol=1e-12,
        label="sample at mu = 2, logvar = 0, eps = 1",
    )
    wrong = sample_without_mu(np.array([2.0]), np.array([0.0]), np.array([1.0]))
    if np.allclose(wrong, np.array([3.0]), atol=1e-8):
        fail(f"sample_without_mu returned {wrong}, expected it to miss the sample 3 at mu = 2")


@check("reparameterise_grad matches a finite difference at a fixed eps, and is 0.5 at the worked point")
def test_reparameterise_grad(sub):
    fn = has(sub, "reparameterise")
    grad = has(sub, "reparameterise_grad")
    _, worked = grad(np.array([0.0]), np.array([0.0]), np.array([1.0]))
    close(worked, np.array([0.5]), atol=1e-12, label="logvar derivative at mu = 0, logvar = 0, eps = 1")
    mu = np.array([0.2, -0.4, 1.5])
    logvar = np.array([0.0, 0.6, -0.3])
    eps = np.array([0.5, -1.2, 0.8])
    got_mu, got_lv = grad(mu, logvar, eps)
    shape(got_mu, (3,), label="derivative with respect to mu")
    shape(got_lv, (3,), label="derivative with respect to logvar")
    want_mu = finite_difference(lambda v: float(np.sum(fn(v, logvar, eps))), mu.copy())
    want_lv = finite_difference(lambda v: float(np.sum(fn(mu, v, eps))), logvar.copy())
    close(got_mu, want_mu, rtol=1e-5, atol=1e-6, label="finite-difference gradient in mu at fixed eps")
    close(got_lv, want_lv, rtol=1e-5, atol=1e-6, label="finite-difference gradient in logvar at fixed eps")


@check("elbo equals log p(x) at the true posterior and is strictly lower for a wider q")
def test_elbo(sub):
    fn = has(sub, "elbo")
    x = 2.0
    truth = log_marginal(x)
    mu_true = 1.0
    logvar_true = np.log(0.5)
    recon_true = expected_loglik(x, mu_true, logvar_true)
    got_true = fn(x, mu_true, logvar_true, recon_true)
    almost(got_true, truth, 1e-9, label="ELBO at the true posterior, x = 2")

    mu_wide = 1.0
    logvar_wide = np.log(2.0)
    recon_wide = expected_loglik(x, mu_wide, logvar_wide)
    want_wide = recon_wide - 0.5 * (mu_wide**2 + np.exp(logvar_wide) - 1.0 - logvar_wide)
    got_wide = fn(x, mu_wide, logvar_wide, recon_wide)
    almost(got_wide, want_wide, 1e-9, label="ELBO at mu = 1, variance = 2")
    if got_wide > truth - 1e-6:
        fail(
            f"wider ELBO is {got_wide:.6g}, expected a value strictly below "
            f"log p(x) = {truth:.6g}"
        )
    wrong = reconstruction_only(x, mu_wide, logvar_wide, recon_wide)
    if abs(wrong - want_wide) <= 1e-8:
        fail(
            f"reconstruction_only returned {wrong:.6g}, expected a gap from the "
            f"wider ELBO {want_wide:.6g}"
        )


@check("linear_autoencoder_fit recovers the top principal component of the worked matrix, up to sign")
def test_linear_autoencoder(sub):
    fn = has(sub, "linear_autoencoder_fit")
    worked = np.array([[3.0, 1.0], [1.0, 3.0]])
    decoder = fn(worked, 1)
    shape(decoder, (2, 1), label="decoder of the 2 by 2 matrix")
    finite(decoder, label="decoder of the 2 by 2 matrix")
    got = projector(decoder)
    want = projector(principal_basis(worked, 1))
    close(got, want, atol=1e-8, label="column space of the worked matrix")
    wrong = projector(mean_direction(worked, 1))
    if np.allclose(wrong, want, atol=1e-6):
        fail(
            f"mean_direction spanned {wrong}, expected it to miss the principal "
            f"projector {want}"
        )
    rng = np.random.default_rng(1)
    data = rng.normal(size=(6, 3))
    fitted = fn(data, 2)
    shape(fitted, (3, 2), label="decoder of a 6 by 3 matrix at rank 2")
    close(
        projector(fitted),
        projector(principal_basis(data, 2)),
        atol=1e-8,
        label="column space of a 6 by 3 matrix",
    )


@check("train_vae samples sit near the cloud after 40 steps at beta = 1")
def test_train_vae_samples(sub):
    fn = has(sub, "train_vae")
    samples, reconstruction, kl = fn(cloud(), 40, 0, 1.0)
    shape_samples = np.asarray(samples)
    if shape_samples.ndim != 2 or shape_samples.shape[1] != 2 or shape_samples.shape[0] < 200:
        fail(
            f"samples have shape {shape_samples.shape}, expected (at least 200, 2)"
        )
    finite(shape_samples, label="decoder samples")
    mean = shape_samples.mean(axis=0)
    var = shape_samples.var(axis=0)
    if not (0.3 <= mean[0] <= 2.0 and -1.2 <= mean[1] <= -0.1):
        fail(
            f"sample mean is {mean}, expected a value inside "
            "[0.3, 2.0] and [-1.2, -0.1]"
        )
    if not (0.35 <= var[0] <= 1.5 and 0.12 <= var[1] <= 0.70):
        fail(
            f"sample variance is {var}, expected a value inside "
            "[0.35, 1.5] and [0.12, 0.70]"
        )
    if not np.isfinite(reconstruction) or not np.isfinite(kl):
        fail(f"reconstruction {reconstruction} and KL {kl} must both be finite")


@check("a large beta raises the reconstruction term and lowers the KL")
def test_train_vae_beta(sub):
    fn = has(sub, "train_vae")
    data = cloud()
    _, recon_one, kl_one = fn(data, 40, 0, 1.0)
    _, recon_big, kl_big = fn(data, 40, 0, 20.0)
    if not (recon_big > recon_one + 1.0):
        fail(
            f"reconstruction at beta = 20 is {recon_big:.6g}, expected a cost at "
            f"least 1 above the beta = 1 cost {recon_one:.6g}"
        )
    if not (kl_big < kl_one - 0.5):
        fail(
            f"KL at beta = 20 is {kl_big:.6g}, expected a value at least 0.5 below "
            f"the beta = 1 KL {kl_one:.6g}"
        )
