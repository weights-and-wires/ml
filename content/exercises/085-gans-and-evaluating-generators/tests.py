"""Checks for the GAN and generator-evaluation exercise.

Three plausible wrong implementations are named below. Each check that would
accept one of them fails it with the expected value and the value it returns:

- fid_without_sqrtm uses trace(S1 + S2 - 2 S1 S2) and drops the matrix square
  root. It is not 0 for two identical Gaussians whose covariance is not the
  identity, so "identical Gaussians" and "scaled isotropic covariances" fail it.
- fid_elementwise_sqrt takes the trace of the elementwise square root of
  S1 S2. It fails the same checks, and the non-commuting check.
- naive_losses writes log(1 - sigmoid(b)) directly. It overflows to infinity
  at a logit of 800, so "the losses stay finite at huge logits" fails it.
"""

import numpy as np

from mlcheck import almost, check, close, fail, finite, finite_difference, has, shape


def fid_without_sqrtm(mu1, s1, mu2, s2):
    """Wrong: trace(S1 + S2 - 2 S1 S2), no matrix square root."""
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2.0 * s1 @ s2))


def fid_elementwise_sqrt(mu1, s1, mu2, s2):
    """Wrong: the elementwise square root of S1 S2 stands in for the matrix one."""
    return float(np.sum((mu1 - mu2) ** 2) + np.trace(s1 + s2 - 2.0 * np.sqrt(np.abs(s1 @ s2))))


def naive_losses(d_real_logit, d_fake_logit):
    """Wrong: log(1 - sigmoid(b)) overflows at large logits."""
    sig = lambda z: 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))
    a, b = np.asarray(d_real_logit, dtype=float), np.asarray(d_fake_logit, dtype=float)
    return (
        float(-np.mean(np.log(sig(a))) - np.mean(np.log(1.0 - sig(b)))),
        float(np.mean(np.log(1.0 - sig(b)))),
        float(-np.mean(np.log(sig(b)))),
    )


def _normal(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2.0 * np.pi))


def _grid():
    x = np.linspace(-6.0, 8.0, 2801)
    dx = x[1] - x[0]
    p, q = _normal(x, 0.0, 1.0), _normal(x, 2.0, 1.0)
    # Renormalise so each density sums to exactly 1 on the grid.
    return x, dx, p / (p.sum() * dx), q / (q.sum() * dx)


@check("optimal_discriminator returns p / (p + q), and 0.5 where both are 0")
def test_optimal_discriminator(sub):
    fn = has(sub, "optimal_discriminator")
    p = np.array([0.4, 0.0, 0.2, 0.0])
    q = np.array([0.1, 0.3, 0.2, 0.0])
    got = np.asarray(fn(p, q), dtype=float)
    finite(got, "optimal_discriminator on a hand array")
    close(got, np.array([0.8, 0.0, 0.5, 0.5]), atol=1e-12, label="D* on [0.4, 0, 0.2, 0] against [0.1, 0.3, 0.2, 0]")
    x = np.array([0.0, 1.0, 2.0])
    got = np.asarray(fn(_normal(x, 0.0, 1.0), _normal(x, 2.0, 1.0)), dtype=float)
    want = 1.0 / (1.0 + np.exp(-(2.0 - 2.0 * x)))
    close(got, want, atol=1e-12, label="D* for N(0, 1) against N(2, 1) equals sigmoid(2 - 2x)")


@check("js_divergence is 0 for equal densities, log 2 for disjoint ones, and symmetric")
def test_js(sub):
    fn = has(sub, "js_divergence")
    x, dx, p, q = _grid()
    almost(fn(p, p, dx), 0.0, 1e-12, "JS of a density with itself")
    a = np.array([0.5, 0.5, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.5, 0.5])
    almost(fn(a, b, 1.0), np.log(2.0), 1e-12, "JS of disjoint densities")
    got = fn(p, q, dx)
    almost(got, 0.33681, 2e-4, "JS(N(0, 1), N(2, 1))")
    almost(fn(q, p, dx), got, 1e-12, "JS(q, p) against JS(p, q)")


@check("The minimax value at D* equals 2 JS - log 4, and no other discriminator beats it")
def test_minimax_value(sub):
    value = has(sub, "minimax_value")
    js = has(sub, "js_divergence")
    opt = has(sub, "optimal_discriminator")
    x, dx, p, q = _grid()
    d_star = np.asarray(opt(p, q), dtype=float)
    got = value(p, q, d_star, dx)
    want = 2.0 * js(p, q, dx) - np.log(4.0)
    almost(got, want, 1e-6, "V(D*) against 2 JS - log 4")
    almost(value(p, q, np.full_like(x, 0.5), dx), -np.log(4.0), 1e-9, "V at D = 1/2 against -log 4")
    for name, other in (
        ("D* shifted by 0.05", np.clip(d_star + 0.05, 0.01, 0.99)),
        ("D* squashed toward 1/2", 0.5 + 0.5 * (d_star - 0.5)),
    ):
        v = value(p, q, other, dx)
        if not v < got:
            fail(f"V({name}) = {v:.6f} should be below V(D*) = {got:.6f}")


@check("The three GAN losses match their definitions on a small batch")
def test_gan_losses(sub):
    fn = has(sub, "gan_losses")
    a = np.array([2.0, -1.0])
    b = np.array([-0.5, 1.5])
    sig = lambda z: 1.0 / (1.0 + np.exp(-z))
    want = (
        float(np.mean(-np.log(sig(a))) + np.mean(-np.log(1.0 - sig(b)))),
        float(np.mean(np.log(1.0 - sig(b)))),
        float(np.mean(-np.log(sig(b)))),
    )
    got = fn(a, b)
    if len(got) != 3:
        fail(f"expected (d_loss, g_saturating, g_nonsaturating), got {len(got)} values")
    close(np.array(got, dtype=float), np.array(want), atol=1e-10, label="(d_loss, g_saturating, g_nonsaturating)")


@check("The losses stay finite at huge logits")
def test_loss_stability(sub):
    fn = has(sub, "gan_losses")
    got = np.array(fn(np.array([800.0, -800.0]), np.array([-800.0, 800.0])), dtype=float)
    finite(got, "losses at logits +-800")
    # d_loss = mean(softplus(-a)) + mean(softplus(b)) = 400 + 400.
    close(got, np.array([800.0, 0.0 - 400.0, 400.0]), atol=1e-8, label="(d_loss, g_saturating, g_nonsaturating) at +-800")


@check("The generator gradients match finite differences, and the saturating one vanishes at D(G(z)) near 0")
def test_generator_grads(sub):
    grads = has(sub, "generator_loss_grads")
    losses = has(sub, "gan_losses")
    b = np.array([-1.2, 0.3, 2.0, -0.4])
    g_sat, g_non = grads(b)
    shape(g_sat, b.shape, "saturating gradient")
    shape(g_non, b.shape, "non-saturating gradient")
    real = np.array([0.0])
    fd_sat = finite_difference(lambda v: losses(real, v)[1], b)
    fd_non = finite_difference(lambda v: losses(real, v)[2], b)
    close(g_sat, fd_sat, atol=1e-6, label="saturating gradient against finite differences")
    close(g_non, fd_non, atol=1e-6, label="non-saturating gradient against finite differences")
    # One fake sample the discriminator is sure about: logit -6, D = 0.00247.
    s, n = grads(np.array([-6.0]))
    if not abs(float(s[0])) < 0.01:
        fail(f"saturating gradient at logit -6 should be near 0 (about -0.00247), got {float(s[0]):.6f}")
    if not abs(float(n[0])) > 0.99:
        fail(f"non-saturating gradient at logit -6 should be near -1 (about -0.99753), got {float(n[0]):.6f}")


@check("sqrtm_psd returns the symmetric square root, including for a rank-1 matrix")
def test_sqrtm(sub):
    fn = has(sub, "sqrtm_psd")
    a = np.array([[2.0, 1.0], [1.0, 2.0]])
    r = np.asarray(fn(a), dtype=float)
    close(r @ r, a, atol=1e-10, label="sqrt(A) @ sqrt(A) for A = [[2, 1], [1, 2]]")
    close(r, r.T, atol=1e-12, label="the root is symmetric")
    r1 = np.asarray(fn(np.array([[1.0, 1.0], [1.0, 1.0]])), dtype=float)
    finite(r1, "root of a singular matrix")
    close(r1, np.full((2, 2), np.sqrt(2.0) / 2.0), atol=1e-8, label="root of [[1, 1], [1, 1]], which is sqrt(2) / 2 in every entry")


@check("frechet_distance is 0 for identical Gaussians and matches the closed forms")
def test_frechet(sub):
    fn = has(sub, "frechet_distance")
    mu = np.array([1.0, -2.0])
    s = np.array([[2.0, 0.5], [0.5, 1.0]])
    almost(fn(mu, s, mu, s), 0.0, 1e-9, "identical Gaussians with covariance [[2, 0.5], [0.5, 1]]")
    shift = np.array([3.0, 4.0])
    almost(fn(mu, s, mu + shift, s), 25.0, 1e-9, "same covariance, means 5 apart")
    # S1 = 4 I and S2 = 9 I in 3 dimensions: 3 (2 - 3)^2 = 3.
    z = np.zeros(3)
    almost(fn(z, 4.0 * np.eye(3), z, 9.0 * np.eye(3)), 3.0, 1e-9, "scaled isotropic covariances 4 I and 9 I")
    almost(fn(np.array([0.0]), np.array([[9.25]]), np.array([3.0]), np.array([[0.25]])), 9.0 + (np.sqrt(9.25) - 0.5) ** 2, 1e-9, "one dimension: shift 3, stds sqrt(9.25) and 0.5")
    # Covariances that do not commute: compare with scipy's matrix square root.
    from scipy.linalg import sqrtm

    s1 = np.array([[2.0, 0.6], [0.6, 1.0]])
    s2 = np.array([[1.0, -0.4], [-0.4, 3.0]])
    m1, m2 = np.array([0.5, 0.0]), np.array([-0.5, 1.0])
    want = float(np.sum((m1 - m2) ** 2) + np.trace(s1 + s2 - 2.0 * np.real(sqrtm(s1 @ s2))))
    almost(fn(m1, s1, m2, s2), want, 1e-8, "covariances that do not commute")
    almost(fn(m2, s2, m1, s1), want, 1e-8, "the distance is symmetric")


@check("A one-dimensional GAN matches the target mean and standard deviation")
def test_train_gan(sub):
    fn = has(sub, "train_gan_1d")
    for target_mean, target_std, seed in ((3.0, 1.5, 0), (-1.0, 0.7, 1)):
        mean, std = fn(target_mean, target_std, 300, seed)
        almost(mean, target_mean, 0.25, f"generator mean for N({target_mean}, {target_std}^2), seed {seed}")
        almost(std, target_std, 0.25, f"generator std for N({target_mean}, {target_std}^2), seed {seed}")
