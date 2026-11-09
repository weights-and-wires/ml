"""Checks for the diffusion exercise.

The check that would accept a missing signal scale fails it with the expected
mean and the mean it returns:

- q_sample_without_signal_scale returns x0 + sqrt(1 - alpha_bar) * eps and
  drops the factor on the clean point. The variance still matches. The Monte
  Carlo mean stays at x0.

Other wrong implementations the same checks reject: alpha_bar that returns
1 - beta without the product, a noise predictor that uses sqrt(alpha_bar) * x_t
instead of sqrt(1 - alpha_bar) * x_t, an ancestral step that uses sqrt(beta)
instead of sqrt(beta_tilde), a deterministic step that returns only the
predicted clean point, and guidance that adds scale * cond without subtracting
the unconditional prediction first.
"""

import numpy as np

from mlcheck import almost, check, close, fail, finite, finite_difference, has, shape


HAND_BETAS = np.array([0.1, 0.2, 0.4])
X0 = np.array([1.5, 0.0])
EPS = np.array([1.0, -1.0])
WEIGHT = np.array([0.4, 0.1])
STEP_NOISE = np.array([0.5, -0.25])
MEANS = np.array([[-1.5, 0.0], [1.5, 0.0]])
SAMPLE_BETAS = np.linspace(1e-4, 0.1, 80)

WORKED_X2 = np.array([1.80194247, -0.52915026])
WORKED_X1 = np.array([1.73925271, -0.31622777])
WORKED_LOSS = 0.26028120267409727
SWAPPED_LOSS = 0.291721024500157
WORKED_GRAD = np.array([-0.5031438, -0.50115026])
WORKED_STEP = np.array([1.72568642, -0.23584616])
OPTIMAL_MC = 0.7183390299092318
LONG_LAST = 4.035829765375676e-05
SAMPLE_LAST = 0.015835450686501273
SEED0_MEAN = np.array([0.01894397, -0.00261531])
SEED0_COV = np.array([[3.10784091, -0.08012027], [-0.08012027, 0.94291266]])


def q_sample_without_signal_scale(x0, t, eps, betas):
    """Wrong: the clean point is not multiplied by sqrt(alpha_bar)."""
    signal = float(np.cumprod(1.0 - np.asarray(betas, dtype=float))[int(t) - 1])
    return np.asarray(x0, dtype=float) + np.sqrt(1.0 - signal) * np.asarray(eps, dtype=float)


def signal_at(t, betas):
    return float(np.cumprod(1.0 - np.asarray(betas, dtype=float))[int(t) - 1])


def closed_form(x0, t, eps, betas):
    signal = signal_at(t, betas)
    return np.sqrt(signal) * np.asarray(x0, dtype=float) + np.sqrt(1.0 - signal) * np.asarray(
        eps, dtype=float
    )


def ancestral_step(x, eps_hat, t, betas, noise):
    schedule = np.asarray(betas, dtype=float)
    cumulative = np.cumprod(1.0 - schedule)
    step = int(t)
    kept = float(cumulative[step - 1])
    beta = float(schedule[step - 1])
    mean = (x - (beta / np.sqrt(1.0 - kept)) * eps_hat) / np.sqrt(1.0 - beta)
    if step == 1:
        return mean
    previous = float(cumulative[step - 2])
    beta_tilde = (1.0 - previous) / (1.0 - kept) * beta
    return mean + np.sqrt(beta_tilde) * noise


def ancestral_with_beta(x, eps_hat, t, betas, noise):
    """Wrong: the added noise has scale sqrt(beta) instead of sqrt(beta_tilde)."""
    schedule = np.asarray(betas, dtype=float)
    cumulative = np.cumprod(1.0 - schedule)
    step = int(t)
    kept = float(cumulative[step - 1])
    beta = float(schedule[step - 1])
    mean = (x - (beta / np.sqrt(1.0 - kept)) * eps_hat) / np.sqrt(1.0 - beta)
    if step == 1:
        return mean
    return mean + np.sqrt(beta) * noise


def reverse_without_alpha(x, eps_hat, t, betas, noise):
    """Wrong: the reverse mean is not divided by sqrt(alpha_t)."""
    schedule = np.asarray(betas, dtype=float)
    cumulative = np.cumprod(1.0 - schedule)
    step = int(t)
    kept = float(cumulative[step - 1])
    beta = float(schedule[step - 1])
    mean = x - (beta / np.sqrt(1.0 - kept)) * eps_hat
    if step == 1:
        return mean
    previous = float(cumulative[step - 2])
    beta_tilde = (1.0 - previous) / (1.0 - kept) * beta
    return mean + np.sqrt(beta_tilde) * noise


def deterministic_step(x, eps_hat, t, betas):
    cumulative = np.cumprod(1.0 - np.asarray(betas, dtype=float))
    step = int(t)
    kept = float(cumulative[step - 1])
    previous = 1.0 if step == 1 else float(cumulative[step - 2])
    clean = (x - np.sqrt(1.0 - kept) * eps_hat) / np.sqrt(kept)
    return np.sqrt(previous) * clean + np.sqrt(1.0 - previous) * eps_hat


def mixture_eps(x, t, betas, means):
    kept = signal_at(t, betas)
    centers = np.sqrt(kept) * means
    gap = x[:, None, :] - centers[None, :, :]
    log_prob = -0.5 * np.sum(gap * gap, axis=-1)
    log_prob -= log_prob.max(axis=1, keepdims=True)
    share = np.exp(log_prob)
    share /= share.sum(axis=1, keepdims=True)
    clean = (1.0 - kept) * (share @ means) + np.sqrt(kept) * x
    return (x - np.sqrt(kept) * clean) / np.sqrt(1.0 - kept)


def population_moments(rows):
    mean = rows.mean(axis=0)
    centered = rows - mean
    cov = (centered.T @ centered) / len(rows)
    return mean, cov


def reference_sample(denoiser, n, betas, rng, step):
    schedule = np.asarray(betas, dtype=float)
    point = rng.normal(size=(int(n), 2))
    for time in range(len(schedule), 0, -1):
        prediction = denoiser(point, time)
        if time == 1:
            noise = np.zeros_like(point)
        else:
            noise = rng.normal(size=point.shape)
        point = step(point, prediction, time, schedule, noise)
    return point


def in_window(mean, cov):
    return (
        abs(float(mean[0])) <= 0.15
        and abs(float(mean[1])) <= 0.15
        and 2.80 <= float(cov[0, 0]) <= 3.55
        and 0.80 <= float(cov[1, 1]) <= 1.15
        and abs(float(cov[0, 1])) <= 0.20
    )


def report_window(mean, cov):
    return (
        f"mean {mean[0]:.6g}, {mean[1]:.6g} (expected each coordinate inside "
        f"[-0.15, 0.15]); covariance {cov[0, 0]:.6g}, {cov[0, 1]:.6g}, "
        f"{cov[1, 1]:.6g} (expected x variance inside [2.80, 3.55], y variance "
        f"inside [0.80, 1.15], off-diagonal inside [-0.20, 0.20])"
    )


@check("alpha_bar is the cumulative product, and the long schedule ends near 0")
def test_alpha_bar(sub):
    fn = has(sub, "alpha_bar")
    got = fn(HAND_BETAS)
    shape(got, (3,), label="alpha_bar on a schedule of length 3")
    close(got, np.array([0.9, 0.72, 0.432]), atol=1e-12, label="alpha_bar on (0.1, 0.2, 0.4)")
    plain = 1.0 - HAND_BETAS
    if np.allclose(plain, np.array([0.9, 0.72, 0.432]), atol=1e-6):
        fail(
            f"1 - beta is {plain}, and that matches the cumulative product "
            f"[0.9, 0.72, 0.432]; the hand schedule would not catch a missing product"
        )
    long = fn(np.linspace(1e-4, 0.02, 1000))
    shape(long, (1000,), label="alpha_bar on a schedule of length 1000")
    almost(float(long[-1]), LONG_LAST, 1e-12, label="alpha_bar at step 1000")
    if not (0.0 < float(long[-1]) < 1e-4):
        fail(f"last alpha_bar is {float(long[-1]):.6g}, expected a value inside (0, 1e-4)")
    sample = fn(SAMPLE_BETAS)
    almost(float(sample[-1]), SAMPLE_LAST, 1e-12, label="alpha_bar at step 80")


@check("q_sample scales the clean point, and the Monte Carlo mean and variance match")
def test_q_sample(sub):
    fn = has(sub, "q_sample")
    close(fn(X0, 2, EPS, HAND_BETAS), WORKED_X2, atol=1e-6, label="q_sample at t = 2")
    close(fn(X0, 1, EPS, HAND_BETAS), WORKED_X1, atol=1e-6, label="q_sample at t = 1")
    rng = np.random.default_rng(0)
    draw = rng.normal(size=(4000, 2))
    samples = fn(X0, 2, draw, HAND_BETAS)
    shape(samples, (4000, 2), label="Monte Carlo q_sample")
    mean = samples.mean(axis=0)
    var = samples.var(axis=0)
    target_mean = np.sqrt(0.72) * X0
    close(mean, target_mean, atol=0.03, label="Monte Carlo mean at t = 2")
    close(var, np.array([0.28, 0.28]), atol=0.03, label="Monte Carlo variance at t = 2")
    wrong = q_sample_without_signal_scale(X0, 2, draw, HAND_BETAS).mean(axis=0)
    if np.allclose(wrong, target_mean, atol=0.03):
        fail(
            f"q_sample_without_signal_scale mean is {wrong}, inside 0.03 of the "
            f"scaled mean {target_mean}; the mean check would not catch a missing signal scale"
        )


@check("ddpm_loss at the optimal denoiser is 0.260281 on the worked pair and near 0.72 in expectation")
def test_ddpm_loss(sub):
    fn = has(sub, "ddpm_loss")

    def optimal(x, t):
        return np.sqrt(1.0 - signal_at(t, HAND_BETAS)) * x

    def swapped(x, t):
        return np.sqrt(signal_at(t, HAND_BETAS)) * x

    got = fn(optimal, X0, 2, EPS, HAND_BETAS)
    almost(got, WORKED_LOSS, 5e-7, label="loss of sqrt(1 - alpha_bar) * x_t on the worked pair")
    swapped_got = fn(swapped, X0, 2, EPS, HAND_BETAS)
    almost(swapped_got, SWAPPED_LOSS, 5e-7, label="loss of sqrt(alpha_bar) * x_t on the worked pair")
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(4000, 2))
    noise = rng.normal(size=(4000, 2))
    batch = fn(optimal, clean, 2, noise, HAND_BETAS)
    almost(batch, OPTIMAL_MC, 1e-6, label="mean loss of the optimal denoiser, seed 0, 4000 pairs")
    almost(batch, 0.72, 0.05, label="mean loss against the expected value 0.72")
    swapped_batch = fn(swapped, clean, 2, noise, HAND_BETAS)
    if abs(swapped_batch - 0.72) <= 0.05:
        fail(
            f"swapped-scale loss is {swapped_batch:.6g}, expected a value more than "
            f"0.05 away from 0.72"
        )


@check("linear_eps_grad matches the worked value and a finite difference, including a batch")
def test_linear_eps_grad(sub):
    grad = has(sub, "linear_eps_grad")
    loss = has(sub, "ddpm_loss")
    got = grad(WEIGHT, X0, 2, EPS, HAND_BETAS)
    shape(got, (2,), label="gradient at the worked weight")
    close(got, WORKED_GRAD, atol=1e-6, label="gradient at weight (0.4, 0.1)")

    def row_loss(weight):
        return loss(lambda x, _t: x * weight, X0, 2, EPS, HAND_BETAS)

    close(
        got,
        finite_difference(row_loss, WEIGHT.copy()),
        rtol=1e-5,
        atol=1e-6,
        label="finite-difference gradient on the worked row",
    )
    rng = np.random.default_rng(1)
    clean = rng.normal(size=(4, 2))
    noise = rng.normal(size=(4, 2))
    batch_weight = np.array([0.3, -0.2])
    got_batch = grad(batch_weight, clean, 2, noise, HAND_BETAS)
    shape(got_batch, (2,), label="gradient on a batch of 4 rows")

    def batch_loss(weight):
        return loss(lambda x, _t: x * weight, clean, 2, noise, HAND_BETAS)

    close(
        got_batch,
        finite_difference(batch_loss, batch_weight.copy()),
        rtol=1e-4,
        atol=1e-5,
        label="finite-difference gradient on a batch of 4 rows",
    )
    finite(got_batch, label="batch gradient")


@check("ddpm_step lands on the worked point, and t = 1 ignores the noise")
def test_ddpm_step(sub):
    fn = has(sub, "ddpm_step")
    got = fn(WORKED_X2, EPS, 2, HAND_BETAS, STEP_NOISE)
    shape(got, (2,), label="ancestral step at t = 2")
    oracle = ancestral_step(WORKED_X2, EPS, 2, HAND_BETAS, STEP_NOISE)
    if not np.allclose(oracle, WORKED_STEP, atol=1e-6):
        fail(
            f"oracle ancestral step is {oracle}, expected the worked step {WORKED_STEP}"
        )
    close(got, WORKED_STEP, atol=1e-6, label="ancestral step at t = 2 with noise (0.5, -0.25)")
    wide = ancestral_with_beta(WORKED_X2, EPS, 2, HAND_BETAS, STEP_NOISE)
    if np.allclose(wide, WORKED_STEP, atol=1e-4):
        fail(
            f"sqrt(beta) step is {wide}, and that matches the worked step {WORKED_STEP}; "
            f"the hand step would not catch the wider noise scale"
        )
    quiet = fn(WORKED_X1, EPS, 1, HAND_BETAS, np.array([0.5, -0.25]))
    loud = fn(WORKED_X1, EPS, 1, HAND_BETAS, np.array([-0.8, 0.3]))
    close(quiet, loud, atol=1e-12, label="t = 1 step with two different noise arguments")
    mean_only = ancestral_step(WORKED_X1, EPS, 1, HAND_BETAS, np.zeros(2))
    close(quiet, mean_only, atol=1e-8, label="t = 1 step against the reverse mean")


@check("ddim_step stays on the closed-form path and does not return only the clean point")
def test_ddim_step(sub):
    fn = has(sub, "ddim_step")
    got = fn(WORKED_X2, EPS, 2, HAND_BETAS)
    shape(got, (2,), label="deterministic step at t = 2")
    oracle = deterministic_step(WORKED_X2, EPS, 2, HAND_BETAS)
    if not np.allclose(oracle, WORKED_X1, atol=1e-6):
        fail(f"oracle deterministic step is {oracle}, expected the closed form {WORKED_X1}")
    close(got, WORKED_X1, atol=1e-6, label="deterministic step at t = 2")
    clean_only = np.array([1.5, 0.0])
    if np.allclose(clean_only, WORKED_X1, atol=1e-4):
        fail(
            f"the predicted clean point {clean_only} matches the deterministic step "
            f"{WORKED_X1}; the check would not catch a step that stops at the clean point"
        )


@check("classifier_free_guidance is the unconditional prediction plus scale times the gap")
def test_classifier_free_guidance(sub):
    fn = has(sub, "classifier_free_guidance")
    cond = np.array([1.0, -0.5])
    uncond = np.array([0.2, 0.0])
    close(fn(cond, uncond, 0.0), uncond, atol=1e-12, label="guidance at scale 0")
    close(fn(cond, uncond, 1.0), cond, atol=1e-12, label="guidance at scale 1")
    close(
        fn(cond, uncond, 3.0),
        np.array([2.6, -1.5]),
        atol=1e-12,
        label="guidance at scale 3",
    )
    forgotten = uncond + 3.0 * cond
    if np.allclose(forgotten, np.array([2.6, -1.5]), atol=1e-8):
        fail(
            f"uncond + scale * cond is {forgotten}, and that matches (2.6, -1.5); "
            f"the scale-3 check would not catch a missing subtraction"
        )


@check("ddpm_sample follows the ancestral step, and the two-blob cloud lands in the window")
def test_ddpm_sample(sub):
    fn = has(sub, "ddpm_sample")

    def zero_denoiser(x, _t):
        return np.zeros_like(x)

    short = fn(zero_denoiser, 2, HAND_BETAS, np.random.default_rng(1))
    shape(short, (2, 2), label="two ancestral samples on the worked schedule")
    oracle = reference_sample(
        zero_denoiser, 2, HAND_BETAS, np.random.default_rng(1), ancestral_step
    )
    finite(short, label="short ancestral sample")
    close(short, oracle, atol=1e-8, label="ancestral path with a zero denoiser, seed 1")

    def denoiser(x, t):
        return mixture_eps(x, t, SAMPLE_BETAS, MEANS)

    rows = fn(denoiser, 2500, SAMPLE_BETAS, np.random.default_rng(0))
    shape(rows, (2500, 2), label="2500 ancestral samples")
    finite(rows, label="two-blob ancestral samples")
    mean, cov = population_moments(rows)
    if not in_window(mean, cov):
        fail(f"two-blob samples are outside the window. {report_window(mean, cov)}")
    reference = reference_sample(
        denoiser, 2500, SAMPLE_BETAS, np.random.default_rng(0), ancestral_step
    )
    ref_mean, ref_cov = population_moments(reference)
    if not np.allclose(ref_mean, SEED0_MEAN, atol=1e-6) or not np.allclose(
        ref_cov, SEED0_COV, atol=1e-6
    ):
        fail(
            f"reference seed-0 moments are mean {ref_mean}, covariance {ref_cov}, "
            f"expected mean {SEED0_MEAN} and covariance {SEED0_COV}"
        )
    broken = reference_sample(
        denoiser, 2500, SAMPLE_BETAS, np.random.default_rng(0), reverse_without_alpha
    )
    broken_mean, broken_cov = population_moments(broken)
    if in_window(broken_mean, broken_cov):
        fail(
            f"dropping 1/sqrt(alpha_t) still lands in the window. "
            f"{report_window(broken_mean, broken_cov)}"
        )
