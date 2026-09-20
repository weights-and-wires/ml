"""Checks for the alignment exercise."""

import numpy as np

from mlcheck import almost, check, close, fail, finite_difference, has, shape


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


@check("bradley_terry_loss matches hand-computed values at equal and separated rewards")
def test_bradley_terry_loss_known_values(sub):
    got = has(sub, "bradley_terry_loss")

    # Equal rewards give even odds: -log(sigmoid(0)) = log(2).
    tied = np.zeros(5)
    almost(got(tied, tied), np.log(2.0), tol=1e-8, label="tied rewards")

    # A wide, correct margin drives the loss toward zero.
    chosen = np.full(5, 10.0)
    rejected = np.zeros(5)
    value = got(chosen, rejected)
    if value > 1e-4:
        fail(f"a reward gap of 10 should give a near-zero loss, got {value:.3g}")

    # A wide, backwards margin should cost close to the gap itself.
    value_backwards = got(rejected, chosen)
    almost(value_backwards, 10.0, tol=0.01, label="a confidently wrong margin")


@check("bradley_terry_grad matches mlcheck.finite_difference")
def test_bradley_terry_grad(sub):
    loss = has(sub, "bradley_terry_loss")
    grad = has(sub, "bradley_terry_grad")

    rng = np.random.default_rng(0)
    n = 6
    reward_chosen = rng.normal(size=n)
    reward_rejected = rng.normal(size=n)
    x0 = np.concatenate([reward_chosen, reward_rejected])

    def f(x):
        return loss(x[:n], x[n:])

    fd = finite_difference(f, x0)
    grad_chosen, grad_rejected = grad(reward_chosen, reward_rejected)
    shape(grad_chosen, (n,), label="grad_chosen")
    shape(grad_rejected, (n,), label="grad_rejected")
    analytic = np.concatenate([grad_chosen, grad_rejected])
    close(analytic, fd, rtol=1e-3, atol=1e-5, label="bradley_terry_grad vs finite differences")


@check("optimal_policy normalises and matches the closed form on a small example")
def test_optimal_policy_closed_form(sub):
    got = has(sub, "optimal_policy")

    rng = np.random.default_rng(1)
    logits = rng.normal(size=(4, 5))
    ref_probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    ref_logprobs = np.log(ref_probs)
    rewards = rng.normal(size=(4, 5))
    beta = 0.7

    pi = got(ref_logprobs, rewards, beta)
    shape(pi, (4, 5), label="optimal_policy output")
    if np.any(pi < -1e-10):
        fail(f"optimal_policy returned a negative probability: min is {pi.min():.3g}")
    close(pi.sum(axis=-1), np.ones(4), atol=1e-8, label="rows of optimal_policy")

    unnormalised = ref_probs * np.exp(rewards / beta)
    manual = unnormalised / unnormalised.sum(axis=-1, keepdims=True)
    close(pi, manual, rtol=1e-6, atol=1e-9, label="against the closed form computed directly")


@check("optimal_policy reduces to the reference as beta grows and to the argmax as beta shrinks")
def test_optimal_policy_limits(sub):
    got = has(sub, "optimal_policy")

    rng = np.random.default_rng(2)
    logits = rng.normal(size=(3, 6))
    ref_probs = np.exp(logits) / np.exp(logits).sum(axis=-1, keepdims=True)
    ref_logprobs = np.log(ref_probs)
    rewards = rng.normal(size=(3, 6))

    large_beta = got(ref_logprobs, rewards, 1e6)
    close(large_beta, ref_probs, atol=1e-5, label="optimal_policy at beta -> infinity")

    small_beta = got(ref_logprobs, rewards, 1e-3)
    best = np.argmax(rewards, axis=-1)
    onehot = np.zeros_like(rewards)
    onehot[np.arange(3), best] = 1.0
    close(small_beta, onehot, atol=1e-3, label="optimal_policy at beta -> 0")


@check("kl_divergence is zero for identical distributions and matches a hand computation")
def test_kl_divergence(sub):
    got = has(sub, "kl_divergence")

    p = np.array([0.2, 0.3, 0.5])
    almost(float(got(p, p)), 0.0, tol=1e-10, label="KL of a distribution with itself")

    q = np.array([0.1, 0.4, 0.5])
    want = float(np.sum(p * np.log(p / q)))
    almost(float(got(p, q)), want, tol=1e-8, label="KL against a hand computation")

    if got(p, q) < -1e-10:
        fail("KL divergence must be non-negative")


@check("dpo_loss equals bradley_terry_loss applied to the implicit rewards")
def test_dpo_loss_is_bradley_terry_on_implicit_rewards(sub):
    dpo = has(sub, "dpo_loss")
    bt = has(sub, "bradley_terry_loss")
    implicit = has(sub, "implicit_reward")

    rng = np.random.default_rng(3)
    n = 10
    beta = 0.3
    policy_w = rng.uniform(-4.0, -0.2, size=n)
    policy_l = rng.uniform(-4.0, -0.2, size=n)
    ref_w = rng.uniform(-4.0, -0.2, size=n)
    ref_l = rng.uniform(-4.0, -0.2, size=n)

    want = bt(implicit(policy_w, ref_w, beta), implicit(policy_l, ref_l, beta))
    almost(dpo(policy_w, policy_l, ref_w, ref_l, beta), want, tol=1e-8,
           label="dpo_loss against bradley_terry_loss of the implicit rewards")


@check("dpo_loss is invariant to shifting both completions' reference log-probs by a constant")
def test_dpo_loss_z_cancellation(sub):
    dpo = has(sub, "dpo_loss")

    rng = np.random.default_rng(4)
    n = 8
    beta = 0.5
    policy_w = rng.uniform(-4.0, -0.2, size=n)
    policy_l = rng.uniform(-4.0, -0.2, size=n)
    ref_w = rng.uniform(-4.0, -0.2, size=n)
    ref_l = rng.uniform(-4.0, -0.2, size=n)

    base = dpo(policy_w, policy_l, ref_w, ref_l, beta)

    # Adding the same per-prompt constant to both completions' reference
    # log-probs is exactly log Z(x) changing; the DPO loss must not move.
    shift = rng.uniform(-5.0, 5.0, size=n)
    shifted = dpo(policy_w, policy_l, ref_w + shift, ref_l + shift, beta)
    almost(shifted, base, tol=1e-8, label="dpo_loss after shifting both reference log-probs")


@check("dpo_grad matches mlcheck.finite_difference")
def test_dpo_grad(sub):
    loss = has(sub, "dpo_loss")
    grad = has(sub, "dpo_grad")

    rng = np.random.default_rng(5)
    n = 6
    beta = 0.4
    policy_w = rng.uniform(-4.0, -0.2, size=n)
    policy_l = rng.uniform(-4.0, -0.2, size=n)
    ref_w = rng.uniform(-4.0, -0.2, size=n)
    ref_l = rng.uniform(-4.0, -0.2, size=n)
    x0 = np.concatenate([policy_w, policy_l])

    def f(x):
        return loss(x[:n], x[n:], ref_w, ref_l, beta)

    fd = finite_difference(f, x0)
    grad_w, grad_l = grad(policy_w, policy_l, ref_w, ref_l, beta)
    analytic = np.concatenate([grad_w, grad_l])
    close(analytic, fd, rtol=1e-3, atol=1e-5, label="dpo_grad vs finite differences")


@check("implicit_reward recovers a planted reward up to a per-prompt constant")
def test_implicit_reward_recovers_planted_reward(sub):
    optimal = has(sub, "optimal_policy")
    implicit = has(sub, "implicit_reward")

    rng = np.random.default_rng(6)
    beta = 0.6
    n_completions = 5
    logits = rng.normal(size=n_completions)
    ref_probs = np.exp(logits) / np.exp(logits).sum()
    ref_logprobs = np.log(ref_probs)
    true_reward = rng.normal(size=n_completions) * 2.0

    # pi* for this planted reward is exactly the optimum implicit_reward
    # should invert, up to the additive beta * log Z(x) the lesson predicts.
    pi_star = optimal(ref_logprobs, true_reward, beta)
    policy_logps = np.log(pi_star)

    recovered = implicit(policy_logps, ref_logprobs, beta)
    # The recovered reward and the planted one must differ by one constant
    # shared across every completion of this prompt.
    diff = recovered - true_reward
    close(diff, np.full(n_completions, diff[0]), atol=1e-6,
          label="recovered reward minus planted reward, expected constant across completions")


@check("A few DPO gradient steps raise the chosen log-prob relative to the rejected one")
def test_dpo_training_raises_relative_preference(sub):
    grad = has(sub, "dpo_grad")
    loss_fn = has(sub, "dpo_loss")

    rng = np.random.default_rng(7)
    n = 8
    beta = 0.2
    ref_w = rng.uniform(-3.0, -0.5, size=n)
    ref_l = rng.uniform(-3.0, -0.5, size=n)
    policy_w = ref_w.copy()
    policy_l = ref_l.copy()

    gap_before = float(np.mean(policy_w - policy_l))
    loss_before = loss_fn(policy_w, policy_l, ref_w, ref_l, beta)

    lr = 0.5
    for _ in range(50):
        grad_w, grad_l = grad(policy_w, policy_l, ref_w, ref_l, beta)
        policy_w = policy_w - lr * grad_w
        policy_l = policy_l - lr * grad_l

    gap_after = float(np.mean(policy_w - policy_l))
    loss_after = loss_fn(policy_w, policy_l, ref_w, ref_l, beta)

    if gap_after <= gap_before + 1e-6:
        fail(
            "DPO training should widen the gap between the chosen and rejected "
            f"log-probabilities: it went from {gap_before:.4f} to {gap_after:.4f}"
        )
    if loss_after >= loss_before:
        fail(
            f"the DPO loss should fall with training: {loss_before:.4f} before, "
            f"{loss_after:.4f} after"
        )
