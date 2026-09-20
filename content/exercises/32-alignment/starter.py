"""Reward models, the KL-optimal policy, and DPO.

Fill in the functions below. They build up the derivation from the lesson in
order: the Bradley-Terry reward loss and its gradient, the closed-form
KL-regularised optimal policy, and the DPO loss that results from substituting
that optimum's inverse back into the Bradley-Terry loss.

Throughout, "logps" means log-probability: log pi(y | x) for some policy pi
and some completion y. A batch of shape (n,) holds one scalar per
prompt-completion pair.
"""

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid, elementwise."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bradley_terry_loss(reward_chosen: np.ndarray, reward_rejected: np.ndarray) -> float:
    """Mean Bradley-Terry negative log-likelihood over a batch of comparisons.

    Args:
        reward_chosen: r(x, y_w) for each pair, shape (n,).
        reward_rejected: r(x, y_l) for each pair, shape (n,).

    Returns:
        The scalar mean of -log(sigmoid(reward_chosen - reward_rejected))
        over the batch.
    """
    raise NotImplementedError("bradley_terry_loss")


def bradley_terry_grad(
    reward_chosen: np.ndarray, reward_rejected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of bradley_terry_loss with respect to both reward arguments.

    Args:
        reward_chosen: r(x, y_w) for each pair, shape (n,).
        reward_rejected: r(x, y_l) for each pair, shape (n,).

    Returns:
        A tuple (grad_chosen, grad_rejected), each shape (n,), the gradient of
        the mean loss with respect to reward_chosen and reward_rejected.
    """
    raise NotImplementedError("bradley_terry_grad")


def optimal_policy(ref_logprobs: np.ndarray, rewards: np.ndarray, beta: float) -> np.ndarray:
    """The closed-form KL-regularised optimal policy over a finite completion set.

    pi*(y | x) = pi_ref(y | x) * exp(r(x, y) / beta) / Z(x), with Z(x) the sum
    over the given completions that makes the result a distribution.

    Args:
        ref_logprobs: log pi_ref(y | x) for each candidate completion, shape
            (..., n_completions). exp(ref_logprobs) sums to 1 along the last
            axis.
        rewards: r(x, y) for the same completions, same shape as ref_logprobs.
        beta: the KL coefficient. Must be positive.

    Returns:
        pi*(y | x), the same shape as the inputs, summing to 1 along the last
        axis.
    """
    raise NotImplementedError("optimal_policy")


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) for discrete distributions along the last axis.

    Args:
        p: a distribution, shape (..., n), summing to 1 along the last axis.
        q: a distribution, same shape as p, summing to 1 along the last axis.

    Returns:
        sum_i p_i * log(p_i / q_i) along the last axis, shape (...). Treat a
        term with p_i == 0 as contributing 0, regardless of q_i.
    """
    raise NotImplementedError("kl_divergence")


def implicit_reward(policy_logps: np.ndarray, ref_logps: np.ndarray, beta: float) -> np.ndarray:
    """The reward DPO trains without ever naming: beta * log(pi / pi_ref).

    Args:
        policy_logps: log pi_theta(y | x), any shape.
        ref_logps: log pi_ref(y | x), the same shape as policy_logps.
        beta: the KL coefficient.

    Returns:
        beta * (policy_logps - ref_logps), elementwise.
    """
    raise NotImplementedError("implicit_reward")


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    """The DPO loss: the Bradley-Terry loss applied to implicit rewards.

    Args:
        policy_chosen_logps: log pi_theta(y_w | x) for each pair, shape (n,).
        policy_rejected_logps: log pi_theta(y_l | x) for each pair, shape (n,).
        ref_chosen_logps: log pi_ref(y_w | x) for each pair, shape (n,).
        ref_rejected_logps: log pi_ref(y_l | x) for each pair, shape (n,).
        beta: the KL coefficient.

    Returns:
        The scalar mean DPO loss over the batch.
    """
    raise NotImplementedError("dpo_loss")


def dpo_grad(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of dpo_loss with respect to the two policy log-probabilities.

    The reference log-probabilities are treated as constants: pi_ref does not
    depend on the parameters being trained.

    Args:
        policy_chosen_logps: log pi_theta(y_w | x) for each pair, shape (n,).
        policy_rejected_logps: log pi_theta(y_l | x) for each pair, shape (n,).
        ref_chosen_logps: log pi_ref(y_w | x) for each pair, shape (n,).
        ref_rejected_logps: log pi_ref(y_l | x) for each pair, shape (n,).
        beta: the KL coefficient.

    Returns:
        A tuple (grad_chosen, grad_rejected), each shape (n,), the gradient of
        the mean DPO loss with respect to policy_chosen_logps and
        policy_rejected_logps.
    """
    raise NotImplementedError("dpo_grad")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 8

    ref_logps_w = rng.uniform(-3.0, -0.5, size=n)
    ref_logps_l = rng.uniform(-3.0, -0.5, size=n)
    beta = 0.2

    # Start the policy equal to the reference, then take a few DPO steps.
    policy_logps_w = ref_logps_w.copy()
    policy_logps_l = ref_logps_l.copy()

    gap_before = float(np.mean(policy_logps_w - policy_logps_l))
    lr = 0.5
    for _ in range(50):
        grad_w, grad_l = dpo_grad(
            policy_logps_w, policy_logps_l, ref_logps_w, ref_logps_l, beta
        )
        policy_logps_w = policy_logps_w - lr * grad_w
        policy_logps_l = policy_logps_l - lr * grad_l

    gap_after = float(np.mean(policy_logps_w - policy_logps_l))
    loss_after = dpo_loss(policy_logps_w, policy_logps_l, ref_logps_w, ref_logps_l, beta)

    print(f"mean log-prob gap before training: {gap_before:.4f}")
    print(f"mean log-prob gap after training:  {gap_after:.4f}")
    print(f"final DPO loss: {loss_after:.4f}")
