"""Build the leash and the best policy.

A policy gives each answer to a prompt a probability. The reference policy is
the supervised model we must not drift far from. You measure the drift with a
KL divergence, find the best policy for "reward minus beta times drift", and
score any policy on that objective.

Arrays have shape (..., n_answers): the last axis runs over the answers to one
prompt, and any leading axes index different prompts.
"""

import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(p || q) for discrete distributions along the last axis.

    Args:
        p: the policy's probabilities, shape (..., n), summing to 1 on the last axis.
        q: the reference probabilities, same shape, positive wherever p is positive.

    Returns:
        sum_i p_i * log(p_i / q_i) on the last axis, shape (...). A term with
        p_i == 0 counts as 0, whatever q_i is. Must not warn or return nan then.
    """
    raise NotImplementedError("kl_divergence")


def optimal_policy(ref_logprobs: np.ndarray, rewards: np.ndarray, beta: float) -> np.ndarray:
    """The best policy for reward minus beta times KL from the reference.

    pi*(y) = pi_ref(y) * exp(r(y) / beta) / Z, with Z the sum over the answers.

    Args:
        ref_logprobs: log pi_ref(y) for each answer, shape (..., n).
        rewards: r(y) for the same answers, same shape.
        beta: the leash price, positive.

    Returns:
        pi*, the same shape, summing to 1 on the last axis. Must stay finite
        when beta is tiny.
    """
    raise NotImplementedError("optimal_policy")


def penalised_objective(
    probs: np.ndarray, ref_probs: np.ndarray, rewards: np.ndarray, beta: float
) -> np.ndarray:
    """Expected reward minus beta times KL(probs || ref_probs).

    Args:
        probs: the policy's probabilities, shape (..., n).
        ref_probs: the reference probabilities, same shape.
        rewards: r(y) for each answer, same shape.
        beta: the leash price, positive.

    Returns:
        The score of each policy, shape (...). Use kl_divergence for the leash.
    """
    raise NotImplementedError("penalised_objective")


if __name__ == "__main__":
    ref = np.array([0.5, 0.3, 0.2])
    rewards = np.array([1.0, 0.0, 0.0])
    beta = 1.0

    best = optimal_policy(np.log(ref), rewards, beta)
    print("best policy:", best.round(4))
    for name, p in [
        ("reference", ref),
        ("[0.7, 0.2, 0.1]", np.array([0.7, 0.2, 0.1])),
        ("best", best),
        ("all on helpful", np.array([1.0, 0.0, 0.0])),
    ]:
        print(f"{name:>16}: KL {kl_divergence(p, ref):.4f}, score {penalised_objective(p, ref, rewards, beta):.4f}")
    print("beta * log Z:", round(beta * np.log(np.sum(ref * np.exp(rewards / beta))), 4))
