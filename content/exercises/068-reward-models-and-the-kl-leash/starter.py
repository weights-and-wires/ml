"""Reward models and the KL leash.

Part one trains a scorer on "B beats A". A reward is one number per answer, and
the chance that a person picks the chosen answer over the rejected one is the
sigmoid of the gap between the two rewards. A batch of shape (n,) holds one
reward per pair. The rewards are free numbers here; in a real reward model they
come from a network.

Part two keeps a policy near its reference. A policy gives each answer to a
prompt a probability, and arrays there have shape (..., n_answers): the last
axis runs over the answers to one prompt, and leading axes index prompts.
"""

import numpy as np


def preference_probability(reward_chosen: np.ndarray, reward_rejected: np.ndarray) -> np.ndarray:
    """The chance that the chosen answer wins, elementwise.

    Args:
        reward_chosen: reward of the chosen answer, any shape.
        reward_rejected: reward of the rejected answer, same shape.

    Returns:
        sigmoid(reward_chosen - reward_rejected), same shape. Must stay finite
        for gaps of +-800.
    """
    raise NotImplementedError("preference_probability")


def bradley_terry_loss(reward_chosen: np.ndarray, reward_rejected: np.ndarray) -> float:
    """Mean negative log of the chance that the chosen answer wins.

    Args:
        reward_chosen: shape (n,).
        reward_rejected: shape (n,).

    Returns:
        The scalar mean of -log(sigmoid(reward_chosen - reward_rejected)).
        Must stay finite when a gap is very negative.
    """
    raise NotImplementedError("bradley_terry_loss")


def bradley_terry_grad(
    reward_chosen: np.ndarray, reward_rejected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of bradley_terry_loss with respect to both arguments.

    Args:
        reward_chosen: shape (n,).
        reward_rejected: shape (n,).

    Returns:
        A tuple (grad_chosen, grad_rejected), each shape (n,), the gradient of
        the mean loss with respect to each reward.
    """
    raise NotImplementedError("bradley_terry_grad")


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
    chosen = np.array([2.0, 0.0])
    rejected = np.array([1.0, 1.0])
    print("chance the chosen answer wins:", preference_probability(chosen, rejected).round(4))
    print("loss:", round(bradley_terry_loss(chosen, rejected), 4))
    grad_c, grad_r = bradley_terry_grad(chosen, rejected)
    print("gradients on the chosen rewards:", grad_c.round(4))
    chosen, rejected = chosen - 2.0 * grad_c, rejected - 2.0 * grad_r
    print("loss after one step with learning rate 2:", round(bradley_terry_loss(chosen, rejected), 4))

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
