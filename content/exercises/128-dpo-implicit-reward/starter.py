"""Train a policy on the implicit reward (DPO).

Run the best-policy formula backwards and the reward of an answer is beta times
the log ratio of the policy to the reference, plus a number that depends only
on the prompt. Bradley-Terry on those implicit rewards trains the policy
straight from preference pairs.

"logps" means log probability: log pi(y | x) for a policy pi and an answer y.
A batch of shape (n,) holds one number per preference pair. The log
probabilities here are free numbers that you move directly; a real policy
moves them through its weights.

The functions from the two earlier chapters are already written below. Use
them. You write the three functions after them.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Written in the chapter "Learning a reward from human preferences".
# ---------------------------------------------------------------------------


def preference_probability(reward_chosen: np.ndarray, reward_rejected: np.ndarray) -> np.ndarray:
    """The chance that the chosen answer wins: sigmoid of the reward gap."""
    z = np.asarray(reward_chosen, dtype=float) - np.asarray(reward_rejected, dtype=float)
    return np.exp(-np.logaddexp(0.0, -z))


def bradley_terry_loss(reward_chosen: np.ndarray, reward_rejected: np.ndarray) -> float:
    """Mean of -log sigmoid(reward_chosen - reward_rejected)."""
    z = np.asarray(reward_chosen, dtype=float) - np.asarray(reward_rejected, dtype=float)
    return float(np.mean(np.logaddexp(0.0, -z)))


def bradley_terry_grad(
    reward_chosen: np.ndarray, reward_rejected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Gradient of bradley_terry_loss with respect to both rewards, each shape (n,)."""
    reward_chosen = np.asarray(reward_chosen, dtype=float)
    n = reward_chosen.shape[0]
    grad_chosen = (preference_probability(reward_chosen, reward_rejected) - 1.0) / n
    return grad_chosen, -grad_chosen


# ---------------------------------------------------------------------------
# Written in the chapter "Why a leash: KL divergence and the best policy".
# ---------------------------------------------------------------------------


def optimal_policy(ref_logprobs: np.ndarray, rewards: np.ndarray, beta: float) -> np.ndarray:
    """The best policy on a KL leash: a softmax of ref_logprobs + rewards / beta."""
    logits = np.asarray(ref_logprobs, dtype=float) + np.asarray(rewards, dtype=float) / beta
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    unnormalised = np.exp(logits)
    return unnormalised / np.sum(unnormalised, axis=-1, keepdims=True)



# ---------------------------------------------------------------------------
# Yours to write in this chapter.
# ---------------------------------------------------------------------------


def implicit_reward(policy_logps: np.ndarray, ref_logps: np.ndarray, beta: float) -> np.ndarray:
    """The reward a policy hides: beta * log(pi / pi_ref).

    Args:
        policy_logps: log pi_theta(y | x), any shape.
        ref_logps: log pi_ref(y | x), the same shape.
        beta: the KL coefficient, positive.

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
    """The DPO loss: the Bradley-Terry loss applied to the implicit rewards.

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
    """Gradient of dpo_loss with respect to the two policy log probabilities.

    The reference log probabilities are constants: the reference does not
    depend on the parameters being trained.

    Args:
        policy_chosen_logps: log pi_theta(y_w | x) for each pair, shape (n,).
        policy_rejected_logps: log pi_theta(y_l | x) for each pair, shape (n,).
        ref_chosen_logps: log pi_ref(y_w | x) for each pair, shape (n,).
        ref_rejected_logps: log pi_ref(y_l | x) for each pair, shape (n,).
        beta: the KL coefficient.

    Returns:
        A tuple (grad_chosen, grad_rejected), each shape (n,).
    """
    raise NotImplementedError("dpo_grad")


if __name__ == "__main__":
    beta, lr = 1.0, 0.1
    ref_w, ref_l = np.array([np.log(0.5)]), np.array([np.log(0.2)])
    pol_w, pol_l = ref_w.copy(), ref_l.copy()

    print("loss at the start:", round(dpo_loss(pol_w, pol_l, ref_w, ref_l, beta), 4))
    for _ in range(10):
        grad_w, grad_l = dpo_grad(pol_w, pol_l, ref_w, ref_l, beta)
        pol_w, pol_l = pol_w - lr * grad_w, pol_l - lr * grad_l

    print("chosen and rejected log probabilities after 10 steps:", pol_w.round(3), pol_l.round(3))
    margin = implicit_reward(pol_w, ref_w, beta) - implicit_reward(pol_l, ref_l, beta)
    print("margin:", margin.round(3))
    print("loss after 10 steps:", round(dpo_loss(pol_w, pol_l, ref_w, ref_l, beta), 4))
