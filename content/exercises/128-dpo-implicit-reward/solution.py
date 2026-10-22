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
# Written in this chapter.
# ---------------------------------------------------------------------------


def implicit_reward(policy_logps: np.ndarray, ref_logps: np.ndarray, beta: float) -> np.ndarray:
    return beta * (np.asarray(policy_logps, dtype=float) - np.asarray(ref_logps, dtype=float))


def dpo_loss(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> float:
    r_chosen = implicit_reward(policy_chosen_logps, ref_chosen_logps, beta)
    r_rejected = implicit_reward(policy_rejected_logps, ref_rejected_logps, beta)
    return bradley_terry_loss(r_chosen, r_rejected)


def dpo_grad(
    policy_chosen_logps: np.ndarray,
    policy_rejected_logps: np.ndarray,
    ref_chosen_logps: np.ndarray,
    ref_rejected_logps: np.ndarray,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    r_chosen = implicit_reward(policy_chosen_logps, ref_chosen_logps, beta)
    r_rejected = implicit_reward(policy_rejected_logps, ref_rejected_logps, beta)
    grad_r_chosen, grad_r_rejected = bradley_terry_grad(r_chosen, r_rejected)
    # Each implicit reward moves by beta per unit of policy log probability.
    return beta * grad_r_chosen, beta * grad_r_rejected


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
