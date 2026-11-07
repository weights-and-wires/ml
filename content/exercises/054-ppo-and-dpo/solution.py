"""Train a policy two ways: PPO's clipped update and DPO's implicit reward.

PPO compares each sampled answer with what you would expect, then nudges the
policy toward answers that did better, while a clip keeps one step from moving
the policy far. DPO runs the best-policy formula backwards, so a policy hides a
reward (beta times its log ratio to the reference), and trains on preference
pairs with no sampling.

"logps" means log probability. Arrays of shape (n,) hold one entry per sampled
token or answer, or one per preference pair. The log probabilities here are free
numbers that you move directly; a real policy moves them through its weights.

The Bradley-Terry functions and optimal_policy are already written below. Use
them. The six functions after them are the reference answers.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Written in the chapter on reward models and the KL leash: the Bradley-Terry functions.
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
# Written in the same chapter: the best policy on a leash.
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


def policy_ratio(new_logps: np.ndarray, old_logps: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(new_logps, dtype=float) - np.asarray(old_logps, dtype=float))


def advantage(rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
    raw = np.asarray(rewards, dtype=float) - np.asarray(values, dtype=float)
    # The small constant keeps a batch with no spread at zero instead of nan.
    return (raw - raw.mean()) / (raw.std() + 1e-8)


def clipped_objective(ratio: np.ndarray, advantage: np.ndarray, eps: float) -> float:
    ratio = np.asarray(ratio, dtype=float)
    advantage = np.asarray(advantage, dtype=float)
    unclipped = ratio * advantage
    clipped = np.clip(ratio, 1.0 - eps, 1.0 + eps) * advantage
    return float(np.mean(np.minimum(unclipped, clipped)))


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
    # PPO: four sampled answers.
    rewards = np.array([3.3, 3.1, 2.4, 2.6])
    values = np.array([2.8, 2.6, 2.9, 3.1])
    adv = advantage(rewards, values)
    print("advantages:", adv.round(3))
    old = np.array([0.5, 0.2, 0.3, 0.2])
    new = np.array([0.55, 0.28, 0.21, 0.26])
    ratio = policy_ratio(np.log(new), np.log(old))
    print("ratios:", ratio.round(3))
    print("clipped objective:", round(clipped_objective(ratio, adv, 0.2), 4))
    print("without the clip: ", round(float(np.mean(ratio * adv)), 4))

    # DPO: one pair, ten steps.
    beta, lr = 1.0, 0.1
    ref_w, ref_l = np.array([np.log(0.5)]), np.array([np.log(0.2)])
    pol_w, pol_l = ref_w.copy(), ref_l.copy()
    print("DPO loss at the start:", round(dpo_loss(pol_w, pol_l, ref_w, ref_l, beta), 4))
    for _ in range(10):
        grad_w, grad_l = dpo_grad(pol_w, pol_l, ref_w, ref_l, beta)
        pol_w, pol_l = pol_w - lr * grad_w, pol_l - lr * grad_l
    margin = implicit_reward(pol_w, ref_w, beta) - implicit_reward(pol_l, ref_l, beta)
    print("margin after 10 steps:", margin.round(3))
    print("DPO loss after 10 steps:", round(dpo_loss(pol_w, pol_l, ref_w, ref_l, beta), 4))
