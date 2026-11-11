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
them. You write the seven functions after them.
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
# Yours to write.
# ---------------------------------------------------------------------------


def policy_ratio(new_logps: np.ndarray, old_logps: np.ndarray) -> np.ndarray:
    """How much more likely the new policy finds each sampled token.

    Args:
        new_logps: log pi_new(token), shape (n,).
        old_logps: log pi_old(token) when the token was sampled, shape (n,).

    Returns:
        exp(new_logps - old_logps), shape (n,). It is 1 where nothing changed.
    """
    raise NotImplementedError("policy_ratio")


def advantage(rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
    """How much better than expected each sample was, rescaled.

    Args:
        rewards: the reward of each sample, shape (n,).
        values: the value head's prediction of each reward, shape (n,).

    Returns:
        (rewards - values), recentred to mean 0 and divided by its standard
        deviation, shape (n,). When every sample matches its prediction the
        result must be all zeros, not nan.
    """
    raise NotImplementedError("advantage")


def clipped_objective(ratio: np.ndarray, advantage: np.ndarray, eps: float) -> float:
    """PPO's clipped objective, the quantity the update tries to raise.

    Args:
        ratio: the probability ratios, shape (n,).
        advantage: the advantages, shape (n,).
        eps: the clip half-width, such as 0.2.

    Returns:
        The mean over the batch of
        min(ratio * advantage, clip(ratio, 1 - eps, 1 + eps) * advantage).
    """
    raise NotImplementedError("clipped_objective")


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


def terminal_reward_gae(
    reward: float, values: np.ndarray, gamma: float = 1.0, lam: float = 1.0
) -> np.ndarray:
    """Per-token advantages for an answer that earns one reward at its last token.

    The reward arrives after the final token, and every earlier token gets
    reward 0. The value after the final token is 0, because the answer ended.
    Work backward from the last token: delta_t = r_t + gamma * V_{t+1} - V_t,
    then A_t = delta_t + gamma * lam * A_{t+1}, with A_{n} = 0.

    Args:
        reward: the single reward paid after the last token.
        values: the value head's guess for each token, shape (n,).
        gamma: the discount, in [0, 1].
        lam: the GAE smoothing parameter, in [0, 1].

    Returns:
        The advantage of each token, shape (n,).
    """
    raise NotImplementedError("terminal_reward_gae")


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
