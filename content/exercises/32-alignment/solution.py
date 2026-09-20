"""Reference solution for the alignment exercise."""

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
    z = np.asarray(reward_chosen, dtype=float) - np.asarray(reward_rejected, dtype=float)
    # -log(sigmoid(z)) = log(1 + exp(-z)), computed stably in both directions.
    per_example = np.where(z >= 0, np.log1p(np.exp(-z)), -z + np.log1p(np.exp(z)))
    return float(np.mean(per_example))


def bradley_terry_grad(
    reward_chosen: np.ndarray, reward_rejected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reward_chosen = np.asarray(reward_chosen, dtype=float)
    reward_rejected = np.asarray(reward_rejected, dtype=float)
    n = reward_chosen.shape[0]
    z = reward_chosen - reward_rejected
    # d/dz [-log sigmoid(z)] = sigmoid(z) - 1 = -sigmoid(-z).
    base = (_sigmoid(z) - 1.0) / n
    return base, -base


def optimal_policy(ref_logprobs: np.ndarray, rewards: np.ndarray, beta: float) -> np.ndarray:
    ref_logprobs = np.asarray(ref_logprobs, dtype=float)
    rewards = np.asarray(rewards, dtype=float)
    logits = ref_logprobs + rewards / beta
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    unnormalised = np.exp(logits)
    return unnormalised / np.sum(unnormalised, axis=-1, keepdims=True)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    terms = np.where(p > 0, p * np.log(p / q), 0.0)
    return np.sum(terms, axis=-1)


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
    # d(bradley_terry_loss)/d(r_chosen), d(...)/d(r_rejected), then chain
    # through implicit_reward's d(r)/d(policy_logps) = beta.
    grad_r_chosen, grad_r_rejected = bradley_terry_grad(r_chosen, r_rejected)
    return beta * grad_r_chosen, beta * grad_r_rejected


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
