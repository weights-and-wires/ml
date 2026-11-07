"""Reference solution: the Bradley-Terry reward loss, the KL leash, and the best policy."""

import numpy as np


def preference_probability(reward_chosen: np.ndarray, reward_rejected: np.ndarray) -> np.ndarray:
    z = np.asarray(reward_chosen, dtype=float) - np.asarray(reward_rejected, dtype=float)
    # sigmoid(z) = 1 / (1 + exp(-z)) = exp(-log(1 + exp(-z))), stable for any z.
    return np.exp(-np.logaddexp(0.0, -z))


def bradley_terry_loss(reward_chosen: np.ndarray, reward_rejected: np.ndarray) -> float:
    z = np.asarray(reward_chosen, dtype=float) - np.asarray(reward_rejected, dtype=float)
    return float(np.mean(np.logaddexp(0.0, -z)))


def bradley_terry_grad(
    reward_chosen: np.ndarray, reward_rejected: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    reward_chosen = np.asarray(reward_chosen, dtype=float)
    n = reward_chosen.shape[0]
    p = preference_probability(reward_chosen, reward_rejected)
    # d/dz of -log sigmoid(z) is sigmoid(z) - 1, and z = chosen - rejected.
    grad_chosen = (p - 1.0) / n
    return grad_chosen, -grad_chosen


def kl_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    # Replace p by 1 where it is 0 so the log is never taken of 0; the where
    # then sets those terms to 0, which is the right value for 0 * log 0.
    safe_p = np.where(p > 0, p, 1.0)
    terms = np.where(p > 0, p * np.log(safe_p / q), 0.0)
    return np.sum(terms, axis=-1)


def optimal_policy(ref_logprobs: np.ndarray, rewards: np.ndarray, beta: float) -> np.ndarray:
    logits = np.asarray(ref_logprobs, dtype=float) + np.asarray(rewards, dtype=float) / beta
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    unnormalised = np.exp(logits)
    return unnormalised / np.sum(unnormalised, axis=-1, keepdims=True)


def penalised_objective(
    probs: np.ndarray, ref_probs: np.ndarray, rewards: np.ndarray, beta: float
) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    expected = np.sum(probs * np.asarray(rewards, dtype=float), axis=-1)
    return expected - beta * kl_divergence(probs, ref_probs)


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
