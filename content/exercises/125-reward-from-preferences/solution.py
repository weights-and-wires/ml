"""Reference solution: the Bradley-Terry reward loss."""

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


if __name__ == "__main__":
    chosen = np.array([2.0, 0.0])
    rejected = np.array([1.0, 1.0])
    print("chance the chosen answer wins:", preference_probability(chosen, rejected).round(4))
    print("loss:", round(bradley_terry_loss(chosen, rejected), 4))

    lr = 2.0
    grad_c, grad_r = bradley_terry_grad(chosen, rejected)
    print("gradients on the chosen rewards:", grad_c.round(4))
    chosen, rejected = chosen - lr * grad_c, rejected - lr * grad_r
    print("rewards after one step:", chosen.round(3), rejected.round(3))
    print("loss after one step:", round(bradley_terry_loss(chosen, rejected), 4))
