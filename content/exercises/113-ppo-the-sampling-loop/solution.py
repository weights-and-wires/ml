"""Reference solution: the advantage and the clipped objective."""

import numpy as np


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


if __name__ == "__main__":
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
