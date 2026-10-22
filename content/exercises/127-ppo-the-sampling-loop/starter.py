"""Build the advantage and the clip.

PPO compares each sampled answer with what you would expect, then nudges the
policy toward answers that did better, while a clip keeps one step from moving
the policy far. You build the three pieces of that update: the probability
ratio, the advantage, and the clipped objective.

Arrays have shape (n,): one entry per sampled token or answer.
"""

import numpy as np


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
