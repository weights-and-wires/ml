"""Train a scorer on 'B beats A'.

A reward is one number per answer. The chance that a person picks the chosen
answer over the rejected one is the sigmoid of the gap between the two rewards.
You build that chance, the loss that rewards a scorer for giving it a high
value, and the gradient that tells each reward which way to move.

A batch of shape (n,) holds one reward per pair. The rewards here are free
numbers. In a real reward model they come from a network.
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
