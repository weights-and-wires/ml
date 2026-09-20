"""Four regularisers, reduced to the same shrinkage idea.

Fill in the six functions below.
"""

import numpy as np


def dropout_forward(x: np.ndarray, p: float, rng: np.random.Generator, training: bool):
    """Apply inverted dropout.

    During training, zero each entry of `x` independently with probability
    `p` and divide the survivors by `1 - p`, so the expected output equals
    `x`. During inference, return `x` unchanged.

    Args:
        x: Input array, any shape.
        p: Probability of dropping a unit, in `[0, 1)`.
        rng: A `numpy.random.Generator` used to draw the mask. Draw the mask
            from `rng.random(x.shape)`, so the mask depends only on the shape
            of `x` and the state of `rng`, not on the values in `x`.
        training: If `False`, return `x` unchanged and a mask of all ones.

    Returns:
        A tuple `(out, mask)`, where `mask` is a same-shaped 0/1 array (all
        ones when `training` is `False`) recording which units survived.
    """
    raise NotImplementedError("dropout_forward")


def dropout_backward(dout: np.ndarray, mask: np.ndarray, p: float) -> np.ndarray:
    """Backpropagate through `dropout_forward`.

    Args:
        dout: Upstream gradient, same shape as the forward output.
        mask: The mask `dropout_forward` returned.
        p: The same drop probability passed to `dropout_forward`.

    Returns:
        The gradient with respect to the forward pass's input, same shape as
        `dout`.
    """
    raise NotImplementedError("dropout_backward")


def weight_decay_step(w: np.ndarray, grad: np.ndarray, lr: float, weight_decay: float) -> np.ndarray:
    """Take one gradient step with weight decay folded in as for plain SGD.

    Args:
        w: Current parameters.
        grad: Gradient of the (unregularised) loss at `w`.
        lr: Learning rate.
        weight_decay: Decay rate.

    Returns:
        `w - lr * grad - lr * weight_decay * w`.
    """
    raise NotImplementedError("weight_decay_step")


def label_smoothing_loss(logits: np.ndarray, targets: np.ndarray, eps: float) -> float:
    """Return the mean label-smoothed cross-entropy loss.

    The smoothed target distribution for a row with class `t` puts
    `1 - eps + eps / K` on class `t` and `eps / K` on every other class,
    where `K` is the number of classes. The loss is the mean, over rows, of
    the cross-entropy between that distribution and `softmax(logits)`.

    Args:
        logits: Unnormalised scores, shape `(N, K)`.
        targets: Integer class labels, shape `(N,)`, values in `[0, K)`.
        eps: Smoothing strength, in `[0, 1)`.

    Returns:
        The mean loss over the batch, as a float.
    """
    raise NotImplementedError("label_smoothing_loss")


def label_smoothing_grad(logits: np.ndarray, targets: np.ndarray, eps: float) -> np.ndarray:
    """Return the gradient of `label_smoothing_loss` with respect to `logits`.

    Args:
        logits: Unnormalised scores, shape `(N, K)`.
        targets: Integer class labels, shape `(N,)`.
        eps: Smoothing strength, matching the value passed to
            `label_smoothing_loss`.

    Returns:
        The gradient, shape `(N, K)`, of the mean loss with respect to
        `logits`.
    """
    raise NotImplementedError("label_smoothing_grad")


def early_stopping_equivalent_ridge(eigenvalue: float, step: float, iterations: int) -> float:
    """Return the ridge penalty that `iterations` gradient steps correspond to.

    For gradient descent from zero on a quadratic with curvature `eigenvalue`
    along this coordinate and learning rate `step`, this is the ridge penalty
    `gamma` such that the ridge solution's shrinkage factor
    `eigenvalue / (eigenvalue + gamma)` equals the gradient-descent
    shrinkage factor `1 - (1 - step * eigenvalue) ** iterations` after that
    many steps.

    Args:
        eigenvalue: The curvature along this coordinate, positive.
        step: The learning rate.
        iterations: The number of gradient steps, a non-negative integer.

    Returns:
        The equivalent ridge penalty, as a float. Return `float("inf")` when
        `iterations` is 0.
    """
    raise NotImplementedError("early_stopping_equivalent_ridge")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.ones(200_000)
    out, mask = dropout_forward(x, p=0.4, rng=rng, training=True)
    print(f"dropout: kept fraction {mask.mean():.3f}, mean output {out.mean():.3f} (should track 1.0)")

    logits = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    targets = np.array([0, 1])
    loss_smoothed = label_smoothing_loss(logits, targets, eps=0.1)
    loss_plain = label_smoothing_loss(logits, targets, eps=0.0)
    print(f"loss with smoothing: {loss_smoothed:.4f}, without: {loss_plain:.4f}")

    for k in (0, 1, 10, 1000):
        gamma = early_stopping_equivalent_ridge(eigenvalue=2.0, step=0.1, iterations=k)
        print(f"iterations={k:>5}: equivalent ridge penalty = {gamma:.4g}")
