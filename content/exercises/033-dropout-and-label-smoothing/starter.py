"""Dropout and label smoothing: two ways to keep a network from trusting itself too much.

Fill in the six functions below. `log_softmax` is written for you.
"""

import numpy as np


def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def dropout_mask(shape, p: float, rng: np.random.Generator) -> np.ndarray:
    """Draw a keep-mask: 1.0 for a unit that stays on, 0.0 for one that is dropped.

    Each entry is dropped independently with probability `p`. Draw from
    `rng.random(shape)`, so the mask depends only on `shape` and on the
    state of `rng`.

    Args:
        shape: The shape of the mask, an int or a tuple of ints.
        p: Probability of dropping a unit, in `[0, 1)`.
        rng: A `numpy.random.Generator`.

    Returns:
        A float array of 0.0s and 1.0s with the given shape.
    """
    raise NotImplementedError("dropout_mask")


def dropout_forward(x: np.ndarray, p: float, rng: np.random.Generator, training: bool):
    """Apply inverted dropout.

    During training, build the mask with `dropout_mask(x.shape, p, rng)`, zero
    the dropped entries of `x`, and scale the survivors so the expected output
    equals `x`. During inference, return `x` unchanged.

    Args:
        x: Activations, any shape.
        p: Probability of dropping a unit, in `[0, 1)`.
        rng: A `numpy.random.Generator` used to draw the mask.
        training: If `False`, return `x` and a mask of all ones, and leave
            `rng` alone.

    Returns:
        A tuple `(out, mask)`, where `mask` is the same-shaped 0/1 array that
        records which units stayed on.
    """
    raise NotImplementedError("dropout_forward")


def dropout_backward(dout: np.ndarray, mask: np.ndarray, p: float) -> np.ndarray:
    """Backpropagate through `dropout_forward`.

    Args:
        dout: Upstream gradient, same shape as the forward output.
        mask: The mask that `dropout_forward` returned.
        p: The same drop probability passed to `dropout_forward`.

    Returns:
        The gradient with respect to the forward pass's input, same shape as
        `dout`.
    """
    raise NotImplementedError("dropout_backward")


def smooth_targets(targets: np.ndarray, num_classes: int, eps: float) -> np.ndarray:
    """Return the smoothed target matrix.

    Each row starts as the one-hot row for its class, scaled down by `1 - eps`,
    with an even share `eps / num_classes` added to every class. A row sums to 1.

    Args:
        targets: Integer class labels, shape `(N,)`, values in `[0, num_classes)`.
        num_classes: The number of classes `K`.
        eps: Smoothing amount, in `[0, 1)`.

    Returns:
        A float array of shape `(N, K)`.
    """
    raise NotImplementedError("smooth_targets")


def label_smoothing_loss(logits: np.ndarray, targets: np.ndarray, eps: float) -> float:
    """Return the mean cross-entropy against the smoothed targets.

    Args:
        logits: Scores, shape `(N, K)`.
        targets: Integer class labels, shape `(N,)`.
        eps: Smoothing amount, in `[0, 1)`.

    Returns:
        The loss, averaged over the `N` rows, as a float. It stays finite when
        the scores are very large.
    """
    raise NotImplementedError("label_smoothing_loss")


def label_smoothing_grad(logits: np.ndarray, targets: np.ndarray, eps: float) -> np.ndarray:
    """Return the gradient of `label_smoothing_loss` with respect to `logits`.

    Args:
        logits: Scores, shape `(N, K)`.
        targets: Integer class labels, shape `(N,)`.
        eps: Smoothing amount, matching the value passed to the loss.

    Returns:
        An array of shape `(N, K)`.
    """
    raise NotImplementedError("label_smoothing_grad")


if __name__ == "__main__":
    x = np.array([2.0, 4.0, 6.0, 8.0])
    rng = np.random.default_rng(0)

    out, mask = dropout_forward(x, 0.5, rng, training=True)
    print("mask:", mask, "out:", out)

    sums = np.array([dropout_forward(x, 0.5, rng, training=True)[0].sum() for _ in range(2000)])
    print(f"mean layer sum over 2000 masks: {sums.mean():.2f} (test time gives {x.sum():.0f})")

    fixed = np.array([1.0, 0.0, 1.0, 1.0])
    print("backward for dout = ones:", dropout_backward(np.ones(4), fixed, 0.5))

    logits = np.array([[2.0, 0.0, 0.0]])
    targets = np.array([0])
    print("smoothed target:", np.round(smooth_targets(targets, 3, 0.1), 4))
    print(f"loss with smoothing {label_smoothing_loss(logits, targets, 0.1):.4f}, without {label_smoothing_loss(logits, targets, 0.0):.4f}")
    print("gradient:", np.round(label_smoothing_grad(logits, targets, 0.1), 4))

    gaps = np.linspace(0.0, 8.0, 801)
    losses = [label_smoothing_loss(np.array([[g, 0.0, 0.0]]), targets, 0.1) for g in gaps]
    print(f"loss is lowest at a gap of {gaps[int(np.argmin(losses))]:.2f} (ln 28 = {np.log(28):.2f})")
