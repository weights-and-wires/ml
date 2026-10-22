"""Label smoothing: ask the model for a little less certainty.

Fill in the three functions below. `log_softmax` is provided.
"""

import numpy as np


# Provided: the max-shifted log-softmax, row by row.
def log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


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
    logits = np.array([[2.0, 0.0, 0.0]])
    targets = np.array([0])
    print("smoothed target:", np.round(smooth_targets(targets, 3, 0.1), 4))
    print(f"loss with smoothing {label_smoothing_loss(logits, targets, 0.1):.4f}, without {label_smoothing_loss(logits, targets, 0.0):.4f}")
    print("gradient:", np.round(label_smoothing_grad(logits, targets, 0.1), 4))

    gaps = np.linspace(0.0, 8.0, 801)
    losses = [label_smoothing_loss(np.array([[g, 0.0, 0.0]]), targets, 0.1) for g in gaps]
    print(f"loss is lowest at a gap of {gaps[int(np.argmin(losses))]:.2f} (ln 28 = {np.log(28):.2f})")
