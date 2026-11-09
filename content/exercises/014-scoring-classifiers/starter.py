"""Score a classifier: counts, an ROC curve, and a temperature.

You write eight functions, in this order:

1. `confusion_counts`: true positives, false positives, false negatives, true negatives.
2. `precision_recall_f1`: the three rates, with 0 when a denominator is 0.
3. `threshold_from_cost`: the probability cut for a false-alarm cost and a miss cost.
4. `roc_curve`: false-positive and true-positive rates from high score to low.
5. `auc`: the chance a positive outranks a negative, with ties worth 1/2.
6. `ece`: expected calibration error over equal-width bins.
7. `temperature_nll_grad`: the slope of the log loss in the temperature.
8. `temperature_scale`: the positive temperature that minimises that loss.

Labels are 0 or 1. For the curve and the area, a higher score means class 1
is more likely. Probabilities lie in [0, 1].
"""

import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Count true positives, false positives, false negatives, and true negatives.

    Args:
        y_true: Labels of 0 or 1, shape (n,). Class 1 is the positive class.
        y_pred: Predictions of 0 or 1, shape (n,).

    Returns:
        Array [tp, fp, fn, tn], shape (4,).
    """
    raise NotImplementedError("write confusion_counts")


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return precision, recall, and F1.

    Args:
        y_true: Labels of 0 or 1, shape (n,).
        y_pred: Predictions of 0 or 1, shape (n,).

    Returns:
        Array [precision, recall, f1], shape (3,). Each rate is 0 when its
        denominator is 0, including precision when no row is predicted positive.
    """
    raise NotImplementedError("write precision_recall_f1")


def threshold_from_cost(cost_fp: float, cost_fn: float) -> float:
    """Return the probability cut for one false alarm and one miss.

    Args:
        cost_fp: Cost of flagging a negative row.
        cost_fn: Cost of missing a positive row.

    Returns:
        cost_fp / (cost_fp + cost_fn). Predict positive when the probability
        is above this value.
    """
    raise NotImplementedError("write threshold_from_cost")


def roc_curve(scores: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trace the ROC by sweeping the score threshold from high to low.

    Args:
        scores: Scores, shape (n,). A higher score favours class 1.
        y: Labels of 0 or 1, shape (n,). Both classes occur.

    Returns:
        fpr and tpr, each starting at 0 and ending at 1. One point is written
        after each distinct score, with every tied row counted before the point.
    """
    raise NotImplementedError("write roc_curve")


def auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Return the chance a positive outranks a negative.

    Args:
        scores: Scores, shape (n,). A higher score favours class 1.
        y: Labels of 0 or 1, shape (n,). Both classes occur.

    Returns:
        The mean over positive-negative pairs of 1 if the positive scores
        higher, 1/2 if the scores tie, and 0 otherwise.
    """
    raise NotImplementedError("write auc")


def ece(probs: np.ndarray, y: np.ndarray, bins: int) -> float:
    """Return the expected calibration error of positive-class probabilities.

    Args:
        probs: Predicted probabilities of class 1, shape (n,), each in [0, 1].
        y: Labels of 0 or 1, shape (n,).
        bins: Number of equal-width bins on [0, 1].

    Returns:
        The sum over bins of (count / n) * abs(fraction of positives - mean
        probability). A probability p goes to bin min(floor(p * bins), bins - 1).
        An empty bin adds nothing.
    """
    raise NotImplementedError("write ece")


def temperature_nll_grad(logits: np.ndarray, y: np.ndarray, temperature: float) -> float:
    """Return the slope of the summed log loss with respect to temperature.

    The loss is sum_i softplus(z_i / T) - y_i z_i / T, with p_i = sigmoid(z_i / T).
    The slope is sum_i (p_i - y_i) * (-z_i / T**2).

    Args:
        logits: Scores before the sigmoid, shape (n,).
        y: Labels of 0 or 1, shape (n,).
        temperature: Positive temperature T.

    Returns:
        One float, the derivative of the sum, not the mean.
    """
    raise NotImplementedError("write temperature_nll_grad")


def temperature_scale(logits: np.ndarray, y: np.ndarray) -> float:
    """Fit the temperature that minimises the summed log loss.

    Args:
        logits: Scores before the sigmoid, shape (n,).
        y: Labels of 0 or 1, shape (n,).

    Returns:
        A positive temperature. Start at 1, step by Newton using
        temperature_nll_grad, and take the bend as a central difference of
        that slope. Stop when the absolute slope is below 1e-8, or after 40
        steps. Halve a step that would make the temperature non-positive.
    """
    raise NotImplementedError("write temperature_scale")


if __name__ == "__main__":
    scores = np.array([0.95, 0.80, 0.70, 0.60, 0.45, 0.30, 0.25, 0.10])
    y = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    pred = (scores >= 0.5).astype(float)
    print("counts", confusion_counts(y, pred))
    print("precision, recall, f1", precision_recall_f1(y, pred))
    print("cost cut", threshold_from_cost(1.0, 9.0))
    fpr, tpr = roc_curve(scores, y)
    print("roc fpr", fpr)
    print("roc tpr", tpr)
    print("auc", auc(scores, y))
    print("ece", ece(scores, y, 4))
