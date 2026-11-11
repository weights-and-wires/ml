"""Checks for scoring classifiers.

Two plausible mistakes are named here and rejected by the checks:

* `precision_nan_when_empty` returns tp / (tp + fp) with no guard. When every
  prediction is negative the denominator is 0 and precision is NaN. The
  function returns 0 instead.
* `auc_ties_as_wins` adds 1 for a tied pair, as if the positive outranked the
  negative. A tie is worth 1/2, which is the value the trapezoid of the ROC
  uses when each tied score is one step.
"""

import numpy as np

from mlcheck import check, close, fail, finite, finite_difference


def _need(module, name):
    if not hasattr(module, name):
        fail(f"expected a function named {name}, got a module with no {name}")
    return getattr(module, name)


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def _overconfident():
    rng = np.random.default_rng(0)
    z = rng.normal(size=500)
    y = (rng.uniform(size=500) < _sigmoid(z)).astype(float)
    return 3.0 * z, y


def _reference_nll(temperature, logits, y):
    u = np.asarray(logits, dtype=float) / float(temperature)
    return float(np.sum(np.logaddexp(0.0, u) - np.asarray(y, dtype=float) * u))


def _mann_whitney(scores, y):
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    pos = scores[y == 1]
    neg = scores[y == 0]
    diff = pos[:, None] - neg[None, :]
    return float(np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5)).mean())


def _auc_ties_as_wins(scores, y):
    """Wrong implementation: a tied pair counts as a correct ranking."""
    scores = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    pos = scores[y == 1]
    neg = scores[y == 0]
    diff = pos[:, None] - neg[None, :]
    return float(np.where(diff >= 0, 1.0, 0.0).mean())


def _trapezoid(fpr, tpr):
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    return float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2.0))


def _reference_ece(probs, y, bins):
    probs = np.asarray(probs, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    index = np.minimum(np.floor(probs * bins).astype(int), bins - 1)
    total = 0.0
    n = probs.size
    for b in range(bins):
        mask = index == b
        count = int(np.sum(mask))
        if count == 0:
            continue
        gap = abs(float(np.mean(y[mask])) - float(np.mean(probs[mask])))
        total += (count / n) * gap
    return float(total)


@check("confusion counts on the eight loans are [3, 1, 1, 3], and a model that flags nobody is [0, 0, 1, 99]")
def test_confusion_counts(sub):
    """The first check calls only confusion_counts."""
    confusion_counts = _need(sub, "confusion_counts")
    y = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    pred = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    close(confusion_counts(y, pred), [3.0, 1.0, 1.0, 3.0], atol=0, rtol=0)

    rare = np.zeros(100)
    rare[0] = 1.0
    close(confusion_counts(rare, np.zeros(100)), [0.0, 0.0, 1.0, 99.0], atol=0, rtol=0)


@check("precision is 0 when nothing is flagged, and F1 is the harmonic mean 0.8 rather than the arithmetic mean")
def test_precision_recall_f1(sub):
    """`precision_nan_when_empty` returns NaN when tp + fp is 0."""
    precision_recall_f1 = _need(sub, "precision_recall_f1")
    y = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    got = np.asarray(precision_recall_f1(y, np.zeros(5)), dtype=float)
    if not np.all(np.isfinite(got)):
        fail(f"expected [0, 0, 0] when no row is predicted positive, got {got}")
    close(got, [0.0, 0.0, 0.0], atol=0, rtol=0)

    # Precision 1, recall 2/3. The arithmetic mean is 5/6; the harmonic mean is 0.8.
    uneven = precision_recall_f1(
        np.array([1.0, 1.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0, 0.0])
    )
    close(uneven, [1.0, 2.0 / 3.0, 0.8], atol=1e-12, rtol=0)

    no_positives = precision_recall_f1(np.zeros(4), np.array([1.0, 0.0, 0.0, 0.0]))
    close(no_positives, [0.0, 0.0, 0.0], atol=0, rtol=0)


@check("the cost cut is cost_fp / (cost_fp + cost_fn): costs 1 and 9 give 0.1, not 9")
def test_threshold_from_cost(sub):
    threshold_from_cost = _need(sub, "threshold_from_cost")
    close(threshold_from_cost(1.0, 9.0), 0.1, atol=1e-12, rtol=0)
    close(threshold_from_cost(1.0, 1.0), 0.5, atol=1e-12, rtol=0)
    close(threshold_from_cost(3.0, 1.0), 0.75, atol=1e-12, rtol=0)


@check("the ROC of the eight loans starts at (0, 0), ends at (1, 1), and matches the hand staircase")
def test_roc_curve(sub):
    roc_curve = _need(sub, "roc_curve")
    scores = np.array([0.95, 0.80, 0.70, 0.60, 0.45, 0.30, 0.25, 0.10])
    y = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    fpr, tpr = roc_curve(scores, y)
    want_fpr = [0.0, 0.0, 0.0, 0.25, 0.25, 0.50, 0.75, 0.75, 1.0]
    want_tpr = [0.0, 0.25, 0.50, 0.50, 0.75, 0.75, 0.75, 1.0, 1.0]
    close(fpr, want_fpr, atol=1e-12, rtol=0)
    close(tpr, want_tpr, atol=1e-12, rtol=0)


@check("AUC is 0.75 on the eight loans, a tie is worth 1/2, and a reversed ranking has area 0")
def test_auc(sub):
    """`auc_ties_as_wins` counts a tie as 1 and scores about 0.639 on the seeded sample."""
    auc = _need(sub, "auc")
    roc_curve = _need(sub, "roc_curve")
    scores = np.array([0.95, 0.80, 0.70, 0.60, 0.45, 0.30, 0.25, 0.10])
    y = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    close(auc(scores, y), 0.75, atol=1e-12, rtol=0)

    rng = np.random.default_rng(0)
    tied = rng.choice(np.array([0.0, 0.5, 1.0, 1.5]), size=40)
    tied_y = rng.integers(0, 2, size=40).astype(float)
    got = float(auc(tied, tied_y))
    trap = _trapezoid(*roc_curve(tied, tied_y))
    half = _mann_whitney(tied, tied_y)
    inflated = _auc_ties_as_wins(tied, tied_y)
    if abs(got - inflated) <= 1e-12 and abs(got - half) > 1e-8:
        fail(
            f"a tie counted as a win gives {inflated}, expected {half} "
            f"(a tie is worth 1/2), got {got}"
        )
    if abs(got - trap) > 1e-12 or abs(got - half) > 1e-12:
        fail(
            f"expected the trapezoid {trap} and the pair average {half} "
            f"to agree to 1e-12, got {got}"
        )

    flat = np.zeros(8)
    flat_y = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    close(auc(flat, flat_y), 0.5, atol=1e-12, rtol=0)

    cancel = np.array([0.0, 1.0, 0.0, 1.0])
    cancel_y = np.array([1.0, 0.0, 0.0, 1.0])
    close(auc(cancel, cancel_y), 0.5, atol=1e-12, rtol=0)

    reversed_scores = np.array([0.0, 0.0, 1.0, 1.0])
    reversed_y = np.array([1.0, 1.0, 0.0, 0.0])
    close(auc(reversed_scores, reversed_y), 0.0, atol=1e-12, rtol=0)


@check("ECE on the eight loans with 4 bins is 0.08125, and perfect probabilities give 0")
def test_ece(sub):
    ece = _need(sub, "ece")
    probs = np.array([0.95, 0.80, 0.70, 0.60, 0.45, 0.30, 0.25, 0.10])
    y = np.array([1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    close(ece(probs, y, 4), 0.08125, atol=1e-12, rtol=0)
    perfect = np.array([1.0, 1.0, 0.0, 0.0])
    close(ece(perfect, perfect, 4), 0.0, atol=1e-12, rtol=0)


@check("the temperature slope matches a finite difference of the summed log loss")
def test_temperature_nll_grad(sub):
    temperature_nll_grad = _need(sub, "temperature_nll_grad")
    logits, y = _overconfident()

    def reference(temperature):
        return _reference_nll(temperature[0], logits, y)

    for temperature in (0.8, 1.5, 2.5):
        got = float(temperature_nll_grad(logits, y, temperature))
        want = float(finite_difference(reference, np.array([temperature]))[0])
        finite(got)
        close(got, want, rtol=1e-5, atol=1e-4, label=f"slope at T={temperature}")


@check("the fitted temperature is above 1, minimises the log loss, and lowers ECE")
def test_temperature_scale(sub):
    temperature_scale = _need(sub, "temperature_scale")
    logits, y = _overconfident()
    temperature = float(temperature_scale(logits, y))
    finite(temperature)
    if temperature <= 1.0:
        fail(
            f"expected a temperature above 1 on overconfident scores, "
            f"got {temperature}"
        )

    def reference(value):
        return _reference_nll(value[0], logits, y)

    slope = float(finite_difference(reference, np.array([temperature]))[0])
    if abs(slope) > 0.05:
        fail(
            f"expected the slope of the log loss near 0 at the fitted "
            f"temperature, got slope {slope} at T = {temperature}"
        )

    loss = reference(np.array([temperature]))
    for factor in (0.7, 0.85, 1.15, 1.4):
        neighbour = reference(np.array([factor * temperature]))
        if neighbour < loss - 0.05:
            fail(
                f"expected the log loss at T = {temperature} ({loss}) to be "
                f"at most the loss at {factor} T ({neighbour})"
            )

    raw = _reference_ece(_sigmoid(logits), y, 10)
    scaled = _reference_ece(_sigmoid(logits / temperature), y, 10)
    if scaled > raw - 0.05:
        fail(
            f"expected ECE to fall by at least 0.05, got raw {raw} and "
            f"scaled {scaled} at T = {temperature}"
        )
