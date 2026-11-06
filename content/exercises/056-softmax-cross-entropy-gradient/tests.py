"""Checks for softmax and cross-entropy."""

import numpy as np

from mlcheck import check, close, fail, finite, finite_difference, has, shape


@check("softmax gives [0.665241, 0.244728, 0.090031] for scores [2, 1, 0], and rows sum to 1")
def test_softmax(sub):
    softmax = has(sub, "softmax")
    got = softmax(np.array([[2.0, 1.0, 0.0]]))
    shape(got, (1, 3), label="softmax output")
    close(got, np.array([[0.665241, 0.244728, 0.090031]]), atol=1e-6, label="worked example")
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 4)) * 3
    p = softmax(z)
    close(p.sum(axis=1), np.ones(5), atol=1e-12, label="row sums")
    if p.min() <= 0:
        fail("softmax produced a non-positive probability")


@check("softmax ignores adding the same number to a row, and survives scores of 1000")
def test_softmax_stable(sub):
    softmax = has(sub, "softmax")
    z = np.array([[1.0, -2.0, 0.5]])
    close(softmax(z + 100.0), softmax(z), atol=1e-12, label="softmax(z + 100) against softmax(z)")
    big = softmax(np.array([[1000.0, 0.0, -1000.0]]))
    finite(big, label="softmax at scores 1000, 0, -1000")
    close(big, np.array([[1.0, 0.0, 0.0]]), atol=1e-12, label="softmax at scores 1000, 0, -1000")


@check("log_softmax equals log(softmax) and stays finite at scores of plus or minus 1000")
def test_log_softmax(sub):
    softmax = has(sub, "softmax")
    log_softmax = has(sub, "log_softmax")
    rng = np.random.default_rng(1)
    z = rng.normal(size=(4, 5)) * 2
    close(log_softmax(z), np.log(softmax(z)), atol=1e-10, label="log_softmax against log of softmax")
    big = log_softmax(np.array([[1000.0, 0.0, -1000.0]]))
    finite(big, label="log_softmax at scores 1000, 0, -1000")
    close(big, np.array([[0.0, -1000.0, -2000.0]]), atol=1e-9, label="log_softmax at scores 1000, 0, -1000")


@check("The loss is 0.407606 for scores [2, 1, 0] and class 0, and grows by 1 per class down")
def test_loss_value(sub):
    Tensor = has(sub, "Tensor")
    sce = has(sub, "softmax_cross_entropy")
    z = np.array([[2.0, 1.0, 0.0]])
    for label, want in [(0, 0.407606), (1, 1.407606), (2, 2.407606)]:
        loss = sce(Tensor(z), np.array([label]))
        close(loss.value, want, atol=1e-6, label=f"loss for class {label}")
    shape(loss.value, (), label="loss")


@check("The slope on the logits is (probabilities - one-hot) / n and matches finite differences")
def test_gradient(sub):
    Tensor = has(sub, "Tensor")
    softmax = has(sub, "softmax")
    sce = has(sub, "softmax_cross_entropy")
    rng = np.random.default_rng(2)
    z = rng.normal(size=(6, 4))
    labels = rng.integers(4, size=6)
    logits = Tensor(z)
    loss = sce(logits, labels)
    loss.backward()
    want = softmax(z)
    want[np.arange(6), labels] -= 1.0
    want /= 6
    close(logits.grad, want, atol=1e-8, label="against (p - onehot) / n")
    fd = finite_difference(lambda v: float(sce(Tensor(v), labels).value), z)
    close(logits.grad, fd, atol=1e-6, label="against finite differences")


@check("The worked example's slope is [-0.334759, 0.244728, 0.090031]")
def test_worked_gradient(sub):
    Tensor = has(sub, "Tensor")
    sce = has(sub, "softmax_cross_entropy")
    logits = Tensor(np.array([[2.0, 1.0, 0.0]]))
    sce(logits, np.array([0])).backward()
    close(logits.grad, np.array([[-0.334759, 0.244728, 0.090031]]), atol=1e-6, label="slope on the logits")


@check("A batch of 6 copies of one example has slope 1/6 of the single example's, and the same loss")
def test_batch_scaling(sub):
    Tensor = has(sub, "Tensor")
    sce = has(sub, "softmax_cross_entropy")
    row = np.array([[2.0, 1.0, 0.0]])
    one = Tensor(row)
    loss1 = sce(one, np.array([0]))
    loss1.backward()
    six = Tensor(np.repeat(row, 6, axis=0))
    loss6 = sce(six, np.zeros(6, dtype=int))
    loss6.backward()
    close(loss6.value, loss1.value, atol=1e-12, label="mean loss of identical rows")
    close(six.grad, np.repeat(one.grad / 6.0, 6, axis=0), atol=1e-12, label="each row's slope")


@check("The loss and slope stay finite and correct for scores of 1000")
def test_large_scores(sub):
    Tensor = has(sub, "Tensor")
    sce = has(sub, "softmax_cross_entropy")
    logits = Tensor(np.array([[1000.0, 0.0, 0.0]]))
    loss = sce(logits, np.array([1]))
    loss.backward()
    finite(loss.value, label="loss at scores 1000, 0, 0")
    close(loss.value, 1000.0, atol=1e-6, label="loss when the right class scores 1000 below the top")
    close(logits.grad, np.array([[1.0, -1.0, 0.0]]), atol=1e-9, label="slope at scores 1000, 0, 0")
