"""Checks for the digit-reader project."""

import copy
import time

import numpy as np

from mlcheck import check, close, fail, finite, has, shape


def _tiny(seed=0, sizes=(6, 5, 4, 3), n=7):
    rng = np.random.default_rng(seed)
    params = [
        {"W": rng.normal(scale=0.7, size=(a, b)), "b": rng.normal(scale=0.3, size=b)}
        for a, b in zip(sizes[:-1], sizes[1:])
    ]
    X = rng.normal(size=(n, sizes[0]))
    y = rng.integers(0, sizes[-1], size=n)
    return params, X, y


def _numeric(sub, params, X, y, eps, i, key, p_drop=0.0, seed=None):
    """Central-difference gradient of the loss with respect to params[i][key]."""
    array = params[i][key]
    out = np.zeros_like(array)
    for index in np.ndindex(*array.shape):
        original = array[index]
        values = []
        for sign in (1.0, -1.0):
            array[index] = original + sign * 1e-5
            rng = np.random.default_rng(seed) if seed is not None else None
            values.append(sub.loss_and_grads(params, X, y, eps, p_drop, rng, seed is not None)[0])
        array[index] = original
        out[index] = (values[0] - values[1]) / 2e-5
    return out


def _rel(a, b):
    scale = np.linalg.norm(a) + np.linalg.norm(b)
    return 0.0 if scale == 0 else float(np.linalg.norm(a - b) / scale)


@check("he_init gives the right shapes, zero biases, and weights with spread sqrt(2 / n_in)")
def test_he_init(sub):
    he_init = has(sub, "he_init")
    sizes = [100, 400, 200, 10]
    params = he_init(sizes, np.random.default_rng(0))
    if len(params) != 3:
        fail(f"expected 3 layers for sizes {sizes}, got {len(params)}")
    for layer, n_in, n_out in zip(params, sizes[:-1], sizes[1:]):
        shape(layer["W"], (n_in, n_out), label=f"W for {n_in} -> {n_out}")
        shape(layer["b"], (n_out,), label=f"b for {n_in} -> {n_out}")
        if np.any(layer["b"] != 0):
            fail("biases must start at zero")
        want = np.sqrt(2.0 / n_in)
        got = layer["W"].std()
        if abs(got / want - 1.0) > 0.05:
            fail(f"W for {n_in} -> {n_out} has std {got:.4f}, expected about {want:.4f} = sqrt(2 / {n_in})")
    again = he_init(sizes, np.random.default_rng(0))
    close(again[0]["W"], params[0]["W"], label="same seed, same weights")


@check("forward returns logits and a cache, and dropout acts only in training, only on hidden layers, with the average unchanged")
def test_forward_and_dropout(sub):
    forward = has(sub, "forward")
    params0, X0, _ = _tiny()
    logits0, cache0 = forward(params0, X0, p_drop=0.5, rng=None, training=False)
    shape(logits0, (7, 3), label="logits")
    if len(cache0) != 3:
        fail(f"cache should hold one entry per layer (3), got {len(cache0)}")
    h = X0
    for layer in params0[:-1]:
        h = np.maximum(h @ layer["W"] + layer["b"], 0.0)
    close(logits0, h @ params0[-1]["W"] + params0[-1]["b"], atol=1e-10, label="logits at evaluation, with rng=None")
    params, X, _ = _tiny(sizes=(6, 50, 3), n=4)
    eval_logits, _ = forward(params, X, p_drop=0.5, rng=np.random.default_rng(1), training=False)
    train_logits, _ = forward(params, X, p_drop=0.5, rng=np.random.default_rng(1), training=True)
    if np.allclose(eval_logits, train_logits):
        fail("training=True with p_drop=0.5 gave the same logits as evaluation, so no unit was dropped")
    zero_p, _ = forward(params, X, p_drop=0.0, rng=np.random.default_rng(1), training=True)
    close(zero_p, eval_logits, atol=1e-10, label="p_drop=0 in training mode")

    rng = np.random.default_rng(2)
    mean = np.mean([forward(params, X, 0.5, rng, True)[0] for _ in range(4000)], axis=0)
    close(mean, eval_logits, atol=0.15, label="logits averaged over 4000 dropout masks (the survivors must be divided by 1 - p)")

    linear = [{"W": params[0]["W"][:, :3].copy(), "b": np.zeros(3)}]
    out_train, _ = forward(linear, X, 0.5, np.random.default_rng(3), True)
    out_eval, _ = forward(linear, X, 0.5, None, False)
    close(out_train, out_eval, atol=1e-12, label="a network with no hidden layer (dropout must not touch inputs or logits)")


@check("smoothed_cross_entropy matches a hand-computed value and its gradient matches finite differences")
def test_loss(sub):
    fn = has(sub, "smoothed_cross_entropy")
    logits = np.array([[2.0, 0.0, 0.0]])
    loss, grad = fn(logits, np.array([0]), 0.1)
    p = np.exp(logits[0]) / np.exp(logits[0]).sum()
    target = np.array([0.9 + 0.1 / 3, 0.1 / 3, 0.1 / 3])
    close(loss, float(-(target * np.log(p)).sum()), atol=1e-10, label="loss at eps=0.1")
    close(grad, ((p - target) / 1)[None, :], atol=1e-10, label="gradient at eps=0.1")

    rng = np.random.default_rng(0)
    z, y = rng.normal(size=(5, 4)) * 3, rng.integers(0, 4, size=5)
    _, g = fn(z, y, 0.2)
    num = np.zeros_like(z)
    for idx in np.ndindex(*z.shape):
        zp, zm = z.copy(), z.copy()
        zp[idx] += 1e-6
        zm[idx] -= 1e-6
        num[idx] = (fn(zp, y, 0.2)[0] - fn(zm, y, 0.2)[0]) / 2e-6
    close(g, num, atol=1e-7, label="gradient against finite differences")
    big, _ = fn(np.array([[900.0, -900.0, 0.0]]), np.array([1]), 0.1)
    finite(np.array(big), label="loss with logits of size 900")


@check("backprop through the whole network matches finite differences for every weight and bias")
def test_backprop(sub):
    loss_and_grads = has(sub, "loss_and_grads")
    params, X, y = _tiny()
    loss, grads = loss_and_grads(params, X, y, 0.1, 0.0, None, False)
    if len(grads) != 3:
        fail(f"expected 3 gradient dicts, got {len(grads)}")
    for i in range(3):
        for key in ("W", "b"):
            shape(grads[i][key], params[i][key].shape, label=f"grad of layer {i} {key}")
            err = _rel(grads[i][key], _numeric(sub, params, X, y, 0.1, i, key))
            if err > 1e-6:
                fail(f"layer {i} {key}: relative error against finite differences is {err:.2e}, expected below 1e-6")


@check("Backprop with dropout reuses the forward pass's mask (the gradient is exact for a fixed mask)")
def test_backprop_dropout(sub):
    loss_and_grads = has(sub, "loss_and_grads")
    params, X, y = _tiny(sizes=(6, 8, 7, 3))
    _, grads = loss_and_grads(params, X, y, 0.1, 0.4, np.random.default_rng(5), True)
    for i in range(3):
        for key in ("W", "b"):
            num = _numeric(sub, params, X, y, 0.1, i, key, p_drop=0.4, seed=5)
            err = _rel(grads[i][key], num)
            if err > 1e-6:
                fail(f"layer {i} {key} with dropout: relative error {err:.2e}, expected below 1e-6. "
                     "The backward pass must use the same mask as the forward pass, and divide by 1 - p.")


@check("gradient_check reports a tiny error for correct gradients and a large one for a broken gradient")
def test_gradient_check(sub):
    gradient_check = has(sub, "gradient_check")
    params, X, y = _tiny(seed=3)
    good = gradient_check(params, X, y, eps=0.1)
    if not good < 1e-6:
        fail(f"gradient_check on backprop's own gradients returned {good:.2e}, expected below 1e-6")
    _, grads = sub.loss_and_grads(params, X, y, 0.1, 0.0, None, False)
    bad = copy.deepcopy(grads)
    bad[0]["W"] = bad[0]["W"] * 1.5
    got = gradient_check(params, X, y, eps=0.1, grads=bad)
    if not got > 0.05:
        fail(f"gradient_check on gradients scaled by 1.5 returned {got:.3g}, expected above 0.05 "
             "(is it comparing the supplied `grads`?)")
    before = copy.deepcopy(params)
    gradient_check(params, X, y, eps=0.1)
    for a, b in zip(before, params):
        close(a["W"], b["W"], atol=0.0, label="params after gradient_check (put each nudged entry back)")


@check("adamw_update takes the first Adam step, decays weights but not biases, and leaves its inputs alone")
def test_adamw_update(sub):
    adamw_update = has(sub, "adamw_update")
    params = [{"W": np.array([[1.0, -2.0]]), "b": np.array([0.5, 0.5])}]
    grads = [{"W": np.array([[0.3, -0.1]]), "b": np.array([0.0, 0.0])}]
    states = [{"W": {}, "b": {}}]
    snapshot = copy.deepcopy(params)
    new, new_states = adamw_update(params, grads, states, lr=0.1, weight_decay=0.5)
    # The first Adam step moves each entry by about lr * sign(grad); decay then shrinks W by lr * wd * w.
    close(new[0]["W"], np.array([[1.0 - 0.1 - 0.1 * 0.5 * 1.0, -2.0 + 0.1 + 0.1 * 0.5 * 2.0]]), atol=1e-6, label="W after one step")
    close(new[0]["b"], np.array([0.5, 0.5]), atol=1e-12, label="b with zero gradient (biases get no decay)")
    close(params[0]["W"], snapshot[0]["W"], atol=0.0, label="the input params")
    if new_states[0]["W"].get("t") != 1:
        fail(f"expected the state to count 1 step, got {new_states[0]['W']}")
    new2, _ = adamw_update(new, grads, new_states, lr=0.1, weight_decay=0.5)
    if np.allclose(new2[0]["W"], new[0]["W"]):
        fail("a second step with the returned states did not move W")


@check("lr_schedule warms up linearly, peaks, then decays along a cosine to zero")
def test_lr_schedule(sub):
    lr_schedule = has(sub, "lr_schedule")
    close(lr_schedule(0, 100, 1000, 0.01), 0.0, atol=1e-12, label="step 0")
    close(lr_schedule(50, 100, 1000, 0.01), 0.005, atol=1e-12, label="halfway through warmup")
    close(lr_schedule(100, 100, 1000, 0.01), 0.01, atol=1e-12, label="end of warmup")
    close(lr_schedule(550, 100, 1000, 0.01), 0.005, atol=1e-9, label="halfway through the decay")
    close(lr_schedule(1000, 100, 1000, 0.01), 0.0, atol=1e-9, label="the final step")
    rates = np.array([lr_schedule(s, 100, 1000, 0.01) for s in range(1000)])
    if np.any(np.diff(rates[100:]) > 1e-15):
        fail("after warmup the rate must never rise")


@check("train stops after `patience` epochs without improvement and returns the best epoch's parameters")
def test_train_early_stopping(sub):
    train = has(sub, "train")
    predict = has(sub, "predict")
    data = sub.make_data(1, n_train=300, n_val=300, n_test=100)
    X_val, y_val = data[2], data[3]
    params, history = train(data, hidden=(64, 32), epochs=150, peak_lr=1e-2, p_drop=0.0,
                            smoothing=0.0, patience=6, seed=0)
    for key in ("train_loss", "val_loss", "val_acc", "lr", "best_epoch"):
        if key not in history:
            fail(f"history is missing {key!r}; it has {sorted(history)}")
    n_run = len(history["val_loss"])
    best = history["best_epoch"]
    if n_run >= 150:
        fail("training ran all 150 epochs with patience 6, but a network this size overfits 300 examples long before that")
    if best != int(np.argmin(history["val_loss"])):
        fail(f"best_epoch is {best}, but the lowest validation loss is at epoch {int(np.argmin(history['val_loss']))}")
    if n_run - 1 - best != 6:
        fail(f"training should stop exactly 6 epochs after the best one: best epoch {best}, ran {n_run} epochs")
    logits, _ = sub.forward(params, X_val, training=False)
    loss = sub.smoothed_cross_entropy(logits, y_val, 0.0)[0]
    close(loss, min(history["val_loss"]), atol=1e-9,
          label="validation loss of the returned parameters (return the best epoch's weights, not the last epoch's)")
    close(float((predict(params, X_val) == y_val).mean()), history["val_acc"][best], atol=1e-12, label="val accuracy at the best epoch")


@check("train follows the warmup-plus-cosine schedule and predict ignores dropout")
def test_train_schedule_and_predict(sub):
    train = has(sub, "train")
    predict = has(sub, "predict")
    data = sub.make_data(2, n_train=200, n_val=100, n_test=100)
    params, history = train(data, hidden=(16,), epochs=8, patience=100, peak_lr=3e-3, p_drop=0.3, seed=0)
    if len(history["val_loss"]) != 8:
        fail(f"with patience 100, expected all 8 epochs, got {len(history['val_loss'])}")
    if not history["lr"][-1] < 0.02 * 3e-3:
        fail(f"the last epoch's final learning rate is {history['lr'][-1]:.2e}; a cosine schedule should be near 0 by then")
    if max(history["lr"]) > 3e-3 * (1 + 1e-9):
        fail(f"the learning rate reached {max(history['lr']):.2e}, above the peak 3e-3")
    X = data[4]
    first, second = predict(params, X), predict(params, X)
    shape(first, (len(X),), label="predictions")
    if not np.array_equal(first, second):
        fail("predict returned different labels on two calls, so dropout is still on at evaluation")
    logits, _ = sub.forward(params, X, training=False)
    if not np.array_equal(first, logits.argmax(axis=1)):
        fail("predict disagrees with the argmax of the evaluation-mode logits")


@check("End to end: the network reads unseen noisy digits with at least 95.5% accuracy within the time budget")
def test_end_to_end_accuracy(sub):
    for seed in (0, 3):
        data = sub.make_data(seed)
        X_test, y_test = data[4], data[5]
        start = time.time()
        params, history = sub.train(data, seed=seed)
        elapsed = time.time() - start
        acc = float((sub.predict(params, X_test) == y_test).mean())
        if acc < 0.955:
            fail(f"seed {seed}: test accuracy {acc:.3f}, expected at least 0.955 (best epoch {history['best_epoch']}, "
                 f"final validation accuracy {history['val_acc'][history['best_epoch']]:.3f})")
        if elapsed > 8.0:
            fail(f"seed {seed}: training took {elapsed:.1f}s, expected under 8s")


@check("End to end: the network beats a linear softmax baseline by at least 5 points of test accuracy")
def test_beats_baseline(sub):
    for seed in (1, 4):
        data = sub.make_data(seed)
        X_test, y_test = data[4], data[5]
        mlp, _ = sub.train(data, seed=seed)
        linear, _ = sub.train(data, hidden=(), p_drop=0.0, seed=seed)
        if len(linear) != 1:
            fail(f"hidden=() should give one layer, got {len(linear)}")
        acc_mlp = float((sub.predict(mlp, X_test) == y_test).mean())
        acc_lin = float((sub.predict(linear, X_test) == y_test).mean())
        if acc_lin < 0.8:
            fail(f"seed {seed}: the linear baseline reached only {acc_lin:.3f}; it should learn most of the task")
        if acc_mlp - acc_lin < 0.05:
            fail(f"seed {seed}: the network reached {acc_mlp:.3f} and the linear model {acc_lin:.3f}; "
                 "expected the network to lead by at least 0.05")
