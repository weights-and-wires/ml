"""Checks for the augmentation, transfer, and patch embedding exercise.

Two plausible wrong implementations are caught on purpose:

* A patchify that flattens each patch channels last, that is, (p, p, C) instead
  of (C, p, p). It passes on one-channel images and fails the two-channel
  patchify check and the patch_embed check against a strided convolution.
* A mixup that blends the inputs but returns the labels unchanged. The mixup
  check expects the blended labels [0.75, 0.25] and [0.25, 0.75].
"""

import numpy as np

from mlcheck import almost, check, close, equal, fail, finite_difference, has, shape


def conv_valid_reference(x, w, stride=1):
    """Plain-loop convolution. x: (C, H, W), w: (D, C, k, k). Returns (D, Ho, Wo)."""
    c, h, wd = x.shape
    d, _, k, _ = w.shape
    ho, wo = (h - k) // stride + 1, (wd - k) // stride + 1
    out = np.zeros((d, ho, wo))
    for i in range(ho):
        for j in range(wo):
            win = x[:, i * stride:i * stride + k, j * stride:j * stride + k]
            out[:, i, j] = np.tensordot(w, win, axes=([1, 2, 3], [0, 1, 2]))
    return out


def make_bars(n, rng, size=8, noise=0.3):
    """Class 0 is a vertical bar, class 1 is a horizontal bar, both with noise.

    Both classes have the same total brightness, so a brightness threshold
    cannot tell them apart.
    """
    X = np.zeros((n, size, size))
    y = np.zeros(n, dtype=int)
    for i in range(n):
        y[i] = i % 2
        c = rng.integers(1, size - 1)
        if y[i] == 0:
            X[i, :, c] = 1.0
        else:
            X[i, c, :] = 1.0
    return X + noise * rng.normal(size=X.shape), y


@check("random_crop_flip follows the draw order: top, left, then flip, with zero padding")
def test_crop_flip_reference(sub):
    fn = has(sub, "random_crop_flip")
    x = np.arange(16, dtype=float).reshape(4, 4) + 1.0
    for seed in range(20):
        for pad in (0, 2):
            ref_rng = np.random.default_rng(seed)
            top = ref_rng.integers(0, 2 * pad + 1)
            left = ref_rng.integers(0, 2 * pad + 1)
            flip = ref_rng.random() < 0.5
            want = np.pad(x, pad)[top:top + 4, left:left + 4]
            if flip:
                want = want[:, ::-1]
            before = x.copy()
            got = fn(x, np.random.default_rng(seed), pad=pad)
            shape(got, (4, 4), label=f"output, seed {seed}, pad {pad}")
            close(got, want, atol=1e-12, label=f"output, seed {seed}, pad {pad}")
            close(x, before, atol=0, label="input image, which must not change")


@check("Crop and flip preserve what the label depends on: the bright mass stays inside the frame, and the outputs vary")
def test_crop_flip_label(sub):
    fn = has(sub, "random_crop_flip")
    x = np.zeros((10, 10))
    x[3:7, 3:7] = 1.0
    x[3, 3] = 5.0  # an asymmetric corner, so a flip changes the image
    mass = x.sum()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(80):
        out = fn(x, rng, pad=2)
        if abs(out.sum() - mass) > 1e-9:
            fail(f"total brightness changed from {mass} to {out.sum()}; the bright block left the frame")
        seen.add(out.tobytes())
    if len(seen) < 8:
        fail(f"expected at least 8 distinct augmented images in 80 draws, saw {len(seen)}")


@check("mixup blends the inputs and the labels with the same weights")
def test_mixup(sub):
    fn = has(sub, "mixup")
    x = np.array([[1.0, 0.0], [0.0, 2.0]])
    y = np.array([[1.0, 0.0], [0.0, 1.0]])
    perm = np.array([1, 0])
    x0, y0 = x.copy(), y.copy()
    xm, ym = fn(x, y, 0.75, perm)
    close(xm, np.array([[0.75, 0.5], [0.25, 1.5]]), atol=1e-12, label="mixed inputs")
    close(ym, np.array([[0.75, 0.25], [0.25, 0.75]]), atol=1e-12, label="mixed labels")
    close(x, x0, atol=0, label="x, which must not change")
    close(y, y0, atol=0, label="y, which must not change")
    xm, ym = fn(x, y, 1.0, perm)
    close(xm, x, atol=1e-12, label="inputs at lam = 1")
    close(ym, y, atol=1e-12, label="labels at lam = 1")
    rng = np.random.default_rng(0)
    xr = rng.normal(size=(6, 3, 3))
    yr = np.eye(4)[rng.integers(0, 4, size=6)]
    _, ymr = fn(xr, yr, 0.3, rng.permutation(6))
    close(ymr.sum(axis=1), np.ones(6), atol=1e-12, label="each mixed label row must still sum to 1")


@check("separable_params equals c_in*k*k + c_in*c_out, and its ratio to a full layer is 1/c_out + 1/k^2")
def test_separable_params(sub):
    fn = has(sub, "separable_params")
    equal(int(fn(64, 64, 3)), 4672, label="separable_params(64, 64, 3)")
    equal(int(fn(3, 8, 3)), 51, label="separable_params(3, 8, 3)")
    for c_in, c_out, k in [(16, 32, 3), (64, 128, 5), (8, 8, 1)]:
        full = c_in * c_out * k * k
        ratio = fn(c_in, c_out, k) / full
        almost(ratio, 1.0 / c_out + 1.0 / k ** 2, 1e-12, label=f"ratio at ({c_in}, {c_out}, {k})")


@check("depthwise_separable_conv equals a full convolution whose kernel is pw[o, c] * dw[c]")
def test_depthwise_conv(sub):
    fn = has(sub, "depthwise_separable_conv")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3, 7, 7))
    dw = rng.normal(size=(3, 3, 3))
    pw = rng.normal(size=(4, 3))
    full = pw[:, :, None, None] * dw[None, :, :, :]
    got = fn(x, dw, pw)
    shape(got, (4, 5, 5), label="output")
    close(got, conv_valid_reference(x, full), atol=1e-10, label="separable versus full convolution")
    # A pointwise row that picks channel 0 must ignore the other channels.
    pick = np.array([[1.0, 0.0, 0.0]])
    x2 = x.copy()
    x2[1:] = rng.normal(size=(2, 7, 7))
    close(fn(x, dw, pick), fn(x2, dw, pick), atol=1e-12, label="output when the pointwise row picks channel 0")


@check("patchify reads patches left to right, top to bottom, each flattened channel first")
def test_patchify(sub):
    fn = has(sub, "patchify")
    x1 = np.arange(16, dtype=float).reshape(1, 4, 4)
    want1 = np.array([[0, 1, 4, 5], [2, 3, 6, 7], [8, 9, 12, 13], [10, 11, 14, 15]], dtype=float)
    close(fn(x1, 2), want1, atol=0, label="one-channel 4 by 4 image, p = 2")
    x2 = np.arange(32, dtype=float).reshape(2, 4, 4)
    got = fn(x2, 2)
    shape(got, (4, 8), label="two-channel patches")
    close(got[0], np.array([0, 1, 4, 5, 16, 17, 20, 21], dtype=float), atol=0, label="patch 0, channel first")
    close(got[1], np.array([2, 3, 6, 7, 18, 19, 22, 23], dtype=float), atol=0, label="patch 1, channel first")


@check("patch_embed equals a convolution with stride p, flattened to tokens")
def test_patch_embed(sub):
    fn = has(sub, "patch_embed")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 6, 6))
    kernel = rng.normal(size=(5, 3, 2, 2))
    bias = rng.normal(size=5)
    conv = conv_valid_reference(x, kernel, stride=2) + bias[:, None, None]
    want = conv.reshape(5, -1).T
    got = fn(x, kernel, bias)
    shape(got, (9, 5), label="tokens")
    close(got, want, atol=1e-10, label="tokens versus strided convolution")


@check("fit_head solves the ridge problem: its gradient is zero at the answer")
def test_fit_head(sub):
    fn = has(sub, "fit_head")
    rng = np.random.default_rng(0)
    F = rng.normal(size=(30, 4))
    y = (rng.random(30) < 0.5).astype(float)
    lam = 0.1
    w = fn(F, y, lam)
    shape(w, (5,), label="head weights")
    F1 = np.hstack([F, np.ones((30, 1))])
    t = 2 * y - 1

    def objective(v):
        return np.mean((F1 @ v - t) ** 2) + lam * np.sum(v ** 2)

    grad = finite_difference(objective, w)
    close(grad, np.zeros(5), atol=1e-5, label="gradient of the objective at your w")


@check("A frozen random extractor plus a trained head beats a brightness threshold on held-out bars")
def test_transfer(sub):
    feats = has(sub, "frozen_features")
    fit = has(sub, "fit_head")
    predict = has(sub, "head_predict")
    x = np.array([[[1.0, -2.0], [3.0, 0.0]]])
    W = np.eye(4)[:, :3]
    close(feats(x, W), np.array([[1.0, 0.0, 3.0]]), atol=1e-12, label="frozen_features on a 2 by 2 image")
    close(predict(np.array([[1.0], [-1.0]]), np.array([2.0, 0.0])), np.array([1, 0]), atol=0, label="head_predict")
    for seed in range(3):
        rng = np.random.default_rng(seed)
        Xtr, ytr = make_bars(200, rng)
        Xte, yte = make_bars(200, rng)
        W = np.random.default_rng(100 + seed).normal(size=(64, 256)) / 8.0
        w = fit(feats(Xtr, W), ytr, 1e-2)
        acc = float(np.mean(predict(feats(Xte, W), w) == yte))
        m = Xtr.mean(axis=(1, 2))
        th = np.median(m)
        base = max(np.mean((m > th) == ytr), np.mean((m <= th) == ytr))
        if acc < 0.9:
            fail(f"seed {seed}: held-out accuracy {acc:.3f}, expected at least 0.9 (a brightness threshold reaches {base:.3f})")


@check("apply_update changes only the unfrozen parameters and leaves its inputs alone")
def test_apply_update(sub):
    fn = has(sub, "apply_update")
    params = {"conv": np.ones((2, 2)), "head": np.array([1.0, 2.0])}
    grads = {"conv": np.full((2, 2), 5.0), "head": np.array([1.0, 1.0])}
    before = {k: v.copy() for k, v in params.items()}
    new = fn(params, grads, 0.5, {"conv"})
    close(new["conv"], np.ones((2, 2)), atol=0, label="frozen conv")
    close(new["head"], np.array([0.5, 1.5]), atol=1e-12, label="head after one step")
    for k in params:
        close(params[k], before[k], atol=0, label=f"params['{k}'], which must not change")
    new = fn(params, grads, 0.5, set())
    close(new["conv"], np.full((2, 2), -1.5), atol=1e-12, label="conv when nothing is frozen")
