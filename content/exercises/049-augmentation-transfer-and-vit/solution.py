"""Augmentation, transfer, depthwise-separable convolution, and patch embedding.

You write ten functions, in the order the checks test them:
`random_crop_flip`, `mixup`, `separable_params`, `depthwise_separable_conv`,
`patchify`, `patch_embed`, `frozen_features`, `fit_head`, `head_predict`,
and `apply_update`.
"""

import numpy as np


def random_crop_flip(x: np.ndarray, rng: np.random.Generator, pad: int = 2) -> np.ndarray:
    """Pad with zeros, crop back to the original size at a random offset, and maybe flip.

    Draw from rng in exactly this order: top = rng.integers(0, 2 * pad + 1),
    left = rng.integers(0, 2 * pad + 1), then flip = rng.random() < 0.5.
    The crop is x_padded[top:top + H, left:left + W]. When flip is True,
    reverse the columns of the crop.

    Args:
        x: One image, shape (H, W).
        rng: A numpy Generator.
        pad: Zero padding added on every side.

    Returns:
        The augmented image, shape (H, W). The input is not modified.
    """
    h, w = x.shape
    padded = np.pad(x, pad)
    top = rng.integers(0, 2 * pad + 1)
    left = rng.integers(0, 2 * pad + 1)
    flip = rng.random() < 0.5
    out = padded[top:top + h, left:left + w]
    return out[:, ::-1].copy() if flip else out.copy()


def mixup(x: np.ndarray, y: np.ndarray, lam: float, perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blend each example with a partner, and blend the labels the same way.

    Args:
        x: Inputs, shape (N, ...).
        y: One-hot labels, shape (N, K).
        lam: Weight on the example itself, between 0 and 1.
        perm: Integer array of length N. Example i's partner is example perm[i].

    Returns:
        (x_mix, y_mix) with x_mix = lam * x + (1 - lam) * x[perm], and y_mix
        built from y with the same weights.
    """
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]


def separable_params(c_in: int, c_out: int, k: int) -> int:
    """Weights of a depthwise-separable layer, without biases.

    A depthwise step has one k by k filter per input channel. A pointwise step
    is a 1 by 1 convolution from c_in channels to c_out channels.

    Returns:
        The total number of weights, as an int.
    """
    return c_in * k * k + c_in * c_out


def depthwise_separable_conv(x: np.ndarray, dw: np.ndarray, pw: np.ndarray) -> np.ndarray:
    """A depthwise convolution followed by a pointwise convolution, no padding, stride 1.

    Args:
        x: Input, shape (C, H, W).
        dw: Depthwise filters, shape (C, k, k). Filter c sees only channel c.
        pw: Pointwise weights, shape (C_out, C).

    Returns:
        Output of shape (C_out, H - k + 1, W - k + 1).
    """
    c, h, w = x.shape
    k = dw.shape[1]
    ho, wo = h - k + 1, w - k + 1
    mid = np.zeros((c, ho, wo))
    for i in range(ho):
        for j in range(wo):
            mid[:, i, j] = np.sum(x[:, i:i + k, j:j + k] * dw, axis=(1, 2))
    return np.tensordot(pw, mid, axes=(1, 0))


def patchify(x: np.ndarray, p: int) -> np.ndarray:
    """Cut an image into non-overlapping p by p patches and flatten each one.

    Patches go in row-major order (left to right, then top to bottom). Each
    patch is flattened in channel-first order, that is, the (C, p, p) block
    read row by row.

    Args:
        x: Image, shape (C, H, W), with H and W divisible by p.
        p: Patch side length.

    Returns:
        Array of shape ((H // p) * (W // p), C * p * p).
    """
    c, h, w = x.shape
    gh, gw = h // p, w // p
    t = x.reshape(c, gh, p, gw, p).transpose(1, 3, 0, 2, 4)
    return t.reshape(gh * gw, c * p * p)


def patch_embed(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Turn each patch into a token with one shared linear map.

    The patch side is p = kernel.shape[2]. The result must equal a convolution
    with stride p whose output grid has been flattened to a list of tokens.

    Args:
        x: Image, shape (C, H, W).
        kernel: Shape (D, C, p, p). The weights of D filters.
        bias: Shape (D,).

    Returns:
        Tokens, shape (number of patches, D).
    """
    d = kernel.shape[0]
    p = kernel.shape[2]
    return patchify(x, p) @ kernel.reshape(d, -1).T + bias


def frozen_features(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Features from a frozen extractor: flatten each image, multiply by W, apply ReLU.

    Args:
        x: Images, shape (N, H, W).
        W: Frozen extractor weights, shape (H * W, D).

    Returns:
        Features, shape (N, D).
    """
    return np.maximum(x.reshape(len(x), -1) @ W, 0.0)


def fit_head(F: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Fit a linear head by ridge regression on targets t = 2 * y - 1.

    Append a column of ones to F for the bias. Minimise
    mean((F1 @ w - t) ** 2) + lam * sum(w ** 2), where the penalty covers every
    entry of w, bias included.

    Args:
        F: Features, shape (N, D).
        y: Labels in {0, 1}, shape (N,).
        lam: Ridge strength, positive.

    Returns:
        Head weights w, shape (D + 1,), bias last.
    """
    f1 = np.hstack([F, np.ones((len(F), 1))])
    t = 2.0 * y - 1.0
    n, d = f1.shape
    return np.linalg.solve(f1.T @ f1 / n + lam * np.eye(d), f1.T @ t / n)


def head_predict(F: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Predict labels in {0, 1}: 1 when the score F1 @ w is above 0.

    Args:
        F: Features, shape (N, D).
        w: Head weights from fit_head, shape (D + 1,).

    Returns:
        Integer array of shape (N,).
    """
    f1 = np.hstack([F, np.ones((len(F), 1))])
    return (f1 @ w > 0).astype(int)


def apply_update(params: dict, grads: dict, lr: float, frozen: set) -> dict:
    """One gradient step that leaves frozen parameters alone.

    Args:
        params: Maps names to arrays.
        grads: Maps the same names to gradients of the same shapes.
        lr: Learning rate.
        frozen: Names that must not change.

    Returns:
        A new dict. A frozen name maps to its unchanged array, and every other
        name maps to params[name] - lr * grads[name]. The inputs are not modified.
    """
    return {k: (v.copy() if k in frozen else v - lr * grads[k]) for k, v in params.items()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    img = np.arange(16, dtype=float).reshape(4, 4)
    print("augmented:\n", random_crop_flip(img, rng, pad=1))
    x = np.arange(32, dtype=float).reshape(2, 4, 4)
    print("patchify shape:", patchify(x, 2).shape)
    print("separable params at 64 -> 64, k=3:", separable_params(64, 64, 3), "versus", 64 * 64 * 9)
