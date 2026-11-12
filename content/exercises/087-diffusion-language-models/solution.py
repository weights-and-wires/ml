"""Masked diffusion for text: forward masking, the 1/t loss, and the sampler."""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def forward_mask(tokens, t, rng, mask_id):
    """Mask each token independently with probability t."""
    tokens = np.asarray(tokens)
    mask = rng.uniform(size=tokens.shape) < t
    return np.where(mask, mask_id, tokens), mask


def masked_diffusion_loss(logits, targets, mask, t):
    """Cross-entropy on masked positions, weighted by 1/t, divided by L."""
    logp = np.log(softmax(logits))
    picked = logp[np.arange(len(targets)), targets]
    return float(-np.sum(mask * picked) / (t * len(targets)))


def masked_diffusion_loss_grad(logits, targets, mask, t):
    """Gradient of masked_diffusion_loss with respect to logits."""
    p = softmax(logits)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(targets)), targets] = 1.0
    return -(mask[:, None] * (onehot - p)) / (t * len(targets))


def num_unmasked_after(length, step, n_steps):
    """Tokens that are unmasked after `step` of `n_steps` (1-based)."""
    return (length * step) // n_steps


def low_confidence_remask_step(x, logits, mask_id, n_unmasked):
    """Fill every masked slot, then keep the `n_unmasked` most confident."""
    x = np.asarray(x)
    probs = softmax(logits)
    pred = probs.argmax(axis=-1)
    conf = probs.max(axis=-1)
    masked = np.flatnonzero(x == mask_id)
    n_commit = max(0, n_unmasked - (len(x) - len(masked)))
    order = masked[np.argsort(-conf[masked], kind="stable")]
    out = x.copy()
    keep = order[:n_commit]
    out[keep] = pred[keep]
    return out


def random_remask_step(x, logits, mask_id, n_unmasked, rng):
    """Same as low_confidence_remask_step, but commit a random subset."""
    x = np.asarray(x)
    pred = logits.argmax(axis=-1)
    masked = np.flatnonzero(x == mask_id)
    n_commit = max(0, n_unmasked - (len(x) - len(masked)))
    keep = rng.permutation(masked)[:n_commit]
    out = x.copy()
    out[keep] = pred[keep]
    return out


def sample(logit_fn, length, n_steps, mask_id):
    """Start fully masked and run the low-confidence sampler."""
    x = np.full(length, mask_id)
    history = [x.copy()]
    for k in range(1, n_steps + 1):
        x = low_confidence_remask_step(
            x, logit_fn(x), mask_id, num_unmasked_after(length, k, n_steps)
        )
        history.append(x.copy())
    return x, history


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    toks = np.array([2, 0, 1, 0, 2, 1])
    noisy, m = forward_mask(toks, 0.5, rng, mask_id=3)
    print("noisy:", noisy, "mask:", m.astype(int))
    print("unmasked after each of 4 steps:", [num_unmasked_after(6, k, 4) for k in range(1, 5)])
