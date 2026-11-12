"""Masked diffusion for text: forward masking, the 1/t loss, and the sampler.

Tokens are integers 0 to V-1. The mask token has the id `mask_id`, which is V,
so it never appears as a target and the model's logits have V columns.
"""

from __future__ import annotations

import numpy as np


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis. Given in full."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def forward_mask(tokens, t, rng, mask_id):
    """Mask each token independently with probability t.

    Args:
        tokens: Integer array of shape (L,).
        t: Masking probability in [0, 1].
        rng: A numpy Generator. Draw one uniform number per position from it.
        mask_id: The integer that stands for the mask token.

    Returns:
        (noisy, mask): `noisy` is `tokens` with masked positions replaced by
        `mask_id`. `mask` is a boolean array of shape (L,), True where masked.
    """
    raise NotImplementedError("write forward_mask")


def masked_diffusion_loss(logits, targets, mask, t):
    """Cross-entropy on masked positions only, weighted by 1/t, divided by L.

    Args:
        logits: Array of shape (L, V), the model's scores for every position.
        targets: Integer array of shape (L,), the clean tokens.
        mask: Boolean array of shape (L,), True where the token was masked.
        t: The masking probability that produced `mask`.

    Returns:
        A float: -(1 / (t * L)) * sum over masked i of log p(targets[i]).
    """
    raise NotImplementedError("write masked_diffusion_loss")


def masked_diffusion_loss_grad(logits, targets, mask, t):
    """Gradient of masked_diffusion_loss with respect to logits.

    Returns:
        An array of shape (L, V). Rows for unmasked positions are zero.
    """
    raise NotImplementedError("write masked_diffusion_loss_grad")


def num_unmasked_after(length, step, n_steps):
    """Tokens that are unmasked after `step` of `n_steps` (step counts from 1).

    Returns:
        An int, floor(length * step / n_steps). Use integer arithmetic.
    """
    raise NotImplementedError("write num_unmasked_after")


def low_confidence_remask_step(x, logits, mask_id, n_unmasked):
    """One sampler step: predict every masked slot, keep the most confident.

    Args:
        x: Integer array of shape (L,) holding tokens and `mask_id`.
        logits: Array of shape (L, V) from the model for this `x`.
        mask_id: The mask token id.
        n_unmasked: How many positions must be unmasked when the step ends.

    Returns:
        A new array of shape (L,). Positions that were already unmasked keep
        their tokens. Among the masked positions, the most confident
        predictions (largest softmax probability of the argmax token) are
        filled in, and the rest stay `mask_id`. Break ties by lower index.
    """
    raise NotImplementedError("write low_confidence_remask_step")


def sample(logit_fn, length, n_steps, mask_id):
    """Start fully masked and run n_steps low-confidence steps.

    Args:
        logit_fn: Function from x of shape (L,) to logits of shape (L, V).
        length: Number of tokens L.
        n_steps: Number of steps N.
        mask_id: The mask token id.

    Returns:
        (x, history): the final tokens, and a list of N + 1 arrays holding x
        before the first step and after each step.
    """
    raise NotImplementedError("write sample")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    toks = np.array([2, 0, 1, 0, 2, 1])
    noisy, m = forward_mask(toks, 0.5, rng, mask_id=3)
    print("noisy:", noisy, "mask:", m.astype(int))
    print("unmasked after each of 4 steps:", [num_unmasked_after(6, k, 4) for k in range(1, 5)])
