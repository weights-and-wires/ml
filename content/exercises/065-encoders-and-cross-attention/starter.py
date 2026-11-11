"""Masks, masked-language-model pretraining, cross-attention, and span corruption.

Boolean masks mean "may attend": True is allowed, False is blocked.
"""

import numpy as np


def padding_mask(lengths, T: int) -> np.ndarray:
    """Return a (B, T) boolean array that is True at real tokens and False at padding.

    Args:
        lengths: Sequence of B integers, the number of real tokens per sequence.
        T: Padded sequence length.

    Returns:
        Boolean array of shape (B, T) where entry [b, t] is True when t < lengths[b].
    """
    raise NotImplementedError("padding_mask")


def attention_allowed(pad: np.ndarray, causal: bool) -> np.ndarray:
    """Return which (query, key) pairs may attend, for a batch of padded sequences.

    A key may be read only if it is a real token. When causal is True, a key
    may also be read only if its position is at or before the query's.
    The padding mask acts on the key axis, so a padded query still has keys.

    Args:
        pad: Boolean array of shape (B, T), True at real tokens.
        causal: True for a decoder mask, False for a bidirectional encoder mask.

    Returns:
        Boolean array of shape (B, T, T), indexed [batch, query, key].
    """
    raise NotImplementedError("attention_allowed")


def masked_softmax(scores: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Softmax over the last axis that gives blocked entries exactly zero weight.

    Args:
        scores: Array of shape (..., n).
        allowed: Boolean array of the same shape, True where an entry may take weight.
            Every row has at least one True.

    Returns:
        Array of the same shape whose rows sum to 1 and are 0 where allowed is False.
    """
    raise NotImplementedError("masked_softmax")


def cross_attention(q_dec, k_enc, v_enc, pad_mask):
    """Scaled dot-product attention from decoder queries to encoder keys and values.

    Args:
        q_dec: Array of shape (T_dec, d), one query per decoder position.
        k_enc: Array of shape (T_enc, d), one key per encoder position.
        v_enc: Array of shape (T_enc, d_v), one value per encoder position.
        pad_mask: Boolean array of shape (T_enc,), True at real encoder tokens.

    Returns:
        A pair (output, weights). output has shape (T_dec, d_v). weights has shape
        (T_dec, T_enc), rows sum to 1, and columns at padded keys are 0.
    """
    raise NotImplementedError("cross_attention")


def mask_tokens(ids, rng, mask_id: int, vocab: int, rate: float = 0.15):
    """Corrupt token ids for masked-language-model pretraining.

    Each position is chosen independently with probability rate. A chosen
    position becomes mask_id with probability 0.8, a uniformly random id in
    range(vocab) with probability 0.1, and stays unchanged with probability 0.1.
    Positions that are not chosen never change.

    Args:
        ids: Integer array of any shape.
        rng: A numpy Generator.
        mask_id: The id of the mask token, outside range(vocab).
        vocab: Number of ordinary token ids.
        rate: Probability that a position is chosen.

    Returns:
        A pair (corrupted, chosen). corrupted is an integer array shaped like ids.
        chosen is a boolean array shaped like ids, True where a position was chosen.
    """
    raise NotImplementedError("mask_tokens")


def masked_lm_loss(logits: np.ndarray, targets: np.ndarray, chosen: np.ndarray):
    """Mean cross-entropy over the chosen positions only, and its gradient.

    Args:
        logits: Array of shape (T, V).
        targets: Integer array of shape (T,), the original token at each position.
        chosen: Boolean array of shape (T,), True where the loss counts.

    Returns:
        A pair (loss, dlogits). loss is a float, the mean over chosen positions
        (0.0 when none is chosen). dlogits has shape (T, V) and is 0 on rows that
        are not chosen.
    """
    raise NotImplementedError("masked_lm_loss")


def span_corrupt(ids, spans, sentinel_start: int):
    """Replace each span with one sentinel and build the matching target.

    Sentinel k has id sentinel_start + k. The input keeps every token outside the
    spans and puts sentinel k where span k was. The target lists, for each span in
    order, sentinel k and then the span's tokens, and ends with one extra sentinel
    after the last span.

    Args:
        ids: Sequence of integer token ids.
        spans: List of (start, length) pairs, sorted and not overlapping.
        sentinel_start: Id of the first sentinel.

    Returns:
        A pair (inputs, targets) of integer numpy arrays.
    """
    raise NotImplementedError("span_corrupt")


def span_corruption_lengths(T: int, noise_density: float = 0.15, mean_span: float = 3.0):
    """Return (input_length, target_length, n_spans) for span corruption of T tokens.

    The number of corrupted tokens is round(T * noise_density). The number of
    spans is round(corrupted / mean_span). The input swaps each span for one
    sentinel. The target holds the corrupted tokens, one sentinel per span, and
    one closing sentinel.

    Args:
        T: Length of the original sequence.
        noise_density: Fraction of tokens that are corrupted.
        mean_span: Average span length.

    Returns:
        A tuple of three ints.
    """
    raise NotImplementedError("span_corruption_lengths")


if __name__ == "__main__":
    pad = padding_mask([3, 2], 4)
    print("padding mask:\n", pad.astype(int))
    print("decoder mask for sequence 0:\n", attention_allowed(pad, True)[0].astype(int))

    q = np.array([[1.0, 0.0]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    V = np.array([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
    out, w = cross_attention(q, K, V, np.array([True, True, False]))
    print("cross-attention weights:", np.round(w, 4), "output:", np.round(out, 4))

    ids = np.arange(20)
    corrupted, chosen = mask_tokens(ids, np.random.default_rng(0), mask_id=99, vocab=50)
    print("corrupted:", corrupted)
    print("chosen   :", chosen.astype(int))

    logits = np.array([[2.0, 0.5, -1.0], [0.0, 0.0, 4.0], [1.0, 1.0, 1.0]])
    loss, grad = masked_lm_loss(logits, np.array([0, 0, 2]), np.array([True, False, True]))
    print("masked-LM loss on chosen positions:", round(loss, 4))

    words = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    print("span corruption:", span_corrupt(words, [(2, 2), (8, 1)], sentinel_start=100))
    print("lengths for T = 512:", span_corruption_lengths(512))
